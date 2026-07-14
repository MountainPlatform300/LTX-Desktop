"""Remote path conventions and trainer command lines.

Pure helpers (no side effects, no state mutation) shared by the LoRA
runner. They encode the deterministic on-remote directory layout and
build the exact `caption_videos.py` / `process_dataset.py` / `train.py`
invocations from the official LTX-2 trainer, so the provider targets
(`TrainerTarget`) stay generic "run this command" plumbing.

Remote layout (rooted at the configured `workspace_dir`, e.g. /workspace):

    {workspace_dir}/
      ltx-trainer/                      # LTX-2 monorepo clone (provisioned)
        packages/ltx-trainer/          # trainer package: scripts/ + uv project
      datasets/{dataset_id}/
        clips/                          # uploaded source clips
        dataset.json                    # caption + media_path metadata
        .precomputed/                   # process_dataset.py output (latents)
      configs/{training_id}.yaml        # generated training config
      outputs/{training_id}/            # train.py output dir
        lora_weights.safetensors        # the artifact we download

All commands run with `workdir` = the trainer package dir (see
`trainer_workdir`) so `uv run python scripts/...` resolves and the uv
workspace finds `ltx-core`. Paths passed to the scripts are absolute.

Requirements on the remote host: the LTX-2 trainer cloned at
`{workspace_dir}/ltx-trainer` with deps installed, plus a local LTX-2
checkpoint and Gemma text encoder at the paths configured in settings.
"""

from __future__ import annotations

import posixpath
import re
import shlex

TRAINER_REPO_DIRNAME = "ltx-trainer"
# Where provisioning downloads the base checkpoint + text encoder, under
# the workspace dir so a mounted network volume caches them across pods.
MODELS_SUBDIR = "models"
# The Gemma text encoder is a whole-repo download into its own directory;
# the LTX-2 checkpoint is a single file dropped alongside it.
TEXT_ENCODER_DIRNAME = "gemma-text-encoder"
# The unified LTX trainer (https://github.com/Lightricks/LTX-2, released
# 2026-06-17) ships as a uv-workspace member inside the monorepo, not at the
# repo root: the trainer's `pyproject.toml` and `scripts/*.py` live under this
# subdirectory. `uv sync` and every `uv run python scripts/...` invocation must
# run from here so the workspace resolves the sibling `ltx-core` package.
TRAINER_PACKAGE_SUBDIR = "packages/ltx-trainer"
LORA_WEIGHTS_FILENAME = "lora_weights.safetensors"
# Idempotency marker dropped at the end of a successful bootstrap so a
# reused pod (or a mounted network volume) skips the install/download on
# subsequent runs. Lives at the workspace root next to the trainer repo.
PROVISION_MARKER_FILENAME = ".ltx-provisioned"
# Separate marker dropped once the trainer's torch install has been pinned to
# the cu128 PyTorch index. Kept distinct from the main provision marker so an
# already-provisioned local workspace (set up before the Blackwell fix) can be
# upgraded in place — patch pyproject + re-sync — without re-cloning the trainer
# or re-downloading the multi-GB checkpoint.
PROVISION_TORCH_INDEX_MARKER_FILENAME = ".ltx-torch-cu128"
# uv is installed *into the volume* (not the default ~/.local/bin on the
# ephemeral container disk) so it survives pod teardown — a reused network
# volume keeps its uv binary and doesn't need a re-install on every fresh pod.
UV_BIN_SUBDIR = "bin"

# The PyTorch CUDA 12.8 wheel index — the only PyTorch build that ships sm_120
# (Blackwell; RTX 5090) kernels. PyPI's torch wheels top out at sm_90, so on a
# 5090 optimum-quanto's kernels launch on unsupported compute kernels. Routing
# torch/torchvision/torchaudio to this index (pinned via `[tool.uv.sources]` in
# the trainer's pyproject) makes `uv sync` resolve a sm_120-capable torch. The
# trainer's `torchcodec<0.10` pin keeps torch on the 2.9 line, so cu128's
# `2.9.1+cu128` is selected — same torch version, just the CUDA 12.8 build
# (ABI-compatible with the locked torchcodec 0.9.x).
#
# This is necessary but not sufficient on Blackwell: optimum-quanto's own
# `quanto_cuda` extension must also JIT-build for sm_120, and that build needs
# gcc-14 (nvcc rejects the gcc-15 WSL ships by default). That toolchain fix
# lives in `_cache_env_exports` (PATH-wrapper to gcc-14) + `install_host_compiler`
# (apt-installs gcc-14 during local provisioning). Without it the extension
# build fails and the quantized matmul dies with `CUDA driver error: device not
# ready` even with cu128 torch.
TORCH_INDEX_NAME = "pytorch-cu128"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Idempotent Python patcher injected into the provision/upgrade shell script.
# Merges the cu128 PyTorch index into the trainer monorepo's workspace-root
# pyproject.toml: appends a `[[tool.uv.index]]` block (always safe — it's an
# array-of-tables, so a second block never clashes with the existing `pypi`
# one) and routes torch/torchvision/torchaudio to it via the existing
# `[tool.uv.sources]` table — inserting the entries right after that table
# header, because the LTX-2 root pyproject already declares `[tool.uv.sources]`
# for its workspace members (ltx-core / ltx-pipelines), so appending a second
# `[tool.uv.sources]` table would be an invalid duplicate. Run as:
#   python3 - <pyproject_path> <index_name> <index_url> <<'PYEOF'  ...  PYEOF
# The quoted heredoc keeps the body literal (no shell expansion), so the `\"`
# and `\'` escapes reach the patcher verbatim and it writes valid TOML.
_PYPROJECT_TORCH_INDEX_PATCHER = r'''import pathlib, sys
pyproject = pathlib.Path(sys.argv[1])
index_name = sys.argv[2]
index_url = sys.argv[3]
s = pyproject.read_text()
if index_name in s:
    sys.exit(0)
header = "[tool.uv.sources]\n"
entry = (
    "torch = [{ index = \"%s\", marker = \"sys_platform == \'linux\' or sys_platform == \'win32\'\" }]\n"
    "torchvision = [{ index = \"%s\", marker = \"sys_platform == \'linux\' or sys_platform == \'win32\'\" }]\n"
    "torchaudio = [{ index = \"%s\", marker = \"sys_platform == \'linux\' or sys_platform == \'win32\'\" }]\n"
) % (index_name, index_name, index_name)
if header in s:
    s = s.replace(header, header + entry, 1)
else:
    s = s.rstrip("\n") + "\n\n" + header + entry
s = s.rstrip("\n") + "\n\n[[tool.uv.index]]\n"
s += "name = \"%s\"\n" % index_name
s += "url = \"%s\"\n" % index_url
s += "explicit = true\n"
pyproject.write_text(s)
'''


def uv_bin_dir(workspace_dir: str) -> str:
    """Persistent install dir for the `uv` binary (`{workspace}/bin`)."""
    return f"{workspace_dir.rstrip('/')}/{UV_BIN_SUBDIR}"


def trainer_repo_dir(workspace_dir: str) -> str:
    """Clone target for the trainer monorepo (`{workspace}/ltx-trainer`)."""
    return f"{workspace_dir.rstrip('/')}/{TRAINER_REPO_DIRNAME}"


def trainer_workdir(workspace_dir: str) -> str:
    """Working directory the trainer scripts run from.

    The cloned repo is the LTX-2 monorepo; the trainer package lives in a
    subdirectory of it. All `uv sync` / `uv run python scripts/...` commands use
    this as their cwd so `scripts/*.py` resolve and the uv workspace can find
    `ltx-core`.
    """
    return f"{trainer_repo_dir(workspace_dir)}/{TRAINER_PACKAGE_SUBDIR}"


def provision_marker_path(workspace_dir: str) -> str:
    return f"{workspace_dir.rstrip('/')}/{PROVISION_MARKER_FILENAME}"


def provision_marker_value(repo_url: str, repo_ref: str) -> str:
    """Identity recorded after provisioning so stale workspaces self-invalidate."""
    return f"{repo_url}@{repo_ref}"


def ensure_ffmpeg_command() -> str:
    """Idempotent one-liner that installs system ffmpeg if the binary is absent.

    `provision_command` installs ffmpeg during a fresh provision, but
    provisioning is marker-gated and runs once per volume — so a workspace
    provisioned before this fix (or one whose marker survived on a reused
    network volume) never got it. Run this on the already-provisioned path too
    so `torchaudio.load(<video>)`'s ffmpeg backend can demux audio out of
    mp4/mov clips (see `provision_command` for the full rationale). Fast no-op
    where ffmpeg already exists.
    """
    return (
        "command -v ffmpeg >/dev/null 2>&1 "
        "|| { apt-get update && apt-get install -y --no-install-recommends ffmpeg; }"
    )


def patch_trainer_audio_fallback_command(workspace_dir: str) -> str:
    """Idempotently patch the trainer's `process_videos.py` so audio extraction
    doesn't depend on `torchaudio.load` (which is broken via torchcodec on
    cu128/WSL — torchaudio 2.9+ routes all I/O through torchcodec, whose
    bundled libav ABI doesn't match the system ffmpeg on WSL2, so
    `torchaudio.load(<mp4>)` raises with no fallback and every clip's audio is
    silently skipped → empty `audio_latents/`).

    Inserts a small monkeypatch right after `import torchaudio` that wraps
    `torchaudio.load`: on success it returns the original result unchanged
    (so RunPod, where torchcodec works, is unaffected), and on failure it
    extracts mono 48kHz f32le PCM via an `ffmpeg` subprocess and returns
    `(waveform, 48000)` — the trainer's `AudioProcessor` resamples to the audio
    VAE's sample rate, so 48kHz is fine. `ffmpeg`/`torch`/`numpy` are all
    already available (provisioning installs ffmpeg; the trainer imports
    torch + numpy as np).

    Best-effort: a missing file or no `import torchaudio` line exits 0 (the
    post-preprocess audio guard still reports a clear failure if audio latents
    come out empty). Idempotent via a marker comment so re-provision / repeated
    calls are no-ops.
    """
    script_path = f"{trainer_workdir(workspace_dir)}/scripts/process_videos.py"
    quoted_path = shlex.quote(script_path)
    # Quoted heredoc ('PYEOF') so the shell doesn't touch the python body; the
    # script path is baked in at build time. `set -e` is inherited, so the
    # caller appends `|| echo ...` to keep a patch failure non-fatal.
    return f"""python3 - <<'PYEOF'
import sys
p = {quoted_path!r}
try:
    s = open(p).read()
except FileNotFoundError:
    sys.exit(0)
MARK = "# --- LTX-Desktop: ffmpeg fallback for torchaudio.load"
if MARK in s:
    sys.exit(0)
if "import torchaudio" not in s:
    sys.exit(0)
BLOCK = '''# --- LTX-Desktop: ffmpeg fallback for torchaudio.load (torchcodec ABI breakage on cu128/WSL) ---
import subprocess as _ltx_subprocess
_orig_torchaudio_load = torchaudio.load
def _ltx_torchaudio_load(filepath, *args, **kwargs):
    try:
        return _orig_torchaudio_load(filepath, *args, **kwargs)
    except Exception:
        proc = _ltx_subprocess.run(
            ["ffmpeg", "-nostdin", "-i", str(filepath), "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
            capture_output=True,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise
        waveform = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.float32)).unsqueeze(0)
        return waveform, 48000
torchaudio.load = _ltx_torchaudio_load
# --- end LTX-Desktop patch ---
'''
s = s.replace("import torchaudio\\n", "import torchaudio\\n" + BLOCK, 1)
open(p, "w").write(s)
PYEOF"""


def torch_index_marker_path(workspace_dir: str) -> str:
    """Marker file recording that the trainer's torch has been pinned to the
    cu128 index. Distinct from `provision_marker_path` so an existing local
    workspace can be upgraded in place (see `ensure_torch_index_command`)."""
    return f"{workspace_dir.rstrip('/')}/{PROVISION_TORCH_INDEX_MARKER_FILENAME}"


def trainer_root_pyproject_path(workspace_dir: str) -> str:
    """The trainer monorepo's workspace-root `pyproject.toml`.

    The cloned repo is the LTX-2 monorepo and its uv workspace root is the repo
    root (the `pyproject.toml` declaring `[tool.uv.workspace]` +
    `[tool.uv.sources]`), NOT the `packages/ltx-trainer` member. uv reads
    `[tool.uv]` index/source config from that root, so the cu128 index + torch
    source routing must be patched here for `uv sync` (run from the member) to
    pick it up.
    """
    return f"{trainer_repo_dir(workspace_dir)}/pyproject.toml"


def _patch_pyproject_torch_index_lines(root_pyproject: str) -> list[str]:
    """Shell lines that idempotently patch the trainer root pyproject to route
    torch/torchvision/torchaudio through the cu128 index (see
    `_PYPROJECT_TORCH_INDEX_PATCHER`). Emit as a quoted heredoc so the patcher's
    quoting survives the shell untouched."""
    return [
        "python3 - "
        + shlex.quote(root_pyproject)
        + " "
        + shlex.quote(TORCH_INDEX_NAME)
        + " "
        + shlex.quote(TORCH_INDEX_URL)
        + " <<'PYEOF'",
        _PYPROJECT_TORCH_INDEX_PATCHER,
        "PYEOF",
    ]


def _cache_env_exports(workspace_dir: str) -> list[str]:
    """`export` lines pinning every cache + tmp dir to the volume.

    Covers the caches that otherwise default to ~/.cache or /tmp on the small
    container disk: HF blobs, uv/pip wheels, Triton + torch.inductor compile
    caches (runtime), and scratch. `XDG_CACHE_HOME` catches the long tail of
    tools that honor it. Applied to BOTH provisioning and every runtime command
    so the container disk never fills.
    """
    ws = workspace_dir.rstrip("/")
    cache = f"{ws}/.cache"
    bin_dir = uv_bin_dir(ws)
    hostcc = f"{ws}/.hostcc"
    return [
        # uv lives on the volume (see UV_BIN_SUBDIR): point its installer here so
        # the binary persists, and put it first on PATH so every command — this
        # provisioning script and all later caption/preprocess/train runs over a
        # fresh non-login SSH shell — resolves `uv` without sourcing a profile.
        f"export UV_INSTALL_DIR={shlex.quote(bin_dir)}",
        f'export PATH={shlex.quote(bin_dir)}:"$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
        f"export XDG_CACHE_HOME={shlex.quote(cache)}",
        f"export HF_HOME={shlex.quote(cache + '/huggingface')}",
        f"export HF_HUB_CACHE={shlex.quote(cache + '/huggingface/hub')}",
        f"export UV_CACHE_DIR={shlex.quote(cache + '/uv')}",
        f"export PIP_CACHE_DIR={shlex.quote(cache + '/pip')}",
        f"export TRITON_CACHE_DIR={shlex.quote(cache + '/triton')}",
        f"export TORCHINDUCTOR_CACHE_DIR={shlex.quote(cache + '/inductor')}",
        f"export TMPDIR={shlex.quote(ws + '/tmp')}",
        # Disable HF's Xet protocol: it keeps a full chunk cache *in addition*
        # to the reconstructed file (~2x disk per download), which overran the
        # volume. Classic download uses ~1x and a temp that's moved into place.
        "export HF_HUB_DISABLE_XET=1",
        # Reduce the CUDA caching allocator's reserved-but-unallocated waste and
        # fragmentation. int8-quanto quantizing the 22B model peaks right at the
        # 32GB-GPU ceiling (on an RTX 5090 it OOMs by ~128MB during `freeze()`);
        # `expandable_segments` — the fix PyTorch's own OOM message names —
        # reclaims enough headroom to clear it. Harmless / beneficial on larger
        # GPUs too, so it's applied to every runtime command. torch renamed the
        # var (PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF; the old one still
        # works but warns), so set both to cover old and new torch.
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        # optimum-quanto JIT-compiles its `quanto_cuda` extension with nvcc the
        # first time a quantized layer is frozen (int8 AND fp8 both pull it in,
        # not just sub-byte). nvcc's host-compiler check rejects gcc > 14, and
        # WSL's Ubuntu ships gcc-15 by default — so the build fails and the
        # quantized matmul falls back to a path that dies on Blackwell (sm_120)
        # with `CUDA driver error: device not ready`. This torch build's
        # cpp_extension does NOT honor CUDAHOSTCXX / -ccbin, so the robust fix is
        # to put gcc-14/g++-14/c++ symlinks FIRST on PATH (nvcc picks its host
        # compiler from PATH) and set CC/CXX for the non-CUDA host files. No-op
        # where gcc-14 isn't installed (e.g. RunPod images whose default gcc is
        # already CUDA-supported); local WSL provisioning installs gcc-14.
        (
            "if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; "
            f"then mkdir -p {shlex.quote(hostcc)} "
            f'&& ln -sf "$(command -v gcc-14)" {shlex.quote(hostcc + "/gcc")} '
            f'&& ln -sf "$(command -v g++-14)" {shlex.quote(hostcc + "/g++")} '
            f'&& ln -sf "$(command -v g++-14)" {shlex.quote(hostcc + "/c++")} '
            f'&& export PATH={shlex.quote(hostcc)}:"$PATH" '
            'CC="$(command -v gcc-14)" CXX="$(command -v g++-14)" '
            'CUDAHOSTCXX="$(command -v g++-14)"; fi'
        ),
        # Help torch's CUDA-extension builder locate the toolkit for that JIT build.
        "if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; fi",
    ]


def cache_env_prefix(workspace_dir: str) -> str:
    """`ulimit ...; export ...; mkdir -p ...; ` to prepend to a runtime remote
    command so its caches/temp land on the volume (not the container disk) and it
    has enough file descriptors."""
    ws = workspace_dir.rstrip("/")
    # Raise the open-file-descriptor limit: PyTorch dataloader workers share
    # tensors over FDs and the trainer opens many files, which overruns the
    # default soft limit (1024) — `OSError: [Errno 24] Too many open files`.
    # Best-effort (try high, fall back), bounded by the process's hard limit
    # (the local systemd unit sets a high LimitNOFILE; RunPod pods already allow
    # it).
    raise_nofile = "ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true"
    lines = (
        [raise_nofile]
        + _cache_env_exports(ws)
        + [f"mkdir -p {shlex.quote(ws + '/.cache')} {shlex.quote(ws + '/tmp')}"]
    )
    return "; ".join(lines) + "; "


def models_dir(workspace_dir: str) -> str:
    """Directory the base checkpoint + text encoder are downloaded into."""
    return f"{workspace_dir.rstrip('/')}/{MODELS_SUBDIR}"


def default_model_path(workspace_dir: str, checkpoint_filename: str) -> str:
    """Local path of the base LTX-2 checkpoint after provisioning.

    The checkpoint is a single file pulled from the (very large) model repo
    into `models/`, so the path is `{workspace}/models/{filename}`.
    """
    return f"{models_dir(workspace_dir)}/{checkpoint_filename}"


def default_text_encoder_path(workspace_dir: str) -> str:
    """Local directory of the Gemma text encoder after provisioning."""
    return f"{models_dir(workspace_dir)}/{TEXT_ENCODER_DIRNAME}"


def _hf_token_prefix(hf_token: str) -> str:
    # The token (when present) is passed via env on this line only — not
    # echoed to the job log, which only captures stdout/stderr.
    return f"HF_TOKEN={shlex.quote(hf_token)} " if hf_token else ""


def _hf_download_repo_command(repo_id: str, dest: str, hf_token: str) -> str:
    """Download an entire HF repo into `dest` (used for the Gemma encoder dir).

    `uvx` runs the HF CLI in an ephemeral env so we don't depend on the
    trainer's own deps including it. Uses the `hf` command — recent
    `huggingface_hub` removed the old `huggingface-cli` entry point.
    """
    return (
        f"{_hf_token_prefix(hf_token)}uvx --from huggingface_hub hf "
        f"download {shlex.quote(repo_id)} --local-dir {shlex.quote(dest)}"
    )


def _hf_download_file_command(
    repo_id: str, filename: str, dest_dir: str, hf_token: str
) -> str:
    """Download a single file from an HF repo into `dest_dir`.

    Critical for the base checkpoint: `Lightricks/LTX-2` is ~314 GB, but the
    trainer only needs one `.safetensors` file. A whole-repo download would
    pull the entire repo, so we fetch just the checkpoint. The HF CLI places
    it at `{dest_dir}/{filename}` (see `default_model_path`).
    """
    return (
        f"{_hf_token_prefix(hf_token)}uvx --from huggingface_hub hf "
        f"download {shlex.quote(repo_id)} {shlex.quote(filename)} "
        f"--local-dir {shlex.quote(dest_dir)}"
    )


def provision_command(
    *,
    workspace_dir: str,
    repo_url: str,
    repo_ref: str,
    marker_path: str,
    model_hf_repo: str = "",
    model_filename: str = "",
    model_path: str = "",
    text_encoder_hf_repo: str = "",
    text_encoder_path: str = "",
    hf_token: str = "",
    torch_cuda_index: str = "",
    install_host_compiler: bool = False,
) -> str:
    """Build the one-shot bootstrap script for a fresh remote workspace.

    Installs `uv`, ensures system `ffmpeg` (torchaudio's audio I/O backend
    needs it to demux audio from mp4/mov clips), clones/updates the LTX-2
    trainer at `{workspace_dir}/ltx-trainer`, runs `uv sync`, and — when the HF
    repos are configured — downloads the base checkpoint (a single file, since
    the LTX-2 repo is huge) and the Gemma text encoder (whole repo). Drops
    `marker_path` on success.

    When `torch_cuda_index` is set (e.g. ``"cu128"`` for local Blackwell/RTX
    5090 training), the trainer's root pyproject is patched to route
    torch/torchvision/torchaudio through that CUDA index *before* `uv sync`,
    so the sync resolves a sm_120-capable torch instead of PyPI's sm_90-only
    build. A second marker (`torch_index_marker_path`) is dropped so the
    in-place upgrade path can tell a cu128-pinned workspace from a legacy one.

    Pure: returns a shell string. The runner gates execution on the
    marker so this only runs once per pod/volume.
    """
    workspace = workspace_dir.rstrip("/")
    repo_dir = trainer_repo_dir(workspace)
    quoted_repo = shlex.quote(repo_dir)
    # `[provision] <step>` markers are echoed before each step so a hang or
    # timeout is diagnosable: the last marker in the remote log tells us exactly
    # which step stalled (clone / deps / model / encoder).
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(workspace)}",
        # Pin caches + tmp + the uv install dir to the volume so the multi-GB
        # downloads (torch/CUDA wheels, HF blobs) and the uv binary don't land on
        # the small ephemeral container disk. bash -c doesn't read profile, so set
        # them here explicitly (not only as pod env vars). This also puts
        # {workspace}/bin on PATH so the uv we install below is found.
        *_cache_env_exports(workspace),
        f"mkdir -p {shlex.quote(workspace + '/.cache')} {shlex.quote(workspace + '/tmp')} "
        f"{shlex.quote(uv_bin_dir(workspace))}",
        # Reclaim any stale Xet chunk cache left by a prior (failed) run with
        # Xet enabled — those chunks are dead weight now that Xet is off.
        f"rm -rf {shlex.quote(workspace + '/.cache/huggingface/xet')}",
        "echo '[provision] installing uv'",
        # UV_INSTALL_DIR (exported above) points the installer at the volume so
        # the binary survives pod teardown; skip if a reused volume already has it.
        "if ! command -v uv >/dev/null 2>&1; then "
        "curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
        "echo '[provision] cloning trainer'",
        f"if [ ! -d {quoted_repo}/.git ]; then "
        f"git clone --filter=blob:none --no-checkout "
        f"{shlex.quote(repo_url)} {quoted_repo}; "
        f"else git -C {quoted_repo} reset --hard "
        f"&& git -C {quoted_repo} clean -fd "
        f"&& git -C {quoted_repo} remote set-url origin {shlex.quote(repo_url)}; fi",
        f"git -C {quoted_repo} fetch --depth 1 origin {shlex.quote(repo_ref)}",
        f"git -C {quoted_repo} checkout --detach FETCH_HEAD",
        f"git -C {quoted_repo} reset --hard FETCH_HEAD",
    ]
    if install_host_compiler:
        # WSL Ubuntu ships gcc-15, which nvcc rejects (gcc > 14). Without
        # gcc-14 the optimum-quanto CUDA extension can't JIT-build and the
        # quantized matmul dies on Blackwell (sm_120) with `device not ready`.
        # Install gcc-14/g++-14 if missing (idempotent; a fast no-op where
        # already present). Skipped for RunPod, whose images ship a
        # CUDA-supported gcc by default. Runs before `uv sync` so the toolchain
        # is ready for the quanto extension's nvcc build during training.
        lines.append(
            "if ! command -v gcc-14 >/dev/null 2>&1 || ! command -v g++-14 >/dev/null 2>&1; "
            "then apt-get update && apt-get install -y gcc-14 g++-14; fi"
        )
    # Ensure a system ffmpeg is present for torchaudio's ffmpeg backend.
    # torchaudio 2.1+ no longer bundles FFmpeg libraries: its `ffmpeg` I/O
    # backend (the only one that can demux audio out of mp4/mov/mkv video
    # containers) loads libavformat/libavcodec/libavutil/libswresample from the
    # system. Without them `torchaudio.load(<video>)` falls back to the
    # `soundfile` backend, which uses libsndfile and CANNOT demux video
    # containers — so `process_dataset.py`'s auto audio extraction raises
    # inside `_extract_audio`, the exception is swallowed at debug level, every
    # clip is silently counted as "no audio track", and `audio_latents/` ends
    # up empty (the run then fails our post-preprocess audio guard). torchvision
    # reads video frames fine regardless because pyav bundles its own ffmpeg
    # libs — which is exactly why video latents cache but audio doesn't.
    # Installing the `ffmpeg` apt package provides the libs torchaudio needs, so
    # the official trainer's `torchaudio.load(video)` path works unchanged.
    # Idempotent (fast no-op where ffmpeg already exists); runs for both RunPod
    # and WSL before `uv sync`. RunPod CUDA images are Ubuntu-based with apt.
    lines.append("echo '[provision] ensuring system ffmpeg (torchaudio audio backend)'")
    lines.append(
        "if ! command -v ffmpeg >/dev/null 2>&1; "
        "then apt-get update && apt-get install -y --no-install-recommends ffmpeg; fi"
    )
    # Patch the trainer's `process_videos.py` so audio extraction falls back to
    # an `ffmpeg` subprocess when `torchaudio.load` raises. On cu128/WSL
    # torchaudio 2.9+ routes I/O through torchcodec, whose bundled libav ABI
    # doesn't match system ffmpeg — so `torchaudio.load(<mp4>)` raises even
    # though ffmpeg the binary is now present, and every clip's audio is
    # silently skipped. The monkeypatch is a no-op where `torchaudio.load`
    # succeeds (RunPod), so it's safe to apply everywhere. Best-effort: a patch
    # failure (e.g. upstream restructured the file) is non-fatal — the
    # post-preprocess audio guard still reports a clear failure. Idempotent via
    # a marker comment. Runs after the clone so `process_videos.py` exists.
    lines.append("echo '[provision] patching trainer audio fallback (ffmpeg subprocess)'")
    audio_patch = patch_trainer_audio_fallback_command(workspace).replace(
        "python3 - <<'PYEOF'",
        "python3 - <<'PYEOF' "
        "|| echo '[provision] trainer audio-fallback patch skipped (non-fatal)'",
        1,
    )
    # A heredoc terminator must be the only token on its line. Keep the
    # best-effort fallback on the invocation line; appending it after the whole
    # command would produce `PYEOF || ...`, causing Bash to consume every
    # remaining provisioning step as Python input.
    lines.append(audio_patch)
    if torch_cuda_index:
        # Blackwell (sm_120) needs a cu128 torch; patch the trainer's root
        # pyproject to route torch/torchvision/torchaudio through the cu128
        # index before `uv sync` resolves them. Idempotent — no-op if the
        # index is already routed (e.g. a re-provision over a patched repo).
        lines.append(
            f"echo '[provision] routing torch to {torch_cuda_index} index "
            "(Blackwell sm_120)'"
        )
        lines.extend(_patch_pyproject_torch_index_lines(trainer_root_pyproject_path(workspace)))
    lines.append("echo '[provision] installing dependencies (uv sync)'")
    lines.append(f"cd {shlex.quote(trainer_workdir(workspace))} && uv sync")
    if model_hf_repo and model_path:
        lines.append("echo '[provision] downloading model checkpoint'")
        if model_filename:
            # Single-file checkpoint download into the models dir (the repo
            # itself is hundreds of GB; we only want this one file).
            lines.append(
                _hf_download_file_command(
                    model_hf_repo,
                    model_filename,
                    posixpath.dirname(model_path.rstrip("/")),
                    hf_token,
                )
            )
        else:
            # Back-compat: whole-repo download into model_path.
            lines.append(_hf_download_repo_command(model_hf_repo, model_path, hf_token))
    if text_encoder_hf_repo and text_encoder_path:
        lines.append("echo '[provision] downloading text encoder'")
        lines.append(
            _hf_download_repo_command(text_encoder_hf_repo, text_encoder_path, hf_token)
        )
    lines.append("echo '[provision] done'")
    if torch_cuda_index:
        # Record that torch was pinned to the CUDA index so the in-place
        # upgrade path (`ensure_torch_index_command`) can skip a workspace this
        # provision already pinned.
        lines.append(f"touch {shlex.quote(torch_index_marker_path(workspace))}")
    lines.append(
        f"printf '%s\\n' {shlex.quote(provision_marker_value(repo_url, repo_ref))} "
        f"> {shlex.quote(marker_path)}"
    )
    return "\n".join(lines)


def ensure_torch_index_command(
    *, workspace_dir: str, marker_path: str, install_host_compiler: bool = False
) -> str:
    """In-place upgrade that pins a provisioned trainer's torch to cu128.

    For a workspace bootstrapped before the Blackwell fix — `uv sync` pulled
    PyPI's sm_90-only torch, whose kernels don't run on a 5090 — this patches
    the cloned trainer's root `pyproject.toml` to route torch/torchvision/
    torchaudio through the cu128 index and re-runs `uv sync`, which re-resolves
    (and re-downloads) the sm_120-capable torch wheels. The trainer's
    `torchcodec<0.10` pin keeps torch on the 2.9 line, so the swap is
    ABI-compatible with the already-installed torchcodec 0.9.x and the rest of
    the locked graph.

    Note: cu128 gets the sm_120 *kernels* in torch, but optimum-quanto's own
    `quanto_cuda` extension must also JIT-build for sm_120 — and that build
    needs gcc-14 (nvcc rejects the gcc-15 WSL ships). When
    `install_host_compiler` is set, install gcc-14/g++-14 here too so an
    upgraded-then-trained workspace isn't left without the toolchain. The
    per-command PATH-wrapper in `_cache_env_exports` then points nvcc at it.

    Pure: returns a shell string. The caller gates execution on `marker_path`
    (the torch-index marker) so this only runs once per workspace — and never
    for a fresh provision that already pinned cu128 via `provision_command`.
    """
    workspace = workspace_dir.rstrip("/")
    lines = [
        "set -euo pipefail",
        *_cache_env_exports(workspace),
        f"mkdir -p {shlex.quote(workspace + '/.cache')} {shlex.quote(workspace + '/tmp')}",
    ]
    if install_host_compiler:
        lines.append(
            "if ! command -v gcc-14 >/dev/null 2>&1 || ! command -v g++-14 >/dev/null 2>&1; "
            "then apt-get update && apt-get install -y gcc-14 g++-14; fi"
        )
    lines += [
        "echo '[torch-index] patching trainer pyproject for cu128 torch (Blackwell sm_120)'",
        *_patch_pyproject_torch_index_lines(trainer_root_pyproject_path(workspace)),
        "echo '[torch-index] re-syncing deps (uv sync) — downloads cu128 torch wheels'",
        f"cd {shlex.quote(trainer_workdir(workspace))} && uv sync",
        "echo '[torch-index] done'",
        f"touch {shlex.quote(marker_path)}",
    ]
    return "\n".join(lines)


def remote_slug(name: str | None, entity_id: str) -> str:
    """Human-readable, collision-free leaf for a remote dir.

    ``{sanitized-name}-{shortid}`` so someone browsing the GPU workspace can
    tell datasets / runs apart at a glance instead of seeing a wall of bare
    UUIDs, while the short id suffix keeps it unique and deterministic. Falls
    back to the bare id when no name is available (older callers / back-compat).
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-._").lower()[:40]
    short = entity_id.replace("-", "")[:8] or entity_id
    return f"{cleaned}-{short}" if cleaned else entity_id


# --- Stored-dir helpers ---------------------------------------------------
# Preprocessing / training resolve paths from the dir recorded in state at
# upload/start time (never recomputed from the id), so renaming a dataset or
# run after upload can never point a later step at the wrong folder.


def dataset_clips_dir_in(remote_dataset_dir: str) -> str:
    return f"{remote_dataset_dir.rstrip('/')}/clips"


def dataset_json_path_in(remote_dataset_dir: str) -> str:
    return f"{remote_dataset_dir.rstrip('/')}/dataset.json"


def precomputed_dir_in(remote_dataset_dir: str) -> str:
    # process_dataset.py writes `.precomputed/` next to the dataset.json.
    return f"{remote_dataset_dir.rstrip('/')}/.precomputed"


def precomputed_run_dir_in(remote_dataset_dir: str, preprocessed_id: str) -> str:
    """Immutable cache root for one preprocessing snapshot."""
    return f"{remote_dataset_dir.rstrip('/')}/.precomputed-{preprocessed_id}"


def lora_weights_path_in(remote_output_dir: str) -> str:
    return f"{remote_output_dir.rstrip('/')}/{LORA_WEIGHTS_FILENAME}"


# The trainer writes adapter weights to `{output_dir}/checkpoints/` as
# `lora_weights_step_{step:05d}.safetensors` (one per checkpoint interval, plus a
# final save at the last step), NOT a single file at the output root.
LORA_CHECKPOINTS_SUBDIR = "checkpoints"


def lora_checkpoints_dir_in(remote_output_dir: str) -> str:
    return f"{remote_output_dir.rstrip('/')}/{LORA_CHECKPOINTS_SUBDIR}"


def lora_checkpoint_filename(step: int) -> str:
    """Trainer's per-step adapter filename, e.g. `lora_weights_step_02000.safetensors`."""
    return f"lora_weights_step_{step:05d}.safetensors"


def lora_checkpoint_path_in(remote_output_dir: str, step: int) -> str:
    return f"{lora_checkpoints_dir_in(remote_output_dir)}/{lora_checkpoint_filename(step)}"


# --- Compute-from-(workspace, id[, name]) helpers -------------------------


def dataset_dir(workspace_dir: str, dataset_id: str, name: str | None = None) -> str:
    return f"{workspace_dir.rstrip('/')}/datasets/{remote_slug(name, dataset_id)}"


def dataset_clips_dir(workspace_dir: str, dataset_id: str, name: str | None = None) -> str:
    return dataset_clips_dir_in(dataset_dir(workspace_dir, dataset_id, name))


def dataset_json_path(workspace_dir: str, dataset_id: str, name: str | None = None) -> str:
    return dataset_json_path_in(dataset_dir(workspace_dir, dataset_id, name))


def precomputed_dir(workspace_dir: str, dataset_id: str, name: str | None = None) -> str:
    return precomputed_dir_in(dataset_dir(workspace_dir, dataset_id, name))


def output_dir(workspace_dir: str, training_id: str, name: str | None = None) -> str:
    return f"{workspace_dir.rstrip('/')}/outputs/{remote_slug(name, training_id)}"


def config_path(workspace_dir: str, training_id: str, name: str | None = None) -> str:
    return f"{workspace_dir.rstrip('/')}/configs/{remote_slug(name, training_id)}.yaml"


def lora_weights_path(workspace_dir: str, training_id: str, name: str | None = None) -> str:
    return lora_weights_path_in(output_dir(workspace_dir, training_id, name))


def caption_command(
    *,
    clips_dir: str,
    dataset_json: str,
    captioner_type: str,
    override: bool = False,
) -> str:
    """`caption_videos.py` over the uploaded clips dir -> dataset.json.

    `caption_videos.py` always captions with audio awareness (it's a
    multimodal captioner), so there's no audio flag to emit — `with_audio`
    only matters later, for `process_dataset.py`'s `--skip-audio`.

    The Gemini API key is NOT passed here: it would land on the command line
    (visible in process listings and our own job-log command echo). The runner
    injects it via the `GEMINI_API_KEY` env var instead (see
    `gemini_key_env_prefix`), which `caption_videos.py` reads natively.

    `qwen_omni` is a two-process flow per the official trainer: a long-lived
    vLLM captioner server (`serve_captioner.py`) plus `caption_videos.py`
    talking to it over HTTP. We fold both into one detached command — start
    the server, wait for its OpenAI-compatible `/v1/models` endpoint, run the
    captioner, then kill the server on exit (trap) so the GPU is free for
    `process_dataset.py`. FP8 quantization (the serve_captioner default) fits
    a 40 GiB card; the model is Qwen3-Omni-30B (~31 GiB FP8, ~65 GB on disk
    for the first download).
    """
    if captioner_type == "qwen_omni":
        return _qwen_omni_caption_script(
            clips_dir=clips_dir, dataset_json=dataset_json, override=override
        )
    command = (
        "uv run python scripts/caption_videos.py "
        f"{shlex.quote(clips_dir)} --output {shlex.quote(dataset_json)} "
        f"--captioner-type {shlex.quote(captioner_type)}"
    )
    return command + (" --override" if override else "")


# vLLM captioner server the `qwen_omni` backend talks to. The official flow is
# two terminals (`serve_captioner.py` + `caption_videos.py`); the script below
# automates both so the runner's single start/poll lifecycle still applies.
VLLM_CAPTIONER_PORT = 8001
VLLM_CAPTIONER_BASE_URL = f"http://127.0.0.1:{VLLM_CAPTIONER_PORT}/v1"
VLLM_CAPTIONER_LOG = "/tmp/ltx-serve-captioner.log"
# How long to wait for the server's first run (the 30B model downloads ~65 GB
# before it can serve). 240 x 5s = 20 min, which covers a slow first download.
_VLLM_READY_PROBES = 240
_VLLM_READY_INTERVAL_S = 5


def serve_captioner_command(
    *, quantization: str = "fp8", port: int = VLLM_CAPTIONER_PORT
) -> str:
    """`serve_captioner.py` — the vLLM HTTP captioner server (qwen_omni only).

    `quantization` is a server-side concern (the client `caption_videos.py`
    has no such flag): `fp8` (~31 GiB weights, fits 40 GiB GPUs) is the
    recommended default; `bf16` (~60 GiB) needs >=66 GiB free VRAM.
    """
    return (
        "uv run python scripts/serve_captioner.py "
        f"--quantization {shlex.quote(quantization)} --port {port}"
    )


def _qwen_omni_caption_script(
    *, clips_dir: str, dataset_json: str, override: bool = False
) -> str:
    """Start the vLLM captioner, wait for it, caption, and tear it down.

    One detached bash script so the runner doesn't need a new state-machine
    phase for the server: `captioning` still maps to a single start/poll.
    `trap cleanup EXIT` guarantees the server is killed before the command
    returns (success or failure), so `process_dataset.py` gets the full GPU.
    """
    serve = serve_captioner_command()
    caption = (
        "uv run python scripts/caption_videos.py "
        f"{shlex.quote(clips_dir)} --output {shlex.quote(dataset_json)} "
        f"--captioner-type qwen_omni --vllm-url {VLLM_CAPTIONER_BASE_URL}"
        + (" --override" if override else "")
    )
    lines = [
        "set -e",
        f"{serve} > {VLLM_CAPTIONER_LOG} 2>&1 &",
        "SERVER_PID=$!",
        'cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }',
        "trap cleanup EXIT",
        f"for i in $(seq 1 {_VLLM_READY_PROBES}); do",
        '  if ! kill -0 "$SERVER_PID" 2>/dev/null; then',
        f'    echo "serve_captioner.py exited early:"; cat {VLLM_CAPTIONER_LOG}',
        "    exit 1",
        "  fi",
        f'  if curl -sf {VLLM_CAPTIONER_BASE_URL}/models >/dev/null 2>&1; then echo "vLLM captioner ready"; break; fi',
        f"  sleep {_VLLM_READY_INTERVAL_S}",
        "done",
        f'if ! curl -sf {VLLM_CAPTIONER_BASE_URL}/models >/dev/null 2>&1; then',
        f'  echo "vLLM captioner did not become ready:"; cat {VLLM_CAPTIONER_LOG}',
        "  exit 1",
        "fi",
        caption,
    ]
    return "\n".join(lines)


def gemini_key_env_prefix(gemini_api_key: str) -> str:
    """`GEMINI_API_KEY=<key> ` to prefix the caption command with, or `` if no key.

    `caption_videos.py` reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the
    environment, so the key never has to appear on the command line — keeping it
    out of process listings and the job-log command echo. Mirrors the
    `HF_TOKEN` env-prefix pattern.
    """
    return f"GEMINI_API_KEY={shlex.quote(gemini_api_key)} " if gemini_api_key else ""


def process_dataset_command(
    *,
    dataset_json: str,
    resolution_buckets: str,
    model_path: str,
    text_encoder_path: str,
    with_audio: bool,
    trigger_word: str | None,
    load_text_encoder_in_8bit: bool = False,
    reference_downscale_factor: int = 1,
    reference_temporal_scale_factor: int = 1,
    output_dir: str | None = None,
) -> str:
    """`process_dataset.py` -> caches latents into `.precomputed/`.

    The trainer auto-detects roles from the `dataset.json` columns, including
    the IC-LoRA `reference_video` column (which it turns into
    `reference_latents/`), so no reference flag is needed for detection. Audio
    is extracted by default; pass `--skip-audio` to disable it for non-audio
    (and IC-LoRA) runs.

    `load_text_encoder_in_8bit` mirrors the official `t2v_lora_low_vram.yaml`'s
    `load_text_encoder_in_8bit: true`: Gemma3 12B is 23 GB in bf16 and OOMs a
    32 GB GPU under WSL2's ~26 GB RAM cap, so the `low_vram` preset loads it in
    8-bit (~12 GB) here — matching what the training stage already does.

    `reference_downscale_factor` / `reference_temporal_scale_factor` mirror the
    official `v2v_ic_lora.yaml` validation `downscale_factor` /
    `temporal_scale_factor` fields, which the trainer's preprocess CLI exposes
    as `--reference-downscale-factor` / `--reference-temporal-scale-factor`.
    IC-LoRA concatenates reference + target tokens, so the `low_vram` preset
    halves reference spatial resolution to keep the doubled sequence's backward
    recompute within a 32 GB card's free VRAM. The caller MUST pass a single
    resolution bucket when either factor is > 1 — `process_dataset.py` rejects
    multiple buckets in that mode.
    """
    parts = [
        "uv run python scripts/process_dataset.py",
        shlex.quote(dataset_json),
        "--resolution-buckets",
        shlex.quote(resolution_buckets),
        "--model-path",
        shlex.quote(model_path),
        "--text-encoder-path",
        shlex.quote(text_encoder_path),
    ]
    if not with_audio:
        parts.append("--skip-audio")
    if trigger_word:
        parts.extend(["--lora-trigger", shlex.quote(trigger_word)])
    if load_text_encoder_in_8bit:
        parts.append("--load-text-encoder-in-8bit")
    if reference_downscale_factor > 1:
        parts.extend(
            ["--reference-downscale-factor", str(reference_downscale_factor)]
        )
    if reference_temporal_scale_factor > 1:
        parts.extend(
            ["--reference-temporal-scale-factor", str(reference_temporal_scale_factor)]
        )
    if output_dir:
        parts.extend(["--output-dir", shlex.quote(output_dir)])
    return " ".join(parts)


def train_command(*, config_yaml_path: str) -> str:
    """`train.py <config>` -> writes lora_weights.safetensors to output_dir.

    `--disable-progress-bars` is required because we redirect stdout to a log
    we poll: train.py's Rich progress bar rewrites one line with carriage
    returns and flushes no parseable step/loss newlines to a redirected log.
    Without the flag the log shows no progress and we can't track steps (per
    the trainer's own train-model skill / launch docs).
    """
    return (
        f"uv run python scripts/train.py {shlex.quote(config_yaml_path)} "
        "--disable-progress-bars"
    )
