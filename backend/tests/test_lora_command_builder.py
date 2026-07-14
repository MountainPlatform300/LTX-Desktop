"""Unit tests for the pure remote-command builders.

Focus on `provision_command` (the auto-provision bootstrap): it must be
deterministic, idempotent-friendly, and only emit a model download when
both an HF repo id and a local destination are configured.
"""

from __future__ import annotations

from handlers import lora_command_builder as paths


def test_provision_marker_path_lives_at_workspace_root() -> None:
    assert paths.provision_marker_path("/workspace/") == "/workspace/.ltx-provisioned"


def test_provision_command_installs_trainer_and_drops_marker() -> None:
    marker = paths.provision_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="v1.2.3",
        marker_path=marker,
    )
    # Installs uv, clones the configured repo/ref into the trainer dir,
    # syncs deps, and writes the marker last.
    assert "astral.sh/uv/install.sh" in script
    assert "https://example.com/trainer.git" in script
    assert "v1.2.3" in script
    assert "/workspace/ltx-trainer" in script
    # The trainer ships as a uv-workspace member inside the cloned monorepo, so
    # `uv sync` must run from that package subdir (not the repo root) for the
    # workspace to resolve `ltx-core`.
    assert f"cd {paths.trainer_workdir('/workspace')} && uv sync" in script
    assert paths.trainer_workdir("/workspace") == "/workspace/ltx-trainer/packages/ltx-trainer"
    assert script.rstrip().endswith(
        "printf '%s\\n' https://example.com/trainer.git@v1.2.3 "
        f"> {marker}"
    )
    # Fetch + detached checkout supports immutable commit SHAs as well as tags
    # and branches; `git clone --branch <sha>` does not.
    assert "fetch --depth 1 origin v1.2.3" in script
    assert "checkout --detach FETCH_HEAD" in script
    # No HF repos configured -> no download step.
    assert "hf download" not in script


def test_provision_command_downloads_models_when_configured() -> None:
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
        model_hf_repo="org/ltx-checkpoint",
        model_path="/workspace/models/ltx",
        text_encoder_hf_repo="org/gemma",
        text_encoder_path="/workspace/models/gemma",
        hf_token="secret-token",
    )
    assert "hf download org/ltx-checkpoint" in script
    assert "--local-dir /workspace/models/ltx" in script
    assert "hf download org/gemma" in script
    assert "--local-dir /workspace/models/gemma" in script
    # Token is forwarded via env on the download lines.
    assert "HF_TOKEN=secret-token" in script


def test_provision_command_downloads_only_checkpoint_file_when_named() -> None:
    # With a model_filename set, the (huge) base repo must NOT be pulled
    # whole — only the one checkpoint file, into the models dir.
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
        model_hf_repo="Lightricks/LTX-2",
        model_filename="ltx-2.3-22b-dev.safetensors",
        model_path="/workspace/models/ltx-2.3-22b-dev.safetensors",
        text_encoder_hf_repo="org/gemma",
        text_encoder_path="/workspace/models/gemma-text-encoder",
    )
    # Single-file form: repo + filename, downloaded into the models dir.
    assert (
        "hf download Lightricks/LTX-2 ltx-2.3-22b-dev.safetensors"
        in script
    )
    assert "--local-dir /workspace/models" in script
    # The encoder is still a whole-repo download into its own dir.
    assert "hf download org/gemma" in script
    assert "--local-dir /workspace/models/gemma-text-encoder" in script


def test_provision_disables_xet_and_reclaims_its_cache() -> None:
    # Xet double-stores (chunk cache + file) and overran the volume; the
    # provision script must disable it and clear any stale chunk cache.
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
    )
    assert "export HF_HUB_DISABLE_XET=1" in script
    assert "rm -rf /workspace/.cache/huggingface/xet" in script


def test_cache_env_prefix_pins_caches_and_tmp_to_volume() -> None:
    # Every runtime command must redirect caches + tmp onto the volume so the
    # small container disk doesn't fill (Triton/inductor/HF runtime caches).
    prefix = paths.cache_env_prefix("/workspace")
    assert "export XDG_CACHE_HOME=/workspace/.cache" in prefix
    assert "export HF_HOME=/workspace/.cache/huggingface" in prefix
    assert "export TRITON_CACHE_DIR=/workspace/.cache/triton" in prefix
    assert "export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor" in prefix
    assert "export TMPDIR=/workspace/tmp" in prefix
    assert prefix.rstrip().endswith(";")  # prepends cleanly before a command


def test_cache_env_prefix_puts_volume_uv_on_path() -> None:
    # uv lives on the volume so it survives pod teardown; every runtime command
    # must find it without sourcing a profile (fresh non-login SSH shells).
    prefix = paths.cache_env_prefix("/workspace")
    assert "export UV_INSTALL_DIR=/workspace/bin" in prefix
    assert "export PATH=/workspace/bin:" in prefix


def test_cache_env_prefix_wraps_path_with_gcc14_for_quanto_build() -> None:
    # optimum-quanto's `quanto_cuda` extension JIT-builds with nvcc, which
    # rejects gcc > 14 (WSL ships gcc-15). nvcc picks its host compiler from
    # PATH, so the env must put gcc-14/g++-14/c++ symlinks first on PATH (the
    # CUDAHOSTCXX env var alone is NOT honored by this torch build). Gated on
    # gcc-14 being present so it's a no-op on RunPod (CUDA-supported default gcc).
    prefix = paths.cache_env_prefix("/workspace")
    assert "command -v gcc-14" in prefix
    assert "command -v g++-14" in prefix
    # Symlinks land in a per-workspace hostcc dir, prepended to PATH.
    assert "/workspace/.hostcc" in prefix
    assert 'ln -sf "$(command -v gcc-14)"' in prefix
    assert 'ln -sf "$(command -v g++-14)"' in prefix
    assert 'export PATH=/workspace/.hostcc:"$PATH"' in prefix
    # CC/CXX cover the non-CUDA host files; CUDAHOSTCXX kept for torch builds
    # that do honor it.
    assert 'CC="$(command -v gcc-14)"' in prefix
    assert 'CXX="$(command -v g++-14)"' in prefix
    assert 'CUDAHOSTCXX="$(command -v g++-14)"' in prefix
    # torch's CUDA-extension builder needs to locate the toolkit.
    assert "export CUDA_HOME=/usr/local/cuda" in prefix


def test_provision_installs_uv_into_volume_and_on_path() -> None:
    # The uv installer must target the volume (UV_INSTALL_DIR) and uv must be on
    # PATH *before* the install/`uv sync` lines run, or a fresh pod reusing a
    # network volume would skip setup yet have no uv (was "uv: command not found").
    command = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/repo.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
    )
    assert "export UV_INSTALL_DIR=/workspace/bin" in command
    assert "export PATH=/workspace/bin:" in command
    # PATH/installer precede the `uv sync` that depends on uv being resolvable.
    assert command.index("export UV_INSTALL_DIR") < command.index("installing uv")
    assert command.index("installing uv") < command.index("uv sync")


def test_default_remote_paths_match_single_file_download() -> None:
    # The runner derives these paths; they must line up with where the
    # single-file download lands (models/<filename>) and the encoder dir.
    assert (
        paths.default_model_path("/workspace", "ltx-2.3-22b-dev.safetensors")
        == "/workspace/models/ltx-2.3-22b-dev.safetensors"
    )
    assert (
        paths.default_text_encoder_path("/workspace")
        == "/workspace/models/gemma-text-encoder"
    )


def test_provision_command_skips_download_without_destination() -> None:
    # Repo id set but no local path -> still skipped (nowhere to write).
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
        model_hf_repo="org/ltx-checkpoint",
        model_path="",
    )
    assert "hf download" not in script


def test_torch_index_marker_path_lives_at_workspace_root() -> None:
    assert paths.torch_index_marker_path("/workspace/") == "/workspace/.ltx-torch-cu128"


def test_provision_command_omits_torch_index_patch_by_default() -> None:
    # The RunPod path calls provision_command without torch_cuda_index; it must
    # NOT rewrite the trainer's pyproject or touch the cu128 marker — RunPod
    # GPUs are sm_90 and PyPI torch works there.
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=paths.provision_marker_path("/workspace"),
    )
    assert paths.TORCH_INDEX_NAME not in script
    assert paths.TORCH_INDEX_URL not in script
    # The pyproject torch-index patcher targets the trainer root pyproject; its
    # absence proves the cu128 rewrite isn't emitted. (Don't assert against
    # `python3 -` itself — the unconditional audio-fallback patch also uses a
    # `python3 - <<'PYEOF'` heredoc.)
    assert paths.trainer_root_pyproject_path("/workspace") not in script
    assert paths.torch_index_marker_path("/workspace") not in script


def test_provision_command_patches_pyproject_for_cu128_when_requested() -> None:
    # Local (Blackwell) provision must route torch through the cu128 index:
    # patch the trainer root pyproject before `uv sync` and drop the cu128
    # marker so the in-place upgrade path skips a fresh cu128 provision.
    marker = paths.provision_marker_path("/workspace")
    torch_marker = paths.torch_index_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=marker,
        torch_cuda_index="cu128",
    )
    # Patcher is invoked against the monorepo root pyproject (not the member).
    root_pyproject = paths.trainer_root_pyproject_path("/workspace")
    assert root_pyproject == "/workspace/ltx-trainer/pyproject.toml"
    assert f"python3 - {root_pyproject} {paths.TORCH_INDEX_NAME} {paths.TORCH_INDEX_URL}" in script
    assert paths.TORCH_INDEX_URL in script
    assert "PYEOF" in script
    # The patcher body routes torch/torchvision/torchaudio to the cu128 index
    # and declares the explicit index table.
    assert "torch = [{" in script
    assert "torchvision = [{" in script
    assert "torchaudio = [{" in script
    assert "[[tool.uv.index]]" in script
    assert "explicit = true" in script
    # The patch runs after clone but before `uv sync` so the sync resolves cu128.
    assert script.index("cloning trainer") < script.index("python3 -")
    assert script.index("python3 -") < script.index("uv sync")
    # The CUDA marker is touched, then the exact trainer source/ref is recorded
    # last in the provision marker.
    assert f"touch {torch_marker}" in script
    assert script.rstrip().endswith(
        "printf '%s\\n' https://example.com/trainer.git@main "
        f"> {marker}"
    )


def test_provision_command_installs_host_compiler_for_local_blackwell() -> None:
    # Local WSL provisioning must install gcc-14/g++-14 (nvcc rejects the
    # gcc-15 Ubuntu ships), or the optimum-quanto CUDA extension can't build and
    # the quantized matmul dies on sm_120 with `device not ready`. Guarded so
    # it's a fast no-op where gcc-14 is already present.
    marker = paths.provision_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=marker,
        torch_cuda_index="cu128",
        install_host_compiler=True,
    )
    assert "apt-get install -y gcc-14 g++-14" in script
    assert "command -v gcc-14" in script  # guarded: only when missing


def test_provision_command_omits_host_compiler_by_default() -> None:
    # RunPod (default) ships a CUDA-supported gcc, so the gcc-14 apt install
    # must be absent unless a caller explicitly opts in. System ffmpeg IS
    # installed by default though — torchaudio's audio backend needs it to
    # demux audio from mp4/mov clips (see `provision_command`).
    marker = paths.provision_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=marker,
    )
    assert "apt-get install -y gcc-14 g++-14" not in script
    # ffmpeg is installed unconditionally (idempotent) for torchaudio audio.
    assert "command -v ffmpeg" in script
    assert "apt-get install -y --no-install-recommends ffmpeg" in script


def test_provision_command_installs_ffmpeg_for_torchaudio_audio() -> None:
    # torchaudio 2.1+ no longer bundles FFmpeg; its ffmpeg I/O backend (the
    # only one that can demux audio out of mp4/mov/mkv) loads libav* from the
    # system. Without it, `process_dataset.py` silently skips every clip's
    # audio and `audio_latents/` comes out empty. Provisioning must install
    # ffmpeg idempotently for both RunPod and WSL.
    marker = paths.provision_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=marker,
    )
    assert "command -v ffmpeg >/dev/null" in script
    assert "apt-get install -y --no-install-recommends ffmpeg" in script
    # Runs before `uv sync` so the libs are present before any trainer code.
    assert script.index("command -v ffmpeg") < script.index("uv sync")


def test_ensure_ffmpeg_command_is_idempotent_one_liner() -> None:
    # The already-provisioned path runs this to backfill ffmpeg on workspaces
    # provisioned before the fix. Must be a guarded no-op where ffmpeg exists.
    cmd = paths.ensure_ffmpeg_command()
    assert "command -v ffmpeg >/dev/null 2>&1" in cmd
    assert "apt-get install -y --no-install-recommends ffmpeg" in cmd
    assert "||" in cmd  # only installs when the binary is missing


def test_provision_command_patches_trainer_audio_fallback() -> None:
    # On WSL cu128, torchaudio 2.9+ routes `torchaudio.load` through torchcodec,
    # whose libav ABI doesn't match system ffmpeg — so `torchaudio.load(<mp4>)`
    # raises even with ffmpeg installed and every clip's audio is silently
    # skipped. Provisioning must patch `process_videos.py` with an
    # ffmpeg-subprocess fallback. Best-effort (non-fatal on failure).
    marker = paths.provision_marker_path("/workspace")
    script = paths.provision_command(
        workspace_dir="/workspace",
        repo_url="https://example.com/trainer.git",
        repo_ref="main",
        marker_path=marker,
    )
    assert "patch_trainer_audio_fallback" not in script  # impl detail
    assert "process_videos.py" in script
    assert "ffmpeg" in script
    assert "torchaudio.load" in script
    # The patch step is best-effort: a failure must not abort provisioning.
    assert "non-fatal" in script
    # The fallback belongs on the heredoc invocation line. A closing delimiter
    # with trailing shell tokens is not recognized and swallows the remaining
    # provisioning script as Python input.
    assert "<<'PYEOF' || echo" in script
    assert "\nPYEOF ||" not in script
    assert "\nPYEOF\n" in script
    # Runs after the clone (so the file exists) — i.e. after `git clone`.
    assert script.index("git clone") < script.index("process_videos.py")


def test_patch_trainer_audio_fallback_command_is_idempotent_and_inserts_block() -> None:
    # The patch command is a `python3 - <<'PYEOF'` heredoc that inserts a
    # monkeypatch after `import torchaudio`. Verify the generated heredoc body
    # actually patches a sample file, is idempotent (marker-guarded no-op), and
    # leaves the file compilable.
    import os
    import tempfile

    workspace = "WS"
    cmd = paths.patch_trainer_audio_fallback_command(workspace)
    assert cmd.startswith("python3 - <<'PYEOF'")
    assert cmd.rstrip().endswith("PYEOF")
    # The target path is baked into the heredoc.
    expected_path = paths.trainer_workdir(workspace) + "/scripts/process_videos.py"
    assert expected_path in cmd

    body = cmd.split("<<'PYEOF'\n", 1)[1].rsplit("PYEOF", 1)[0]
    d = tempfile.mkdtemp()
    fake = os.path.join(d, "process_videos.py")
    open(fake, "w").write(
        "import torch\nimport torchaudio\nimport numpy as np\n\nprint('hi')\n"
    )
    body = body.replace(
        "p = '" + expected_path + "'",
        "p = r'" + fake + "'",
    )

    def run() -> None:
        try:
            exec(body, {"__name__": "__main__"})
        except SystemExit:
            pass

    run()
    patched = open(fake).read()
    compile(patched, fake, "exec")  # patched file is valid python
    assert "LTX-Desktop: ffmpeg fallback" in patched
    assert "torchaudio.load = _ltx_torchaudio_load" in patched
    assert patched.index("import torchaudio") < patched.index("_ltx_torchaudio_load")
    run()  # second run is a marker-guarded no-op
    assert open(fake).read() == patched


def test_ensure_torch_index_command_patches_and_resyncs() -> None:
    # The in-place upgrade (for workspaces provisioned before the Blackwell
    # fix) patches the trainer pyproject and re-runs `uv sync`, then drops the
    # cu128 marker. Self-contained: sets the same cache env as provisioning.
    torch_marker = paths.torch_index_marker_path("/workspace")
    script = paths.ensure_torch_index_command(
        workspace_dir="/workspace", marker_path=torch_marker
    )
    assert f"python3 - {paths.trainer_root_pyproject_path('/workspace')}" in script
    assert paths.TORCH_INDEX_URL in script
    assert f"cd {paths.trainer_workdir('/workspace')} && uv sync" in script
    assert "PYEOF" in script
    assert script.rstrip().endswith(f"touch {torch_marker}")
    # Reuses the volume-pinned cache env so the cu128 wheel download lands on
    # the workspace, not the small WSL root disk.
    assert "export UV_INSTALL_DIR=/workspace/bin" in script
    assert "export XDG_CACHE_HOME=/workspace/.cache" in script


def test_ensure_torch_index_command_installs_host_compiler_when_requested() -> None:
    # An old workspace upgraded in place must also get gcc-14, or the cu128
    # torch upgrade alone leaves the quanto extension un-buildable on Blackwell.
    torch_marker = paths.torch_index_marker_path("/workspace")
    script = paths.ensure_torch_index_command(
        workspace_dir="/workspace",
        marker_path=torch_marker,
        install_host_compiler=True,
    )
    assert "apt-get install -y gcc-14 g++-14" in script
    # The host-compiler install precedes the uv sync that depends on the toolchain
    # being ready (and on a clean distro apt-get update must run first).
    assert script.index("apt-get install") < script.index("uv sync")


def test_remote_slug_is_readable_and_unique() -> None:
    slug = paths.remote_slug("Xray LoRA!", "a3f8c2e1d4b54f6a9c0e")
    # Sanitized, lowercased name + short id suffix so folders are browseable
    # yet collision-free.
    assert slug == "xray-lora-a3f8c2e1"


def test_remote_slug_falls_back_to_id_without_name() -> None:
    assert paths.remote_slug(None, "abc123") == "abc123"
    assert paths.remote_slug("", "abc123") == "abc123"


def test_dataset_dir_uses_readable_slug() -> None:
    assert (
        paths.dataset_dir("/workspace", "a3f8c2e1d4b5", "Xray LoRA")
        == "/workspace/datasets/xray-lora-a3f8c2e1"
    )


def test_stored_dir_helpers_derive_paths_from_recorded_dir() -> None:
    remote = "/home/user/datasets/xray-lora-a3f8c2e1"
    assert paths.dataset_clips_dir_in(remote) == f"{remote}/clips"
    assert paths.dataset_json_path_in(remote) == f"{remote}/dataset.json"
    assert paths.precomputed_dir_in(remote) == f"{remote}/.precomputed"
    assert (
        paths.precomputed_run_dir_in(remote, "prep-123")
        == f"{remote}/.precomputed-prep-123"
    )
    assert (
        paths.lora_weights_path_in("/home/user/outputs/run-7c44d9a2")
        == "/home/user/outputs/run-7c44d9a2/lora_weights.safetensors"
    )


def test_lora_checkpoint_path_matches_trainer_layout() -> None:
    # The trainer writes adapters to checkpoints/lora_weights_step_NNNNN (5-digit
    # zero-padded), not a single file at the output root. The download path must
    # match exactly or the finished LoRA can't be retrieved.
    out = "/workspace/outputs/cleanplate-6ea8e1e6"
    assert paths.lora_checkpoint_filename(2000) == "lora_weights_step_02000.safetensors"
    assert (
        paths.lora_checkpoint_path_in(out, 2000)
        == f"{out}/checkpoints/lora_weights_step_02000.safetensors"
    )
    assert paths.lora_checkpoints_dir_in(out) == f"{out}/checkpoints"


def test_caption_command_never_emits_audio_flag() -> None:
    # caption_videos.py has no --no-audio / --with-audio option (it always
    # captions with audio awareness); emitting one aborts captioning with
    # "No such option". Audio is controlled later by process_dataset.py.
    cmd = paths.caption_command(
        clips_dir="/clips",
        dataset_json="/ds/dataset.json",
        captioner_type="gemini_flash",
    )
    assert "--no-audio" not in cmd
    assert "--with-audio" not in cmd
    assert "--captioner-type gemini_flash" in cmd
    # The Gemini key is passed via env, never on the command line (it would
    # leak in process listings + the job-log command echo).
    assert "--api-key" not in cmd


def test_caption_command_qwen_omni_starts_server_and_caps_with_vllm_url() -> None:
    # qwen_omni is a two-process flow: serve_captioner.py (vLLM server) +
    # caption_videos.py talking to it over --vllm-url. The command must start
    # the server, wait for /v1/models, caption, and tear it down — with NO
    # --use-8bit (that isn't a caption_videos.py option; quantization is a
    # serve_captioner.py flag).
    cmd = paths.caption_command(
        clips_dir="/clips",
        dataset_json="/ds/dataset.json",
        captioner_type="qwen_omni",
    )
    assert "serve_captioner.py" in cmd
    assert "--quantization fp8" in cmd
    assert "--vllm-url http://127.0.0.1:8001/v1" in cmd
    assert "caption_videos.py" in cmd
    assert "--captioner-type qwen_omni" in cmd
    # Server is killed on exit so process_dataset.py gets the full GPU.
    assert "trap cleanup EXIT" in cmd
    # The invalid legacy flag must never appear.
    assert "--use-8bit" not in cmd


def test_caption_command_gemini_flash_is_a_single_cloud_call() -> None:
    # gemini_flash is cloud-based: no vLLM server, no --use-8bit, no --vllm-url.
    cmd = paths.caption_command(
        clips_dir="/clips",
        dataset_json="/ds/dataset.json",
        captioner_type="gemini_flash",
    )
    assert "caption_videos.py" in cmd
    assert "--captioner-type gemini_flash" in cmd
    assert "serve_captioner.py" not in cmd
    assert "--vllm-url" not in cmd
    assert "--use-8bit" not in cmd


def test_caption_command_override_recaptions_for_both_backends() -> None:
    for captioner in ("gemini_flash", "qwen_omni"):
        cmd = paths.caption_command(
            clips_dir="/clips",
            dataset_json="/ds/dataset.json",
            captioner_type=captioner,
            override=True,
        )
        assert "--override" in cmd


def test_gemini_key_env_prefix() -> None:
    assert paths.gemini_key_env_prefix("k") == "GEMINI_API_KEY=k "
    # Empty key -> no prefix (so we don't export an empty var).
    assert paths.gemini_key_env_prefix("") == ""
    # Shell-hostile characters are quoted so the key can't break out of the env
    # assignment or inject a command.
    assert paths.gemini_key_env_prefix("a b") == "GEMINI_API_KEY='a b' "


def test_process_dataset_command_never_emits_reference_column() -> None:
    # The trainer auto-detects the reference_video column; there is no
    # --reference-column flag, so we must never emit one.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word="TOK",
    )
    assert "--reference-column" not in cmd
    assert "--lora-trigger TOK" in cmd


def test_process_dataset_command_skips_audio_when_disabled() -> None:
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word=None,
    )
    assert "--skip-audio" in cmd
    assert "--with-audio" not in cmd


def test_process_dataset_command_keeps_audio_when_enabled() -> None:
    # Audio is on by default in the trainer, so neither flag should appear.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=True,
        trigger_word=None,
    )
    assert "--skip-audio" not in cmd
    assert "--with-audio" not in cmd


def test_process_dataset_command_loads_text_encoder_in_8bit_when_requested() -> None:
    # low_vram preset -> 8-bit text encoder (Gemma3 12B is 23 GB in bf16 and
    # OOMs a 32 GB GPU); the flag must reach process_dataset.py.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word="TOK",
        load_text_encoder_in_8bit=True,
    )
    assert "--load-text-encoder-in-8bit" in cmd


def test_process_dataset_command_omits_8bit_flag_by_default() -> None:
    # standard preset -> bf16 text encoder; never emit the 8-bit flag by default.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word=None,
    )
    assert "--load-text-encoder-in-8bit" not in cmd


def test_process_dataset_command_emits_reference_downscale_when_requested() -> None:
    # IC-LoRA low_vram halves reference resolution via the official lever; the
    # flag (and only the spatial one) must reach process_dataset.py.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word="TOK",
        reference_downscale_factor=2,
    )
    assert "--reference-downscale-factor 2" in cmd
    assert "--reference-temporal-scale-factor" not in cmd


def test_process_dataset_command_emits_reference_temporal_scale_when_requested() -> None:
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word=None,
        reference_temporal_scale_factor=2,
    )
    assert "--reference-temporal-scale-factor 2" in cmd
    assert "--reference-downscale-factor" not in cmd


def test_process_dataset_command_omits_reference_downscale_by_default() -> None:
    # standard preset / text-to-video -> full-size references; never emit the
    # reference flags by default.
    cmd = paths.process_dataset_command(
        dataset_json="/ds/dataset.json",
        resolution_buckets="768x448x89",
        model_path="/m.safetensors",
        text_encoder_path="/te",
        with_audio=False,
        trigger_word=None,
    )
    assert "--reference-downscale-factor" not in cmd
    assert "--reference-temporal-scale-factor" not in cmd


def test_process_dataset_command_supports_isolated_output_dir() -> None:
    cmd = paths.process_dataset_command(
        dataset_json="/workspace/dataset.json",
        resolution_buckets="768x448x49",
        model_path="/workspace/model.safetensors",
        text_encoder_path="/workspace/gemma",
        with_audio=False,
        trigger_word=None,
        output_dir="/workspace/dataset/.precomputed-prep-123",
    )
    assert "--output-dir /workspace/dataset/.precomputed-prep-123" in cmd
