"""RunPod `TrainerTarget` (BYOK on-demand GPU pods).

Uses the official `runpod` Python SDK (lazily imported) for the pod
lifecycle — create / inspect / terminate — and the shared `SSHRemote`
helper for command execution and file transfer over the pod's exposed
SSH port.

What this needs on the RunPod side:
- An API key with pod-management scope (the BYOK key in settings).
- A pod image that has the LTX-2 trainer installed at
  ``{workspace_dir}/ltx-trainer``, plus the LTX-2 checkpoint and Gemma
  encoder at the configured remote paths, and an SSH server on port 22.
- The account's SSH public key registered with RunPod so key-based auth
  succeeds (we authenticate via the local SSH agent / default keys).

This implementation is structured for review and live validation: the
SDK surface and port-mapping shape can vary by template, so the SSH
endpoint extraction is defensive and raises actionable
`TrainerTargetError`s rather than guessing.
"""

from __future__ import annotations

import logging
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from handlers import lora_command_builder as paths
from secret_redaction import redact_text
from services.trainer_target.ssh_keys import SshKeyManager
from services.trainer_target.ssh_remote import (
    SSHConnection,
    SSHRemote,
    SshHostTrustStore,
)
from services.trainer_target.trainer_target import (
    AccountInfo,
    GpuOffer,
    GpuTelemetry,
    NetworkVolume,
    NVIDIA_SMI_GPU_QUERY,
    PodInfo,
    ProgressCallback,
    RemoteCommandStatus,
    RegionHealth,
    TrainerCredentials,
    TrainerTargetError,
    ValidationArtifact,
    parse_nvidia_smi_output,
    parse_samples_listing,
    samples_dir_for,
    checkpoints_dir_for,
    parse_checkpoints_listing,
)
from state.lora_training_state import TargetHandle

# Logs flow to stdout, which Electron captures into the session log file
# surfaced by the in-app log viewer. Prefix messages with "RunPod:" so the
# pod lifecycle is easy to spot/filter there. Never log the API key.
logger = logging.getLogger(__name__)

# Base CUDA image used when no override is configured. With
# auto-provisioning on, the trainer + models are installed onto this
# generic image at runtime, so a stock pytorch image is sufficient. A
# deployment can still pin a pre-baked image via `runpod_image`.
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
_POD_READY_TIMEOUT_SECONDS = 600
_POD_POLL_INTERVAL_SECONDS = 10
# Bootstrap (clone trainer + uv sync + optional multi-GB model download)
# can be slow on a cold pod; generous ceiling, gated by the marker so it
# only ever runs once per pod/volume.
_PROVISION_TIMEOUT_SECONDS = 2400
# Poll provisioning fairly often so the live download % visibly climbs on the
# card/log rather than feeling stuck between ticks.
_PROVISION_POLL_INTERVAL_SECONDS = 8

# Network-volume management isn't in the `runpod` SDK (1.9.x), so the
# connect flow talks to RunPod's REST API directly for list/create.
_RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
_REST_TIMEOUT_SECONDS = 30
_REGIONAL_QUERY_LIMIT = 4
# Datacenter used to create a fresh network volume when the account has
# none yet and RunPod doesn't report a storage-capable region.
_DEFAULT_VOLUME_DATACENTER = "EU-RO-1"

# Minimum VRAM to even show a GPU in the connect picker: the LTX-2 trainer's
# low-VRAM preset needs ~32 GB, so anything smaller can't train a LoRA and is
# filtered out (otherwise the list is ~49 mostly-unusable cards).
LORA_MIN_GPU_VRAM_GB = 32
# Preferred VRAM when choosing a region for a new volume: the standard config
# wants 80GB, so we favor a datacenter that has an 80GB+ card in stock.
_MIN_PREFERRED_GPU_VRAM_GB = 80
# Container disk for created pods. The big stuff (weights, venv, caches) lives
# on the network volume; this is headroom for build temp + base image layers
# so the container disk doesn't fill (the 106%-disk failure we saw).
_CONTAINER_DISK_GB = 100
# Fallback /workspace size for the ephemeral (no network volume) path when the
# caller didn't pass a size. Must fit the ~68GB models + some data.
_DEFAULT_EPHEMERAL_VOLUME_GB = 500
# RunPod tags pods the app created with this name so the connect UI can flag
# which active pods are ours (safe to reuse/terminate) vs the user's own.
_APP_POD_NAME = "ltx-desktop-lora"
_APP_VOLUME_NAME_PREFIX = "ltx-desktop-lora"
# Substrings RunPod uses when a GPU type can't be allocated right now.
_UNAVAILABLE_MARKERS = (
    "not have the resources",
    "no longer any instances",
    "no instances available",
    "out of stock",
    "no gpus available",
)
# Substrings RunPod uses when a pod no longer exists (terminate is then a no-op).
_POD_GONE_MARKERS = ("not found", "does not exist", "no pod")


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _stock_is_available(
    status_value: Any,
    available_gpu_counts: Any = None,
) -> bool:
    """Interpret RunPod's stock signal without truthiness false positives.

    RunPod returns the strings ``High``, ``Medium``, ``Low``, or ``None``.
    ``bool("None")`` is True, which previously made every out-of-stock GPU look
    selectable. When counts are supplied, a one-GPU pod must also be listed.
    """
    status = str(status_value or "").strip().lower()
    if status not in {"high", "medium", "low"}:
        return False
    if isinstance(available_gpu_counts, list) and available_gpu_counts:
        try:
            return 1 in {int(count) for count in available_gpu_counts}
        except (TypeError, ValueError):
            return False
    return True


def _latest_progress_line(lines: list[str]) -> str:
    """Last meaningful line of a remote log tail, for live progress.

    `read_logs` already splits on `\\r` (tqdm/git progress bars use it), so the
    last non-empty entry is the most recent progress update. Truncated so a
    long bar doesn't blow up the UI.
    """
    for raw in reversed(lines):
        line = raw.strip()
        if line:
            return line if len(line) <= 140 else line[:139] + "…"
    return ""


def _runpod(credentials: TrainerCredentials) -> Any:
    if not credentials.runpod_api_key:
        raise TrainerTargetError("RunPod API key is not configured", retryable=False)
    try:
        import runpod  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise TrainerTargetError(
            "The 'runpod' package is required for the RunPod target but is "
            "not installed in the backend environment",
            retryable=False,
        ) from exc
    runpod.api_key = credentials.runpod_api_key
    return runpod


def _ssh_endpoint(pod: dict[str, Any]) -> tuple[str, int]:
    """Extract the public SSH (ip, port) from a pod's runtime ports."""
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for port in ports:
        if str(port.get("type")) == "tcp" and int(port.get("privatePort", 0)) == 22:
            ip = port.get("ip")
            public_port = port.get("publicPort")
            if ip and public_port:
                return str(ip), int(public_port)
    raise TrainerTargetError(
        "RunPod pod has no public SSH port yet; ensure the image exposes "
        "port 22 over public TCP",
        retryable=True,
    )


class RunPodTrainerTarget:
    def __init__(self, ssh_key_dir: Path | None = None) -> None:
        # When a key dir is provided the app owns a dedicated keypair and
        # injects its public half into new pods (RunPod's `PUBLIC_KEY`),
        # so auth is fully hands-off. Without one we fall back to the
        # local SSH agent / default keys (the prior behaviour).
        self._key_manager = SshKeyManager(ssh_key_dir) if ssh_key_dir else None
        self._host_trust = (
            SshHostTrustStore(ssh_key_dir / "trusted_hosts.json")
            if ssh_key_dir
            else None
        )

    def _key_filename(self) -> str | None:
        if self._key_manager is None:
            return None
        private_path, _ = self._key_manager.ensure_keypair()
        return private_path

    def _remote(self, credentials: TrainerCredentials, handle: TargetHandle) -> SSHRemote:
        if handle.pod_id is None:
            raise TrainerTargetError("RunPod handle is missing a pod id", retryable=False)
        runpod = _runpod(credentials)
        pod = runpod.get_pod(handle.pod_id)
        if not pod:
            raise TrainerTargetError(
                f"RunPod pod {handle.pod_id} not found", retryable=False
            )
        host, port = _ssh_endpoint(pod)
        return SSHRemote(
            SSHConnection(
                host=host, port=port, username="root", key_filename=self._key_filename()
            ),
            trust_store=self._host_trust,
            trust_identity=f"runpod:{handle.pod_id}",
        )

    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        runpod = _runpod(credentials)
        try:
            runpod.get_pods()
        except Exception as exc:
            raise TrainerTargetError(
                f"RunPod API check failed: {exc}", retryable=True
            ) from exc

    # ------------------------------------------------------------------
    # Connect flow: GPU discovery + network-volume management
    # ------------------------------------------------------------------

    def connect_account(self, *, credentials: TrainerCredentials) -> AccountInfo:
        logger.info("RunPod: connecting — discovering GPU types and volumes")
        runpod = _runpod(credentials)
        volumes = self._list_volumes(credentials)
        # The picker is GPU-first: always discover global stock. Existing
        # storage regions are queried separately below so the UI can annotate
        # which globally available GPUs can reuse a ready model cache.
        target_dc = ""
        if credentials.runpod_network_volume_id:
            for vol in volumes:
                if vol.id == credentials.runpod_network_volume_id:
                    target_dc = vol.datacenter_id
                    break
        offers = self._discover_gpus_region_aware(credentials, runpod, "")
        region_health = tuple(
            self._region_health(credentials, dc)
            for dc in sorted({v.datacenter_id for v in volumes if v.datacenter_id})
        )
        pods = self._list_pods(runpod)
        available_count = sum(1 for o in offers if o.available)
        logger.info(
            "RunPod: connected — %d training-capable GPU types (%d available%s), "
            "%d network volume(s), %d active pod(s)",
            len(offers),
            available_count,
            " globally",
            len(volumes),
            len(pods),
        )
        return AccountInfo(
            gpus=tuple(offers),
            volumes=tuple(volumes),
            pods=tuple(pods),
            datacenter=target_dc,
            region_health=region_health,
        )

    def _discover_gpus_region_aware(
        self, credentials: TrainerCredentials, runpod: Any, active_dc: str
    ) -> list[GpuOffer]:
        """Report stock in the selected region and in alternative regions."""
        active = self._discover_gpus_graphql(credentials, active_dc)
        if active is None:
            return self._discover_gpus(credentials, runpod, active_dc)
        by_id = {offer.id: offer for offer in active}
        best: dict[str, tuple[str, float | None]] = {}
        regions = self._storage_datacenters(credentials)[:_REGIONAL_QUERY_LIMIT]
        query_regions = [dc for dc in regions if dc != active_dc]
        if query_regions:
            with ThreadPoolExecutor(max_workers=len(query_regions)) as pool:
                regional_results = tuple(
                    zip(
                        query_regions,
                        pool.map(
                            lambda dc: self._discover_gpus_graphql(credentials, dc),
                            query_regions,
                        ),
                        strict=True,
                    )
                )
        else:
            regional_results = ()
        for dc, regional in regional_results:
            if regional is None:
                continue
            for offer in regional:
                if not offer.available:
                    continue
                previous = best.get(offer.id)
                if previous is None or (
                    offer.price_per_hr is not None
                    and (previous[1] is None or offer.price_per_hr < previous[1])
                ):
                    best[offer.id] = (dc, offer.price_per_hr)
                by_id.setdefault(offer.id, offer)
        offers: list[GpuOffer] = []
        for offer in by_id.values():
            elsewhere = offer.id in best and best[offer.id][0] != active_dc
            available = offer.available if active_dc else offer.available or offer.id in best
            price = offer.price_per_hr
            if not available and offer.id in best:
                price = best[offer.id][1]
            offers.append(
                GpuOffer(
                    id=offer.id,
                    label=offer.label,
                    memory_gb=offer.memory_gb,
                    price_per_hr=price,
                    available=available,
                    active_region_available=offer.available if active_dc else None,
                    available_elsewhere=elsewhere,
                    best_available_region=(
                        active_dc
                        if active_dc and offer.available
                        else best.get(offer.id, (None, None))[0]
                    ),
                )
            )
        suitable = [
            offer
            for offer in offers
            if (offer.available or offer.available_elsewhere) and offer.memory_gb >= 80
        ] or [
            offer for offer in offers if offer.available or offer.available_elsewhere
        ]
        recommended_id = (
            min(
                suitable,
                key=lambda item: (
                    item.price_per_hr
                    if item.price_per_hr is not None
                    else float("inf"),
                    -item.memory_gb,
                ),
            ).id
            if suitable
            else ""
        )
        result = [
            GpuOffer(
                id=offer.id,
                label=offer.label,
                memory_gb=offer.memory_gb,
                price_per_hr=offer.price_per_hr,
                available=offer.available,
                active_region_available=offer.active_region_available,
                available_elsewhere=offer.available_elsewhere,
                best_available_region=offer.best_available_region,
                recommended=offer.id == recommended_id,
            )
            for offer in offers
        ]
        result.sort(key=lambda item: item.memory_gb, reverse=True)
        return result

    def _discover_gpus(
        self, credentials: TrainerCredentials, runpod: Any, datacenter: str
    ) -> list[GpuOffer]:
        """Training-capable GPUs (>=32GB) with accurate availability.

        Uses the GraphQL `lowestPrice` query for real stock — per-datacenter
        when a volume pins a region, else global. Only if GraphQL fails do we
        fall back to the SDK's per-GPU stock detail.
        """
        graphql_offers = self._discover_gpus_graphql(credentials, datacenter)
        if graphql_offers is not None:
            return graphql_offers
        logger.warning(
            "RunPod: availability query failed; listing GPUs without a stock "
            "signal (the create attempt will validate)"
        )
        return self._discover_gpus_global(runpod)

    def _discover_gpus_graphql(
        self, credentials: TrainerCredentials, datacenter: str
    ) -> list[GpuOffer] | None:
        """GPU stock via the GraphQL API (per-DC when given). None on failure."""
        import requests  # noqa: PLC0415

        query = (
            "query($dc:String){ gpuTypes { id displayName memoryInGb "
            "lowestPrice(input:{gpuCount:1, dataCenterId:$dc}) "
            "{ stockStatus availableGpuCounts uninterruptablePrice } } }"
        )
        try:
            resp = requests.post(
                "https://api.runpod.io/graphql",
                params={"api_key": credentials.runpod_api_key},
                # null dataCenterId = global availability.
                json={"query": query, "variables": {"dc": datacenter or None}},
                timeout=_REST_TIMEOUT_SECONDS,
            )
        except Exception:
            return None
        if resp.status_code >= 400:
            return None
        body = resp.json()
        if not isinstance(body, dict) or body.get("errors"):
            return None
        gpu_types = ((body.get("data") or {}).get("gpuTypes")) or []
        offers: list[GpuOffer] = []
        for gpu in gpu_types:
            gid = str(gpu.get("id") or "").strip()
            memory_gb = int(gpu.get("memoryInGb") or 0)
            if not gid or memory_gb < LORA_MIN_GPU_VRAM_GB:
                continue
            lowest = gpu.get("lowestPrice") or {}
            available = _stock_is_available(
                lowest.get("stockStatus"),
                lowest.get("availableGpuCounts"),
            )
            offers.append(
                GpuOffer(
                    id=gid,
                    label=str(gpu.get("displayName") or gid),
                    memory_gb=memory_gb,
                    price_per_hr=_as_float(lowest.get("uninterruptablePrice")),
                    available=available,
                )
            )
        offers.sort(key=lambda o: o.memory_gb, reverse=True)
        return offers

    def _region_health(
        self, credentials: TrainerCredentials, datacenter: str
    ) -> RegionHealth:
        offers = self._discover_gpus_graphql(credentials, datacenter)
        if offers is None:
            return RegionHealth(
                datacenter_id=datacenter,
                status="unknown",
                qualifying_gpu_available=False,
            )
        available = tuple(o.id for o in offers if o.available)
        return RegionHealth(
            datacenter_id=datacenter,
            status="healthy" if available else "no_stock",
            qualifying_gpu_available=bool(available),
            available_gpu_ids=available,
        )

    def _discover_gpus_global(self, runpod: Any) -> list[GpuOffer]:
        try:
            raw_gpus = runpod.get_gpus()
        except Exception as exc:
            logger.error("RunPod: failed to list GPU types: %s", exc)
            raise TrainerTargetError(
                f"Failed to list RunPod GPU types: {exc}", retryable=True
            ) from exc
        offers: list[GpuOffer] = []
        for gpu in raw_gpus or []:
            gid = str(gpu.get("id") or "").strip()
            memory_gb = int(gpu.get("memoryInGb") or 0)
            if not gid or memory_gb < LORA_MIN_GPU_VRAM_GB:
                continue
            price, available = self._gpu_price_and_stock(runpod, gid)
            # A failed/unknown stock probe is not presented as available. It is
            # safer to ask the user to refresh than to repeatedly offer GPUs
            # that RunPod will reject after they start a paid workflow.
            offers.append(
                GpuOffer(
                    id=gid,
                    label=str(gpu.get("displayName") or gid),
                    memory_gb=memory_gb,
                    price_per_hr=price,
                    available=available,
                )
            )
        offers.sort(key=lambda o: o.memory_gb, reverse=True)
        return offers

    def _list_pods(self, runpod: Any) -> list[PodInfo]:
        """List pods currently on the account (best-effort) for the connect UI."""
        try:
            raw = runpod.get_pods()
        except Exception as exc:
            logger.warning("RunPod: failed to list pods: %s", exc)
            return []
        pods: list[PodInfo] = []
        for pod in raw or []:
            pid = str(pod.get("id") or "")
            if not pid:
                continue
            machine = pod.get("machine") or {}
            gpu = str(
                pod.get("gpuTypeId")
                or machine.get("gpuDisplayName")
                or pod.get("gpuDisplayName")
                or ""
            )
            cost = pod.get("costPerHr")
            try:
                cost_val = float(cost) if cost is not None else None
            except (TypeError, ValueError):
                cost_val = None
            name = str(pod.get("name") or "")
            desired = str(pod.get("desiredStatus") or "")
            status_str = desired or str(pod.get("lastStatusChange") or "")
            runtime = pod.get("runtime") or {}
            raw_uptime = (
                runtime.get("uptimeInSeconds")
                or runtime.get("uptimeSeconds")
                or pod.get("uptimeInSeconds")
                or pod.get("uptimeSeconds")
            )
            try:
                uptime_seconds = (
                    max(0, int(float(raw_uptime))) if raw_uptime is not None else None
                )
            except (TypeError, ValueError):
                uptime_seconds = None
            last_started = (
                runtime.get("lastStartedAt")
                or pod.get("lastStartedAt")
                or pod.get("lastStatusChange")
            )
            pods.append(
                PodInfo(
                    id=pid,
                    name=name,
                    gpu=gpu,
                    status=status_str,
                    cost_per_hr=cost_val,
                    created_by_app=name == _APP_POD_NAME,
                    desired_status=desired,
                    running=desired == "RUNNING",
                    uptime_seconds=uptime_seconds,
                    last_started_at=str(last_started) if last_started else None,
                )
            )
        return pods

    def list_pods(self, *, credentials: TrainerCredentials) -> list[PodInfo]:
        """Standalone pod list for the Trainer compute panel (no GPU discovery)."""
        runpod = _runpod(credentials)
        return self._list_pods(runpod)

    @staticmethod
    def _require_app_owned_pod(
        runpod: Any, pod_id: str, *, action: str, allow_gone: bool
    ) -> dict[str, Any] | None:
        try:
            pod = runpod.get_pod(pod_id)
        except Exception as exc:
            if allow_gone and any(
                marker in str(exc).lower() for marker in _POD_GONE_MARKERS
            ):
                return None
            raise TrainerTargetError(
                f"Failed to verify RunPod pod {pod_id} before {action}: {exc}",
                retryable=True,
            ) from exc
        if not pod:
            if allow_gone:
                return None
            raise TrainerTargetError(
                f"RunPod pod {pod_id} no longer exists",
                retryable=False,
            )
        if str(pod.get("name") or "") != _APP_POD_NAME:
            raise TrainerTargetError(
                f"Refusing to {action} RunPod pod {pod_id}: it was not created "
                "by LTX Desktop",
                retryable=False,
                code="ownership_violation",
            )
        return pod

    def stop_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        """Pause a running pod — stops GPU billing, keeps the container disk."""
        runpod = _runpod(credentials)
        if self._require_app_owned_pod(
            runpod, pod_id, action="stop", allow_gone=True
        ) is None:
            return
        logger.info("RunPod: stopping pod %s", pod_id)
        try:
            runpod.stop_pod(pod_id)
        except Exception as exc:
            # Stopping an already-stopped or gone pod is success (idempotent):
            # the goal — pod not running — is achieved. Swallow it so the UI
            # refresh doesn't error on a pod that changed state between list+act.
            if any(m in str(exc).lower() for m in _POD_GONE_MARKERS):
                logger.info("RunPod: pod %s already gone — treating as stopped", pod_id)
                return
            logger.error("RunPod: failed to stop pod %s: %s", pod_id, exc)
            raise TrainerTargetError(
                f"Failed to stop RunPod pod {pod_id}: {exc}", retryable=True
            ) from exc
        logger.info("RunPod: pod %s stopped", pod_id)

    def resume_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        """Start a stopped pod — resumes GPU billing."""
        runpod = _runpod(credentials)
        self._require_app_owned_pod(
            runpod, pod_id, action="resume", allow_gone=False
        )
        logger.info("RunPod: resuming pod %s", pod_id)
        try:
            runpod.resume_pod(pod_id)
        except Exception as exc:
            logger.error("RunPod: failed to resume pod %s: %s", pod_id, exc)
            raise TrainerTargetError(
                f"Failed to resume RunPod pod {pod_id}: {exc}", retryable=True
            ) from exc
        logger.info("RunPod: pod %s resumed", pod_id)

    def _gpu_price_and_stock(self, runpod: Any, gpu_id: str) -> tuple[float | None, bool]:
        """Best-effort on-demand $/hr + current availability for one GPU type.

        Tolerant of the SDK's shifting response shape: any missing field
        degrades to (None, True) rather than failing the whole connect.
        """
        try:
            detail = runpod.get_gpu(gpu_id)
        except Exception:
            return None, False
        if not isinstance(detail, dict):
            return None, False
        lowest = detail.get("lowestPrice") or {}
        # Do NOT OR with secureCloud/communityCloud: those fields indicate that
        # a type exists, not that an instance can currently be allocated.
        available = _stock_is_available(
            lowest.get("stockStatus"),
            lowest.get("availableGpuCounts"),
        )
        return _as_float(lowest.get("uninterruptablePrice")), available

    def _rest_headers(self, credentials: TrainerCredentials) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {credentials.runpod_api_key}",
            "Content-Type": "application/json",
        }

    def _list_volumes(self, credentials: TrainerCredentials) -> list[NetworkVolume]:
        import requests  # noqa: PLC0415

        try:
            resp = requests.get(
                f"{_RUNPOD_REST_BASE}/networkvolumes",
                headers=self._rest_headers(credentials),
                timeout=_REST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise TrainerTargetError(
                f"Failed to list RunPod network volumes: {exc}", retryable=True
            ) from exc
        if resp.status_code >= 400:
            raise TrainerTargetError(
                f"RunPod volume list failed ({resp.status_code}): {resp.text[:300]}",
                retryable=resp.status_code >= 500,
            )
        out: list[NetworkVolume] = []
        for vol in resp.json() or []:
            vid = str(vol.get("id") or "")
            if not vid:
                continue
            out.append(
                NetworkVolume(
                    id=vid,
                    name=str(vol.get("name") or ""),
                    size_gb=int(vol.get("size") or 0),
                    datacenter_id=str(vol.get("dataCenterId") or ""),
                    created_by_app=str(vol.get("name") or "").startswith(
                        _APP_VOLUME_NAME_PREFIX
                    ),
                )
            )
        return out

    def ensure_network_volume(
        self,
        *,
        credentials: TrainerCredentials,
        name: str,
        size_gb: int,
        datacenter_id: str | None = None,
    ) -> NetworkVolume:
        # Reuse a same-named volume if one already exists so repeated connects
        # are idempotent (and don't rack up storage cost with duplicates).
        for existing in self._list_volumes(credentials):
            if existing.name == name:
                logger.info(
                    "RunPod: reusing network volume %s (%s, %dGB) in %s",
                    existing.id,
                    existing.name,
                    existing.size_gb,
                    existing.datacenter_id or "unknown dc",
                )
                return existing
        import requests  # noqa: PLC0415

        datacenter = datacenter_id or self._pick_volume_datacenter(credentials)
        if datacenter_id:
            storage_dcs = self._storage_datacenters(credentials)
            if datacenter not in storage_dcs:
                raise TrainerTargetError(
                    f"RunPod datacenter {datacenter!r} does not support network "
                    "volumes. Refresh account regions and choose a storage-capable "
                    "datacenter.",
                    retryable=False,
                )
            health = self._region_health(credentials, datacenter)
            if health.status != "healthy":
                reason = (
                    "has no qualifying 32GB+ GPU stock"
                    if health.status == "no_stock"
                    else "GPU stock could not be verified"
                )
                raise TrainerTargetError(
                    f"Refusing to create a paid cache volume in {datacenter}: "
                    f"the region {reason}. Refresh RunPod availability or choose "
                    "another healthy region.",
                    retryable=False,
                )
        logger.info(
            "RunPod: creating %dGB network volume '%s' in datacenter %s",
            size_gb,
            name,
            datacenter,
        )
        try:
            resp = requests.post(
                f"{_RUNPOD_REST_BASE}/networkvolumes",
                headers=self._rest_headers(credentials),
                json={"name": name, "size": size_gb, "dataCenterId": datacenter},
                timeout=_REST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise TrainerTargetError(
                f"Failed to create RunPod network volume: {exc}", retryable=True
            ) from exc
        if resp.status_code >= 400:
            raise TrainerTargetError(
                f"RunPod volume create failed ({resp.status_code}): {resp.text[:300]}",
                retryable=False,
            )
        vol = resp.json() or {}
        created = NetworkVolume(
            id=str(vol.get("id") or ""),
            name=str(vol.get("name") or name),
            size_gb=int(vol.get("size") or size_gb),
            datacenter_id=str(vol.get("dataCenterId") or datacenter),
            created_by_app=True,
        )
        logger.info(
            "RunPod: created network volume %s in %s",
            created.id,
            created.datacenter_id,
        )
        return created

    def _pick_volume_datacenter(self, credentials: TrainerCredentials) -> str:
        """Choose a datacenter for a new network volume.

        Critically, the volume pins the region for ALL future pods, so we pick a
        storage-capable datacenter that ALSO currently has training GPUs in
        stock — otherwise the user ends up with a volume in a region where no
        pod can ever launch (exactly the EU-RO-1 dead end). Preference order:
        a DC with an 80GB+ GPU available, then any 32GB+ GPU, then any
        storage-capable DC, then a known-good default.
        """
        storage_dcs = self._storage_datacenters(credentials)
        fallback_32 = ""
        for dc in storage_dcs:
            offers = self._discover_gpus_graphql(credentials, dc)
            if offers is None:
                continue
            available = [o for o in offers if o.available]
            if any(o.memory_gb >= _MIN_PREFERRED_GPU_VRAM_GB for o in available):
                logger.info("RunPod: selecting datacenter %s (80GB+ GPU in stock)", dc)
                return dc
            if not fallback_32 and available:
                fallback_32 = dc
        if fallback_32:
            logger.info(
                "RunPod: selecting datacenter %s (training GPU in stock, <80GB)",
                fallback_32,
            )
            return fallback_32
        raise TrainerTargetError(
            "No storage-capable RunPod datacenter currently has qualifying "
            "32GB+ GPU stock. No paid cache volume was created. Refresh "
            "availability later or run this pipeline uncached in any region.",
            retryable=False,
        )

    def delete_network_volume(
        self, *, credentials: TrainerCredentials, volume_id: str
    ) -> None:
        volumes = self._list_volumes(credentials)
        volume = next((v for v in volumes if v.id == volume_id), None)
        if volume is None:
            return
        if not volume.created_by_app:
            raise TrainerTargetError(
                f"Refusing to delete RunPod volume {volume_id}: it was not "
                "created by LTX Desktop",
                retryable=False,
            )
        import requests  # noqa: PLC0415

        try:
            resp = requests.delete(
                f"{_RUNPOD_REST_BASE}/networkvolumes/{volume_id}",
                headers=self._rest_headers(credentials),
                timeout=_REST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise TrainerTargetError(
                f"Failed to delete RunPod network volume: {exc}", retryable=True
            ) from exc
        if resp.status_code not in (200, 202, 204, 404):
            raise TrainerTargetError(
                f"RunPod volume delete failed ({resp.status_code}): "
                f"{resp.text[:300]}",
                retryable=resp.status_code >= 500,
            )

    def _storage_datacenters(self, credentials: TrainerCredentials) -> list[str]:
        """Storage-capable datacenter ids (best-effort)."""
        import requests  # noqa: PLC0415

        try:
            resp = requests.get(
                f"{_RUNPOD_REST_BASE}/datacenters",
                headers=self._rest_headers(credentials),
                timeout=_REST_TIMEOUT_SECONDS,
            )
            if resp.status_code < 400:
                return [
                    str(dc["id"])
                    for dc in (resp.json() or [])
                    if dc.get("storageSupport") and dc.get("id")
                ]
        except Exception:
            pass
        return [_DEFAULT_VOLUME_DATACENTER]

    def ensure_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle | None
    ) -> TargetHandle:
        runpod = _runpod(credentials)
        if handle is not None and handle.pod_id is not None:
            pod = runpod.get_pod(handle.pod_id)
            if pod:
                desired = str(pod.get("desiredStatus") or "").upper()
                if desired and desired != "RUNNING":
                    self._require_app_owned_pod(
                        runpod,
                        handle.pod_id,
                        action="resume",
                        allow_gone=False,
                    )
                    logger.info(
                        "RunPod: existing pod %s is %s; resuming it",
                        handle.pod_id,
                        desired,
                    )
                    try:
                        runpod.resume_pod(handle.pod_id)
                    except Exception as exc:
                        raise TrainerTargetError(
                            f"Failed to resume RunPod pod {handle.pod_id}: {exc}",
                            retryable=True,
                        ) from exc
                logger.info("RunPod: reusing existing pod %s", handle.pod_id)
                self._wait_until_ready(runpod, handle.pod_id)
                return handle
            logger.info(
                "RunPod: previous pod %s is gone; creating a new one", handle.pod_id
            )
        if credentials.runpod_network_volume_id:
            selected = next(
                (
                    volume
                    for volume in self._list_volumes(credentials)
                    if volume.id == credentials.runpod_network_volume_id
                ),
                None,
            )
            if selected is None:
                raise TrainerTargetError(
                    "Selected RunPod cache volume no longer exists",
                    retryable=False,
                )
            if (
                credentials.runpod_datacenter
                and selected.datacenter_id
                and credentials.runpod_datacenter != selected.datacenter_id
            ):
                raise TrainerTargetError(
                    "Selected datacenter does not match the cache volume region",
                    retryable=False,
                )
        # Auto-pick the cheapest in-stock GPU when the user hasn't chosen one,
        # so the happy path needs zero GPU selection.
        gpu_type = credentials.runpod_gpu_type or self._cheapest_available_gpu(
            credentials, runpod
        )
        if not gpu_type:
            raise TrainerTargetError(
                "No training GPU is in stock right now — try again shortly.",
                retryable=True,
            )
        workspace = credentials.workspace_dir or "/workspace"
        # Pin HF/uv/pip caches + tmp to the network volume. RunPod's container
        # disk is small (~20GB), and these caches (torch/CUDA wheels, HF blobs)
        # otherwise fill it — which is exactly what pushed the pod to 106% disk.
        cache_root = f"{workspace}/.cache"
        env: dict[str, str] = {
            "XDG_CACHE_HOME": cache_root,
            "HF_HOME": f"{cache_root}/huggingface",
            "HF_HUB_CACHE": f"{cache_root}/huggingface/hub",
            "UV_CACHE_DIR": f"{cache_root}/uv",
            "PIP_CACHE_DIR": f"{cache_root}/pip",
            "TRITON_CACHE_DIR": f"{cache_root}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/inductor",
            "TMPDIR": f"{workspace}/tmp",
        }
        # Inject our public key so the new pod accepts key-based SSH with
        # no manual key registration on the RunPod account.
        if self._key_manager is not None:
            _, public_key = self._key_manager.ensure_keypair()
            env["PUBLIC_KEY"] = public_key
        create_kwargs: dict[str, Any] = {
            "name": _APP_POD_NAME,
            "image_name": credentials.runpod_image or DEFAULT_RUNPOD_IMAGE,
            "gpu_type_id": gpu_type,
            "support_public_ip": True,
            "ports": "22/tcp",
            # Generous container disk as a safety net for anything that still
            # lands outside the volume (build temp, base image layers).
            "container_disk_in_gb": _CONTAINER_DISK_GB,
            "env": env,
        }
        if credentials.runpod_datacenter and not credentials.runpod_network_volume_id:
            create_kwargs["data_center_id"] = credentials.runpod_datacenter
        # /workspace storage. CRITICAL: always set volume_mount_path to the
        # workspace, for BOTH the network-volume and ephemeral cases. Without
        # it the volume mounted at RunPod's default path (not /workspace), so
        # everything we wrote to /workspace landed on the small container disk
        # — that's why the container disk filled (87%/100GB) while the network
        # volume sat at 0%, and why downloads/latents ran out of space.
        volume_gb = credentials.volume_size_gb or _DEFAULT_EPHEMERAL_VOLUME_GB
        create_kwargs["volume_mount_path"] = workspace
        if credentials.runpod_network_volume_id:
            # Persistent volume: survives teardown, so the marker-gated
            # install/download only runs once.
            create_kwargs["network_volume_id"] = credentials.runpod_network_volume_id
        else:
            # Ephemeral pod-local volume (caching off), sized for models + data.
            create_kwargs["volume_in_gb"] = volume_gb
        return self._create_pod(runpod, credentials, create_kwargs)

    def _create_pod(
        self,
        runpod: Any,
        credentials: TrainerCredentials,
        create_kwargs: dict[str, Any],
    ) -> TargetHandle:
        """Create a pod on the user's chosen GPU only.

        We deliberately do NOT silently fall back to a different GPU type when
        the chosen one is out of stock — GPUs differ in price, so the choice is
        the user's. On unavailability we raise an actionable error listing the
        training-capable GPUs that *are* in stock so the training flow can ask
        the user to choose again without losing the dataset or job.
        """
        gpu_id = str(create_kwargs.get("gpu_type_id") or "")
        volume = credentials.runpod_network_volume_id
        logger.info(
            "RunPod: creating pod on GPU '%s'%s",
            gpu_id,
            f" (network volume {volume})" if volume else " (ephemeral, no volume)",
        )
        try:
            created = runpod.create_pod(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - classify by message below
            if any(m in str(exc).lower() for m in _UNAVAILABLE_MARKERS):
                available = self._available_training_gpus(runpod, exclude=gpu_id)
                options = (
                    "; ".join(available[:6]) if available else "none right now"
                )
                region_note = (
                    " (the saved-model storage selected for this run is tied "
                    "to one datacenter)"
                    if volume
                    else ""
                )
                logger.warning(
                    "RunPod: requested GPU '%s' is unavailable. In stock: %s",
                    gpu_id,
                    options,
                )
                stock_message = (
                    f"Currently available alternatives include: {options}. "
                    if available
                    else "No compatible training GPU is currently in stock. "
                )
                raise TrainerTargetError(
                    f"GPU '{gpu_id}' is unavailable right now{region_note}. "
                    f"{stock_message}"
                    "Return to GPU selection and choose another available GPU, "
                    "or refresh and retry later. Your dataset and progress are "
                    "preserved.",
                    retryable=False,
                    code="capacity_unavailable",
                ) from exc
            logger.error("RunPod: pod create failed on '%s': %s", gpu_id, exc)
            raise TrainerTargetError(
                f"Failed to create RunPod pod: {exc}", retryable=True
            ) from exc
        raw_pod_id = created.get("id") if isinstance(created, dict) else created
        pod_id = str(raw_pod_id or "").strip()
        if not pod_id:
            raise TrainerTargetError(
                "RunPod created a pod but returned no pod id; check the account "
                "before retrying to avoid duplicate billing.",
                retryable=False,
            )
        logger.info("RunPod: pod %s created on GPU '%s'; waiting for ready", pod_id, gpu_id)
        try:
            self._wait_until_ready(runpod, pod_id)
        except Exception:
            # Creation succeeded, so this pod is billable even if readiness
            # polling/SSH setup fails. Tear it down before surfacing the error;
            # otherwise every reconcile retry can create another paid orphan.
            logger.warning(
                "RunPod: pod %s failed readiness; terminating it to stop billing",
                pod_id,
            )
            try:
                runpod.terminate_pod(pod_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.error(
                    "RunPod: CRITICAL — failed to terminate unready pod %s: %s",
                    pod_id,
                    cleanup_exc,
                )
            raise
        return TargetHandle(provider="runpod", pod_id=pod_id)

    def _cheapest_available_gpu(self, credentials: TrainerCredentials, runpod: Any) -> str:
        """The cheapest in-stock training GPU id, or "" if none available."""
        offers = self._discover_gpus(credentials, runpod, "")
        available = [o for o in offers if o.available]
        if not available:
            return ""
        # Cheapest by price (then smallest VRAM as a tiebreak); GPUs without a
        # reported price sort last.
        cheapest = min(
            available,
            key=lambda o: (
                o.price_per_hr if o.price_per_hr is not None else float("inf"),
                o.memory_gb,
            ),
        )
        logger.info(
            "RunPod: auto-selected cheapest in-stock GPU '%s'%s",
            cheapest.id,
            f" (${cheapest.price_per_hr:.2f}/hr)" if cheapest.price_per_hr else "",
        )
        return cheapest.id

    def _available_training_gpus(self, runpod: Any, *, exclude: str) -> list[str]:
        """`"<id> ($X/hr)"` for in-stock, training-capable GPUs (for error text)."""
        try:
            gpus = runpod.get_gpus() or []
        except Exception:
            return []
        out: list[tuple[int, str]] = []
        for gpu in gpus:
            gid = str(gpu.get("id") or "")
            memory_gb = int(gpu.get("memoryInGb") or 0)
            if not gid or gid == exclude or memory_gb < LORA_MIN_GPU_VRAM_GB:
                continue
            price, available = self._gpu_price_and_stock(runpod, gid)
            if available:
                label = f"{gid} (${price:.2f}/hr)" if price is not None else gid
                out.append((memory_gb, label))
        out.sort(key=lambda item: item[0])
        return [label for _, label in out]

    def ensure_provisioned(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        progress: ProgressCallback | None = None,
    ) -> None:
        if not credentials.auto_provision:
            return
        if not credentials.trainer_repo_url:
            raise TrainerTargetError(
                "Auto-provisioning is enabled but no trainer repo URL is "
                "configured (set loraTrainerRepoUrl or disable loraAutoProvision)",
                retryable=False,
            )
        workspace = credentials.workspace_dir or "/workspace"
        marker = paths.provision_marker_path(workspace)
        marker_value = paths.provision_marker_value(
            credentials.trainer_repo_url,
            credentials.trainer_repo_ref or "main",
        )
        uv_bin = paths.uv_bin_dir(workspace)
        remote = self._remote(credentials, handle)
        # The marker records the exact trainer source/ref contract. A plain
        # legacy marker or a marker for another revision forces reprovisioning;
        # this prevents a reused network volume from silently running stale
        # trainer code. uv must also be runnable because older provisions put it
        # on ephemeral container storage.
        _, out, _ = remote.run(
            f'export PATH={shlex.quote(uv_bin)}:"$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; '
            f"test -f {shlex.quote(marker)} "
            f"&& test \"$(cat {shlex.quote(marker)})\" = {shlex.quote(marker_value)} "
            "&& command -v uv >/dev/null 2>&1 "
            "&& echo __present__ || echo __absent__"
        )
        if "__present__" in out:
            logger.info(
                "RunPod: workspace already provisioned (marker + uv present) — "
                "skipping setup"
            )
            # A workspace provisioned before the ffmpeg fix (or one whose
            # marker survived on a reused network volume) never got the system
            # ffmpeg torchaudio's audio backend needs. Idempotent no-op
            # otherwise; never fatal — a missing ffmpeg only breaks audio
            # training, which the post-preprocess audio guard reports clearly.
            try:
                remote.run(paths.ensure_ffmpeg_command())
            except Exception as exc:  # noqa: BLE001
                logger.warning("RunPod: ensure-ffmpeg skipped (%s)", exc)
            # Backfill the audio-fallback monkeypatch too (see
            # `patch_trainer_audio_fallback_command`): a workspace provisioned
            # before this patch has a `process_videos.py` whose
            # `torchaudio.load` raises on cu128/WSL torchcodec, silently
            # skipping every clip's audio. Idempotent + best-effort.
            try:
                remote.run(paths.patch_trainer_audio_fallback_command(workspace))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RunPod: trainer audio-fallback patch skipped (%s)", exc)
            return
        logger.info(
            "RunPod: provisioning workspace at %s — cloning trainer (%s@%s), "
            "uv sync, downloading checkpoint '%s' + encoder (this can take several "
            "minutes on a cold pod)",
            workspace,
            credentials.trainer_repo_url,
            credentials.trainer_repo_ref or "main",
            credentials.model_filename or credentials.model_hf_repo,
        )
        command = paths.provision_command(
            workspace_dir=workspace,
            repo_url=credentials.trainer_repo_url,
            repo_ref=credentials.trainer_repo_ref or "main",
            marker_path=marker,
            model_hf_repo=credentials.model_hf_repo,
            model_filename=credentials.model_filename,
            model_path=credentials.model_path,
            text_encoder_hf_repo=credentials.text_encoder_hf_repo,
            text_encoder_path=credentials.text_encoder_path,
            hf_token=credentials.hf_token,
        )
        job_id = remote.run_detached(command=command, workdir=workspace)
        # Immediate heartbeat so the card updates the instant setup starts,
        # rather than sitting static until the first poll.
        if progress is not None:
            progress("Cloning trainer & installing dependencies…")
        deadline = time.monotonic() + _PROVISION_TIMEOUT_SECONDS
        last_reported = ""
        while time.monotonic() < deadline:
            status = remote.poll(job_id)
            if status.state == "succeeded":
                logger.info("RunPod: workspace provisioning complete")
                if progress is not None:
                    progress("Setup complete")
                return
            if status.state == "failed":
                tail = remote.read_logs(job_id, 40)
                detail = redact_text(
                    " | ".join(tail[-12:])
                    if tail
                    else (status.error or "unknown error")
                )
                logger.error("RunPod: workspace provisioning failed: %s", detail)
                raise TrainerTargetError(
                    f"Workspace provisioning failed: {detail}", retryable=False
                )
            # Surface live setup progress (git clone %, uv sync, HF download
            # bar) from the remote log so the user sees it's actually working.
            line = redact_text(_latest_progress_line(remote.read_logs(job_id, 8)))
            if line and line != last_reported:
                last_reported = line
                logger.info("RunPod setup: %s", line)
                if progress is not None:
                    progress(line)
            time.sleep(_PROVISION_POLL_INTERVAL_SECONDS)
        # Include the remote log tail so the timeout says WHERE it stalled
        # (the last `[provision] <step>` marker), not just "timed out".
        tail = remote.read_logs(job_id, 30)
        where = redact_text(
            " | ".join(t for t in tail[-8:] if t.strip()) or "no remote output"
        )
        logger.error("RunPod: workspace provisioning timed out — last: %s", where)
        raise TrainerTargetError(
            f"Workspace provisioning timed out. Last remote output: {where}",
            retryable=True,
        )

    def _wait_until_ready(self, runpod: Any, pod_id: str) -> None:
        deadline = time.monotonic() + _POD_READY_TIMEOUT_SECONDS
        waited = 0
        while time.monotonic() < deadline:
            pod = runpod.get_pod(pod_id)
            runtime = (pod or {}).get("runtime") if isinstance(pod, dict) else None
            if runtime and (runtime.get("ports")):
                logger.info("RunPod: pod %s is ready (SSH exposed)", pod_id)
                return
            if waited and waited % 60 == 0:
                logger.info(
                    "RunPod: still waiting for pod %s to start (%ds elapsed)",
                    pod_id,
                    waited,
                )
            time.sleep(_POD_POLL_INTERVAL_SECONDS)
            waited += _POD_POLL_INTERVAL_SECONDS
        logger.error("RunPod: pod %s did not become ready in time", pod_id)
        raise TrainerTargetError(
            f"RunPod pod {pod_id} did not become ready in time", retryable=True
        )

    def upload_directory(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        local_dir: str,
        remote_dir: str,
    ) -> None:
        self._remote(credentials, handle).upload_directory(
            local_dir=local_dir, remote_dir=remote_dir
        )

    def download_file(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        self._remote(credentials, handle).download_file(
            remote_path=remote_path, local_path=local_path
        )

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        return self._remote(credentials, handle).run_detached(
            command=command, workdir=workdir
        )

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        return self._remote(credentials, handle).poll(remote_job_id)

    def read_logs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
        tail: int,
    ) -> list[str]:
        return self._remote(credentials, handle).read_logs(remote_job_id, tail)

    def terminate(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> None:
        self._remote(credentials, handle).terminate(remote_job_id)

    def release_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> None:
        if handle.pod_id is None:
            return
        # Cached workspaces keep their network volume and stop the pod so GPU
        # billing ends reversibly. Ephemeral workspaces have no durable volume
        # and are terminated below.
        if credentials.runpod_network_volume_id:
            self.stop_pod(credentials=credentials, pod_id=handle.pod_id)
            return
        runpod = _runpod(credentials)
        # Scope the API-level pod kill to pods LTX Desktop created. The pod's
        # name is the app-ownership signal (`_APP_POD_NAME`, set at create time
        # and surfaced as `created_by_app` in the connect UI). Refusing a
        # same-account pod we didn't spawn prevents a corrupted/hand-crafted
        # handle or a stray connect-UI request from terminating the user's
        # other RunPod workloads. A pod that's already gone (not found / null)
        # is treated as released (idempotent), matching the prior behaviour.
        try:
            pod = runpod.get_pod(handle.pod_id)
        except Exception as exc:
            if any(m in str(exc).lower() for m in _POD_GONE_MARKERS):
                logger.info(
                    "RunPod: pod %s already gone — treating as released",
                    handle.pod_id,
                )
                return
            logger.error(
                "RunPod: failed to look up pod %s: %s", handle.pod_id, exc
            )
            raise TrainerTargetError(
                f"Failed to look up RunPod pod {handle.pod_id}: {exc}",
                retryable=True,
            ) from exc
        if not pod:
            logger.info(
                "RunPod: pod %s already gone — treating as released", handle.pod_id
            )
            return
        if str(pod.get("name") or "") != _APP_POD_NAME:
            logger.warning(
                "RunPod: refusing to terminate pod %s not owned by LTX Desktop "
                "(name=%r)",
                handle.pod_id,
                pod.get("name"),
            )
            raise TrainerTargetError(
                f"Refusing to terminate RunPod pod {handle.pod_id}: it was not "
                "created by LTX Desktop",
                retryable=False,
                code="ownership_violation",
            )
        logger.info("RunPod: terminating pod %s", handle.pod_id)
        try:
            runpod.terminate_pod(handle.pod_id)
        except Exception as exc:
            # Terminating an already-gone pod is success (idempotent): the goal
            # — pod not running — is achieved. Swallow it so the caller clears
            # the handle instead of retrying the terminate every tick forever.
            if any(m in str(exc).lower() for m in _POD_GONE_MARKERS):
                logger.info(
                    "RunPod: pod %s already gone — treating as released",
                    handle.pod_id,
                )
                return
            logger.error(
                "RunPod: failed to terminate pod %s: %s", handle.pod_id, exc
            )
            raise TrainerTargetError(
                f"Failed to terminate RunPod pod {handle.pod_id}: {exc}",
                retryable=True,
            ) from exc
        logger.info("RunPod: pod %s terminated", handle.pod_id)

    def query_gpu(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> GpuTelemetry:
        code, out, err = self._remote(credentials, handle).run(NVIDIA_SMI_GPU_QUERY)
        if code != 0:
            raise TrainerTargetError(
                "RunPod: nvidia-smi failed on pod: "
                f"{err.strip() or 'command failed'}",
                retryable=True,
            )
        return parse_nvidia_smi_output(out)

    def list_validation_outputs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
        since_step: int,
    ) -> list[ValidationArtifact]:
        samples_dir = samples_dir_for(remote_output_dir)
        # Fixed `ls -1` against the run's own samples dir (server-derived).
        # shlex-quote the path defensively; `|| true` -> empty list when the
        # dir doesn't exist yet (validation hasn't run).
        code, out, _ = self._remote(credentials, handle).run(
            f"ls -1 {shlex.quote(samples_dir)} 2>/dev/null || true"
        )
        if code != 0:
            return []
        return [
            a for a in parse_samples_listing(out, samples_dir) if a.step > since_step
        ]

    def list_checkpoints(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
    ) -> list[int]:
        ckpt_dir = checkpoints_dir_for(remote_output_dir)
        # Fixed `ls -1` against the run's own checkpoints dir (server-derived).
        # shlex-quote the path defensively; `|| true` -> empty list when the
        # dir doesn't exist yet (training hasn't checkpointed).
        code, out, _ = self._remote(credentials, handle).run(
            f"ls -1 {shlex.quote(ckpt_dir)} 2>/dev/null || true"
        )
        if code != 0:
            return []
        return parse_checkpoints_listing(out)

    def count_precomputed_source(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        precomputed_dir: str,
        source: str,
    ) -> int:
        src_dir = f"{precomputed_dir.rstrip('/')}/{source}"
        # Recursive count of .pt files under the source dir. `2>/dev/null` so a
        # not-yet-created source yields 0 (process_dataset.py wrote nothing for
        # it) rather than an error; `|| true` keeps the pipe's exit status 0 so
        # a missing dir doesn't surface as a transport failure.
        code, out, _ = self._remote(credentials, handle).run(
            f"find {shlex.quote(src_dir)} -type f -name '*.pt' 2>/dev/null | wc -l"
        )
        if code != 0:
            return 0
        try:
            return max(0, int(out.strip()))
        except ValueError:
            return 0

    def delete_remote_paths(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        paths: list[str],
    ) -> None:
        if not paths:
            return
        # `rm -rf` each quoted path; `|| true` so a missing path (a prior run
        # that died before writing anything) isn't an error — reset must
        # succeed regardless of how far the previous run got.
        quoted = " ".join(shlex.quote(p) for p in paths)
        code, _, err = self._remote(credentials, handle).run(
            f"rm -rf {quoted} 2>/dev/null || true"
        )
        if code != 0:
            raise TrainerTargetError(
                "RunPod: failed to delete remote paths: "
                f"{err.strip() or 'rm failed'}",
                retryable=True,
            )
