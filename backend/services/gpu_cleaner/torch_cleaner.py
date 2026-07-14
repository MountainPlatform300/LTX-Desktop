"""GPU cleanup helper service."""

from __future__ import annotations

import gc

import torch

from services.services_utils import empty_device_cache


class TorchCleaner:
    """Wraps GPU memory cleanup operations."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = device

    def cleanup(self) -> None:
        # Order matters: collect BEFORE emptying the cache.
        #
        # diffusers pipelines run with enable_model_cpu_offload keep reference
        # cycles (module <-> forward hooks <-> hook closures) that stop Python's
        # refcount from deallocating their CUDA tensors when the pipeline is
        # dropped. gc.collect() breaks those cycles so the tensor storages are
        # freed back to the caching allocator; only then can empty_cache() hand
        # those blocks back to the driver.
        #
        # The previous order (empty_cache -> gc) returned nothing on the first
        # call because the tensors were still live, then freed the cycles with
        # no follow-up empty_cache — leaving VRAM pinned even after the pipeline
        # was unloaded. That was the root cause of the Klein -> LTX OOM and the
        # "GPU still full when nothing is running" symptom. Two collects catch
        # tensors released by finalizers that run during the first sweep.
        gc.collect()
        gc.collect()
        empty_device_cache(self._device)
