# Copyright (c) 2026 BAAI. All rights reserved.

"""MUSA worker device management patches.

Provides ``patch_worker_for_musa()`` which monkey-patches the Worker class's
``_setup_device`` method to handle MUSA-specific device isolation.

Background:
    Ray sets CUDA_VISIBLE_DEVICES for GPU resource isolation, but torch_musa
    ignores it (reads MUSA_VISIBLE_DEVICES instead). Unlike CUDA/NCCL,
    MCCL/FlagCX does not work when MUSA_VISIBLE_DEVICES restricts each
    process to a single device. Instead, keep all devices visible and
    use set_device() with the Ray-assigned physical device index.

Usage in recipe:
    from verl_plugin_fl.worker_patch import patch_worker_for_musa
    patch_worker_for_musa()  # call before creating workers
"""

import logging
import os

import ray

from verl.utils.device import get_device_name, get_torch_device

logger = logging.getLogger(__name__)


def setup_musa_device_for_worker():
    """Configure MUSA device isolation for the current Ray worker.

    Should be called inside a Ray actor's setup phase (e.g., in _setup_device
    or at the beginning of worker initialization).

    This handles the MUSA-specific case where:
    - Ray sets CUDA_VISIBLE_DEVICES but MUSA ignores it
    - MCCL/FlagCX needs all devices visible
    - We use set_device() with the physical device index instead
    """
    if get_device_name() != "musa":
        return

    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if cuda_val:
        musa_device_index = int(cuda_val)
    else:
        # Ray may not set CUDA_VISIBLE_DEVICES for MUSA workers.
        # Fall back to Ray runtime context to get the assigned device.
        try:
            gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
            musa_device_index = int(gpu_ids[0]) if gpu_ids else 0
        except Exception:
            musa_device_index = 0

    # Remove MUSA_VISIBLE_DEVICES to keep all devices visible
    os.environ.pop("MUSA_VISIBLE_DEVICES", None)
    os.environ["LOCAL_RANK"] = str(musa_device_index)
    get_torch_device().set_device(musa_device_index)

    logger.debug("MUSA worker: set_device(%d), LOCAL_RANK=%d", musa_device_index, musa_device_index)


def setup_musa_for_vllm_rollout():
    """Configure MUSA-specific settings for vLLM rollout workers.

    Returns the local_rank to use for vLLM initialization.
    Also disables torch.compile/inductor on MUSA (triton NVIDIA backend
    tries to link libcuda.so which doesn't exist on MUSA nodes).

    Returns:
        int or None: The local_rank for MUSA, or None if not on MUSA platform.
    """
    import torch

    if get_device_name() != "musa":
        return None

    # MUSA workers see all devices (no MUSA_VISIBLE_DEVICES isolation),
    # so use the LOCAL_RANK set by worker setup (physical device index).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Disable torch.compile/inductor on MUSA: the standard triton NVIDIA
    # backend tries to link libcuda.so which doesn't exist on MUSA nodes.
    torch._dynamo.config.disable = True

    logger.debug("MUSA vLLM rollout: local_rank=%d, dynamo disabled", local_rank)
    return local_rank
