# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from verl/utils/device.py in verl-physics PR #10.

"""Device utilities for multi-chip platforms.

Provides ``get_dist_backend()`` which returns the compound backend string
for ``torch.distributed.init_process_group`` that works across all platforms
(CUDA, NPU, MUSA, CPU).
"""

from verl.plugin.platform import get_platform


def get_dist_backend() -> str:
    """Get the compound backend string for ``init_process_group``.

    Returns ``"gloo"`` on CPU, otherwise ``"cpu:gloo,<device>:<comm>"``
    (e.g. ``"cpu:gloo,cuda:nccl"``, ``"cpu:gloo,musa:flagcx"``).
    Works for any device type — no device-specific branches needed.
    """
    platform = get_platform()
    device_name = platform.device_name
    if device_name == "cpu":
        return "gloo"
    comm_backend = platform.communication_backend_name()
    return f"cpu:gloo,{device_name}:{comm_backend}"
