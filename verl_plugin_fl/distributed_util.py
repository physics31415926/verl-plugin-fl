# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from verl/utils/distributed.py with FlagCX support added.

"""Distributed communication utilities for multi-chip platforms.

Provides ``stateless_init_process_group`` which creates a stateless
communicator for weight synchronization with vLLM workers, selecting the
appropriate backend (FlagCX / HCCL / NCCL) based on the active platform.

This extends the main line's ``verl.utils.distributed.stateless_init_process_group``
with FlagCX support for MUSA and other multi-chip platforms.
"""

import logging
import socket
from datetime import timedelta

from verl.utils.device import get_device_name, get_nccl_backend, is_npu_available
from verl.utils.net_utils import is_ipv6

logger = logging.getLogger(__name__)


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Create a stateless communicator for weight synchronization with vLLM workers.

    This is an extended version of ``verl.utils.distributed.stateless_init_process_group``
    that adds FlagCX support for multi-chip platforms (MUSA, etc.).

    Uses vLLM's ``StatelessProcessGroup`` for TCP rendezvous, then initialises
    the appropriate data-plane communicator based on the platform backend:
    - FlagCX: :class:`PyFlagcxCommunicator` from vllm_fl
    - NPU (Ascend): ``PyHcclCommunicator`` from vllm_ascend
    - CUDA: ``PyNcclCommunicator`` from vllm
    """
    from torch.distributed import TCPStore
    from vllm.distributed.utils import StatelessProcessGroup

    comm_backend = get_nccl_backend()
    logger.info(
        "stateless_init_process_group: backend=%s, rank=%d, world_size=%d, device=%s",
        comm_backend,
        rank,
        world_size,
        device,
    )

    # Create process group with IPv6 support (copied from main line)
    def create_process_group(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
    ) -> "StatelessProcessGroup":
        """Create StatelessProcessGroup with IPv6 support."""
        launch_server = rank == 0
        if launch_server:
            # listen on the specified interface (instead of 0.0.0.0)
            if is_ipv6(master_address):
                listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            else:
                listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
            listen_fd = listen_socket.fileno()
        else:
            listen_socket = None
            listen_fd = None

        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=launch_server,
            timeout=timedelta(seconds=store_timeout),
            use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
            master_listen_fd=listen_fd,
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            socket=listen_socket,
            data_expiration_seconds=data_expiration_seconds,
        )

    pg = create_process_group(host=master_address, port=master_port, rank=rank, world_size=world_size)

    # Select communicator based on backend
    if comm_backend == "flagcx":
        from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

        # Convert int device to device string (e.g., 0 -> "musa:0" or "cuda:0")
        if isinstance(device, int):
            device_name = get_device_name()
            device = f"{device_name}:{device}"
        return PyFlagcxCommunicator(pg, device=device)
    elif is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator(pg, device=device)
