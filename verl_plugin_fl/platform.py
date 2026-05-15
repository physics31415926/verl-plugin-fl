# Copyright (c) 2026 BAAI. All rights reserved.

"""FL platform registration.

Importing this module ensures the platform singleton is initialised early
and logs the FL multi-chip configuration.  The built-in PlatformCUDA /
PlatformNPU already support FlagCX (via ``USE_FLAGCX`` env var), so no
custom subclass is needed — we just trigger ``get_platform()`` and attach
FL-specific diagnostics.

This module also registers the MUSA platform (from platform_musa.py) if
torch_musa is available.

Usage:
    export VERL_USE_EXTERNAL_MODULES=verl_plugin_fl
"""

import logging

from verl.plugin.platform import PlatformRegistry, get_platform
from verl_plugin_fl.utils.config_manager import FLEnvManager

logger = logging.getLogger(__name__)


def register_platform() -> None:
    """Eagerly initialise the platform and log FL status.

    Also registers MUSA platform if torch_musa is available.
    """
    # Register MUSA platform (import triggers @PlatformRegistry.register)
    try:
        from verl_plugin_fl.platform_musa import PlatformMUSA  # noqa: F401

        logger.debug("MUSA platform registered via verl_plugin_fl")
    except ImportError:
        pass

    platform = get_platform()

    # Call ensure_initialized() if the platform supports it (MUSA only)
    if hasattr(platform, "ensure_initialized"):
        platform.ensure_initialized()
        logger.debug("Platform %s initialized via ensure_initialized()", platform.device_name)

    logger.info(
        "FL platform: %s (%s) | registered platforms: %s | %s",
        platform.device_name,
        type(platform).__name__,
        PlatformRegistry.registered_names(),
        FLEnvManager.get_summary(),
    )
