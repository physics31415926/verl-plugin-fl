# Copyright (c) 2026 BAAI. All rights reserved.

"""FL engine plugin - importing this module registers FL engines via @EngineRegistry.register().

The plugin calls get_device_name() at import time and uses "last writer wins" to override
the default engine for the detected hardware. No environment variables needed — just
configure custom_engine_module.

Usage:
    In verl config YAML:
        fsdp_config:
            custom_engine_module: "pkg://verl_plugin_fl.engine"
"""

from verl_plugin_fl.engine.fsdp_fl import FSDPFLEngineWithLMHead, FSDPFLEngineWithValueHead  # noqa: F401
from verl_plugin_fl.engine.megatron_fl import MegatronFLEngineWithLMHead  # noqa: F401

__all__ = ["FSDPFLEngineWithLMHead", "FSDPFLEngineWithValueHead", "MegatronFLEngineWithLMHead"]
