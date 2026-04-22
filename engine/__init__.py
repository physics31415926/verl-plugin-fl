# Copyright (c) 2026 BAAI. All rights reserved.

"""FL engine plugin - importing this module registers FL engines via @EngineRegistry.register().

Usage:
    In verl config YAML:
        engine:
            custom_engine_module: "pkg://verl_plugin_fl.engine"
"""

from verl_plugin_fl.engine.fsdp_fl import FSDPFLEngineWithLMHead, FSDPFLEngineWithValueHead  # noqa: F401
from verl_plugin_fl.engine.megatron_fl import MegatronFLEngineWithLMHead  # noqa: F401

__all__ = ["FSDPFLEngineWithLMHead", "FSDPFLEngineWithValueHead", "MegatronFLEngineWithLMHead"]
