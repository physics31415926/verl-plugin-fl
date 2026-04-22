# Copyright (c) 2026 BAAI. All rights reserved.

"""FL engine plugin - importing this module registers FL engines via @EngineRegistry.register().

Usage:
    In verl config YAML:
        fsdp_config:
            custom_engine_module: "pkg://verl_plugin_fl.engine"
"""

import os

# Auto-set VERL_ENGINE_DEVICE to "flagos" if not already configured.
# This allows users to only configure custom_engine_module without
# needing to manually export VERL_ENGINE_DEVICE=flagos.
os.environ.setdefault("VERL_ENGINE_DEVICE", "flagos")

from verl_plugin_fl.engine.fsdp_fl import FSDPFLEngineWithLMHead, FSDPFLEngineWithValueHead  # noqa: F401
from verl_plugin_fl.engine.megatron_fl import MegatronFLEngineWithLMHead  # noqa: F401

__all__ = ["FSDPFLEngineWithLMHead", "FSDPFLEngineWithValueHead", "MegatronFLEngineWithLMHead"]
