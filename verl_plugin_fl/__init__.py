# Copyright (c) 2026 BAAI. All rights reserved.

"""verl-plugin-fl: FlagOS multi-chip engine plugin for verl.

Usage:
    export VERL_USE_EXTERNAL_MODULES=verl_plugin_fl

Importing this package:
1. Registers the platform (eagerly initialises get_platform() + logs FL config)
2. Registers FL engines via @EngineRegistry.register() ("last writer wins")
"""

from verl_plugin_fl.platform import register_platform  # noqa: F401

# 1. Platform: eagerly initialise and log FL status
register_platform()

# 2. Engines: import triggers @EngineRegistry.register() overrides
from verl_plugin_fl.engine import FSDPFLEngineWithLMHead, FSDPFLEngineWithValueHead, MegatronFLEngineWithLMHead  # noqa: F401, E402
