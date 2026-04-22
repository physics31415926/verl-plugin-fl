# Copyright (c) 2026 BAAI. All rights reserved.

"""verl-plugin-fl: FlagOS multi-chip engine plugin for verl.

Usage in verl config:
    fsdp_config:
        custom_engine_module: "pkg://verl_plugin_fl.engine"

VERL_ENGINE_DEVICE is auto-set to "flagos" on plugin import.
No manual environment variable configuration needed.
"""
