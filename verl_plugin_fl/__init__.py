# Copyright (c) 2026 BAAI. All rights reserved.

"""verl-plugin-fl: FlagOS multi-chip engine plugin for verl.

Usage in verl config:
    fsdp_config:
        custom_engine_module: "pkg://verl_plugin_fl.engine"

The plugin overrides default engines for the detected hardware via
"last writer wins" registration. No extra environment variables needed.
"""
