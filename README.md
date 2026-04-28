# verl-plugin-fl

FlagOS multi-chip engine plugin for [verl](https://github.com/verl-project/verl).

## Package Structure

```
verl_plugin_fl/
├── __init__.py            # Entry point: registers platform + engines
├── platform.py            # Platform registration (eagerly init + FL diagnostics)
├── pyproject.toml
├── engine/
│   ├── __init__.py        # Import triggers @EngineRegistry.register()
│   ├── fsdp_fl.py         # FSDPFLEngineWithLMHead / FSDPFLEngineWithValueHead
│   └── megatron_fl.py     # MegatronFLEngineWithLMHead
├── utils/
│   ├── __init__.py
│   └── config_manager.py  # FLEnvManager, may_enable_flag_gems
└── configs/
    └── vllm_plugin_fl_dispatch.yaml
```

## Installation

```bash
# From local directory (development mode)
cd verl-plugin-fl && pip install -e .

# From GitHub
pip install git+https://github.com/flagos-ai/verl-plugin-fl.git
```

## How It Works

1. `VERL_USE_EXTERNAL_MODULES=verl_plugin_fl` triggers `importlib.import_module("verl_plugin_fl")`
2. `verl_plugin_fl.__init__` calls `register_platform()` to eagerly initialise the platform singleton and log FL config
3. Then imports `verl_plugin_fl.engine`, which imports `fsdp_fl.py` and `megatron_fl.py`
4. Each engine file calls `get_device_name()` at import time to detect the current hardware (e.g. `"cuda"`, `"npu"`)
5. The `@EngineRegistry.register(device=_device)` decorators override the default engine for the detected hardware ("last writer wins")
6. `EngineRegistry.get_engine_cls()` uses the same `get_device_name()` to look up the engine — which now resolves to the FL version

This means **only `VERL_USE_EXTERNAL_MODULES` needs to be set** — one environment variable to load the plugin.

## Usage

```bash
# Set before launching training
export VERL_USE_EXTERNAL_MODULES=verl_plugin_fl
```

Multiple external modules can be comma-separated:

```bash
export VERL_USE_EXTERNAL_MODULES=verl_plugin_fl,other_plugin.module
```

## Environment Variables

### Training Phase (TE-FL / Megatron / FSDP)

```bash
# TE-FL backend priority: flagos / vendor / reference
export TE_FL_PREFER=flagos

# Strict mode (no fallback): 1 / 0
export TE_FL_STRICT=0

# FlagGems operator switch
export USE_FLAGGEMS=true

# FlagGems operator whitelist/blacklist (comma separated, optional)
export TRAINING_FL_FLAGOS_WHITELIST="..."
export TRAINING_FL_FLAGOS_BLACKLIST="..."

# FlagCX communication
export USE_FLAGCX=1
export FLAGCX_PATH=/path/FlagCX
```

### Rollout Phase (vLLM-FL)

```bash
export VLLM_PLUGINS="fl"
export USE_FLAGGEMS=true
export VLLM_FL_OOT_ENABLED=1
export VLLM_FL_FLAGOS_BLACKLIST="where_scalar_other,where_scalar_self,where_self,where_self_out,pad"
export USE_FLAGCX=1
export FLAGCX_PATH=/path/FlagCX
```

See [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL/pull/4) and [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/dispatch/README.md) for detailed explanations.

## Registered Engines

| Engine Class | model_type | backend | device |
|---|---|---|---|
| `FSDPFLEngineWithLMHead` | `language_model` | `fsdp`, `fsdp2` | auto (`get_device_name()`) |
| `FSDPFLEngineWithValueHead` | `value_model` | `fsdp`, `fsdp2` | auto (`get_device_name()`) |
| `MegatronFLEngineWithLMHead` | `language_model` | `megatron` | auto (`get_device_name()`) |

## Dependencies

- `verl >= 0.7.0`
- [FlagGems](https://github.com/FlagOpen/FlagGems) (optional, for Triton operator acceleration)
- [FlagCX](https://github.com/FlagOpen/FlagCX) (optional, for communication)
- [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) (for Megatron engine)
- [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) (for rollout)
