# verl-plugin-fl

FlagOS multi-chip engine plugin for [verl](https://github.com/verl-project/verl).

## Package Structure

```
verl_plugin_fl/
├── __init__.py
├── pyproject.toml
├── engine/
│   ├── __init__.py          # 入口：import 即触发 @EngineRegistry.register()
│   ├── fsdp_fl.py           # FSDPFLEngineWithLMHead / FSDPFLEngineWithValueHead
│   └── megatron_fl.py       # MegatronFLEngineWithLMHead
├── utils/
│   ├── __init__.py
│   └── config_manager.py    # FLEnvManager, may_enable_flag_gems
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

1. `custom_engine_module` triggers `load_module("pkg://verl_plugin_fl.engine")`
2. This imports `verl_plugin_fl.engine.__init__`, which imports `fsdp_fl.py` and `megatron_fl.py`
3. The `@EngineRegistry.register(device="flagos")` decorators register FL engines
4. `VERL_ENGINE_DEVICE=flagos` makes `EngineRegistry.get_engine_cls()` select the FL engine

## verl Config

`custom_engine_module` supports two loading modes: `pkg://` (installed package) and `file://` (local file path).

### Method 1: Package Mode (`pkg://`)

Requires `pip install` first. Suitable for production.

Hydra overrides:

```bash
actor_rollout_ref.actor.fsdp_config.custom_engine_module='pkg://verl_plugin_fl.engine'
actor_rollout_ref.ref.fsdp_config.custom_engine_module='pkg://verl_plugin_fl.engine'
```

Or in YAML config:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      custom_engine_module: "pkg://verl_plugin_fl.engine"
  ref:
    fsdp_config:
      custom_engine_module: "pkg://verl_plugin_fl.engine"
```

### Method 2: File Path Mode (`file://`)

No installation needed. Load directly from local file. Suitable for development and debugging.

Hydra overrides:

```bash
actor_rollout_ref.actor.fsdp_config.custom_engine_module='file:///path/to/verl_plugin_fl/engine/__init__.py'
actor_rollout_ref.ref.fsdp_config.custom_engine_module='file:///path/to/verl_plugin_fl/engine/__init__.py'
```

Or in YAML config:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      custom_engine_module: "file:///path/to/verl_plugin_fl/engine/__init__.py"
  ref:
    fsdp_config:
      custom_engine_module: "file:///path/to/verl_plugin_fl/engine/__init__.py"
```

You can also omit the `file://` prefix and use the path directly:

```bash
actor_rollout_ref.actor.fsdp_config.custom_engine_module='/path/to/verl_plugin_fl/engine/__init__.py'
```

## Environment Variables

### Required

```bash
# Tell EngineRegistry to use flagos device
export VERL_ENGINE_DEVICE=flagos
```

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
| `FSDPFLEngineWithLMHead` | `language_model` | `fsdp`, `fsdp2` | `flagos` |
| `FSDPFLEngineWithValueHead` | `value_model` | `fsdp`, `fsdp2` | `flagos` |
| `MegatronFLEngineWithLMHead` | `language_model` | `megatron` | `flagos` |

## Dependencies

- `verl >= 0.7.0`
- [FlagGems](https://github.com/FlagOpen/FlagGems) (optional, for Triton operator acceleration)
- [FlagCX](https://github.com/FlagOpen/FlagCX) (optional, for communication)
- [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) (for Megatron engine)
- [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) (for rollout)
