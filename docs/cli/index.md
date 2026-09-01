# CLI reference

```
ctd-processing [OPTIONS] COMMAND [ARGS]...
```

`ctd-processing` provides four commands, one per pipeline stage plus
project setup:

| Command | Purpose |
| --- | --- |
| [`init`](init.md) | Write a starter `config.toml` for a new project. |
| [`process`](process.md) | Extract profiles from raw `.rsk` deployments. *(stub — not yet implemented)* |
| [`bin`](bin.md) | Bin one or more deployments' profiles onto a common grid. |
| [`concatenate`](concatenate.md) | Merge every binned deployment into one dataset. |

## Options shared by every command

| Option | Default | Description |
| --- | --- | --- |
| `--log-level` | `INFO` | Minimum log level to emit: `DEBUG`, `VERBOSE`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive). |
| `--verbose` | off | Shortcut for `VERBOSE`-level logging. Overridden by `--debug`. |
| `--debug` | off | Shortcut for `DEBUG`-level logging (implies `--verbose`). |
| `--no-stdout-log` | off | Disable writing log records to stdout. |

`process`, `bin`, and `concatenate` — every command that loads a
[`Settings`](../api/config.md) object — additionally share:

| Option | Default | Description |
| --- | --- | --- |
| `--config`, `-c` | `config.toml` | Path to a TOML configuration file. Must exist. |
| `--set` | — | Override a config option, e.g. `--set section.key=value`. Repeatable. See [Configuration](../configuration.md#overrides). |
| `--target`, `-t` | — | Restrict the command to specific deployments/targets instead of auto-discovering everything. Repeatable; meaning is command-specific. |
