# `concatenate`

```
ctd-processing concatenate [OPTIONS]
```

Loads each resolved deployment's binned file from `paths.binned_directory`,
merges them via
[`ctd_processing.concatenate.concatenate_deployments`](../api/misc.md) —
dropping any profile that shares an exact `time` with another (e.g. from an
instrument whose onboard memory wasn't wiped between deployments) and
sorting the result by `time` ascending — and writes the result as a single
CF-compliant netCDF file to `paths.concatenated_file`.

`paths.concatenated_file` has no default and must be set in `config.toml`
before this command can run — see
[Configuration → `[paths]`](../configuration.md#paths).

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--target`, `-t` | — | Deployment stem to concatenate, i.e. the filename (without extension) of a `binned_directory` file. Repeatable. If omitted, every deployment in `binned_directory` is concatenated. |
| `--config`, `-c` | `config.toml` | Path to a TOML configuration file. Must exist. |
| `--set` | — | Override a config option. Repeatable. |

Plus the [shared logging options](index.md#options-shared-by-every-command).

## Example

```bash
ctd-processing concatenate --config config.toml
```
