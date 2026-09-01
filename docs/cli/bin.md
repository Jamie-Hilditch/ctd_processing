# `bin`

```
ctd-processing bin [OPTIONS]
```

For each resolved deployment (a subdirectory of `paths.profiles_directory`),
loads every profile file inside it, bins and combines them onto a common
grid via [`ctd_processing.bin.bin_deployment`](../api/bin.md), and writes
the result to `paths.binned_directory / f"{stem}.{extension}"` (`.nc` for
`bin.output_format = "netcdf"`, `.zarr` for `"zarr"`). Every resolved
deployment is attempted regardless of an earlier one's failure; failures
are collected and reported together as a single non-zero exit.

See [Configuration → `[bin]`](../configuration.md#bin) for grid and
compression settings.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--target`, `-t` | — | Deployment stem to bin, i.e. the name of a subdirectory of `profiles_directory`. Repeatable. If omitted, every top-level subdirectory is binned. |
| `--config`, `-c` | `config.toml` | Path to a TOML configuration file. Must exist. |
| `--set` | — | Override a config option, e.g. `--set bin.channel=sea_pressure`. Repeatable. |

Plus the [shared logging options](index.md#options-shared-by-every-command).

## Example

```bash
ctd-processing bin --config config.toml --set bin.step=0.5
```
