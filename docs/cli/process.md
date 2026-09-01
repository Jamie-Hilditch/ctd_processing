# `process`

```
ctd-processing process [OPTIONS]
```

!!! warning "Not yet implemented"
    `process` is currently a scaffolding stub. It validates its arguments,
    loads configuration, and resolves which `.rsk` files it would act on
    and dispatches them for reading, but profile extraction itself is not
    yet implemented — it always exits with a non-zero status after
    reporting the resolved deployment files. This lets the command be
    registered, documented, and tested ahead of the real
    `pyrsktools`/`gsw`-based implementation.

Once implemented, `process` will read each raw `.rsk` deployment, identify
its individual profiles (casts), apply configured raw-channel corrections
and despiking, attach a position, compute TEOS-10 derived variables, and
write one file per profile into `paths.profiles_directory`. See
[Concepts → the processing pipeline](../concepts/pipeline.md) for the full
internal flow.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--target`, `-t` | — | Filename of a `.rsk` file to process, relative to `paths.rsk_directory`. Repeatable. If omitted, every top-level `.rsk` file in `rsk_directory` is processed. |
| `--config`, `-c` | `config.toml` | Path to a TOML configuration file. Must exist. |
| `--set` | — | Override a config option, e.g. `--set process.atmospheric_pressure=10.1325`. Repeatable. |

Plus the [shared logging options](index.md#options-shared-by-every-command).

## Example

```bash
ctd-processing process --config config.toml --target 243188_20260809_0304.rsk
```
