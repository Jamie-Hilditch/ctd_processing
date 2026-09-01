# Quickstart

This walks through setting up a new `ctd-processing` project and running the
full pipeline against a directory of `.rsk` deployment files.

## 1. Initialize a project

```bash
ctd-processing init \
  --name "my_survey" \
  --rsk-directory rsk_files \
  --profiles-directory profiles \
  --binned-directory binned \
  --working-dir my_survey_project
```

This writes `my_survey_project/config.toml` and creates the
`rsk_files/`, `profiles/`, and `binned/` directories underneath it. Every
option is optional except `--working-dir` matters for where things land;
see [`init`](cli/init.md) for the full option list and defaults.

Copy your instrument's `.rsk` files into `my_survey_project/rsk_files/`.

## 2. Configure geolocation

`init` cannot know your deployment's real position, so
`[process.geolocation]` is left commented out in the written `config.toml`.
Open it and set either a fixed reference position:

```toml
[process.geolocation]
reference_latitude = 45.0
reference_longitude = -125.0
```

or an external GPS track to interpolate from:

```toml
[process.geolocation]
external_dataset_path = "gps_track.nc"
```

The config won't validate for `process`/`bin`/`concatenate` until this is
set. See [Configuration](configuration.md#geolocation) for details.

## 3. Tune profile detection (if needed)

The bundled `[process.profiles]` defaults are tuned for a multi-hundred-dbar
oceanic deployment. If your instrument profiles a shallower range or at a
different rate, adjust `peak_height`, `peak_distance`, `peak_width`, and
`peak_prominence` accordingly — see
[Configuration](configuration.md#profile-detection).

## 4. Process, bin, and concatenate

From inside `my_survey_project/`:

```bash
ctd-processing process --config config.toml
ctd-processing bin --config config.toml
ctd-processing concatenate --config config.toml
```

Each command defaults to `--config config.toml` in the current directory, so
`--config` can be omitted when run from the project directory. Every command
also accepts `--target/-t` (repeatable) to act on a subset of deployments
instead of everything discovered automatically. See the
[CLI reference](cli/index.md) for the full set of options.

!!! note
    `process` is currently a scaffolding stub: it validates configuration
    and reports which `.rsk` files it would process, but does not yet write
    profile files. See [`process`](cli/process.md).

## 5. Override settings without editing the file

Any option can be overridden from the command line with repeatable
`--set section.key=value` flags, e.g.:

```bash
ctd-processing bin --config config.toml --set bin.channel=sea_pressure --set bin.step=0.5
```

See [Configuration](configuration.md#overrides) for the full syntax,
including per-instrument and per-deployment overrides.
