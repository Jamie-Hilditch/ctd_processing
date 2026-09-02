# Configuration

Every `ctd-processing` project is configured by a single `config.toml` file,
loaded via `--config`/`-c` (defaulting to `config.toml` in the current
directory). The full, documented schema is
[`ctd_processing.config.Settings`](api/config.md); this page is a narrative
tour of it. `ctd-processing init` writes a starter file with every option
commented out and documented inline — see [`init`](cli/init.md).

## Path resolution

A project's `config.toml` location defines that project's working
directory. Relative paths inside the file (e.g. `paths.rsk_directory`,
`process.geolocation.external_dataset_path`) are resolved relative to the
directory containing that `config.toml`, not the process's current working
directory — so a project's config loads correctly regardless of where
`ctd-processing` is invoked from. Paths may be given as either relative or
absolute.

When configuration comes purely from `--set`, with no `--config` file,
relative paths are instead resolved against the current working directory,
since there is no config file location to anchor to.

`ctd-processing init` writes `--rsk-directory` (and the other directory
options) into `config.toml` exactly as given, and creates the corresponding
directory relative to `--working-dir` — so the written path and the
directory `init` creates always line up when the file is later loaded from
its own directory.

## `[project]`

Project-level metadata attached to every output file, currently just
`name` (defaults to `"my_ctd_processing_project"`).

## `[paths]`

Directory locations for each pipeline stage — see
[`PathsSettings`](api/config.md). `rsk_directory`, `profiles_directory`, and
`binned_directory` are required, with no defaults. `concatenated_file`,
`log_file`, and `error_log_file` are optional and unset by default;
`concatenated_file` must be set before `concatenate` can run.

## `[process]`

Settings for the `process` command — see
[`ProcessSettings`](api/config.md).

- **`read_channels`** restricts which RBR channels are extracted from the
  `.rsk` file at all. Defaults to every channel present.
- **`atmospheric_pressure`**, if set, forces `sea_pressure` to be
  recomputed from `absolute_pressure`; left unset, the dataset's own
  `sea_pressure` channel is trusted.
- **`profile_format`** — `"parquet"` (default) or `"netcdf"` — selects the
  extracted-profile file format, each with its own compression table
  (`parquet_compression` / `netcdf_compression`).
- **`output_dtype`** is the project-wide default numpy floating dtype
  (`"float32"` by default) every channel is cast to on write.

### Raw channel corrections

`[process.raw_channels.<name>]` configures per-channel corrections applied
to a *raw* channel before anything is derived from it — see
[`RawChannelSettings`](api/config.md): `remove_holds` (repeated-value
stretches), `shift` (a pandas-`.shift()`-style sample shift), and `offset`
(a fixed additive bias).

### Profile detection

`[process.profiles]` configures identifying individual casts from the
`sea_pressure` channel via [`profinder`](https://pypi.org/project/profinder/)
— see [`ProfileSettings`](api/config.md). The defaults are tuned for a
multi-hundred-dbar oceanic deployment; re-tune `peak_height`,
`peak_distance`, `peak_width`, and `peak_prominence` for a shallower
deployment or a different sampling rate. `direction` (`"down"` by default)
selects which cast(s) of each identified turnaround are written out as
separate profiles.

### CT lag correction

`[process.ct_lag]` configures the conductivity/temperature lag correction —
see [`CTLagSettings`](api/config.md). Disabled (`enabled = false`) by
default; when enabled, a single sample shift is grid-searched for the whole
deployment to minimize salinity spiking, over `[min_lag, max_lag]`.

### Geolocation

`[process.geolocation]` attaches a position to every extracted profile — see
[`GeolocationSettings`](api/config.md). Unlike every other `[process.*]`
section, this one is **required**: exactly one of the following must be
configured, with no numeric fallback.

=== "Fixed position"

    ```toml
    [process.geolocation]
    reference_latitude = 45.0
    reference_longitude = -125.0
    ```

=== "External GPS track"

    ```toml
    [process.geolocation]
    external_dataset_path = "gps_track.nc"
    latitude_variable = "latitude"
    longitude_variable = "longitude"
    time_variable = "time"
    ```

### Derived variables

`[process.derived_variables]` toggles which TEOS-10 quantities (computed via
`gsw`) are attached to each profile — see
[`DerivedVariablesSettings`](api/config.md). `z`, `practical_salinity`,
`absolute_salinity`, `conservative_temperature`, and `potential_density` are
enabled by default; `potential_temperature`, `sound_speed`, `density`,
`spiciness`, `freezing_point`, `thermal_expansion`, `haline_contraction`,
and `oxygen_concentration` are opt-in.

### Despiking

`[process.despiking]` sets the project-wide rolling-median despike defaults
(`threshold`, `window_length`, `iterations`) — see
[`DespikeSettings`](api/config.md). These are only defaults: a channel is
only actually despiked if enabled per-channel below.

### Per-channel output settings

`[process.channels.<name>]` enables despiking (`despike = true`, with
optional `despiking` overrides of the project-wide defaults) and sets a
per-channel `output_dtype` override — see
[`ChannelSettings`](api/config.md). `<name>` uses the same key namespace as
`raw_channels`: both raw channels (e.g. `"temperature"`) and derived
variable output keys (e.g. `"practical_salinity"`).

### Compression

Profile-level netCDF/Parquet output is compressed per
`[process.netcdf_compression]` / `[process.parquet_compression]` — see
[`NetcdfCompressionSettings`](api/config.md) and
[`ParquetCompressionSettings`](api/config.md). Both default to enabled,
matching the package's previous hardcoded behavior.

## `[bin]`

Settings for the `bin` command — see [`BinSettings`](api/config.md):
`channel` (default `"z"`) and `step` (default `1.0`) define the common grid
every profile is averaged onto; `first`/`last` default to the observed data
range. `output_format` (`"netcdf"` or `"zarr"`) selects the combined
deployment file's format, each with its own compression table
(`netcdf_compression` / `zarr_compression`).

## Per-instrument and per-deployment overrides

`[instruments.<serial_number>.process]` and `[deployments.<stem>.process]`
override `[process]` for one physical instrument (identified by serial
number, read from the `.rsk` file itself — never inferred from a filename)
or one specific deployment (identified by its `.rsk` filename stem) — see
[`InstrumentSettings`](api/config.md) and
[`DeploymentSettings`](api/config.md). Only the given fields are
overridden; nested tables merge field-by-field. Where both apply,
`[deployments.*]` wins over `[instruments.*]`, which wins over `[process]`.

```toml
[instruments.208532.process]
atmospheric_pressure = 10.1

[instruments.208532.process.raw_channels.temperature]
shift = 2

[deployments.243188_20260809_0304.process]
atmospheric_pressure = 10.1325
```

## Overrides

Any option can be overridden from the command line, without editing
`config.toml`, using repeatable `--set section.key=value` flags:

```bash
ctd-processing bin --set bin.channel=sea_pressure --set bin.step=0.5
```

Each value is parsed as TOML, so strings must be quoted
(`--set project.name='"my_survey"'`) and dotted keys build nested tables.
`--set` overrides are applied on top of `--config` (or on top of every
field's default, if no `--config` is given) — see
[`parse_overrides`](api/config.md) and [`merge_overrides`](api/config.md).
