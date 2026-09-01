# The processing pipeline

This page walks through what happens internally when `process`, `bin`, and
`concatenate` run, tying the CLI commands to the modules that implement
them. For the configuration options mentioned below, see
[Configuration](../configuration.md); for full API details, see the
[API reference](../api/config.md).

## `Dataset` and `Channel`

Internally, one deployment (or profile) is represented by a
[`Dataset`](../api/process.md) — a time-indexed collection of
[`Channel`](../api/process.md) objects, each holding one variable's data,
metadata, and processing history. Every step below reads from and writes to
a `Dataset`, and every mutating operation (`record`, `add_channel`,
`remove_channel`) appends to that channel's history — visible later as CF
global/variable attributes on written output.

## `process`: deployment → profiles

[`process_deployment_files`](../api/process.md) is the entry point
`ctd-processing process` dispatches to. It copies each `.rsk` deployment
into a private temporary directory (so later steps can safely run
write-capable `pyrsktools.RSK` methods against the copy) and processes
deployments concurrently via a thread pool, since copying is I/O-bound.

Each deployment then runs through
[`process_deployment`](../api/process.md):

1. **Read** — [`read_rsk`](../api/process.md) opens the `.rsk` file
   read-only via `pyrsktools`.
2. **Resolve settings** — the instrument's serial number
   (`rsk.instrument.serialID`) and the deployment's filename stem resolve
   the effective [`ProcessSettings`](../api/config.md) for this specific
   deployment, applying any matching `[instruments.*]`/`[deployments.*]`
   overrides (see
   [Configuration → per-instrument and per-deployment overrides](../configuration.md#per-instrument-and-per-deployment-overrides)).
   This happens *before* building the dataset, so `read_channels` can
   restrict which channels are even read from disk.
3. **Build** — [`build_dataset`](../api/process.md) constructs a `Dataset`
   from the opened `RSK`, mapping RBR channel `longName`s to CF metadata and
   short storage keys via
   [`cf_channels`](../api/process.md).
4. **Raw channel corrections and despiking** —
   [`process_raw_channels`](../api/process.md) applies `remove_holds`,
   `shift`, and `offset` per `[process.raw_channels.<name>]`, then despikes
   any channel enabled in `[process.channels.<name>]`.
5. **Sea pressure** — [`compute_sea_pressure`](../api/process.md) either
   trusts the dataset's existing `sea_pressure` channel or recomputes it
   from `absolute_pressure` minus `atmospheric_pressure`.
6. **Profile identification** —
   [`find_profiles`](../api/process.md) identifies turnaround cycles from
   `sea_pressure` via `profinder` (both the downcast and upcast of every
   cycle, regardless of `direction`), and
   [`resolve_cast_slices`](../api/process.md) selects the cast(s)
   `[process.profiles].direction` configures to be written out — never
   including the dwell between a cycle's downcast and upcast.
7. **CT lag** — if `[process.ct_lag].enabled`,
   [`process_ct_lag`](../api/process.md) grid-searches a single
   conductivity/temperature sample shift for the whole deployment,
   evaluated only over the resolved cast slices, to minimize salinity
   spiking.
8. **Per-profile processing** — each resolved cast is extracted
   (`Dataset.subset`) and passed through
   [`process_profile`](../api/process.md):
      - [`attach_geolocation`](../api/process.md) attaches the profile's
        canonical start/end time and a `latitude`/`longitude` position,
        per `[process.geolocation]`.
      - [`compute_derived_variables`](../api/process.md) computes the
        TEOS-10 chain via `gsw`, per `[process.derived_variables]`, then
        despikes any newly derived channel that's enabled for it.
9. **Save** — [`save_profile`](../api/process.md) writes each processed
   profile to `paths.profiles_directory`, in `profile_format` (`"parquet"`
   or `"netcdf"`; see [`save_parquet`](../api/process.md) /
   [`save_netcdf`](../api/process.md)).

## `bin`: profiles → one deployment dataset

[`bin_deployment`](../api/bin.md) is the entry point `ctd-processing bin`
dispatches to, given every profile `Dataset` for one deployment (loaded
back via [`load_profile`](../api/process.md)):

1. Profiles are sorted by their canonical start time.
2. [`compute_bin_edges`](../api/bin.md) computes a common grid from
   `[bin].channel`/`step`/`first`/`last` and every profile's data.
3. [`bin_profile`](../api/bin.md) bins each profile onto those edges,
   averaging every other channel within each bin.
4. [`combine_binned_profiles`](../api/bin.md) stacks the binned profiles
   along a new `profile` dimension into one combined `xarray.Dataset`.
5. [`save_binned_dataset`](../api/bin.md) writes the result to
   `paths.binned_directory`, in `[bin].output_format` (`"netcdf"` or
   `"zarr"`).

## `concatenate`: deployments → one dataset

[`concatenate_deployments`](../api/misc.md) merges every deployment's
binned dataset (loaded via
[`load_binned_dataset`](../api/bin.md)) into one, dropping any profile
sharing an exact `time` with another and sorting the result ascending —
then writes it as a single CF netCDF file to `paths.concatenated_file`.

## CF conventions and TEOS-10

Every written dataset — profile-level or binned — carries CF-compliant
`standard_name`/`long_name`/`units` attributes, assembled by
[`cf_attrs`](../api/misc.md) from each channel's metadata and processing
history. Oceanographic quantities are computed via
[`gsw`](https://teos-10.github.io/GSW-Python/), in line with the
[TEOS-10](https://www.teos-10.org/) standard, so downstream tools that
expect practical/absolute salinity, conservative temperature, and related
quantities receive values computed the same way the wider oceanographic
community does.
