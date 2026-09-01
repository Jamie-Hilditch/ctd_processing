# ctd-processing

`ctd-processing` is a command line application for turning raw RBR CTD
deployments (`.rsk` files) into individually extracted profiles and, from
there, binned and concatenated oceanographic datasets.

It relies on [`pyrsktools`](https://pypi.org/project/pyrsktools/) for the
low-level `.rsk` reading, but the post-processing pipeline is a custom
implementation built on top of it:

- Every oceanographic variable is computed in line with the
  [TEOS-10](https://www.teos-10.org/) standard, via the
  [`gsw`](https://teos-10.github.io/GSW-Python/) package.
- Every output file follows the
  [CF Metadata Conventions](https://cfconventions.org/), with `standard_name`,
  `long_name`, and `units` attached to every variable.
- Configuration is explicit and version-controllable: a single `config.toml`
  per project, with per-instrument and per-deployment overrides where needed.

## Pipeline

The pipeline runs in three stages, each its own CLI command:

1. **`process`** — read a deployment's `.rsk` file, identify individual
   profiles (casts), correct and despike the raw channels, attach a
   position, compute TEOS-10 derived variables, and write one file per
   profile.
2. **`bin`** — read back a deployment's profiles, bin each one onto a common
   grid, and combine them into a single dataset for that deployment.
3. **`concatenate`** — merge every deployment's binned dataset into one
   time-ordered, deduplicated CF netCDF file.

See [Quickstart](quickstart.md) for a walkthrough, or the
[CLI reference](cli/index.md) for full command documentation.

!!! note
    `process` is currently a scaffolding stub — it validates configuration
    and resolves which `.rsk` files it would act on, but does not yet write
    profile files. See [`process`](cli/process.md) for details.

## Where to go next

- [Installation](installation.md) — install `ctd-processing`.
- [Quickstart](quickstart.md) — set up a project and run the pipeline.
- [Configuration](configuration.md) — the full `config.toml` reference.
- [Concepts](concepts/pipeline.md) — how the processing pipeline works
  internally.
- [API reference](api/config.md) — generated from the package's docstrings.
