# Contributing

## Setup

```bash
git clone https://github.com/Jamie-Hilditch/ctd_processing.git
cd ctd_processing
uv sync
```

## Checks

CI runs the following; run them locally before opening a pull request:

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

Some tests require real `.rsk` example data in `tests/example_data/`, which
is not checked into git; those are marked `requires_example_data` and skip
automatically without it.

## Conventions

- **Docstrings** — every public (and most private) function, class, and
  method has a complete [numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html)
  docstring.
- **Configuration** — every configuration option has a documented entry in
  both [`ctd_processing.config`](api/config.md) and the bundled starter
  template (`src/ctd_processing/cli/templates/config.toml`).
- **TEOS-10** — oceanographic variables are computed via
  [`gsw`](https://teos-10.github.io/GSW-Python/), in line with the TEOS-10
  standard.
- **CF conventions** — `xarray`/netCDF outputs follow the
  [CF Metadata Conventions](https://cfconventions.org/): every variable
  carries `standard_name`, `long_name`, and `units`.
- **`__repr__`/`__str__`** — every plain class has a useful, explicit
  `__repr__` (developer-facing) and `__str__` (concise summary), except
  where inherited from a base class that already provides one (e.g.
  `pydantic.BaseModel`, `enum`).
- **Logging** — every call site that records a `Dataset`/`Channel` action
  (`record`, `add_channel`, `remove_channel`) also logs the same action at
  `VERBOSE` level via
  [`ctd_processing.logging_utils.log_verbose`](api/misc.md), unless already
  logged at a higher level there.

## Building these docs

This site is built with [Zensical](https://zensical.org/). To preview it
locally:

```bash
uv run zensical serve
```

or build a static copy to `site/`:

```bash
uv run zensical build
```

API reference pages under `docs/api/` are generated at build time from the
package's docstrings via the `mkdocstrings` plugin (configured in
`zensical.toml`) — there's no need to hand-maintain them when a
function's signature or docstring changes.
