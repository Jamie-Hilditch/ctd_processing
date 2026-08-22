# CTD Processing

This package is a command line application for processing RBR CTD data, i.e. .rsk files. The command line interface is built upon Typer and we use pydantic_settings for handling. The low level interface for handling rsk files relies on pyrsktools. However, we quickly switch to custom implementations for post-processing rather than rely on the pyrsktool implementations.

The following conventions should be adopted for all classes and functions in the package.
- Complete numpy style docstrings
- Any configuration options must have an entry in the configuration file
- Oceanographic variables should be computed inline with the TEOS-10 standards. In practice this means using the gsw package.
- Outputs, particularly when in the form of netcdf or xarray datasets, should the Climate and Forecast (CF) Metadata Conventions.
  - `standard_name`, `long_name`, and `units` are critical metadata for all variables.

CI is built on the UV ecosystem and pytest
- uv run ruff format
- uv run ruff check
- uv run ty check
- uv run pytest
