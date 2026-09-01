# Installation

`ctd-processing` requires Python 3.12 or later.

## With uv (recommended)

```bash
uv add ctd-processing
```

Or, to try the CLI without adding it to a project:

```bash
uvx ctd-processing --help
```

## With pip

```bash
pip install ctd-processing
```

## Verify the install

```bash
ctd-processing --help
```

This should print the top-level command list (`init`, `process`, `bin`,
`concatenate`). See the [Quickstart](quickstart.md) to set up your first
project.

## Development install

To work on `ctd-processing` itself, clone the repository and install with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Jamie-Hilditch/ctd_processing.git
cd ctd_processing
uv sync
```

See [Contributing](contributing.md) for the development workflow (linting,
type checking, tests, and building these docs).
