"""``init`` command: write a starter configuration file."""

import tomllib
from importlib import resources
from pathlib import Path
from typing import Annotated

import tomli_w
import typer
from pydantic import ValidationError

from ctd_processing.cli._options import SetOption
from ctd_processing.config import Settings, merge_overrides

_TEMPLATE_PACKAGE = "ctd_processing.cli.templates"
_TEMPLATE_NAME = "config.toml"


def init_command(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Directory in which to write config.toml. Created if "
            "it does not exist."
        ),
    ],
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            exists=True,
            dir_okay=False,
            help="Use this TOML file as the starting point instead of "
            "the bundled default.",
        ),
    ] = None,
    set_: SetOption = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite an existing config.toml in `directory` if present.",
        ),
    ] = False,
) -> None:
    """Write a starter ``config.toml`` into a directory.

    Copies a configuration template into `directory` so users have a
    documented, editable starting point. The template is either the
    package's bundled default, or a caller-supplied file via `template`.
    Individual options in the written file can be overridden with `set_`.

    Parameters
    ----------
    directory : pathlib.Path
        Destination directory for the written ``config.toml``. Created
        (including parents) if it does not already exist.
    template : pathlib.Path or None, optional
        Path to a TOML file to use instead of the bundled default
        template.
    set_ : list of str, optional
        ``--set key=value`` overrides to apply to the template before
        writing it out. Applying any override re-serializes the file,
        which loses comments from the original template.
    force : bool, optional
        If ``False`` (default) and ``directory/config.toml`` already
        exists, the command aborts without touching the existing file.
        If ``True``, the existing file is overwritten.

    Raises
    ------
    typer.Exit
        Raised with a non-zero exit code if ``config.toml`` already
        exists in `directory` and `force` is ``False``, or if `set_`
        contains a malformed or invalid override.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _TEMPLATE_NAME

    if destination.exists() and not force:
        typer.echo(
            f"{destination} already exists; use --force to overwrite.", err=True
        )
        raise typer.Exit(code=1)

    if template is not None:
        template_text = template.read_text(encoding="utf-8")
    else:
        template_resource = resources.files(_TEMPLATE_PACKAGE).joinpath(
            _TEMPLATE_NAME
        )
        template_text = template_resource.read_text(encoding="utf-8")

    if set_:
        try:
            merged = merge_overrides(tomllib.loads(template_text), set_ or [])
            Settings.model_validate(merged)
        except (ValueError, ValidationError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        output_text = tomli_w.dumps(merged)
    else:
        output_text = template_text

    destination.write_text(output_text, encoding="utf-8")
    typer.echo(f"Wrote configuration to {destination}")
