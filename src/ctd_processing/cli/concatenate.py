"""``concatenate`` command (stub): merge multiple CTD casts into one dataset."""

from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._options import SetOption


def concatenate_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Directory containing multiple CTD casts to merge "
            "into one dataset.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            help="Path to a TOML configuration file. If omitted, "
            "built-in defaults are used.",
        ),
    ] = None,
    set_: SetOption = None,
) -> None:
    """Concatenate multiple CTD casts into a single dataset.

    Not yet implemented. This is a scaffolding stub: it validates its
    arguments, loads configuration, and reports that no concatenation
    has occurred, so that the command is registered, documented, and
    testable ahead of the real implementation.

    Parameters
    ----------
    input_path : pathlib.Path
        Directory containing multiple CTD casts to merge into one
        dataset.
    config : pathlib.Path or None, optional
        Path to a TOML configuration file. If not given, built-in
        `ctd_processing.config.Settings` defaults are used.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.

    Raises
    ------
    typer.Exit
        Always raised with exit code 1, since concatenation is not yet
        implemented.
    """
    settings = resolve_settings(config, set_ or [])
    typer.echo(
        f"'concatenate' is not yet implemented "
        f"(input_path={input_path}, settings={settings!r})."
    )
    raise typer.Exit(code=1)
