"""``bin`` command (stub): bin processed CTD data to pressure/depth bins."""

from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._options import SetOption


def bin_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Path to a processed data file, or a directory of such "
            "files, to bin.",
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
    """Bin processed CTD data onto pressure or depth bins.

    Not yet implemented. This is a scaffolding stub: it validates its
    arguments, loads configuration, and reports that no binning has
    occurred, so that the command is registered, documented, and
    testable ahead of the real implementation.

    Parameters
    ----------
    input_path : pathlib.Path
        A processed data file, or a directory of such files, to bin.
    config : pathlib.Path or None, optional
        Path to a TOML configuration file. If not given, built-in
        `ctd_processing.config.Settings` defaults are used.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.

    Raises
    ------
    typer.Exit
        Always raised with exit code 1, since binning is not yet
        implemented.
    """
    settings = resolve_settings(config, set_ or [])
    typer.echo(
        f"'bin' is not yet implemented "
        f"(input_path={input_path}, settings={settings!r})."
    )
    raise typer.Exit(code=1)
