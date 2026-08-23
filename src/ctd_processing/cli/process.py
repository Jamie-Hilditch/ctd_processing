"""``process`` command (stub): process a raw RBR .rsk file or directory."""

from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._options import SetOption


def process_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Path to a .rsk file, or a directory containing .rsk "
            "files, to process.",
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
    """Process a raw RBR .rsk file into derived oceanographic variables.

    Not yet implemented. This is a scaffolding stub: it validates its
    arguments, loads configuration, and reports that no processing has
    occurred, so that the command is registered, documented, and
    testable ahead of the real pyrsktools/gsw-based implementation.

    Parameters
    ----------
    input_path : pathlib.Path
        A ``.rsk`` file, or a directory of ``.rsk`` files, to process.
    config : pathlib.Path or None, optional
        Path to a TOML configuration file. If not given, built-in
        `ctd_processing.config.Settings` defaults are used.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.

    Raises
    ------
    typer.Exit
        Always raised with exit code 1, since processing is not yet
        implemented.
    """
    settings = resolve_settings(config, set_ or [])
    typer.echo(
        f"'process' is not yet implemented "
        f"(input_path={input_path}, settings={settings!r})."
    )
    raise typer.Exit(code=1)
