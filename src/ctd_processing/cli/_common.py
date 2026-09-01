"""Shared helpers for ctd_processing CLI commands."""

from pathlib import Path

import typer
from pydantic import ValidationError

from ctd_processing.cli._logging import configure_file_logging
from ctd_processing.config import Settings, load_settings


def resolve_settings(config: Path, set_: list[str]) -> Settings:
    """Load :class:`Settings`, converting loading errors into a clean CLI exit.

    Also configures file logging from the loaded settings'
    ``paths.log_file``/``paths.error_log_file`` (see
    :func:`ctd_processing.cli._logging.configure_file_logging`), so every
    command that calls this function gets consistent file-logging setup.

    Parameters
    ----------
    config : pathlib.Path
        Path to a TOML configuration file. Existence is already
        validated by the CLI's ``--config`` option (see
        `ctd_processing.cli._options.ConfigOption`) before this is
        called.
    set_ : list of str
        ``--set key=value`` override strings to apply on top of `config`.

    Returns
    -------
    Settings
        The loaded and validated settings.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if an override in `set_` is malformed,
        or the merged configuration fails validation. In each case a
        description of the problem is printed to stderr first.
    """
    try:
        settings = load_settings(config, set_=set_)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    configure_file_logging(
        settings.paths.log_file, settings.paths.error_log_file
    )
    return settings
