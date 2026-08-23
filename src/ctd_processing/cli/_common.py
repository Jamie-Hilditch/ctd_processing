"""Shared helpers for ctd_processing CLI commands."""

from pathlib import Path

import typer
from pydantic import ValidationError

from ctd_processing.config import Settings, load_settings


def resolve_settings(config: Path | None, set_: list[str]) -> Settings:
    """Load :class:`Settings`, converting loading errors into a clean CLI exit.

    Parameters
    ----------
    config : pathlib.Path or None
        Path to a TOML configuration file, or ``None`` to use field
        defaults only.
    set_ : list of str
        ``--set key=value`` override strings to apply on top of `config`.

    Returns
    -------
    Settings
        The loaded and validated settings.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `config` does not exist, an override
        in `set_` is malformed, or the merged configuration fails
        validation. In each case a description of the problem is printed
        to stderr first.
    """
    try:
        return load_settings(config, set_=set_)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
