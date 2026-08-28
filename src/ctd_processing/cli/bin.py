"""``bin`` command: bin one deployment's profiles onto a common grid."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.bin import bin_deployment
from ctd_processing.bin.save import binned_filename, save_binned_dataset
from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._options import SetOption
from ctd_processing.process.save import load_profile

logger = logging.getLogger(__name__)

__all__ = ["bin_command", "resolve_profile_files"]

_PROFILE_EXTENSIONS = (".nc", ".parquet")


def resolve_profile_files(input_path: Path) -> list[Path]:
    """Resolve the profile file(s) a `bin` run should act on.

    Parameters
    ----------
    input_path : pathlib.Path
        A single profile file (written by
        `ctd_processing.process.save.save_profile`), or a directory of
        such files.

    Returns
    -------
    list[pathlib.Path]
        ``[input_path]`` if it is a file. If it is a directory, every
        top-level ``*.nc``/``*.parquet`` file directly inside it, sorted
        by filename.

    Raises
    ------
    ValueError
        If `input_path` is a directory containing no ``*.nc``/``*.parquet``
        files.
    """
    if input_path.is_file():
        return [input_path]

    discovered = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix in _PROFILE_EXTENSIONS
    )
    if not discovered:
        raise ValueError(f"No profile files found in {input_path}.")
    return discovered


def bin_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Path to a processed profile file, or a directory of "
            "such files belonging to one deployment, to bin.",
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
    """Bin one deployment's profiles onto a common grid and combine them.

    Loads every profile file resolved from `input_path` (see
    `resolve_profile_files`), bins and combines them via
    `ctd_processing.bin.bin_deployment`, and writes the result into
    ``paths.binned_directory`` in ``bin.output_format``.

    Parameters
    ----------
    input_path : pathlib.Path
        A processed profile file, or a directory of such files, all
        belonging to one deployment.
    config : pathlib.Path or None, optional
        Path to a TOML configuration file. If not given, built-in
        `ctd_processing.config.Settings` defaults are used.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `config`/`set_` produce invalid
        settings, if `input_path` cannot be resolved into any profile
        files (see `resolve_profile_files`), or if the resolved profiles
        do not form a single, bin-able deployment (see
        `ctd_processing.bin.bin_deployment`).
    """
    settings = resolve_settings(config, set_ or [])

    try:
        profile_paths = resolve_profile_files(input_path)
        profiles = [load_profile(path) for path in profile_paths]
        combined = bin_deployment(profiles, settings.bin)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    extension = "nc" if settings.bin.output_format == "netcdf" else "zarr"
    filename = binned_filename(profiles[0], extension)
    output_path = save_binned_dataset(
        combined,
        settings.paths.binned_directory,
        filename,
        settings.bin.output_format,
    )

    logger.info(
        "Binned %d profile(s) from %s into %s",
        len(profiles),
        input_path,
        output_path,
    )
    typer.echo(
        f"Wrote binned dataset ({len(profiles)} profile(s)) to {output_path}"
    )
