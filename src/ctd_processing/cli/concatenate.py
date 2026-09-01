"""``concatenate`` command: merge every binned deployment into one dataset."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.bin.save import load_binned_dataset, save_binned_dataset
from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._logging import configure_cli_logging
from ctd_processing.cli._options import (
    ConfigOption,
    DebugOption,
    LogLevel,
    LogLevelOption,
    NoStdoutLogOption,
    SetOption,
    VerboseOption,
)
from ctd_processing.concatenate import concatenate_deployments

logger = logging.getLogger(__name__)

__all__ = ["concatenate_command", "resolve_binned_files"]


def resolve_binned_files(
    binned_directory: Path, targets: list[str] | None, extension: str
) -> list[Path]:
    """Resolve the binned deployment files a `concatenate` run should act on.

    Parameters
    ----------
    binned_directory : pathlib.Path
        Directory containing one file per deployment (see
        `ctd_processing.bin.save.save_binned_dataset`), named
        ``f"{deployment_stem}.{extension}"``.
    targets : list of str or None
        Deployment stems to concatenate, i.e. filenames (without
        extension) directly inside `binned_directory`. May be repeated.
        If ``None`` or empty, every top-level file matching
        ``f"*.{extension}"`` is returned instead.
    extension : str
        The filename extension `binned_directory`'s files use, without
        a leading dot (``"nc"`` or ``"zarr"`` -- see
        `ctd_processing.config.BinSettings.output_format`). A ``"zarr"``
        entry is a directory, not a plain file; existence is checked
        accordingly.

    Returns
    -------
    list[pathlib.Path]
        Resolved, absolute paths to the binned files to concatenate. If
        `targets` was given, paths are returned in the same order; if
        `targets` was omitted, paths are auto-discovered and sorted by
        filename.

    Raises
    ------
    ValueError
        If `binned_directory` does not exist or is not a directory; if
        a target resolves outside `binned_directory`; if a target does
        not exist; or if no matching files are found during
        auto-discovery.
    """
    if not binned_directory.exists():
        raise ValueError(f"binned_directory does not exist: {binned_directory}")
    if not binned_directory.is_dir():
        raise ValueError(
            f"binned_directory is not a directory: {binned_directory}"
        )

    root = binned_directory.resolve()

    if targets:
        resolved = []
        for target in targets:
            candidate = (root / f"{target}.{extension}").resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(
                    f"Target {target!r} resolves outside binned_directory "
                    f"({binned_directory})."
                )
            if not candidate.exists():
                raise ValueError(
                    f"Target {target!r} does not exist in {binned_directory}."
                )
            resolved.append(candidate)
        return resolved

    discovered = sorted(root.glob(f"*.{extension}"), key=lambda path: path.name)
    if not discovered:
        raise ValueError(f"No .{extension} files found in {binned_directory}.")
    return discovered


def concatenate_command(
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Deployment stem to concatenate, i.e. the filename "
            "(without extension) of a binned_directory file. May be "
            "repeated. If omitted, every deployment in binned_directory "
            "is concatenated.",
        ),
    ] = None,
    config: ConfigOption = Path("config.toml"),
    set_: SetOption = None,
    log_level: LogLevelOption = LogLevel.INFO,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
    no_stdout_log: NoStdoutLogOption = False,
) -> None:
    """Concatenate every resolved deployment's binned dataset into one.

    Loads each resolved deployment's binned file from `binned_directory`
    (see `resolve_binned_files`), concatenates them via
    `ctd_processing.concatenate.concatenate_deployments` -- which drops
    any profile sharing an exact ``time`` with another (e.g. from an
    instrument whose onboard memory wasn't wiped between deployments, so
    the next one's raw data repeats the tail of the previous one) and
    sorts the result by ``time`` ascending -- and writes it as a single
    CF-compliant netCDF file to ``paths.concatenated_file``.

    Parameters
    ----------
    target : list of str or None, optional
        Deployment stems to concatenate, i.e. filenames (without
        extension) directly inside ``settings.paths.binned_directory``.
        May be repeated. If not given, every deployment found in
        ``binned_directory`` is concatenated.
    config : pathlib.Path, optional
        Path to a TOML configuration file. Defaults to ``config.toml`` in
        the current directory; Typer validates that this path exists
        (whether given explicitly or left at its default) before this
        function runs.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.
    log_level : LogLevel, optional
        Minimum log level to emit. Overridden by `verbose`/`debug` when
        either is given.
    verbose : bool, optional
        If given, emit VERBOSE-level (and above) log records regardless
        of `log_level`. Overridden by `debug`.
    debug : bool, optional
        If given, emit DEBUG-level (and above) log records regardless of
        `log_level`/`verbose` -- DEBUG is more detailed than VERBOSE, so
        this includes every VERBOSE record too.
    no_stdout_log : bool, optional
        If given, log records are not written to stdout. Independent
        of any ``paths.log_file``/``paths.error_log_file`` configured
        for this command.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `set_` produces invalid settings, if
        ``paths.concatenated_file`` is unset, if `target` cannot be
        resolved against ``binned_directory`` (see
        `resolve_binned_files`), or if the resolved files could not be
        concatenated (see
        `ctd_processing.concatenate.concatenate_deployments`).
    """
    configure_cli_logging(log_level, verbose, debug, no_stdout_log)
    settings = resolve_settings(config, set_ or [])

    if settings.paths.concatenated_file is None:
        typer.echo(
            "paths.concatenated_file must be set to run 'concatenate'.",
            err=True,
        )
        raise typer.Exit(code=1)

    extension = "nc" if settings.bin.output_format == "netcdf" else "zarr"
    try:
        binned_files = resolve_binned_files(
            settings.paths.binned_directory, target, extension
        )
        datasets = [
            load_binned_dataset(path, settings.bin.output_format)
            for path in binned_files
        ]
        combined = concatenate_deployments(datasets)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    output_path = save_binned_dataset(
        combined,
        settings.paths.concatenated_file.parent,
        settings.paths.concatenated_file.name,
        settings.bin.model_copy(update={"output_format": "netcdf"}),
    )

    logger.info(
        "Concatenated %d deployment(s) (%d profile(s)) into %s",
        len(datasets),
        combined.sizes["profile"],
        output_path,
    )
    typer.echo(
        f"Wrote concatenated dataset ({len(datasets)} deployment(s), "
        f"{combined.sizes['profile']} profile(s)) to {output_path}"
    )
