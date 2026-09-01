"""``bin`` command: bin one or more deployments' profiles onto a common grid."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.bin import bin_deployment
from ctd_processing.bin.save import save_binned_dataset
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
from ctd_processing.process.save import load_profile

logger = logging.getLogger(__name__)

__all__ = ["bin_command", "resolve_deployment_stems", "resolve_profile_files"]

_PROFILE_EXTENSIONS = (".nc", ".parquet")


def resolve_deployment_stems(
    profiles_directory: Path, targets: list[str] | None
) -> list[str]:
    """Resolve the deployment stems a `bin` run should act on.

    Parameters
    ----------
    profiles_directory : pathlib.Path
        Directory containing one subdirectory per deployment (see
        `ctd_processing.process.save.save_profile`).
    targets : list of str or None
        Deployment stems to bin, i.e. subdirectory names directly
        inside `profiles_directory`. May be repeated. If ``None`` or
        empty, every top-level subdirectory of `profiles_directory` is
        returned instead.

    Returns
    -------
    list[str]
        The deployment stems to bin. If `targets` was given, returned
        in the same order; if `targets` was omitted, auto-discovered
        and sorted by name.

    Raises
    ------
    ValueError
        If `profiles_directory` does not exist or is not a directory;
        if a target resolves outside `profiles_directory`; if a target
        does not exist as a subdirectory of it; or if no subdirectories
        are found during auto-discovery.
    """
    if not profiles_directory.exists():
        raise ValueError(
            f"profiles_directory does not exist: {profiles_directory}"
        )
    if not profiles_directory.is_dir():
        raise ValueError(
            f"profiles_directory is not a directory: {profiles_directory}"
        )

    root = profiles_directory.resolve()

    if targets:
        for target in targets:
            candidate = (root / target).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(
                    f"Target {target!r} resolves outside profiles_directory "
                    f"({profiles_directory})."
                )
            if not candidate.is_dir():
                raise ValueError(
                    f"Target {target!r} does not exist in {profiles_directory}."
                )
        return list(targets)

    discovered = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not discovered:
        raise ValueError(
            f"No deployment subdirectories found in {profiles_directory}."
        )
    return discovered


def resolve_profile_files(input_path: Path) -> list[Path]:
    """Resolve the profile file(s) a `bin` run should act on.

    Parameters
    ----------
    input_path : pathlib.Path
        A single profile file (written by
        `ctd_processing.process.save.save_profile`), or a directory of
        such files -- typically ``profiles_directory / deployment_stem``
        (see `resolve_deployment_stems`).

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
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Deployment stem to bin, i.e. the name of a subdirectory "
            "of profiles_directory. May be repeated. If omitted, every "
            "top-level subdirectory of profiles_directory is binned.",
        ),
    ] = None,
    config: ConfigOption = Path("config.toml"),
    set_: SetOption = None,
    log_level: LogLevelOption = LogLevel.INFO,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
    no_stdout_log: NoStdoutLogOption = False,
) -> None:
    """Bin one or more deployments' profiles onto a common grid each.

    For each resolved deployment stem (see `resolve_deployment_stems`),
    loads every profile file in ``profiles_directory / stem`` (see
    `resolve_profile_files`), bins and combines them via
    `ctd_processing.bin.bin_deployment`, and writes the result to
    ``binned_directory / f"{stem}.{extension}"`` (``.nc`` for
    ``bin.output_format`` ``"netcdf"``, ``.zarr`` for ``"zarr"``). Every
    resolved deployment is attempted regardless of an earlier one's
    failure; failures are collected and reported together, once every
    deployment has been attempted, as a single non-zero exit.

    Parameters
    ----------
    target : list of str or None, optional
        Deployment stems to bin, i.e. subdirectory names directly
        inside ``settings.paths.profiles_directory``. May be repeated.
        If not given, every top-level subdirectory of
        ``profiles_directory`` is binned.
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
        `target` cannot be resolved against ``profiles_directory`` (see
        `resolve_deployment_stems`), or if any resolved deployment
        failed to resolve into profile files (see `resolve_profile_files`)
        or did not form a single, bin-able deployment (see
        `ctd_processing.bin.bin_deployment`).
    """
    configure_cli_logging(log_level, verbose, debug, no_stdout_log)
    settings = resolve_settings(config, set_ or [])

    try:
        deployment_stems = resolve_deployment_stems(
            settings.paths.profiles_directory, target
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    extension = "nc" if settings.bin.output_format == "netcdf" else "zarr"
    errors: list[str] = []
    for stem in deployment_stems:
        try:
            profile_paths = resolve_profile_files(
                settings.paths.profiles_directory / stem
            )
            profiles = [load_profile(path) for path in profile_paths]
            combined = bin_deployment(profiles, settings.bin)
        except ValueError as exc:
            errors.append(f"{stem}: {exc}")
            continue

        output_path = save_binned_dataset(
            combined,
            settings.paths.binned_directory,
            f"{stem}.{extension}",
            settings.bin,
        )
        logger.info(
            "Binned %d profile(s) from %s into %s",
            len(profiles),
            stem,
            output_path,
        )
        typer.echo(
            f"Wrote binned dataset ({len(profiles)} profile(s)) to "
            f"{output_path}"
        )

    if errors:
        for error in errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
