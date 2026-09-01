"""``process`` command (stub): process raw RBR .rsk deployment files."""

from pathlib import Path
from typing import Annotated

import typer

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
from ctd_processing.process import process_deployment_files


def resolve_deployment_files(
    rsk_directory: Path, targets: list[str] | None
) -> list[Path]:
    """Resolve the ``.rsk`` deployment files a `process` run should act on.

    Parameters
    ----------
    rsk_directory : pathlib.Path
        Directory containing the raw ``.rsk`` deployment files.
    targets : list of str or None
        Filenames of specific deployments to process, relative to
        `rsk_directory`. If ``None`` or empty, every top-level ``.rsk``
        file directly inside `rsk_directory` is returned instead.

    Returns
    -------
    list[pathlib.Path]
        Resolved, absolute paths to the deployment files to process. If
        `targets` was given, paths are returned in the same order; if
        `targets` was omitted, paths are auto-discovered and sorted by
        filename.

    Raises
    ------
    ValueError
        If `rsk_directory` does not exist or is not a directory; if a
        target resolves outside `rsk_directory`; if a target does not
        have a ``.rsk`` extension; if a target does not exist as a
        file; or if no ``.rsk`` files are found during auto-discovery.
    """
    if not rsk_directory.exists():
        raise ValueError(f"rsk_directory does not exist: {rsk_directory}")
    if not rsk_directory.is_dir():
        raise ValueError(f"rsk_directory is not a directory: {rsk_directory}")

    root = rsk_directory.resolve()

    if targets:
        resolved = []
        for target in targets:
            candidate = (root / target).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(
                    f"Target {target!r} resolves outside rsk_directory "
                    f"({rsk_directory})."
                )
            if candidate.suffix.lower() != ".rsk":
                raise ValueError(f"Target {target!r} is not a .rsk file.")
            if not candidate.is_file():
                raise ValueError(
                    f"Target {target!r} does not exist in {rsk_directory}."
                )
            resolved.append(candidate)
        return resolved

    discovered = sorted(
        (path for path in root.glob("*.rsk") if path.is_file()),
        key=lambda path: path.name,
    )
    if not discovered:
        raise ValueError(f"No .rsk files found in {rsk_directory}.")
    return discovered


def process_command(
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Filename of a .rsk file to process, relative to "
            "rsk_directory. May be repeated. If omitted, every "
            "top-level .rsk file in rsk_directory is processed.",
        ),
    ] = None,
    config: ConfigOption = Path("config.toml"),
    set_: SetOption = None,
    log_level: LogLevelOption = LogLevel.INFO,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
    no_stdout_log: NoStdoutLogOption = False,
) -> None:
    """Process raw RBR .rsk deployments into derived oceanographic variables.

    Not yet implemented. This is a scaffolding stub: it validates its
    arguments, loads configuration, resolves which deployment files
    would be processed, dispatches them to
    :func:`ctd_processing.process.process_deployment_files` (which
    currently only reads each deployment; profile extraction is not
    yet implemented), and reports that no processing has occurred, so
    that the command is registered, documented, and testable ahead of
    the real pyrsktools/gsw-based implementation.

    Parameters
    ----------
    target : list of str or None, optional
        Filenames of specific ``.rsk`` deployments to process, relative
        to ``settings.paths.rsk_directory``. May be repeated. If not
        given, every top-level ``.rsk`` file in ``rsk_directory`` is
        processed.
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
        `target` cannot be resolved against ``rsk_directory`` (see
        :func:`resolve_deployment_files`), or unconditionally once
        deployment files have been resolved, since
        processing is not yet implemented.
    """
    configure_cli_logging(log_level, verbose, debug, no_stdout_log)
    settings = resolve_settings(config, set_ or [])

    try:
        deployment_files = resolve_deployment_files(
            settings.paths.rsk_directory, target
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    process_deployment_files(
        deployment_files,
        settings.paths.profiles_directory,
        settings,
    )

    listing = "\n".join(f"  {path}" for path in deployment_files)
    typer.echo(
        f"'process' is not yet implemented (settings={settings!r}). "
        f"Resolved deployment files:\n{listing}"
    )
    raise typer.Exit(code=1)
