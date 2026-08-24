"""``process`` command (stub): process raw RBR .rsk deployment files."""

from pathlib import Path
from typing import Annotated

import typer

from ctd_processing.cli._common import resolve_settings
from ctd_processing.cli._options import SetOption
from ctd_processing.config import ProcessSettings, ProjectSettings


def process_deployment(
    file: Path,
    profiles_directory: Path,
    settings: ProcessSettings,
    project: ProjectSettings,
) -> None:
    """Process one ``.rsk`` deployment into extracted profile files.

    Not yet implemented; currently a no-op. This defines the interface
    ahead of the real pyrsktools/gsw-based implementation. `project`
    metadata (e.g. `name`) is intended to be attached to every output
    file's metadata once implemented.

    Parameters
    ----------
    file : pathlib.Path
        The ``.rsk`` deployment file to process.
    profiles_directory : pathlib.Path
        Directory to write extracted profile files into.
    settings : ProcessSettings
        Process-specific settings (currently none defined).
    project : ProjectSettings
        Project metadata to attach to every output file.
    """


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
    """Process raw RBR .rsk deployments into derived oceanographic variables.

    Not yet implemented. This is a scaffolding stub: it validates its
    arguments, loads configuration, resolves which deployment files
    would be processed, calls :func:`process_deployment` for each one
    (currently a no-op), and reports that no processing has occurred,
    so that the command is registered, documented, and testable ahead
    of the real pyrsktools/gsw-based implementation.

    Parameters
    ----------
    target : list of str or None, optional
        Filenames of specific ``.rsk`` deployments to process, relative
        to ``settings.paths.rsk_directory``. May be repeated. If not
        given, every top-level ``.rsk`` file in ``rsk_directory`` is
        processed.
    config : pathlib.Path or None, optional
        Path to a TOML configuration file. If not given, built-in
        `ctd_processing.config.Settings` defaults are used.
    set_ : list of str or None, optional
        ``--set key=value`` overrides to apply on top of `config`.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `config`/`set_` produce invalid
        settings, if `target` cannot be resolved against
        ``rsk_directory`` (see :func:`resolve_deployment_files`), or
        unconditionally once deployment files have been resolved, since
        processing is not yet implemented.
    """
    settings = resolve_settings(config, set_ or [])

    try:
        deployment_files = resolve_deployment_files(
            settings.paths.rsk_directory, target
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for deployment_file in deployment_files:
        process_deployment(
            deployment_file,
            settings.paths.profiles_directory,
            settings.process,
            settings.project,
        )

    listing = "\n".join(f"  {path}" for path in deployment_files)
    typer.echo(
        f"'process' is not yet implemented (settings={settings!r}). "
        f"Resolved deployment files:\n{listing}"
    )
    raise typer.Exit(code=1)
