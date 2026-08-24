"""``init`` command: write a starter configuration file."""

import json
import tomllib
from importlib import resources
from pathlib import Path
from typing import Annotated

import tomli_w
import typer
from pydantic import ValidationError

from ctd_processing.cli._options import SetOption
from ctd_processing.config import Settings, merge_overrides

_TEMPLATE_PACKAGE = "ctd_processing.cli.templates"
_TEMPLATE_NAME = "config.toml"


def init_command(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Human-readable name for this project, stored as "
            "project.name.",
        ),
    ] = "my_ctd_processing_project",
    rsk_directory: Annotated[
        Path,
        typer.Option(
            "--rsk-directory",
            help="Directory for raw .rsk deployment files, stored as "
            "paths.rsk_directory. If relative, resolved and created "
            "relative to --working-dir.",
        ),
    ] = Path("rsk_files"),
    profiles_directory: Annotated[
        Path,
        typer.Option(
            "--profiles-directory",
            help="Directory for extracted profile files, stored as "
            "paths.profiles_directory. If relative, resolved and "
            "created relative to --working-dir.",
        ),
    ] = Path("profiles"),
    binned_directory: Annotated[
        Path,
        typer.Option(
            "--binned-directory",
            help="Directory for binned profile files, stored as "
            "paths.binned_directory. If relative, resolved and "
            "created relative to --working-dir.",
        ),
    ] = Path("binned"),
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            help="File to write log records below ERROR level to, "
            "stored as paths.log_file. If omitted, no such file is "
            "configured. If relative, its parent directory is created "
            "relative to --working-dir.",
        ),
    ] = None,
    error_log_file: Annotated[
        Path | None,
        typer.Option(
            "--error-log-file",
            help="File to write log records at ERROR level and above "
            "to, stored as paths.error_log_file. If omitted, no such "
            "file is configured. If relative, its parent directory is "
            "created relative to --working-dir.",
        ),
    ] = None,
    working_dir: Annotated[
        Path | None,
        typer.Option(
            "--working-dir",
            help="Directory in which to write config.toml, and against "
            "which a relative --rsk-directory is resolved. Defaults to "
            "the current working directory. Created if it does not "
            "exist.",
        ),
    ] = None,
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            exists=True,
            dir_okay=False,
            help="Use this TOML file as the starting point instead of "
            "the bundled default.",
        ),
    ] = None,
    set_: SetOption = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite an existing config.toml in the working "
            "directory if present.",
        ),
    ] = False,
) -> None:
    """Write a starter ``config.toml`` for a new ctd_processing project.

    Builds ``config.toml`` from a template (the package's bundled
    default, or a caller-supplied file via `template`), with `name`
    written into its ``[project]`` table and `rsk_directory`,
    `profiles_directory`, `binned_directory`, and (if given) `log_file`/
    `error_log_file` written into its ``[paths]`` table, `set_`
    overrides applied on top, and the corresponding directories created
    on disk. Since these options are always applied, the output is
    always re-serialized and does not preserve comments from the
    template.

    Parameters
    ----------
    name : str, optional
        Human-readable name for this project, written as
        ``project.name``.
    rsk_directory : pathlib.Path, optional
        Directory for raw ``.rsk`` deployment files, written as
        ``paths.rsk_directory`` exactly as given (relative or
        absolute). If relative, it is resolved against `working_dir`
        both to create the directory here and later, when
        `ctd_processing.config.load_settings` resolves it against the
        directory containing this ``config.toml``.
    profiles_directory : pathlib.Path, optional
        Directory for extracted profile files, written as
        ``paths.profiles_directory``. Resolved and created the same
        way as `rsk_directory`.
    binned_directory : pathlib.Path, optional
        Directory for binned profile files, written as
        ``paths.binned_directory``. Resolved and created the same way
        as `rsk_directory`.
    log_file : pathlib.Path or None, optional
        File to write log records below ``ERROR`` level to, written as
        ``paths.log_file``. If omitted, no such file is configured. If
        given, its parent directory is resolved against `working_dir`
        and created the same way as `rsk_directory`.
    error_log_file : pathlib.Path or None, optional
        File to write log records at ``ERROR`` level and above to,
        written as ``paths.error_log_file``. If omitted, no such file
        is configured. Resolved and created the same way as `log_file`.
    working_dir : pathlib.Path or None, optional
        Directory in which to write ``config.toml``, and against which
        relative `rsk_directory`, `profiles_directory`, and
        `binned_directory` values are resolved and created. Created
        (including parents) if it does not already exist. Defaults to
        the current working directory.
    template : pathlib.Path or None, optional
        Path to a TOML file to use instead of the bundled default
        template.
    set_ : list of str, optional
        ``--set key=value`` overrides to apply on top of `name`,
        `rsk_directory`, `profiles_directory`, and `binned_directory`.
    force : bool, optional
        If ``False`` (default) and ``config.toml`` already exists in
        `working_dir`, the command aborts without touching the existing
        file. If ``True``, the existing file is overwritten.

    Raises
    ------
    typer.Exit
        Raised with a non-zero exit code if ``config.toml`` already
        exists in `working_dir` and `force` is ``False``, or if `set_`
        contains a malformed or invalid override, or if the merged
        configuration is otherwise invalid.
    """
    resolved_working_dir = (
        working_dir if working_dir is not None else Path.cwd()
    )
    resolved_working_dir.mkdir(parents=True, exist_ok=True)
    destination = resolved_working_dir / _TEMPLATE_NAME

    if destination.exists() and not force:
        typer.echo(
            f"{destination} already exists; use --force to overwrite.", err=True
        )
        raise typer.Exit(code=1)

    if template is not None:
        template_text = template.read_text(encoding="utf-8")
    else:
        template_resource = resources.files(_TEMPLATE_PACKAGE).joinpath(
            _TEMPLATE_NAME
        )
        template_text = template_resource.read_text(encoding="utf-8")

    project_overrides = [
        f"project.name={json.dumps(name)}",
        f"paths.rsk_directory={json.dumps(rsk_directory.as_posix())}",
        f"paths.profiles_directory={json.dumps(profiles_directory.as_posix())}",
        f"paths.binned_directory={json.dumps(binned_directory.as_posix())}",
    ]
    if log_file is not None:
        project_overrides.append(
            f"paths.log_file={json.dumps(log_file.as_posix())}"
        )
    if error_log_file is not None:
        project_overrides.append(
            f"paths.error_log_file={json.dumps(error_log_file.as_posix())}"
        )

    try:
        merged = merge_overrides(
            tomllib.loads(template_text), project_overrides + (set_ or [])
        )
        settings = Settings.model_validate(merged)
    except (ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    paths = settings.paths
    created_directories = {
        "rsk_directory": resolved_working_dir / paths.rsk_directory,
        "profiles_directory": resolved_working_dir / paths.profiles_directory,
        "binned_directory": resolved_working_dir / paths.binned_directory,
    }
    if paths.log_file is not None:
        created_directories["log_file"] = (
            resolved_working_dir / paths.log_file
        ).parent
    if paths.error_log_file is not None:
        created_directories["error_log_file"] = (
            resolved_working_dir / paths.error_log_file
        ).parent
    for directory in created_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    destination.write_text(tomli_w.dumps(merged), encoding="utf-8")
    created_summary = "\n".join(
        f"  {field}: {directory}"
        for field, directory in created_directories.items()
    )
    typer.echo(
        f"Wrote configuration to {destination}\nCreated:\n{created_summary}"
    )
