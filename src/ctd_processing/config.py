"""Configuration model and loading utilities for ctd_processing."""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ProjectSettings(BaseModel):
    """Project-level metadata attached to every output file.

    Attributes
    ----------
    name : str
        Human-readable name for this project. Intended to be attached
        to every output file's metadata (e.g. as a CF global attribute).
        Defaults to ``"my_ctd_processing_project"``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "my_ctd_processing_project"


class PathsSettings(BaseModel):
    """Directory locations for a ctd_processing project's pipeline stages.

    Attributes
    ----------
    rsk_directory : pathlib.Path
        Directory containing the raw ``.rsk`` deployment files to
        process. Required; there is no default. May be given as a
        relative or absolute path; :func:`load_settings` resolves a
        relative path against the directory containing the loaded
        config file (or the current working directory if none was
        given). ``process`` resolves ``--target`` filenames relative to
        this directory, or auto-discovers every top-level ``*.rsk``
        file inside it when no target is given. Existence of the
        directory is validated by the CLI at call time, not by this
        model.
    profiles_directory : pathlib.Path
        Directory for extracted profile files produced from
        `rsk_directory` deployments. Required; there is no default.
        Resolved the same way as `rsk_directory`.
    binned_directory : pathlib.Path
        Directory for pressure/depth-binned profile files produced from
        `profiles_directory`. Required; there is no default. Resolved
        the same way as `rsk_directory`.
    log_file : pathlib.Path or None
        File to write log records below ``ERROR`` level to. Optional;
        defaults to ``None``, meaning no such file is written. Resolved
        the same way as `rsk_directory` when given.
    error_log_file : pathlib.Path or None
        File to write log records at ``ERROR`` level and above to.
        Optional; defaults to ``None``, meaning no such file is
        written. Resolved the same way as `rsk_directory` when given.
    """

    model_config = ConfigDict(extra="forbid")

    rsk_directory: Path
    profiles_directory: Path
    binned_directory: Path
    log_file: Path | None = None
    error_log_file: Path | None = None


class ProcessSettings(BaseModel):
    """Settings specific to the ``process`` command.

    Currently defines no fields. Extension point for process-specific
    options (e.g. TEOS-10 conversion parameters) once
    `ctd_processing.cli.process.process_deployment` needs them.
    """

    model_config = ConfigDict(extra="forbid")


class Settings(BaseSettings):
    """Runtime configuration for ctd_processing.

    This model is the single extension point for every configuration
    option that ``process``, ``bin``, and ``concatenate`` need. Per
    repository convention, every field added here must have a
    corresponding, documented entry in the bundled starter template at
    ``ctd_processing/cli/templates/config.toml``.

    Attributes
    ----------
    project : ProjectSettings
        Project-level metadata attached to every output file, e.g.
        `name`. Optional; every field of `ProjectSettings` has a
        default.
    paths : PathsSettings
        Directory locations for the project's pipeline stages, e.g.
        `rsk_directory`. Required, since none of its fields have a
        default.
    process : ProcessSettings
        Settings specific to the ``process`` command. Optional;
        currently has no fields.
    """

    model_config = SettingsConfigDict(extra="forbid")

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathsSettings
    process: ProcessSettings = Field(default_factory=ProcessSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Restrict settings sources to explicit constructor keyword arguments.

        ctd_processing sources all configuration from an explicitly loaded
        TOML file plus ``--set`` overrides (see :func:`load_settings`), so
        environment variables, ``.env`` files, and secrets directories are
        deliberately excluded to avoid surprising, implicit configuration.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The class being instantiated.
        init_settings : PydanticBaseSettingsSource
            Source representing explicit constructor keyword arguments.
        env_settings : PydanticBaseSettingsSource
            Unused.
        dotenv_settings : PydanticBaseSettingsSource
            Unused.
        file_secret_settings : PydanticBaseSettingsSource
            Unused.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Single-element tuple containing only `init_settings`.
        """
        return (init_settings,)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    r"""Parse ``--set`` command line overrides into a nested dictionary.

    Each pair must have the form ``key=value`` or ``section.key=value``,
    where dotted keys build nested dictionaries. `value` is parsed using
    TOML syntax, so it follows the same conventions as the configuration
    file itself (e.g. strings must be quoted, `true`/`false` for booleans).

    Parameters
    ----------
    pairs : list of str
        Override strings as supplied on the command line, e.g.
        ``["section.key=1", "other=\"text\""]``.

    Returns
    -------
    dict[str, Any]
        Nested dictionary of overrides.

    Raises
    ------
    ValueError
        If a pair does not contain ``=``, or if its value is not valid
        TOML syntax.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        if not separator:
            raise ValueError(
                f"Invalid --set value {pair!r}; expected key=value."
            )

        try:
            value = tomllib.loads(f"_ = {raw_value}")["_"]
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"Invalid TOML value in --set {pair!r}: {exc}"
            ) from exc

        target = overrides
        keys = key.split(".")
        for nested_key in keys[:-1]:
            target = target.setdefault(nested_key, {})
        target[keys[-1]] = value

    return overrides


def _deep_merge(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge `overrides` into `base`, returning a new dictionary.

    Parameters
    ----------
    base : dict[str, Any]
        The base dictionary.
    overrides : dict[str, Any]
        Dictionary of overrides to merge on top of `base`. Nested dicts
        are merged recursively; other values overwrite the base value.

    Returns
    -------
    dict[str, Any]
        A new, merged dictionary. Neither input is mutated.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_overrides(data: dict[str, Any], pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set`` overrides and deep-merge them into `data`.

    Parameters
    ----------
    data : dict[str, Any]
        The base configuration dictionary, e.g. parsed from a TOML file.
    pairs : list of str
        Override strings as supplied on the command line. See
        :func:`parse_overrides`.

    Returns
    -------
    dict[str, Any]
        A new dictionary with `pairs` merged on top of `data`. `data` is
        not mutated.

    Raises
    ------
    ValueError
        If any element of `pairs` is malformed. See :func:`parse_overrides`.
    """
    return _deep_merge(data, parse_overrides(pairs))


def load_settings(
    config_path: Path | None = None, set_: list[str] | None = None
) -> Settings:
    """Load :class:`Settings` from a TOML configuration file and CLI overrides.

    Parameters
    ----------
    config_path : pathlib.Path or None, optional
        Path to a TOML configuration file to load. If ``None`` (default),
        no file is read.
    set_ : list of str or None, optional
        ``--set key=value`` override strings to apply on top of
        `config_path` (or on top of field defaults if `config_path` is
        ``None``). See :func:`parse_overrides` for syntax.

    Returns
    -------
    Settings
        The loaded and validated settings. Relative
        ``paths.rsk_directory``, ``paths.profiles_directory``,
        ``paths.binned_directory``, and (when given) ``paths.log_file``
        and ``paths.error_log_file`` values are resolved against the
        directory containing `config_path` (or the current working
        directory if `config_path` is ``None``), so a project's config
        resolves correctly regardless of where it is loaded from.

    Raises
    ------
    FileNotFoundError
        If `config_path` is given but does not point to an existing file.
    ValueError
        If `set_` contains a malformed override. See :func:`parse_overrides`.
    pydantic.ValidationError
        If the merged configuration contains unknown or invalid keys.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    if set_:
        data = merge_overrides(data, set_)

    settings = Settings.model_validate(data)

    base_dir = config_path.parent if config_path is not None else Path.cwd()
    settings.paths.rsk_directory = base_dir / settings.paths.rsk_directory
    settings.paths.profiles_directory = (
        base_dir / settings.paths.profiles_directory
    )
    settings.paths.binned_directory = base_dir / settings.paths.binned_directory
    if settings.paths.log_file is not None:
        settings.paths.log_file = base_dir / settings.paths.log_file
    if settings.paths.error_log_file is not None:
        settings.paths.error_log_file = base_dir / settings.paths.error_log_file

    return settings
