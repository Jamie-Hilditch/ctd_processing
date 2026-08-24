"""Tests for ctd_processing.config."""

import pytest
from pydantic import ValidationError

from ctd_processing.config import (
    PathsSettings,
    ProcessSettings,
    ProjectSettings,
    Settings,
    load_settings,
    merge_overrides,
    parse_overrides,
)


def _other_paths(tmp_path) -> list[str]:
    """--set args for the required profiles_directory/binned_directory."""
    return [
        f'paths.profiles_directory="{(tmp_path / "profiles").as_posix()}"',
        f'paths.binned_directory="{(tmp_path / "binned").as_posix()}"',
    ]


def test_parse_overrides_flat_key() -> None:
    """A flat key=value pair should parse to a single-level dict."""
    assert parse_overrides(["name=1"]) == {"name": 1}


def test_parse_overrides_nested_dotted_key() -> None:
    """Dotted keys should build nested dictionaries."""
    assert parse_overrides(["section.key=1"]) == {"section": {"key": 1}}


def test_parse_overrides_deeply_nested_dotted_key() -> None:
    """Multiple dots should build multiple levels of nesting."""
    assert parse_overrides(["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", 1),
        ("1.5", 1.5),
        ("true", True),
        ("false", False),
        ('"text"', "text"),
        ("[1, 2, 3]", [1, 2, 3]),
    ],
)
def test_parse_overrides_type_coercion(
    raw_value: str, expected: object
) -> None:
    """Values should be parsed using TOML syntax, matching the config file."""
    assert parse_overrides([f"key={raw_value}"]) == {"key": expected}


def test_parse_overrides_missing_equals_raises() -> None:
    """A pair with no '=' should raise ValueError."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_overrides(["not-a-pair"])


def test_parse_overrides_invalid_value_raises() -> None:
    """A value that isn't valid TOML syntax should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid TOML value"):
        parse_overrides(["key=not valid toml"])


def test_merge_overrides_deep_merges_nested_dicts() -> None:
    """Overrides should merge into nested dicts rather than replacing them."""
    base = {"section": {"a": 1, "b": 2}}
    merged = merge_overrides(base, ["section.b=3"])
    assert merged == {"section": {"a": 1, "b": 3}}


def test_merge_overrides_does_not_mutate_input() -> None:
    """merge_overrides should return a new dict, leaving the input untouched."""
    base = {"section": {"a": 1}}
    merge_overrides(base, ["section.a=2"])
    assert base == {"section": {"a": 1}}


def test_load_settings_missing_paths_raises() -> None:
    """With no config path or overrides, [paths] is missing and fails."""
    with pytest.raises(ValidationError):
        load_settings()


def test_load_settings_project_and_process_default_without_paths_set(
    tmp_path,
) -> None:
    """project/process are optional; only [paths] is required."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.project == ProjectSettings()
    assert settings.process == ProcessSettings()


@pytest.mark.parametrize(
    "field", ["rsk_directory", "profiles_directory", "binned_directory"]
)
def test_load_settings_missing_required_paths_field_raises(
    tmp_path, field: str
) -> None:
    """Each of the three paths fields is individually required."""
    fields = {"rsk_directory", "profiles_directory", "binned_directory"} - {
        field
    }
    set_ = [f'paths.{f}="{(tmp_path / f).as_posix()}"' for f in fields]
    with pytest.raises(ValidationError):
        load_settings(set_=set_)


def test_load_settings_reads_toml_file(tmp_path) -> None:
    """A config file supplying [paths] should load it correctly."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    assert load_settings(config_path) == Settings(
        paths=PathsSettings(
            rsk_directory=rsk_dir,
            profiles_directory=profiles_dir,
            binned_directory=binned_dir,
        )
    )


def test_load_settings_resolves_relative_paths_against_config_parent(
    tmp_path,
) -> None:
    """Relative paths resolve against the config file's parent directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.rsk_directory == project_dir / "rsk_files"
    assert settings.paths.profiles_directory == project_dir / "profiles_files"
    assert settings.paths.binned_directory == project_dir / "binned_files"


def test_load_settings_keeps_absolute_paths_from_file(
    tmp_path,
) -> None:
    """Absolute paths in the config file are left untouched."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    rsk_dir = tmp_path / "elsewhere" / "rsk"
    profiles_dir = tmp_path / "elsewhere" / "profiles"
    binned_dir = tmp_path / "elsewhere" / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.rsk_directory == rsk_dir
    assert settings.paths.profiles_directory == profiles_dir
    assert settings.paths.binned_directory == binned_dir


def test_load_settings_resolves_relative_paths_against_cwd(
    monkeypatch, tmp_path
) -> None:
    """With no config file, relative paths resolve against cwd."""
    monkeypatch.chdir(tmp_path)

    settings = load_settings(
        set_=[
            'paths.rsk_directory="rsk_files"',
            'paths.profiles_directory="profiles_files"',
            'paths.binned_directory="binned_files"',
        ]
    )

    assert settings.paths.rsk_directory == tmp_path / "rsk_files"
    assert settings.paths.profiles_directory == tmp_path / "profiles_files"
    assert settings.paths.binned_directory == tmp_path / "binned_files"


def test_load_settings_accepts_rsk_directory_via_set(tmp_path) -> None:
    """rsk_directory can be supplied purely via --set, with no config file."""
    rsk_dir = tmp_path / "rsk"
    settings = load_settings(
        set_=[f'paths.rsk_directory="{rsk_dir.as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.paths.rsk_directory == rsk_dir


def test_project_name_defaults_when_omitted(tmp_path) -> None:
    """project.name defaults to 'my_ctd_processing_project' when unset."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.project.name == "my_ctd_processing_project"


def test_project_name_can_be_overridden(tmp_path) -> None:
    """project.name can be overridden via --set."""
    settings = load_settings(
        set_=[
            f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"',
            'project.name="a custom name"',
        ]
        + _other_paths(tmp_path)
    )
    assert settings.project.name == "a custom name"


def test_load_settings_missing_file_raises(tmp_path) -> None:
    """A nonexistent config path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml")


def test_load_settings_rejects_unknown_key_from_file(tmp_path) -> None:
    """An unknown key in the config file should fail validation."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("not_a_real_option = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_load_settings_rejects_unknown_key_from_overrides() -> None:
    """An unknown key from --set should fail validation."""
    with pytest.raises(ValidationError):
        load_settings(set_=["not_a_real_option=1"])


def test_load_settings_propagates_malformed_override() -> None:
    """A malformed --set pair raises ValueError, not a validation error."""
    with pytest.raises(ValueError, match="expected key=value"):
        load_settings(set_=["not-a-pair"])
