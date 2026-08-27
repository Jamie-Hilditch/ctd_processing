"""Tests for ctd_processing.config."""

import pytest
from pydantic import ValidationError

from ctd_processing.config import (
    PathsSettings,
    ProcessSettings,
    ProfileSettings,
    ProjectSettings,
    RawChannelSettings,
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


def test_load_settings_log_files_default_to_none(tmp_path) -> None:
    """log_file/error_log_file default to None when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.paths.log_file is None
    assert settings.paths.error_log_file is None


def test_load_settings_resolves_relative_log_files_against_config_parent(
    tmp_path,
) -> None:
    """Relative log_file/error_log_file resolve against the config parent."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        'log_file = "logs/ctd.log"\n'
        'error_log_file = "logs/ctd.error.log"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.log_file == project_dir / "logs" / "ctd.log"
    assert (
        settings.paths.error_log_file == project_dir / "logs" / "ctd.error.log"
    )


def test_load_settings_keeps_absolute_log_files_from_file(tmp_path) -> None:
    """Absolute log_file/error_log_file values are left untouched."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    log_file = tmp_path / "elsewhere" / "ctd.log"
    error_log_file = tmp_path / "elsewhere" / "ctd.error.log"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        f'log_file = "{log_file.as_posix()}"\n'
        f'error_log_file = "{error_log_file.as_posix()}"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.log_file == log_file
    assert settings.paths.error_log_file == error_log_file


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


def test_process_raw_channels_defaults_to_empty(tmp_path) -> None:
    """process.raw_channels defaults to {} when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.raw_channels == {}


def test_process_raw_channels_section_defaults_remove_holds_true(
    tmp_path,
) -> None:
    """An empty raw-channel section still gets remove_holds=True."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.raw_channels.sea_water_temperature]\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.raw_channels == {
        "sea_water_temperature": RawChannelSettings()
    }
    assert (
        settings.process.raw_channels["sea_water_temperature"].remove_holds
        is True
    )
    assert settings.process.raw_channels["sea_water_temperature"].offset is None
    assert settings.process.raw_channels["sea_water_temperature"].shift is None


def test_process_raw_channels_remove_holds_can_be_disabled(tmp_path) -> None:
    """remove_holds = false parses correctly for a named raw channel."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.raw_channels.sea_water_temperature]\n"
        "remove_holds = false\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert (
        settings.process.raw_channels["sea_water_temperature"].remove_holds
        is False
    )


def test_process_raw_channels_offset_can_be_set(tmp_path) -> None:
    """Offset = 1.5 parses correctly for a named raw channel."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.raw_channels.sea_water_temperature]\n"
        "offset = 1.5\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.raw_channels["sea_water_temperature"].offset == 1.5


@pytest.mark.parametrize("shift", [3, -2])
def test_process_raw_channels_shift_can_be_set(tmp_path, shift: int) -> None:
    """A positive or negative shift value parses correctly."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.raw_channels.sea_water_temperature]\n"
        f"shift = {shift}\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.raw_channels["sea_water_temperature"].shift == shift


def test_process_raw_channels_rejects_unknown_key(tmp_path) -> None:
    """An unknown key inside a raw-channel section fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.raw_channels.sea_water_temperature]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_atmospheric_pressure_defaults_to_none(tmp_path) -> None:
    """process.atmospheric_pressure defaults to None (trust sea_pressure)."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.atmospheric_pressure is None


def test_process_atmospheric_pressure_can_be_overridden(tmp_path) -> None:
    """process.atmospheric_pressure can be overridden via --set."""
    settings = load_settings(
        set_=[
            f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"',
            "process.atmospheric_pressure=10.05",
        ]
        + _other_paths(tmp_path)
    )
    assert settings.process.atmospheric_pressure == 10.05


def test_process_profiles_defaults(tmp_path) -> None:
    """process.profiles defaults to ProfileSettings() when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.profiles == ProfileSettings()


def test_process_profiles_fields_can_be_set(tmp_path) -> None:
    """[process.profiles] fields parse correctly from a config file."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.profiles]\n"
        "min_pressure = 0.5\n"
        "peak_height = 10.0\n"
        'direction = "both"\n'
        "apply_speed_threshold = true\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.profiles.min_pressure == 0.5
    assert settings.process.profiles.peak_height == 10.0
    assert settings.process.profiles.direction == "both"
    assert settings.process.profiles.apply_speed_threshold is True


def test_process_profiles_rejects_unknown_key(tmp_path) -> None:
    """An unknown key inside [process.profiles] fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.profiles]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_profiles_rejects_invalid_direction(tmp_path) -> None:
    """An invalid direction value fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.profiles]\n"
        'direction = "sideways"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)
