"""Tests for ctd_processing.config."""

import pytest
from pydantic import ValidationError

from ctd_processing.config import (
    BinSettings,
    ChannelSettings,
    CTLagSettings,
    DeploymentSettings,
    DerivedVariablesSettings,
    DespikeChannelOverride,
    DespikeSettings,
    GeolocationSettings,
    InstrumentSettings,
    NetcdfCompressionSettings,
    ParquetCompressionSettings,
    PathsSettings,
    ProcessSettings,
    ProfileSettings,
    ProjectSettings,
    RawChannelSettings,
    Settings,
    ZarrCompressionSettings,
    load_settings,
    merge_overrides,
    parse_overrides,
    resolve_despike_settings,
    resolve_output_dtype,
    resolve_process_settings,
)

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)


def _other_paths(tmp_path) -> list[str]:
    """--set args for profiles_directory/binned_directory/geolocation.

    `geolocation` is included here (not just the two required `paths`
    fields) since `[process.geolocation]` is itself required -- see
    `GeolocationSettings`.
    """
    return [
        f'paths.profiles_directory="{(tmp_path / "profiles").as_posix()}"',
        f'paths.binned_directory="{(tmp_path / "binned").as_posix()}"',
        "process.geolocation.reference_latitude=0.0",
        "process.geolocation.reference_longitude=0.0",
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


def test_load_settings_project_defaults_without_project_set(tmp_path) -> None:
    """Project is optional; [paths]/[process.geolocation] are required."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.project == ProjectSettings()
    assert settings.process == ProcessSettings(geolocation=_GEOLOCATION)


def test_load_settings_missing_process_geolocation_raises(tmp_path) -> None:
    """Omitting [process.geolocation] fails validation, like missing [paths]."""
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    with pytest.raises(ValidationError):
        load_settings(
            set_=[
                f'paths.rsk_directory="{rsk_dir.as_posix()}"',
                f'paths.profiles_directory="{profiles_dir.as_posix()}"',
                f'paths.binned_directory="{binned_dir.as_posix()}"',
            ]
        )


def test_geolocation_settings_rejects_neither_source_set() -> None:
    """Neither external_dataset_path nor a reference position set fails."""
    with pytest.raises(ValidationError):
        GeolocationSettings()


def test_geolocation_settings_rejects_both_sources_set() -> None:
    """Setting both external_dataset_path and a reference position fails."""
    with pytest.raises(ValidationError):
        GeolocationSettings(
            external_dataset_path="gps.nc",
            reference_latitude=0.0,
            reference_longitude=0.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_latitude": 0.0},
        {"reference_longitude": 0.0},
    ],
)
def test_geolocation_settings_rejects_partial_reference_position(
    kwargs: dict,
) -> None:
    """reference_latitude/reference_longitude must be set together."""
    with pytest.raises(ValidationError):
        GeolocationSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_latitude": 91.0, "reference_longitude": 0.0},
        {"reference_latitude": -91.0, "reference_longitude": 0.0},
        {"reference_latitude": 0.0, "reference_longitude": 181.0},
        {"reference_latitude": 0.0, "reference_longitude": -181.0},
    ],
)
def test_geolocation_settings_rejects_out_of_range_position(
    kwargs: dict,
) -> None:
    """reference_latitude/reference_longitude outside their range fail."""
    with pytest.raises(ValidationError):
        GeolocationSettings(**kwargs)


def test_geolocation_settings_accepts_external_dataset_only() -> None:
    """external_dataset_path alone, with no reference position, is valid."""
    settings = GeolocationSettings(external_dataset_path="gps.nc")
    assert settings.reference_latitude is None
    assert settings.reference_longitude is None


def test_geolocation_settings_accepts_reference_position_only() -> None:
    """A complete reference position alone, with no dataset, is valid."""
    settings = GeolocationSettings(
        reference_latitude=45.0, reference_longitude=-125.0
    )
    assert settings.external_dataset_path is None


def test_load_settings_resolves_relative_external_dataset_path(
    tmp_path,
) -> None:
    """A relative external_dataset_path resolves against the config parent."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        "[process.geolocation]\n"
        'external_dataset_path = "gps.nc"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert (
        settings.process.geolocation.external_dataset_path
        == project_dir / "gps.nc"
    )


def test_load_settings_keeps_absolute_external_dataset_path(tmp_path) -> None:
    """An absolute external_dataset_path in the config is left untouched."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    gps_path = tmp_path / "elsewhere" / "gps.nc"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        "[process.geolocation]\n"
        f'external_dataset_path = "{gps_path.as_posix()}"\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.geolocation.external_dataset_path == gps_path


def test_resolve_process_settings_geolocation_instrument_override(
    tmp_path,
) -> None:
    """A partial instrument override merges field-by-field onto geolocation."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            geolocation=GeolocationSettings(
                reference_latitude=0.0, reference_longitude=0.0
            )
        ),
        instruments={
            "208532": InstrumentSettings(
                process={"geolocation": {"reference_latitude": 45.0}}
            )
        },
    )

    resolved = resolve_process_settings(settings, serial_number="208532")

    assert resolved.geolocation.reference_latitude == 45.0
    assert resolved.geolocation.reference_longitude == 0.0


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
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
        encoding="utf-8",
    )
    assert load_settings(config_path) == Settings(
        paths=PathsSettings(
            rsk_directory=rsk_dir,
            profiles_directory=profiles_dir,
            binned_directory=binned_dir,
        ),
        process=ProcessSettings(geolocation=_GEOLOCATION),
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
        'binned_directory = "binned_files"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
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
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
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
            "process.geolocation.reference_latitude=0.0",
            "process.geolocation.reference_longitude=0.0",
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
        'error_log_file = "logs/ctd.error.log"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
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
        f'error_log_file = "{error_log_file.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.log_file == log_file
    assert settings.paths.error_log_file == error_log_file


def test_load_settings_concatenated_file_defaults_to_none(tmp_path) -> None:
    """concatenated_file defaults to None when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.paths.concatenated_file is None


def test_load_settings_resolves_relative_concatenated_file(
    tmp_path,
) -> None:
    """A relative concatenated_file resolves against the config parent."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        'concatenated_file = "concatenated.nc"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.concatenated_file == project_dir / "concatenated.nc"


def test_load_settings_keeps_absolute_concatenated_file_from_file(
    tmp_path,
) -> None:
    """An absolute concatenated_file value is left untouched."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = project_dir / "config.toml"
    concatenated_file = tmp_path / "elsewhere" / "concatenated.nc"
    config_path.write_text(
        "[paths]\n"
        'rsk_directory = "rsk_files"\n'
        'profiles_directory = "profiles_files"\n'
        'binned_directory = "binned_files"\n'
        f'concatenated_file = "{concatenated_file.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.concatenated_file == concatenated_file


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


def test_process_read_channels_defaults_to_empty(tmp_path) -> None:
    """process.read_channels defaults to [] when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.read_channels == []


def test_process_read_channels_can_be_set(tmp_path) -> None:
    """process.read_channels parses as a list of RBR channel longNames."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process]\n"
        'read_channels = ["temperature", "conductivity"]\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.read_channels == ["temperature", "conductivity"]


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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.profiles]\n"
        "min_pressure = 0.5\n"
        "peak_height = 10.0\n"
        'direction = "both"\n'
        'speed_threshold_direction = "down"\n'
        "apply_speed_threshold = true\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.profiles.min_pressure == 0.5
    assert settings.process.profiles.peak_height == 10.0
    assert settings.process.profiles.direction == "both"
    assert settings.process.profiles.speed_threshold_direction == "down"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
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
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.profiles]\n"
        'direction = "sideways"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_profiles_rejects_invalid_speed_threshold_direction(
    tmp_path,
) -> None:
    """An invalid speed_threshold_direction value fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.profiles]\n"
        'speed_threshold_direction = "sideways"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_settings_profile_format_defaults_to_parquet() -> None:
    """profile_format defaults to "parquet" when unset."""
    assert ProcessSettings(geolocation=_GEOLOCATION).profile_format == "parquet"


def test_process_settings_rejects_invalid_profile_format(tmp_path) -> None:
    """An invalid profile_format value fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process]\n"
        'profile_format = "csv"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_ct_lag_defaults(tmp_path) -> None:
    """process.ct_lag defaults to CTLagSettings() when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.ct_lag == CTLagSettings()
    assert settings.process.ct_lag.enabled is False


def test_process_ct_lag_fields_can_be_set(tmp_path) -> None:
    """[process.ct_lag] fields parse correctly from a config file."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.ct_lag]\n"
        "enabled = true\n"
        "sea_pressure_min = 1.0\n"
        "sea_pressure_max = 500.0\n"
        "window_length = 15\n"
        "min_lag = -10\n"
        "max_lag = 10\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.ct_lag.enabled is True
    assert settings.process.ct_lag.sea_pressure_min == 1.0
    assert settings.process.ct_lag.sea_pressure_max == 500.0
    assert settings.process.ct_lag.window_length == 15
    assert settings.process.ct_lag.min_lag == -10
    assert settings.process.ct_lag.max_lag == 10


def test_process_ct_lag_rejects_unknown_key(tmp_path) -> None:
    """An unknown key inside [process.ct_lag] fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.ct_lag]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_process_ct_lag_rejects_even_window_length() -> None:
    """An even window_length fails validation."""
    with pytest.raises(ValidationError):
        CTLagSettings(window_length=20)


def test_process_ct_lag_rejects_min_lag_greater_than_max_lag() -> None:
    """min_lag greater than max_lag fails validation."""
    with pytest.raises(ValidationError):
        CTLagSettings(min_lag=5, max_lag=-5)


def test_process_derived_variables_defaults(tmp_path) -> None:
    """process.derived_variables defaults to DerivedVariablesSettings()."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.derived_variables == DerivedVariablesSettings()


def test_derived_variables_settings_core_five_default_true() -> None:
    """The core five derived variables default to enabled."""
    settings = DerivedVariablesSettings()
    assert settings.z is True
    assert settings.practical_salinity is True
    assert settings.absolute_salinity is True
    assert settings.conservative_temperature is True
    assert settings.potential_density is True


def test_derived_variables_settings_extras_default_false() -> None:
    """The optional extras default to disabled."""
    settings = DerivedVariablesSettings()
    assert settings.potential_temperature is False
    assert settings.sound_speed is False
    assert settings.density is False
    assert settings.spiciness is False
    assert settings.freezing_point is False
    assert settings.thermal_expansion is False
    assert settings.haline_contraction is False
    assert settings.oxygen_concentration is False


def test_process_derived_variables_fields_can_be_set(tmp_path) -> None:
    """[process.derived_variables] fields parse correctly from a config file."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.derived_variables]\n"
        "z = false\n"
        "sound_speed = true\n"
        "oxygen_concentration = true\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.process.derived_variables.z is False
    assert settings.process.derived_variables.sound_speed is True
    assert settings.process.derived_variables.oxygen_concentration is True
    assert settings.process.derived_variables.practical_salinity is True


def test_process_derived_variables_rejects_unknown_key(tmp_path) -> None:
    """An unknown key inside [process.derived_variables] fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[process.derived_variables]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_despike_settings_defaults() -> None:
    """window_length matches pyrsktools'; threshold/iterations don't."""
    settings = DespikeSettings()
    assert settings.threshold == 3.0
    assert settings.window_length == 3
    assert settings.iterations == 1


def test_despike_settings_rejects_even_window_length() -> None:
    """An even window_length fails validation."""
    with pytest.raises(ValidationError):
        DespikeSettings(window_length=4)


def test_process_despiking_defaults(tmp_path) -> None:
    """process.despiking defaults to DespikeSettings(), channels to {}."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.process.despiking == DespikeSettings()
    assert settings.process.channels == {}


def test_resolve_despike_settings_omits_unconfigured_channels() -> None:
    """A channel with despike left at False has no resolved settings."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        channels={"practical_salinity": ChannelSettings(despike=True)},
    )

    resolved = resolve_despike_settings(process_settings)

    assert set(resolved) == {"practical_salinity"}
    assert "sea_water_temperature" not in resolved


def test_resolve_despike_settings_uses_defaults_for_plain_true() -> None:
    """Despike = true uses despiking's project-wide defaults as-is."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        despiking=DespikeSettings(threshold=3.0, window_length=5),
        channels={"practical_salinity": ChannelSettings(despike=True)},
    )

    resolved = resolve_despike_settings(process_settings)

    assert resolved["practical_salinity"] == DespikeSettings(
        threshold=3.0, window_length=5
    )


def test_resolve_despike_settings_merges_partial_override() -> None:
    """A partial per-channel override changes only the given fields."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        despiking=DespikeSettings(threshold=2.0, window_length=5),
        channels={
            "practical_salinity": ChannelSettings(
                despike=True,
                despiking=DespikeChannelOverride(threshold=4.0),
            )
        },
    )

    resolved = resolve_despike_settings(process_settings)

    assert resolved["practical_salinity"].threshold == 4.0
    assert resolved["practical_salinity"].window_length == 5


def test_resolve_despike_settings_ignores_despiking_when_despike_false() -> (
    None
):
    """A despiking override has no effect unless despike is True."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        channels={
            "practical_salinity": ChannelSettings(
                despiking=DespikeChannelOverride(threshold=4.0)
            )
        },
    )

    resolved = resolve_despike_settings(process_settings)

    assert resolved == {}


def test_load_settings_rejects_invalid_despike_channel_override(
    tmp_path,
) -> None:
    """A bad per-channel despiking override in an instrument override raises."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[instruments.208532.process.channels.practical_salinity]\n"
        "despike = true\n"
        "[instruments.208532.process.channels.practical_salinity.despiking]\n"
        "window_length = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_channel_settings_despike_bool_defaults_false() -> None:
    """ChannelSettings.despike defaults to False (not despiked)."""
    assert ChannelSettings().despike is False


def test_channel_settings_despiking_defaults_empty() -> None:
    """ChannelSettings.despiking defaults to an all-None override."""
    assert ChannelSettings().despiking == DespikeChannelOverride()


def test_channel_settings_output_dtype_defaults_none() -> None:
    """output_dtype defaults to None, meaning use the project default."""
    assert ChannelSettings().output_dtype is None


def test_channel_settings_rejects_invalid_output_dtype() -> None:
    """A non-numpy-dtype output_dtype string fails validation."""
    with pytest.raises(ValidationError):
        ChannelSettings(output_dtype="not_a_dtype")


def test_channel_settings_rejects_non_floating_output_dtype() -> None:
    """An integer output_dtype fails validation."""
    with pytest.raises(ValidationError):
        ChannelSettings(output_dtype="int32")


def test_process_settings_output_dtype_defaults_to_float32() -> None:
    """ProcessSettings.output_dtype defaults to 'float32'."""
    assert ProcessSettings(geolocation=_GEOLOCATION).output_dtype == "float32"


def test_process_settings_rejects_non_floating_output_dtype() -> None:
    """A non-floating project-wide output_dtype fails validation."""
    with pytest.raises(ValidationError):
        ProcessSettings(geolocation=_GEOLOCATION, output_dtype="int64")


def test_netcdf_compression_settings_defaults() -> None:
    """Defaults match this package's previous hardcoded netCDF behavior."""
    settings = NetcdfCompressionSettings()
    assert settings.enabled is True
    assert settings.complevel == 4
    assert settings.shuffle is True


def test_netcdf_compression_settings_rejects_out_of_range_complevel() -> None:
    """Complevel must be within HDF5's 0-9 deflate range."""
    with pytest.raises(ValidationError):
        NetcdfCompressionSettings(complevel=-1)
    with pytest.raises(ValidationError):
        NetcdfCompressionSettings(complevel=10)


def test_parquet_compression_settings_defaults() -> None:
    """Defaults match this package's previous hardcoded parquet behavior."""
    settings = ParquetCompressionSettings()
    assert settings.enabled is True
    assert settings.level is None


def test_parquet_compression_settings_rejects_out_of_range_level() -> None:
    """Level must be within zstd's documented 1-22 range."""
    with pytest.raises(ValidationError):
        ParquetCompressionSettings(level=0)
    with pytest.raises(ValidationError):
        ParquetCompressionSettings(level=23)


def test_zarr_compression_settings_defaults() -> None:
    """Defaults are the new, deliberate replacement for zarr's own default."""
    settings = ZarrCompressionSettings()
    assert settings.enabled is True
    assert settings.cname == "zstd"
    assert settings.clevel == 5
    assert settings.shuffle is None


def test_zarr_compression_settings_rejects_out_of_range_clevel() -> None:
    """Clevel must be within Blosc's 0-9 range."""
    with pytest.raises(ValidationError):
        ZarrCompressionSettings(clevel=-1)
    with pytest.raises(ValidationError):
        ZarrCompressionSettings(clevel=10)


def test_process_settings_compression_fields_default_to_enabled() -> None:
    """ProcessSettings' compression fields default to enabled settings."""
    settings = ProcessSettings(geolocation=_GEOLOCATION)
    assert settings.netcdf_compression == NetcdfCompressionSettings()
    assert settings.parquet_compression == ParquetCompressionSettings()


def test_bin_settings_compression_fields_default_to_enabled() -> None:
    """BinSettings' compression fields default to enabled settings."""
    settings = BinSettings()
    assert settings.netcdf_compression == NetcdfCompressionSettings()
    assert settings.zarr_compression == ZarrCompressionSettings()


def test_process_and_bin_netcdf_compression_are_independent_instances() -> None:
    """Overriding one's netcdf_compression does not affect the other's."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        netcdf_compression=NetcdfCompressionSettings(complevel=9),
    )
    bin_settings = BinSettings()

    assert process_settings.netcdf_compression.complevel == 9
    assert bin_settings.netcdf_compression.complevel == 4


def test_resolve_output_dtype_falls_back_to_project_default() -> None:
    """A channel with no output_dtype override uses the project default."""
    process_settings = ProcessSettings(geolocation=_GEOLOCATION)

    assert resolve_output_dtype(process_settings, "temperature") == "float32"


def test_resolve_output_dtype_uses_channel_override() -> None:
    """A channel's own output_dtype overrides the project default."""
    process_settings = ProcessSettings(
        geolocation=_GEOLOCATION,
        output_dtype="float32",
        channels={
            "practical_salinity": ChannelSettings(output_dtype="float64")
        },
    )

    assert (
        resolve_output_dtype(process_settings, "practical_salinity")
        == "float64"
    )
    assert resolve_output_dtype(process_settings, "temperature") == "float32"


def test_instruments_and_deployments_default_to_empty(tmp_path) -> None:
    """instruments/deployments default to {} when omitted."""
    settings = load_settings(
        set_=[f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"']
        + _other_paths(tmp_path)
    )
    assert settings.instruments == {}
    assert settings.deployments == {}


def test_instruments_section_parses_from_file(tmp_path) -> None:
    """[instruments.<serial>.process] parses into Settings.instruments."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[instruments.208532.process]\n"
        "atmospheric_pressure = 10.1\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.instruments == {
        "208532": InstrumentSettings(process={"atmospheric_pressure": 10.1})
    }


def test_deployments_section_parses_from_file(tmp_path) -> None:
    """[deployments.<stem>.process] parses into Settings.deployments."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[deployments.243188_20260809_0304.process]\n"
        "atmospheric_pressure = 10.1325\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.deployments == {
        "243188_20260809_0304": DeploymentSettings(
            process={"atmospheric_pressure": 10.1325}
        )
    }


def test_instruments_rejects_unknown_key(tmp_path) -> None:
    """An unknown key under [instruments.<serial>] fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[instruments.208532]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_deployments_rejects_unknown_key(tmp_path) -> None:
    """An unknown key under [deployments.<stem>] fails validation."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[deployments.243188_20260809_0304]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_load_settings_rejects_invalid_instrument_override(tmp_path) -> None:
    """A bad field inside an instrument override fails at load time."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[instruments.208532.process]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_load_settings_rejects_invalid_deployment_override(tmp_path) -> None:
    """A bad field inside a deployment override fails at load time."""
    config_path = tmp_path / "config.toml"
    rsk_dir = tmp_path / "rsk"
    profiles_dir = tmp_path / "profiles"
    binned_dir = tmp_path / "binned"
    config_path.write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{profiles_dir.as_posix()}"\n'
        f'binned_directory = "{binned_dir.as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n"
        "[deployments.243188_20260809_0304.process]\n"
        "not_a_real_option = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_instruments_override_via_set(tmp_path) -> None:
    """--set instruments.<serial>.process.<key>=<value> works."""
    settings = load_settings(
        set_=[
            f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"',
            "instruments.208532.process.atmospheric_pressure=10.1",
        ]
        + _other_paths(tmp_path)
    )
    assert settings.instruments["208532"].process == {
        "atmospheric_pressure": 10.1
    }


def test_deployments_override_via_set(tmp_path) -> None:
    """--set deployments.<stem>.process.<key>=<value> works."""
    settings = load_settings(
        set_=[
            f'paths.rsk_directory="{(tmp_path / "rsk").as_posix()}"',
            "deployments.243188_20260809_0304.process.atmospheric_pressure=10.1325",
        ]
        + _other_paths(tmp_path)
    )
    assert settings.deployments["243188_20260809_0304"].process == {
        "atmospheric_pressure": 10.1325
    }


def test_resolve_process_settings_returns_project_settings_unchanged() -> None:
    """With no matching instrument/deployment, project settings pass through."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            atmospheric_pressure=10.0, geolocation=_GEOLOCATION
        ),
    )

    resolved = resolve_process_settings(
        settings, serial_number="999999", stem="unmatched"
    )

    assert resolved == settings.process


def test_resolve_process_settings_applies_instrument_override() -> None:
    """An instrument override merges onto the project-level settings."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            atmospheric_pressure=10.0, geolocation=_GEOLOCATION
        ),
        instruments={
            "208532": InstrumentSettings(
                process={
                    "raw_channels": {"sea_water_temperature": {"shift": 2}}
                }
            )
        },
    )

    resolved = resolve_process_settings(settings, serial_number="208532")

    assert resolved.atmospheric_pressure == 10.0
    assert resolved.raw_channels["sea_water_temperature"] == RawChannelSettings(
        shift=2
    )


def test_resolve_process_settings_applies_deployment_override() -> None:
    """A deployment override merges onto the project-level settings."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            atmospheric_pressure=10.0, geolocation=_GEOLOCATION
        ),
        deployments={
            "243188_20260809_0304": DeploymentSettings(
                process={"atmospheric_pressure": 10.5}
            )
        },
    )

    resolved = resolve_process_settings(settings, stem="243188_20260809_0304")

    assert resolved.atmospheric_pressure == 10.5


def test_resolve_process_settings_deployment_wins_over_instrument() -> None:
    """A deployment override wins over an instrument override, same field."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            atmospheric_pressure=10.0, geolocation=_GEOLOCATION
        ),
        instruments={
            "208532": InstrumentSettings(process={"atmospheric_pressure": 10.1})
        },
        deployments={
            "243188_20260809_0304": DeploymentSettings(
                process={"atmospheric_pressure": 10.5}
            )
        },
    )

    resolved = resolve_process_settings(
        settings, serial_number="208532", stem="243188_20260809_0304"
    )

    assert resolved.atmospheric_pressure == 10.5


def test_resolve_process_settings_combines_disjoint_overrides() -> None:
    """Instrument and deployment overrides on different fields both apply."""
    settings = Settings(
        paths=PathsSettings(
            rsk_directory="rsk",
            profiles_directory="profiles",
            binned_directory="binned",
        ),
        process=ProcessSettings(
            atmospheric_pressure=10.0, geolocation=_GEOLOCATION
        ),
        instruments={
            "208532": InstrumentSettings(
                process={
                    "raw_channels": {"sea_water_temperature": {"shift": 2}}
                }
            )
        },
        deployments={
            "243188_20260809_0304": DeploymentSettings(
                process={"atmospheric_pressure": 10.5}
            )
        },
    )

    resolved = resolve_process_settings(
        settings, serial_number="208532", stem="243188_20260809_0304"
    )

    assert resolved.atmospheric_pressure == 10.5
    assert resolved.raw_channels["sea_water_temperature"] == RawChannelSettings(
        shift=2
    )
