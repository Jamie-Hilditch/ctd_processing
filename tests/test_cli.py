"""Tests for ctd_processing.cli."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ctd_processing.cli import app
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save import save_profile

runner = CliRunner()


def _write_profile(
    directory: Path,
    index: int,
    total: int,
    *,
    start: str,
    serial: int = 208532,
    source_file: str = "243188_20260809_0304.rsk",
) -> Path:
    """Write one small, already-processed profile file for bin CLI tests."""
    n = 5
    time = Channel(
        data=np.datetime64(start) + np.arange(n) * np.timedelta64(1, "s")
    )
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {"instrument_serial_number": serial, "source_file": source_file}
    )
    dataset.add_channel(
        "z", Channel(data=-np.linspace(0, 4, n), metadata={"units": "m"})
    )
    dataset.add_channel(
        "sea_water_temperature",
        Channel(data=np.linspace(10, 11, n), metadata={"units": "degree_C"}),
    )
    dataset.metadata.update(
        {
            "profile_start_time": time.data[0],
            "profile_end_time": time.data[-1],
            "latitude": 45.0,
            "longitude": -125.0,
        }
    )
    return save_profile(dataset, dataset, index, total, directory, "netcdf")


def _other_paths(tmp_path: Path) -> list[str]:
    """CLI args for profiles_directory/binned_directory/geolocation.

    `geolocation` is included here (not just the two required `paths`
    fields) since `[process.geolocation]` is itself required -- see
    `ctd_processing.config.GeolocationSettings`.
    """
    return [
        "--set",
        f'paths.profiles_directory="{(tmp_path / "profiles").as_posix()}"',
        "--set",
        f'paths.binned_directory="{(tmp_path / "binned").as_posix()}"',
        "--set",
        "process.geolocation.reference_latitude=0.0",
        "--set",
        "process.geolocation.reference_longitude=0.0",
    ]


def test_help_lists_all_commands() -> None:
    """--help should list all four top-level commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("init", "process", "bin", "concatenate"):
        assert command in result.stdout


def test_init_default_populates_project_and_creates_directories(
    tmp_path: Path,
) -> None:
    """Init with no options writes CLI defaults and creates all project dirs."""
    result = runner.invoke(app, ["init", "--working-dir", str(tmp_path)])

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["project"] == {"name": "my_ctd_processing_project"}
    assert config["paths"] == {
        "rsk_directory": "rsk_files",
        "profiles_directory": "profiles",
        "binned_directory": "binned",
    }
    assert (tmp_path / "rsk_files").is_dir()
    assert (tmp_path / "profiles").is_dir()
    assert (tmp_path / "binned").is_dir()


def test_init_project_directory_options_are_overridable(
    tmp_path: Path,
) -> None:
    """--rsk-directory/--profiles-directory/--binned-directory are honored."""
    result = runner.invoke(
        app,
        [
            "init",
            "--working-dir",
            str(tmp_path),
            "--name",
            "Cruise 1",
            "--rsk-directory",
            "data/rsk",
            "--profiles-directory",
            "data/profiles",
            "--binned-directory",
            "data/binned",
        ],
    )

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["project"] == {"name": "Cruise 1"}
    assert config["paths"] == {
        "rsk_directory": "data/rsk",
        "profiles_directory": "data/profiles",
        "binned_directory": "data/binned",
    }
    assert (tmp_path / "data" / "rsk").is_dir()
    assert (tmp_path / "data" / "profiles").is_dir()
    assert (tmp_path / "data" / "binned").is_dir()


def test_init_log_file_options_are_written_and_directories_created(
    tmp_path: Path,
) -> None:
    """--log-file/--error-log-file are written and their parents created."""
    result = runner.invoke(
        app,
        [
            "init",
            "--working-dir",
            str(tmp_path),
            "--log-file",
            "logs/ctd.log",
            "--error-log-file",
            "logs/ctd.error.log",
        ],
    )

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["paths"]["log_file"] == "logs/ctd.log"
    assert config["paths"]["error_log_file"] == "logs/ctd.error.log"
    assert (tmp_path / "logs").is_dir()


def test_init_defaults_working_dir_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --working-dir, config.toml is written into the cwd."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (tmp_path / "config.toml").is_file()
    assert (tmp_path / "rsk_files").is_dir()


def test_init_creates_working_dir_if_missing(tmp_path: Path) -> None:
    """--working-dir is created (with parents) if it doesn't exist."""
    working_dir = tmp_path / "sub" / "dir"

    result = runner.invoke(app, ["init", "--working-dir", str(working_dir)])

    assert result.exit_code == 0
    assert (working_dir / "config.toml").is_file()


def test_init_custom_template_project_fields_are_overridden(
    tmp_path: Path,
) -> None:
    """--name/--rsk-directory always win over a template's own values.

    They're always applied on top of the template, even at their
    defaults, so a template's own [project]/[paths] values are never
    used.
    """
    template_path = tmp_path / "custom.toml"
    template_path.write_text(
        "[project]\n"
        'name = "template-name"\n'
        "[paths]\n"
        'rsk_directory = "template-rsk"\n'
        'profiles_directory = "template-profiles"\n'
        'binned_directory = "template-binned"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "init",
            "--working-dir",
            str(tmp_path),
            "--template",
            str(template_path),
        ],
    )

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["project"] == {"name": "my_ctd_processing_project"}
    assert config["paths"] == {
        "rsk_directory": "rsk_files",
        "profiles_directory": "profiles",
        "binned_directory": "binned",
    }


def test_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """Init should not overwrite an existing config.toml without --force."""
    (tmp_path / "config.toml").write_text("sentinel\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--working-dir", str(tmp_path)])

    assert result.exit_code != 0
    assert (tmp_path / "config.toml").read_text(
        encoding="utf-8"
    ) == "sentinel\n"


def test_init_force_overwrites(tmp_path: Path) -> None:
    """Init --force should overwrite an existing config.toml."""
    (tmp_path / "config.toml").write_text("sentinel\n", encoding="utf-8")

    result = runner.invoke(
        app, ["init", "--working-dir", str(tmp_path), "--force"]
    )

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["name"] == "my_ctd_processing_project"


def test_init_set_malformed_pair_errors(tmp_path: Path) -> None:
    """Init --set with a malformed pair should exit non-zero with a message."""
    result = runner.invoke(
        app,
        ["init", "--working-dir", str(tmp_path), "--set", "not-a-pair"],
    )

    assert result.exit_code != 0
    assert "expected key=value" in result.stderr
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / "rsk_files").exists()


def test_init_set_unknown_key_errors(tmp_path: Path) -> None:
    """Init --set with an unknown key exits non-zero."""
    result = runner.invoke(
        app,
        [
            "init",
            "--working-dir",
            str(tmp_path),
            "--set",
            "not_a_real_option=1",
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / "rsk_files").exists()


def test_init_set_overrides_name_option(tmp_path: Path) -> None:
    """--set is applied on top of --name, so --set wins if both are given."""
    result = runner.invoke(
        app,
        [
            "init",
            "--working-dir",
            str(tmp_path),
            "--name",
            "from-option",
            "--set",
            'project.name="from-set"',
        ],
    )

    assert result.exit_code == 0
    config = tomllib.loads(
        (tmp_path / "config.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["name"] == "from-set"


def test_bin_combines_profile_directory_into_one_dataset(
    tmp_path: Path,
) -> None:
    """Bin on a directory of profiles writes one combined file, exit 0."""
    profiles_dir = tmp_path / "profiles"
    _write_profile(profiles_dir, 0, 2, start="2026-08-09T03:04:00")
    _write_profile(profiles_dir, 1, 2, start="2026-08-09T04:04:00")
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            str(profiles_dir),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 0, result.stderr
    written = list((tmp_path / "binned").glob("*.nc"))
    assert len(written) == 1
    assert "2 profile(s)" in result.stdout


def test_bin_accepts_single_profile_file(tmp_path: Path) -> None:
    """Bin on a single profile file (not a directory) also works."""
    profiles_dir = tmp_path / "profiles"
    path = _write_profile(profiles_dir, 0, 1, start="2026-08-09T03:04:00")
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            str(path),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 0, result.stderr
    assert list((tmp_path / "binned").glob("*.nc"))


def test_bin_empty_directory_errors(tmp_path: Path) -> None:
    """A directory with no profile files exits 1 with a clean message."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            str(profiles_dir),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "No profile files found" in result.stderr


def test_bin_mixed_deployment_input_errors(tmp_path: Path) -> None:
    """Profiles from different deployments exit 1 with a clean message."""
    profiles_dir = tmp_path / "profiles"
    _write_profile(profiles_dir, 0, 2, start="2026-08-09T03:04:00")
    _write_profile(
        profiles_dir,
        1,
        2,
        start="2026-08-09T04:04:00",
        serial=999999,
        source_file="other_deployment.rsk",
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            str(profiles_dir),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "multiple deployments" in result.stderr


def test_concatenate_stub_reports_not_implemented(tmp_path: Path) -> None:
    """Concatenate stub should report 'not yet implemented' and exit 1."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "concatenate",
            str(tmp_path),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.stdout


@pytest.mark.parametrize("command", ["bin", "concatenate"])
def test_command_missing_input_path_errors(
    tmp_path: Path, command: str
) -> None:
    """A nonexistent input_path fails validation before the command runs."""
    result = runner.invoke(app, [command, str(tmp_path / "does-not-exist")])

    assert result.exit_code != 0


@pytest.mark.parametrize("command", ["bin", "concatenate"])
def test_command_set_unknown_key_errors(tmp_path: Path, command: str) -> None:
    """--set with an unknown key exits non-zero with a clean message."""
    input_path = tmp_path / "cast.rsk"
    input_path.write_text("", encoding="utf-8")

    result = runner.invoke(
        app, [command, str(input_path), "--set", "not_a_real_option=1"]
    )

    assert result.exit_code == 1
    assert result.stderr != ""


def test_process_explicit_targets_reports_resolved_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process --target should resolve only the named .rsk files."""
    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files", lambda *a: None
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    (rsk_dir / "a.rsk").write_text("", encoding="utf-8")
    (rsk_dir / "b.rsk").write_text("", encoding="utf-8")
    (rsk_dir / "c.txt").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path)
        + ["--target", "a.rsk", "--target", "b.rsk"],
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.stdout
    assert "a.rsk" in result.stdout
    assert "b.rsk" in result.stdout
    assert "c.txt" not in result.stdout


def test_process_auto_discovers_targets_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process with no --target auto-discovers top-level .rsk files."""
    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files", lambda *a: None
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    (rsk_dir / "b.rsk").write_text("", encoding="utf-8")
    (rsk_dir / "a.rsk").write_text("", encoding="utf-8")
    (rsk_dir / "c.txt").write_text("", encoding="utf-8")
    nested_dir = rsk_dir / "sub"
    nested_dir.mkdir()
    (nested_dir / "nested.rsk").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "a.rsk" in result.stdout
    assert "b.rsk" in result.stdout
    assert "nested.rsk" not in result.stdout
    assert result.stdout.index("a.rsk") < result.stdout.index("b.rsk")


def test_process_set_unknown_key_errors(tmp_path: Path) -> None:
    """--set with an unknown key exits non-zero with a clean message."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path)
        + ["--set", "not_a_real_option=1"],
    )

    assert result.exit_code == 1
    assert "not yet implemented" not in result.stdout
    assert result.stderr != ""


def test_process_missing_rsk_directory_errors(tmp_path: Path) -> None:
    """A nonexistent rsk_directory should exit 1 with a clean message."""
    missing_dir = tmp_path / "does-not-exist"

    result = runner.invoke(
        app,
        [
            "process",
            "--set",
            f'paths.rsk_directory="{missing_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "not yet implemented" not in result.stdout
    assert str(missing_dir) in result.stderr
