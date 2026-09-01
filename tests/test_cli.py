"""Tests for ctd_processing.cli."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from ctd_processing.bin.save import save_binned_dataset
from ctd_processing.cli import app
from ctd_processing.config import (
    BinSettings,
    GeolocationSettings,
    ProcessSettings,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save import profile_filename
from ctd_processing.process.save_netcdf import write_netcdf

runner = CliRunner()

_PROCESS_SETTINGS = ProcessSettings(
    geolocation=GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )
)


def _write_profile(
    directory: Path,
    index: int,
    *,
    start: str,
    serial: int = 208532,
    source_file: str = "243188_20260809_0304.rsk",
) -> Path:
    """Write one small, already-processed profile file into `directory`.

    Unlike `ctd_processing.process.save.save_profile`, this does not nest
    the file under a deployment-stem subdirectory -- these bin CLI tests
    exercise `bin`'s own handling of a profile directory, independent of
    where `process` happens to write its output.
    """
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
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / profile_filename(dataset, index, "nc")
    write_netcdf(dataset, path, _PROCESS_SETTINGS)
    return path


def _write_binned(directory: Path, stem: str, times: list[str]) -> Path:
    """Write a minimal binned dataset, for concatenate CLI tests."""
    n = len(times)
    dataset = xr.Dataset(
        {"temperature": (("profile", "z"), np.zeros((n, 2)))},
        coords={
            "z": ("z", [-0.5, -1.5]),
            "time": ("profile", np.array(times, dtype="datetime64[s]")),
            "latitude": ("profile", np.full(n, 45.0)),
            "longitude": ("profile", np.full(n, -125.0)),
        },
    )
    return save_binned_dataset(dataset, directory, f"{stem}.nc", BinSettings())


def _other_paths(tmp_path: Path) -> list[str]:
    """CLI args for --config plus profiles/binned directories and geolocation.

    An empty ``config.toml`` is written into `tmp_path` and pointed to
    via ``--config`` since that option now defaults to ``config.toml``
    and must exist. `geolocation` is included here (not just the two
    required `paths` fields) since `[process.geolocation]` is itself
    required -- see `ctd_processing.config.GeolocationSettings`.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    return [
        "--config",
        str(config_path),
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


def test_bin_target_bins_one_named_deployment(tmp_path: Path) -> None:
    """--target names one deployment stem, binned from its subdirectory."""
    profiles_dir = tmp_path / "profiles"
    deployment_dir = profiles_dir / "243188_20260809_0304"
    _write_profile(deployment_dir, 0, start="2026-08-09T03:04:00")
    _write_profile(deployment_dir, 1, start="2026-08-09T04:04:00")
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--target",
            "243188_20260809_0304",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "binned" / "243188_20260809_0304.nc").is_file()
    assert "2 profile(s)" in result.stdout


def test_bin_no_target_bins_every_deployment(tmp_path: Path) -> None:
    """Omitting --target bins every deployment subdirectory found."""
    profiles_dir = tmp_path / "profiles"
    _write_profile(
        profiles_dir / "243188_20260809_0304", 0, start="2026-08-09T03:04:00"
    )
    _write_profile(
        profiles_dir / "999999_20260810_0000",
        0,
        start="2026-08-10T00:00:00",
        serial=999999,
        source_file="999999_20260810_0000.rsk",
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 0, result.stderr
    written = sorted(path.name for path in (tmp_path / "binned").glob("*.nc"))
    assert written == ["243188_20260809_0304.nc", "999999_20260810_0000.nc"]


def test_bin_unknown_target_errors(tmp_path: Path) -> None:
    """A --target naming a nonexistent deployment exits 1 with a message."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--target",
            "does-not-exist",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "does not exist" in result.stderr


def test_bin_empty_deployment_directory_errors(tmp_path: Path) -> None:
    """An empty deployment subdirectory exits 1 with a clean message."""
    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "243188_20260809_0304").mkdir(parents=True)
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--target",
            "243188_20260809_0304",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "No profile files found" in result.stderr


def test_bin_mixed_deployment_input_errors(tmp_path: Path) -> None:
    """Profiles from different deployments in one subdirectory error cleanly.

    Continues on to bin any other resolved deployment rather than
    aborting the whole run.
    """
    profiles_dir = tmp_path / "profiles"
    mixed_dir = profiles_dir / "mixed"
    _write_profile(mixed_dir, 0, start="2026-08-09T03:04:00")
    _write_profile(
        mixed_dir,
        1,
        start="2026-08-09T04:04:00",
        serial=999999,
        source_file="other_deployment.rsk",
    )
    good_dir = profiles_dir / "243188_20260809_0304"
    _write_profile(good_dir, 0, start="2026-08-09T03:04:00")
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "multiple deployments" in result.stderr
    assert (tmp_path / "binned" / "243188_20260809_0304.nc").is_file()


def _concatenated_paths(tmp_path: Path) -> list[str]:
    """`_other_paths` plus rsk_directory/concatenated_file, for concatenate."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir(exist_ok=True)
    concatenated_file = tmp_path / "concatenated.nc"
    return _other_paths(tmp_path) + [
        "--set",
        f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        "--set",
        f'paths.concatenated_file="{concatenated_file.as_posix()}"',
    ]


def test_concatenate_target_concatenates_named_deployments(
    tmp_path: Path,
) -> None:
    """--target names which deployments to concatenate, in time order."""
    binned_dir = tmp_path / "binned"
    _write_binned(binned_dir, "b", ["2026-08-10T00:00:00"])
    _write_binned(
        binned_dir, "a", ["2026-08-09T03:04:00", "2026-08-09T04:04:00"]
    )

    result = runner.invoke(
        app,
        ["concatenate", "--target", "a", "--target", "b"]
        + _concatenated_paths(tmp_path),
    )

    assert result.exit_code == 0, result.stderr
    output = tmp_path / "concatenated.nc"
    assert output.is_file()
    with xr.open_dataset(output) as combined:
        assert combined.sizes["profile"] == 3
        times = combined["time"].values
        assert list(times) == sorted(times)
    assert "3 profile(s)" in result.stdout


def test_concatenate_no_target_concatenates_every_deployment(
    tmp_path: Path,
) -> None:
    """Omitting --target concatenates every deployment in binned_directory."""
    binned_dir = tmp_path / "binned"
    _write_binned(binned_dir, "a", ["2026-08-09T03:04:00"])
    _write_binned(binned_dir, "b", ["2026-08-10T00:00:00"])

    result = runner.invoke(app, ["concatenate"] + _concatenated_paths(tmp_path))

    assert result.exit_code == 0, result.stderr
    with xr.open_dataset(tmp_path / "concatenated.nc") as combined:
        assert combined.sizes["profile"] == 2


def test_concatenate_removes_duplicate_profiles_and_sorts_by_time(
    tmp_path: Path,
) -> None:
    """A profile repeated in a later deployment (unwiped memory) is dropped.

    Simulates forgetting to wipe an instrument's memory between
    deployments: the second deployment's binned file repeats the first
    deployment's last profile time in addition to a genuinely new one.
    """
    binned_dir = tmp_path / "binned"
    _write_binned(
        binned_dir, "first", ["2026-08-09T03:04:00", "2026-08-09T04:04:00"]
    )
    _write_binned(
        binned_dir, "second", ["2026-08-09T04:04:00", "2026-08-10T00:00:00"]
    )

    result = runner.invoke(app, ["concatenate"] + _concatenated_paths(tmp_path))

    assert result.exit_code == 0, result.stderr
    with xr.open_dataset(tmp_path / "concatenated.nc") as combined:
        assert combined.sizes["profile"] == 3
        times = list(combined["time"].values)
        assert times == sorted(times)
        assert len(set(times)) == 3


def test_concatenate_missing_concatenated_file_setting_errors(
    tmp_path: Path,
) -> None:
    """paths.concatenated_file left unset exits 1 with a clean message."""
    binned_dir = tmp_path / "binned"
    binned_dir.mkdir()
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "concatenate",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "concatenated_file" in result.stderr


def test_concatenate_unknown_target_errors(tmp_path: Path) -> None:
    """A --target naming a nonexistent deployment exits 1 with a message."""
    (tmp_path / "binned").mkdir()

    result = runner.invoke(
        app,
        ["concatenate", "--target", "does-not-exist"]
        + _concatenated_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "does not exist" in result.stderr


def test_concatenate_set_unknown_key_errors(tmp_path: Path) -> None:
    """--set with an unknown key exits non-zero with a clean message."""
    (tmp_path / "binned").mkdir()

    result = runner.invoke(
        app,
        ["concatenate"]
        + _concatenated_paths(tmp_path)
        + ["--set", "not_a_real_option=1"],
    )

    assert result.exit_code == 1
    assert result.stderr != ""


def test_bin_set_unknown_key_errors(tmp_path: Path) -> None:
    """--set with an unknown key exits non-zero with a clean message."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path)
        + ["--set", "not_a_real_option=1"],
    )

    assert result.exit_code == 1
    assert result.stderr != ""


def test_process_explicit_targets_reports_resolved_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process --target should dispatch only the named .rsk files."""
    calls: list[list[Path]] = []
    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files",
        lambda deployment_files, *a: calls.append(deployment_files),
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

    assert result.exit_code == 0
    assert "Processed 2 deployment(s)" in result.stdout
    assert [path.name for path in calls[0]] == ["a.rsk", "b.rsk"]


def test_process_auto_discovers_targets_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process with no --target auto-discovers top-level .rsk files."""
    calls: list[list[Path]] = []
    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files",
        lambda deployment_files, *a: calls.append(deployment_files),
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

    assert result.exit_code == 0
    assert "Processed 2 deployment(s)" in result.stdout
    assert [path.name for path in calls[0]] == ["a.rsk", "b.rsk"]


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
    assert result.stderr != ""


def test_process_uses_default_config_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --config, a config.toml present in the cwd is loaded."""
    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files", lambda *a: None
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    (rsk_dir / "deployment.rsk").write_text("", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'rsk_directory = "{rsk_dir.as_posix()}"\n'
        f'profiles_directory = "{(tmp_path / "profiles").as_posix()}"\n'
        f'binned_directory = "{(tmp_path / "binned").as_posix()}"\n'
        "[process.geolocation]\n"
        "reference_latitude = 0.0\n"
        "reference_longitude = 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["process"])

    assert result.exit_code == 0
    assert "Processed 1 deployment(s)" in result.stdout


def test_process_reports_failed_deployments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed deployment is reported on stderr with a non-zero exit."""

    def fake_process_deployment_files(
        deployment_files, profiles_directory, settings
    ):
        error = ValueError("boom")
        error.add_note("while processing deployment: bad.rsk")
        raise ExceptionGroup("Failed to process 1 of 1 deployment(s).", [error])

    monkeypatch.setattr(
        "ctd_processing.cli.process.process_deployment_files",
        fake_process_deployment_files,
    )
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    (rsk_dir / "bad.rsk").write_text("", encoding="utf-8")

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
    assert "boom" in result.stderr
    assert "while processing deployment: bad.rsk" in result.stderr
    assert "Processed" not in result.stdout


def test_process_missing_default_config_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --config, a missing config.toml in the cwd exits non-zero."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["process"])

    assert result.exit_code != 0
    assert "config.toml" in result.stderr


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
    assert str(missing_dir) in result.stderr
