"""Tests for ctd_processing.cli."""

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctd_processing.cli import app

runner = CliRunner()


def _other_paths(tmp_path: Path) -> list[str]:
    """CLI args for the required profiles_directory/binned_directory."""
    return [
        "--set",
        f'paths.profiles_directory="{(tmp_path / "profiles").as_posix()}"',
        "--set",
        f'paths.binned_directory="{(tmp_path / "binned").as_posix()}"',
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


def test_bin_stub_reports_not_implemented(tmp_path: Path) -> None:
    """Bin stub should report 'not yet implemented' and exit 1."""
    input_path = tmp_path / "cast.rsk"
    input_path.write_text("", encoding="utf-8")
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "bin",
            str(input_path),
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.stdout


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
def test_stub_command_missing_input_path_errors(
    tmp_path: Path, command: str
) -> None:
    """A nonexistent input_path fails validation before the stub body runs."""
    result = runner.invoke(app, [command, str(tmp_path / "does-not-exist")])

    assert result.exit_code != 0


@pytest.mark.parametrize("command", ["bin", "concatenate"])
def test_stub_command_set_unknown_key_errors(
    tmp_path: Path, command: str
) -> None:
    """--set with an unknown key exits non-zero with a clean message."""
    input_path = tmp_path / "cast.rsk"
    input_path.write_text("", encoding="utf-8")

    result = runner.invoke(
        app, [command, str(input_path), "--set", "not_a_real_option=1"]
    )

    assert result.exit_code == 1
    assert "not yet implemented" not in result.stdout
    assert result.stderr != ""


def test_process_explicit_targets_reports_resolved_files(
    tmp_path: Path,
) -> None:
    """Process --target should resolve only the named .rsk files."""
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
    tmp_path: Path,
) -> None:
    """Process with no --target auto-discovers top-level .rsk files."""
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
