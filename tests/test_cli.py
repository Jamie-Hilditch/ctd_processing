"""Tests for ctd_processing.cli."""

from importlib import resources
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctd_processing.cli import app

runner = CliRunner()

BUNDLED_TEMPLATE = (
    resources.files("ctd_processing.cli.templates")
    .joinpath("config.toml")
    .read_text(encoding="utf-8")
)


def test_help_lists_all_commands() -> None:
    """--help should list all four top-level commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("init", "process", "bin", "concatenate"):
        assert command in result.stdout


def test_init_writes_bundled_template_by_default(tmp_path: Path) -> None:
    """Init with no options writes the bundled default template verbatim."""
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "config.toml").read_text(
        encoding="utf-8"
    ) == BUNDLED_TEMPLATE


def test_init_with_custom_template(tmp_path: Path) -> None:
    """Init --template uses the supplied file instead of the bundled default."""
    template_path = tmp_path / "custom.toml"
    template_path.write_text("# a custom template\n", encoding="utf-8")
    destination = tmp_path / "out"

    result = runner.invoke(
        app, ["init", str(destination), "--template", str(template_path)]
    )

    assert result.exit_code == 0
    assert (destination / "config.toml").read_text(
        encoding="utf-8"
    ) == "# a custom template\n"


def test_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """Init should not overwrite an existing config.toml without --force."""
    (tmp_path / "config.toml").write_text("sentinel\n", encoding="utf-8")

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code != 0
    assert (tmp_path / "config.toml").read_text(
        encoding="utf-8"
    ) == "sentinel\n"


def test_init_force_overwrites(tmp_path: Path) -> None:
    """Init --force should overwrite an existing config.toml."""
    (tmp_path / "config.toml").write_text("sentinel\n", encoding="utf-8")

    result = runner.invoke(app, ["init", str(tmp_path), "--force"])

    assert result.exit_code == 0
    assert (tmp_path / "config.toml").read_text(
        encoding="utf-8"
    ) == BUNDLED_TEMPLATE


def test_init_set_malformed_pair_errors(tmp_path: Path) -> None:
    """Init --set with a malformed pair should exit non-zero with a message."""
    result = runner.invoke(app, ["init", str(tmp_path), "--set", "not-a-pair"])

    assert result.exit_code != 0
    assert "expected key=value" in result.stderr
    assert not (tmp_path / "config.toml").exists()


def test_init_set_unknown_key_errors(tmp_path: Path) -> None:
    """Init --set with an unknown key exits non-zero (no fields exist yet)."""
    result = runner.invoke(
        app, ["init", str(tmp_path), "--set", "not_a_real_option=1"]
    )

    assert result.exit_code != 0
    assert not (tmp_path / "config.toml").exists()


@pytest.mark.parametrize("command", ["process", "bin"])
def test_stub_file_command_reports_not_implemented(
    tmp_path: Path, command: str
) -> None:
    """process/bin stubs should report 'not yet implemented' and exit 1."""
    input_path = tmp_path / "cast.rsk"
    input_path.write_text("", encoding="utf-8")

    result = runner.invoke(app, [command, str(input_path)])

    assert result.exit_code == 1
    assert "not yet implemented" in result.stdout


def test_concatenate_stub_reports_not_implemented(tmp_path: Path) -> None:
    """Concatenate stub should report 'not yet implemented' and exit 1."""
    result = runner.invoke(app, ["concatenate", str(tmp_path)])

    assert result.exit_code == 1
    assert "not yet implemented" in result.stdout


@pytest.mark.parametrize("command", ["process", "bin", "concatenate"])
def test_stub_command_missing_input_path_errors(
    tmp_path: Path, command: str
) -> None:
    """A nonexistent input_path fails validation before the stub body runs."""
    result = runner.invoke(app, [command, str(tmp_path / "does-not-exist")])

    assert result.exit_code != 0


@pytest.mark.parametrize("command", ["process", "bin", "concatenate"])
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
