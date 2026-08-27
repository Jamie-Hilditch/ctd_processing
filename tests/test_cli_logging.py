"""Tests for ctd_processing.cli._logging and the CLI's logging options."""

import logging
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctd_processing.cli import app
from ctd_processing.cli._logging import (
    PACKAGE_LOGGER_NAME,
    _MaxLevelFilter,
    configure_file_logging,
    configure_stdout_logging,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_logging():
    """Ensure no handlers leak between tests, regardless of outcome."""
    yield
    configure_stdout_logging(level=logging.INFO, enable_stdout=False)
    configure_file_logging(None, None)


def _other_paths(tmp_path: Path) -> list[str]:
    """CLI args for the required profiles_directory/binned_directory."""
    return [
        "--set",
        f'paths.profiles_directory="{(tmp_path / "profiles").as_posix()}"',
        "--set",
        f'paths.binned_directory="{(tmp_path / "binned").as_posix()}"',
    ]


def test_max_level_filter_rejects_at_and_above_max() -> None:
    """_MaxLevelFilter admits records strictly below max_level."""
    filter_ = _MaxLevelFilter(logging.ERROR)
    below = logging.LogRecord("x", logging.WARNING, "", 0, "msg", None, None)
    at = logging.LogRecord("x", logging.ERROR, "", 0, "msg", None, None)

    assert filter_.filter(below) is True
    assert filter_.filter(at) is False


def test_max_level_filter_repr_and_str() -> None:
    """repr()/str() summarize max_level without a bare object-id repr."""
    filter_ = _MaxLevelFilter(logging.ERROR)

    assert repr(filter_) == "_MaxLevelFilter(max_level=40)"
    assert str(filter_) == "reject records at or above ERROR"


def test_configure_stdout_logging_writes_to_stdout(
    capsys: pytest.CaptureFixture,
) -> None:
    """A record at/above the configured level is written to stdout."""
    configure_stdout_logging(level=logging.INFO, enable_stdout=True)
    logging.getLogger(f"{PACKAGE_LOGGER_NAME}.somemodule").info("hello")

    assert "hello" in capsys.readouterr().out


def test_configure_stdout_logging_disabled_writes_nothing(
    capsys: pytest.CaptureFixture,
) -> None:
    """enable_stdout=False attaches no stdout handler."""
    configure_stdout_logging(level=logging.INFO, enable_stdout=False)
    logging.getLogger(f"{PACKAGE_LOGGER_NAME}.somemodule").info("hello")

    assert capsys.readouterr().out == ""


def test_configure_stdout_logging_does_not_accumulate_handlers() -> None:
    """Repeated calls replace, rather than add to, the stdout handler."""
    for _ in range(3):
        configure_stdout_logging(level=logging.INFO, enable_stdout=True)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    stdout_handlers = [
        h for h in logger.handlers if h.name == "ctd_processing.stdout"
    ]
    assert len(stdout_handlers) == 1


def test_configure_file_logging_splits_by_level(tmp_path: Path) -> None:
    """Below-ERROR records land in log_file; ERROR+ land in error_log_file."""
    log_file = tmp_path / "ctd.log"
    error_log_file = tmp_path / "ctd.error.log"
    configure_file_logging(log_file, error_log_file)

    logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.somemodule")
    logger.setLevel(logging.DEBUG)
    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(logging.DEBUG)
    logger.info("routine message")
    logger.error("boom")

    log_contents = log_file.read_text(encoding="utf-8")
    error_contents = error_log_file.read_text(encoding="utf-8")
    assert "routine message" in log_contents
    assert "boom" not in log_contents
    assert "boom" in error_contents
    assert "routine message" not in error_contents


def test_configure_file_logging_none_removes_previous_handlers(
    tmp_path: Path,
) -> None:
    """Calling with both None tears down previously-attached file handlers."""
    log_file = tmp_path / "ctd.log"
    configure_file_logging(log_file, None)
    configure_file_logging(None, None)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    file_handler_names = {
        "ctd_processing.log_file",
        "ctd_processing.error_log_file",
    }
    assert not any(h.name in file_handler_names for h in logger.handlers)


@pytest.mark.requires_example_data
def test_cli_log_level_controls_emitted_records(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--log-level DEBUG surfaces process's DEBUG-level read_rsk log line."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--log-level",
            "DEBUG",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "Opening RSK file" in result.stdout


@pytest.mark.requires_example_data
def test_cli_no_stdout_log_silences_log_records(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--no-stdout-log suppresses log output but not the stub's own message."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--no-stdout-log",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "Reading deployment" not in result.stdout
    assert "not yet implemented" in result.stdout


@pytest.mark.requires_example_data
def test_cli_verbose_shows_verbose_level_messages(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--verbose surfaces build_dataset's VERBOSE-level history logs."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--verbose",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "added channel" in result.stdout
    assert "Opening RSK file" not in result.stdout


@pytest.mark.requires_example_data
def test_cli_default_level_hides_verbose_messages(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """Without --verbose/--debug, VERBOSE-level history logs are hidden."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "added channel" not in result.stdout


@pytest.mark.requires_example_data
def test_cli_debug_implies_verbose(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--debug shows both DEBUG and VERBOSE-level messages."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--debug",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "Opening RSK file" in result.stdout
    assert "added channel" in result.stdout


@pytest.mark.requires_example_data
def test_cli_debug_overrides_verbose_when_both_given(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--verbose --debug together behave like --debug alone."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--verbose",
            "--debug",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "Opening RSK file" in result.stdout


@pytest.mark.requires_example_data
def test_cli_debug_overrides_explicit_log_level(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--debug wins even over an explicit, less verbose --log-level."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--log-level",
            "ERROR",
            "--debug",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "Opening RSK file" in result.stdout


@pytest.mark.requires_example_data
def test_cli_log_level_verbose_choice(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """--log-level VERBOSE shows VERBOSE messages but not DEBUG ones."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    shutil.copy(example_rsk_path, rsk_dir / "deployment.rsk")

    result = runner.invoke(
        app,
        [
            "--log-level",
            "VERBOSE",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert "added channel" in result.stdout
    assert "Opening RSK file" not in result.stdout


def test_cli_process_writes_split_log_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """paths.log_file/error_log_file receive non-overlapping records."""
    import ctd_processing.process as process_module

    def failing_process_deployment(file, profiles_directory, settings):
        process_module.logger.info("about to fail")
        raise ValueError("boom")

    monkeypatch.setattr(
        process_module, "process_deployment", failing_process_deployment
    )

    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    (rsk_dir / "deployment.rsk").write_text("", encoding="utf-8")
    log_file = tmp_path / "ctd.log"
    error_log_file = tmp_path / "ctd.error.log"

    result = runner.invoke(
        app,
        [
            "--log-level",
            "DEBUG",
            "process",
            "--set",
            f'paths.rsk_directory="{rsk_dir.as_posix()}"',
            "--set",
            f'paths.log_file="{log_file.as_posix()}"',
            "--set",
            f'paths.error_log_file="{error_log_file.as_posix()}"',
        ]
        + _other_paths(tmp_path),
    )

    assert result.exit_code != 0
    log_contents = log_file.read_text(encoding="utf-8")
    error_contents = error_log_file.read_text(encoding="utf-8")
    assert "about to fail" in log_contents
    assert "boom" not in log_contents
    assert "Failed to process deployment" in error_contents
    assert "ValueError: boom" in error_contents
