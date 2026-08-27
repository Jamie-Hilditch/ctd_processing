"""Tests for ctd_processing.process.process_deployment_files."""

import threading
from pathlib import Path

import pytest

import ctd_processing.process as process_module
from ctd_processing.config import (
    GeolocationSettings,
    PathsSettings,
    ProcessSettings,
    Settings,
)
from ctd_processing.process import process_deployment_files

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        paths=PathsSettings(
            rsk_directory=tmp_path / "rsk",
            profiles_directory=tmp_path / "profiles",
            binned_directory=tmp_path / "binned",
        ),
        process=ProcessSettings(geolocation=_GEOLOCATION),
    )


def test_process_deployment_files_copies_and_dispatches_each_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each deployment is copied (keeping its filename) and dispatched."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    a = _touch(rsk_dir / "a.rsk")
    b = _touch(rsk_dir / "b.rsk")

    calls: list[Path] = []
    lock = threading.Lock()

    def fake_process_deployment(file, profiles_directory, settings):
        with lock:
            calls.append(file)

    monkeypatch.setattr(
        process_module, "process_deployment", fake_process_deployment
    )

    process_deployment_files([a, b], tmp_path / "profiles", _settings(tmp_path))

    assert {path.name for path in calls} == {"a.rsk", "b.rsk"}
    for path in calls:
        assert path != a
        assert path != b
        assert not path.is_relative_to(rsk_dir)
        assert not path.exists()  # temp dir is cleaned up afterwards


def test_process_deployment_files_continues_after_one_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure on one deployment doesn't stop the rest; errors collect."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    good = _touch(rsk_dir / "good.rsk")
    bad = _touch(rsk_dir / "bad.rsk")

    calls: list[Path] = []
    lock = threading.Lock()

    def fake_process_deployment(file, profiles_directory, settings):
        with lock:
            calls.append(file)
        if file.name == "bad.rsk":
            raise ValueError("boom")

    monkeypatch.setattr(
        process_module, "process_deployment", fake_process_deployment
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        process_deployment_files(
            [good, bad], tmp_path / "profiles", _settings(tmp_path)
        )

    assert {path.name for path in calls} == {"good.rsk", "bad.rsk"}
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)
    assert str(bad) in "".join(exc_info.value.exceptions[0].__notes__)

    [failure_record] = [
        record for record in caplog.records if record.levelname == "ERROR"
    ]
    assert "Failed to process deployment" in failure_record.getMessage()
    assert failure_record.exc_info is not None
