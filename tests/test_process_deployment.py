"""Tests for ctd_processing.process."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import ctd_processing.process as process_module
from ctd_processing.config import (
    DeploymentSettings,
    InstrumentSettings,
    PathsSettings,
    ProcessSettings,
    Settings,
)
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process import process_deployment


def _settings(
    tmp_path: Path,
    instruments: dict[str, InstrumentSettings] | None = None,
    deployments: dict[str, DeploymentSettings] | None = None,
) -> Settings:
    """Build a minimal Settings for process_deployment tests."""
    return Settings(
        paths=PathsSettings(
            rsk_directory=tmp_path / "rsk",
            profiles_directory=tmp_path / "profiles",
            binned_directory=tmp_path / "binned",
        ),
        process=ProcessSettings(),
        instruments=instruments or {},
        deployments=deployments or {},
    )


@pytest.mark.requires_example_data
def test_process_deployment_reads_and_returns_none(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """process_deployment reads the deployment (step 1) and returns None.

    Profile extraction is not yet implemented, so a real, readable
    ``.rsk`` deployment currently produces no further effect.
    """
    result = process_deployment(
        example_rsk_path, tmp_path / "profiles", _settings(tmp_path)
    )

    assert result is None


@pytest.mark.requires_example_data
def test_process_deployment_applies_raw_channel_processing(
    tmp_path: Path,
    example_rsk_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment's dataset has had remove_holds applied to it.

    The example file's temperature channel has known zero-order holds
    (verified in earlier work), so a VERBOSE record from `remove_holds`
    appearing here proves the full read_rsk -> build_dataset ->
    process_raw_channels chain actually ran.
    """
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    process_deployment(
        example_rsk_path, tmp_path / "profiles", _settings(tmp_path)
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("zero-order hold value(s)" in m for m in messages)


@pytest.mark.requires_example_data
def test_process_deployment_reuses_logged_sea_pressure(
    tmp_path: Path,
    example_rsk_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment's compute_sea_pressure step runs against real data.

    The example file already logs a sea_pressure channel directly
    (verified via `build_dataset`), and default settings leave
    `atmospheric_pressure` unset, so `compute_sea_pressure` should trust
    that channel as-is rather than recompute it -- proving that step ran
    against real data.
    """
    caplog.set_level(logging.INFO, logger="ctd_processing.process.sea_pressure")

    process_deployment(
        example_rsk_path, tmp_path / "profiles", _settings(tmp_path)
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("sea_pressure channel already present" in m for m in messages)


@pytest.mark.requires_example_data
def test_process_deployment_identifies_profiles(
    tmp_path: Path,
    example_rsk_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment logs the number of profiles it identified."""
    caplog.set_level(logging.INFO, logger="ctd_processing.process")

    process_deployment(
        example_rsk_path, tmp_path / "profiles", _settings(tmp_path)
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Identified" in m and "profile(s)" in m for m in messages)


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, serial_number: str) -> dict:
    """Stub read_rsk/build_dataset/... so process_deployment needs no real .rsk.

    Returns a dict that `captured["process_settings"]` is filled with once
    `process_deployment` calls the (stubbed) `process_raw_channels`, so
    tests can inspect exactly which resolved `ProcessSettings` reached it.
    """
    captured: dict = {}

    monkeypatch.setattr(process_module, "read_rsk", lambda file: object())
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project: SimpleNamespace(
            metadata={"instrument_serial_number": serial_number}
        ),
    )

    def fake_process_raw_channels(dataset, settings):
        captured["process_settings"] = settings
        return dataset

    monkeypatch.setattr(
        process_module, "process_raw_channels", fake_process_raw_channels
    )
    monkeypatch.setattr(
        process_module,
        "compute_sea_pressure",
        lambda dataset, atmospheric_pressure: dataset,
    )
    monkeypatch.setattr(
        process_module, "find_profiles", lambda dataset, settings: []
    )
    monkeypatch.setattr(
        process_module,
        "process_ct_lag",
        lambda dataset, profiles, settings: dataset,
    )

    return captured


def test_process_deployment_applies_instrument_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An [instruments.<serial>] override reaches raw-channel processing."""
    captured = _stub_pipeline(monkeypatch, serial_number="208532")
    settings = _settings(
        tmp_path,
        instruments={
            "208532": InstrumentSettings(process={"atmospheric_pressure": 10.1})
        },
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert captured["process_settings"].atmospheric_pressure == 10.1


def test_process_deployment_deployment_override_wins_over_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching deployment override wins over an instrument override."""
    captured = _stub_pipeline(monkeypatch, serial_number="208532")
    settings = _settings(
        tmp_path,
        instruments={
            "208532": InstrumentSettings(process={"atmospheric_pressure": 10.1})
        },
        deployments={
            "243188_20260809_0304": DeploymentSettings(
                process={"atmospheric_pressure": 10.5}
            )
        },
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert captured["process_settings"].atmospheric_pressure == 10.5


def test_process_deployment_applies_ct_lag_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [process.ct_lag] override reaches the resolved ProcessSettings."""
    captured = _stub_pipeline(monkeypatch, serial_number="208532")
    settings = _settings(
        tmp_path,
        instruments={
            "208532": InstrumentSettings(process={"ct_lag": {"enabled": True}})
        },
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert captured["process_settings"].ct_lag.enabled is True
