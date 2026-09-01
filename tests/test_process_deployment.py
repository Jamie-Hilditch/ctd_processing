"""Tests for ctd_processing.process."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr

import ctd_processing.process as process_module
from ctd_processing.config import (
    ChannelSettings,
    DeploymentSettings,
    DerivedVariablesSettings,
    DespikeChannelOverride,
    GeolocationSettings,
    InstrumentSettings,
    PathsSettings,
    ProcessSettings,
    Settings,
)
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process import process_deployment, process_profile
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.profiles import Profile
from ctd_processing.process.save import load_profile

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)


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
        process=ProcessSettings(geolocation=_GEOLOCATION),
        instruments=instruments or {},
        deployments=deployments or {},
    )


@pytest.mark.requires_example_data
def test_process_deployment_reads_and_returns_none(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """process_deployment reads and processes the deployment, returning None."""
    result = process_deployment(
        example_rsk_path, tmp_path / "profiles", _settings(tmp_path)
    )

    assert result is None


@pytest.mark.requires_example_data
@pytest.mark.parametrize(
    ("profile_format", "extension"),
    [("parquet", ".parquet"), ("netcdf", ".nc")],
)
def test_process_deployment_writes_one_file_per_profile(
    tmp_path: Path,
    example_rsk_path: Path,
    caplog: pytest.LogCaptureFixture,
    profile_format: Literal["netcdf", "parquet"],
    extension: str,
) -> None:
    """Every profile identified is written to profiles_directory, in format."""
    caplog.set_level(logging.INFO, logger="ctd_processing.process")
    profiles_directory = tmp_path / "profiles"
    settings = _settings(tmp_path)
    settings.process.profile_format = profile_format

    process_deployment(example_rsk_path, profiles_directory, settings)

    messages = [record.getMessage() for record in caplog.records]
    identified = next(
        int(m.split()[1])
        for m in messages
        if "Identified" in m and "profile(s)" in m
    )
    written_files = list(profiles_directory.rglob(f"*{extension}"))
    assert identified > 0
    assert len(written_files) == identified
    deployment_directory = profiles_directory / example_rsk_path.stem
    assert all(f.parent == deployment_directory for f in written_files)

    if profile_format == "parquet":
        metadata = json.loads(
            pq.read_schema(written_files[0]).metadata[
                b"ctd_processing.dataset_metadata"
            ]
        )
    else:
        with xr.open_dataset(written_files[0], engine="h5netcdf") as ds:
            metadata = dict(ds.attrs)
    assert metadata["latitude"] == 0.0
    assert metadata["longitude"] == 0.0
    assert metadata["position_source"] == "reference position"


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


def _count_identified_and_written(
    caplog: pytest.LogCaptureFixture, profiles_directory: Path
) -> tuple[int, int]:
    """Parse the logged "Identified N profile(s)" count; count written files."""
    messages = [record.getMessage() for record in caplog.records]
    identified = next(
        int(m.split()[1])
        for m in messages
        if "Identified" in m and "profile(s)" in m
    )
    written = len(list(profiles_directory.rglob("*.parquet")))
    return identified, written


@pytest.mark.requires_example_data
def test_process_deployment_handles_fluorometer_instrument(
    tmp_path: Path,
    example_rsk_path_fluorometer: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment handles a second instrument with no sea_pressure.

    This instrument has no onboard sea_pressure channel, so
    atmospheric_pressure must be configured for compute_sea_pressure to
    derive one from absolute_pressure -- otherwise the pipeline would
    raise. How many profiles (if any) this deployment actually contains
    is not assumed ahead of time.
    """
    caplog.set_level(logging.INFO, logger="ctd_processing.process")
    profiles_directory = tmp_path / "profiles"
    settings = _settings(tmp_path)
    settings.process.atmospheric_pressure = 10.1325

    process_deployment(
        example_rsk_path_fluorometer, profiles_directory, settings
    )

    identified, written = _count_identified_and_written(
        caplog, profiles_directory
    )
    assert written == identified


@pytest.mark.requires_example_data
def test_process_deployment_handles_oxygen_instrument(
    tmp_path: Path,
    example_rsk_path_oxygen: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment handles a third instrument, carrying real oxygen data.

    This instrument also has no onboard sea_pressure channel, so
    atmospheric_pressure must be configured. How many profiles (if any)
    this deployment actually contains is not assumed ahead of time.
    """
    caplog.set_level(logging.INFO, logger="ctd_processing.process")
    profiles_directory = tmp_path / "profiles"
    settings = _settings(tmp_path)
    settings.process.atmospheric_pressure = 10.1325

    process_deployment(example_rsk_path_oxygen, profiles_directory, settings)

    identified, written = _count_identified_and_written(
        caplog, profiles_directory
    )
    assert written == identified


@pytest.mark.requires_example_data
def test_process_deployment_computes_oxygen_concentration_from_saturation(
    tmp_path: Path,
    example_rsk_path_oxygen: Path,
) -> None:
    """derived_variables.oxygen_concentration derives concentration end-to-end.

    Reads a written profile back and confirms the real
    dissolved_oxygen_saturation channel on this instrument was used to
    derive an oxygen_concentration_from_saturation channel.
    """
    profiles_directory = tmp_path / "profiles"
    settings = _settings(tmp_path)
    settings.process.atmospheric_pressure = 10.1325
    settings.process.derived_variables = DerivedVariablesSettings(
        oxygen_concentration=True
    )

    process_deployment(example_rsk_path_oxygen, profiles_directory, settings)

    written_files = list(profiles_directory.rglob("*.parquet"))
    assert written_files
    profile_dataset = load_profile(written_files[0])

    oxygen = profile_dataset.channels["oxygen_concentration_from_saturation"]
    assert (
        oxygen.metadata["standard_name"]
        == "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
    )
    assert np.isfinite(oxygen.data).any()


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, serial_number: str) -> dict:
    """Stub read_rsk/build_dataset/... so process_deployment needs no real .rsk.

    Returns a dict that `captured["process_settings"]` is filled with once
    `process_deployment` calls the (stubbed) `process_raw_channels`, so
    tests can inspect exactly which resolved `ProcessSettings` reached it.
    `find_profiles` returns an empty list by default, so the profile loop
    itself never executes -- tests that need it to should override
    `find_profiles`/`build_dataset` after calling this.
    """
    captured: dict = {}

    fake_rsk = SimpleNamespace(
        instrument=SimpleNamespace(serialID=serial_number)
    )
    monkeypatch.setattr(process_module, "read_rsk", lambda file: fake_rsk)
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project, read_channels=None: SimpleNamespace(
            metadata={"instrument_serial_number": serial_number}
        ),
    )

    def fake_process_raw_channels(dataset, settings, despike=None):
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
        lambda dataset, spans, settings: dataset,
    )

    def fake_process_profile(
        dataset, geolocation, external_dataset, derived_variables, despike=None
    ):
        return dataset

    monkeypatch.setattr(process_module, "process_profile", fake_process_profile)
    monkeypatch.setattr(
        process_module,
        "save_profile",
        lambda dataset, profile_dataset, index, directory, format: (
            directory / f"profile_{index}.{format}"
        ),
    )

    return captured


def _real_deployment_dataset(n: int = 10) -> Dataset:
    """Build a real, subset-able Dataset for the loop tests below."""
    time = Channel(
        data=(
            np.datetime64("2026-08-09T03:04:12", "ms")
            + np.arange(n) * np.timedelta64(1, "s")
        )
    )
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {
            "source_file": "/data/rsk/243188_20260809_0304.rsk",
            "instrument_serial_number": "208532",
        }
    )
    return dataset


def _two_profiles(n: int) -> list[Profile]:
    """Two disjoint profiles spanning the first and second halves of `n`."""
    half = n // 2
    return [
        Profile(down_start=0, down_end=half, up_start=half, up_end=half),
        Profile(down_start=half, down_end=n, up_start=n, up_end=n),
    ]


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


def test_process_deployment_applies_derived_variables_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [process.derived_variables] override reaches ProcessSettings."""
    captured = _stub_pipeline(monkeypatch, serial_number="208532")
    settings = _settings(
        tmp_path,
        instruments={
            "208532": InstrumentSettings(
                process={"derived_variables": {"sound_speed": True}}
            )
        },
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert captured["process_settings"].derived_variables.sound_speed is True


def test_process_deployment_applies_despike_channel_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [process.channels.<name>.despiking] override reaches settings."""
    captured = _stub_pipeline(monkeypatch, serial_number="208532")
    settings = _settings(
        tmp_path,
        instruments={
            "208532": InstrumentSettings(
                process={
                    "channels": {
                        "practical_salinity": {
                            "despike": True,
                            "despiking": {"threshold": 3.0},
                        }
                    }
                }
            )
        },
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert captured["process_settings"].channels == {
        "practical_salinity": ChannelSettings(
            despike=True,
            despiking=DespikeChannelOverride(threshold=3.0),
        )
    }


def test_process_deployment_calls_process_profile_and_save_profile_per_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop extracts, processes, then saves each identified profile."""
    _stub_pipeline(monkeypatch, serial_number="208532")
    dataset = _real_deployment_dataset(10)
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project, read_channels=None: dataset,
    )
    profiles = _two_profiles(10)
    monkeypatch.setattr(
        process_module, "find_profiles", lambda dataset, settings: profiles
    )

    process_profile_calls: list[Dataset] = []
    save_profile_calls: list[tuple] = []

    def fake_process_profile(
        pd, geolocation, external_dataset, derived_variables, despike=None
    ):
        process_profile_calls.append(pd)
        return pd

    monkeypatch.setattr(process_module, "process_profile", fake_process_profile)
    monkeypatch.setattr(
        process_module,
        "save_profile",
        lambda deployment_dataset, pd, index, directory, format: (
            save_profile_calls.append((pd, index)),
            directory / f"p{index}.{format}",
        )[1],
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        _settings(tmp_path),
    )

    assert [pd.length for pd in process_profile_calls] == [5, 5]
    assert [index for (_, index) in save_profile_calls] == [0, 1]


def test_process_deployment_direction_both_splits_into_two_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """direction="both" writes the downcast and upcast as two profiles.

    Uses a profile with a real (non-empty) dwell between down_end and
    up_start, and asserts neither extracted profile is as long as the
    full down_start:up_end span would be -- proving the dwell itself was
    excluded, not just that two profiles were produced.
    """
    _stub_pipeline(monkeypatch, serial_number="208532")
    dataset = _real_deployment_dataset(12)
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project, read_channels=None: dataset,
    )
    profile = Profile(down_start=0, down_end=4, up_start=6, up_end=10)
    monkeypatch.setattr(
        process_module, "find_profiles", lambda dataset, settings: [profile]
    )

    process_profile_calls: list[Dataset] = []

    def fake_process_profile(
        pd, geolocation, external_dataset, derived_variables, despike=None
    ):
        process_profile_calls.append(pd)
        return pd

    monkeypatch.setattr(process_module, "process_profile", fake_process_profile)
    monkeypatch.setattr(
        process_module,
        "save_profile",
        lambda deployment_dataset, pd, index, directory, format: (
            directory / f"p{index}.{format}"
        ),
    )

    settings = _settings(tmp_path)
    settings.process.profiles.direction = "both"

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert [pd.length for pd in process_profile_calls] == [4, 4]


def test_process_deployment_passes_resolved_spans_to_ct_lag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_ct_lag receives the resolved output spans, not the full cycle.

    With direction="down" (the default), the dwell between down_end and
    up_start, and the upcast itself, must be excluded from the span
    handed to process_ct_lag -- it should only ever see the same data
    that will actually be written out as a profile.
    """
    _stub_pipeline(monkeypatch, serial_number="208532")
    dataset = _real_deployment_dataset(12)
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project, read_channels=None: dataset,
    )
    profile = Profile(down_start=0, down_end=4, up_start=6, up_end=10)
    monkeypatch.setattr(
        process_module, "find_profiles", lambda dataset, settings: [profile]
    )

    ct_lag_calls: list[list[slice]] = []
    monkeypatch.setattr(
        process_module,
        "process_ct_lag",
        lambda dataset, spans, settings: (
            ct_lag_calls.append(spans),
            dataset,
        )[1],
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        _settings(tmp_path),
    )

    assert ct_lag_calls == [[slice(0, 4)]]


def test_process_deployment_opens_external_geolocation_dataset_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external geolocation dataset is opened once per deployment."""
    _stub_pipeline(monkeypatch, serial_number="208532")
    dataset = _real_deployment_dataset(10)
    monkeypatch.setattr(
        process_module,
        "build_dataset",
        lambda rsk, file, project, read_channels=None: dataset,
    )
    monkeypatch.setattr(
        process_module,
        "find_profiles",
        lambda dataset, settings: _two_profiles(10),
    )

    def fake_process_profile(
        pd, geolocation, external_dataset, derived_variables, despike=None
    ):
        return pd

    monkeypatch.setattr(process_module, "process_profile", fake_process_profile)
    monkeypatch.setattr(
        process_module,
        "save_profile",
        lambda deployment_dataset, pd, index, directory, format: (
            directory / f"p{index}"
        ),
    )

    open_calls: list = []
    close_calls: list = []

    class _FakeExternalDataset:
        def close(self) -> None:
            close_calls.append(1)

    def fake_open_dataset(path):
        open_calls.append(path)
        return _FakeExternalDataset()

    monkeypatch.setattr(process_module.xr, "open_dataset", fake_open_dataset)

    settings = _settings(tmp_path)
    settings.process.geolocation = GeolocationSettings(
        external_dataset_path="gps.nc"
    )

    process_deployment(
        tmp_path / "rsk" / "243188_20260809_0304.rsk",
        tmp_path / "profiles",
        settings,
    )

    assert len(open_calls) == 1
    assert len(close_calls) == 1


def test_process_profile_attaches_geolocation_before_derived_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_profile geolocates before computing derived variables."""
    calls: list[str] = []

    def fake_attach_geolocation(dataset, settings, external_dataset):
        calls.append("attach_geolocation")
        return dataset

    def fake_compute_derived_variables(dataset, settings, despike=None):
        calls.append("compute_derived_variables")
        return dataset

    monkeypatch.setattr(
        process_module, "attach_geolocation", fake_attach_geolocation
    )
    monkeypatch.setattr(
        process_module,
        "compute_derived_variables",
        fake_compute_derived_variables,
    )

    sentinel = _real_deployment_dataset(1)
    result = process_profile(
        sentinel, _GEOLOCATION, None, DerivedVariablesSettings()
    )

    assert calls == ["attach_geolocation", "compute_derived_variables"]
    assert result is sentinel
