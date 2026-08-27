"""Tests for ctd_processing.process.save."""

from pathlib import Path

import numpy as np
import pytest

import ctd_processing.process.save as save_module
from ctd_processing.config import GeolocationSettings
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.profiles import Profile
from ctd_processing.process.save import profile_filename, save_profiles

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)


def _dataset(n: int = 10) -> Dataset:
    """Build a small Dataset with one float channel and deployment metadata."""
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
            "instrument_serial_number": 208532,
        }
    )
    dataset.add_channel(
        "sea_water_temperature",
        Channel(
            data=np.linspace(10.0, 12.0, n),
            metadata={"units": "degree_C", "long_name": "Temperature"},
        ),
    )
    return dataset


def test_profile_filename_shape() -> None:
    """profile_filename builds the expected serial_stem_pIDX_start.ext shape."""
    dataset = _dataset()
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    filename = profile_filename(dataset, profile_dataset, 0, 1, "parquet")

    assert (
        filename == "208532_243188_20260809_0304_p000_20260809T030412.parquet"
    )


def test_profile_filename_differs_by_index() -> None:
    """Two profiles from the same dataset get different filenames."""
    dataset = _dataset()
    first = dataset.subset(slice(0, 5), "first")
    second = dataset.subset(slice(5, 10), "second")

    name0 = profile_filename(dataset, first, 0, 2, "nc")
    name1 = profile_filename(dataset, second, 1, 2, "nc")

    assert name0 != name1
    assert name0.endswith(".nc")


def test_profile_filename_pads_index_to_total_width() -> None:
    """Index padding widens beyond 3 digits when total requires it."""
    dataset = _dataset()
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    filename = profile_filename(dataset, profile_dataset, 7, 1234, "parquet")

    assert "_p0007_" in filename


def _stub_writers(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace write_netcdf/write_parquet with recording stubs."""
    calls: dict = {"netcdf": [], "parquet": []}

    def fake_write_netcdf(dataset: Dataset, path: Path) -> Path:
        calls["netcdf"].append((dataset, path))
        return path

    def fake_write_parquet(dataset: Dataset, path: Path) -> Path:
        calls["parquet"].append((dataset, path))
        return path

    monkeypatch.setattr(save_module, "write_netcdf", fake_write_netcdf)
    monkeypatch.setattr(save_module, "write_parquet", fake_write_parquet)
    return calls


def _profiles(n: int) -> list[Profile]:
    """Two disjoint profiles spanning the first and second halves of `n`."""
    half = n // 2
    return [
        Profile(down_start=0, down_end=half, up_start=half, up_end=half),
        Profile(down_start=half, down_end=n, up_start=n, up_end=n),
    ]


def test_save_profiles_dispatches_to_parquet_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format="parquet" calls write_parquet, not write_netcdf."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)

    paths = save_profiles(dataset, profiles, tmp_path, "parquet", _GEOLOCATION)

    assert len(paths) == 2
    assert len(calls["parquet"]) == 2
    assert not calls["netcdf"]
    assert all(path.suffix == ".parquet" for path in paths)


def test_save_profiles_dispatches_to_netcdf_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format="netcdf" calls write_netcdf, not write_parquet."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)

    paths = save_profiles(dataset, profiles, tmp_path, "netcdf", _GEOLOCATION)

    assert len(paths) == 2
    assert len(calls["netcdf"]) == 2
    assert not calls["parquet"]
    assert all(path.suffix == ".nc" for path in paths)


def test_save_profiles_extracts_expected_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each profile's extracted Dataset spans down_start:up_end of `dataset`."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)

    save_profiles(dataset, profiles, tmp_path, "parquet", _GEOLOCATION)

    first_dataset, _ = calls["parquet"][0]
    second_dataset, _ = calls["parquet"][1]
    assert first_dataset.length == 5
    assert second_dataset.length == 5
    np.testing.assert_array_equal(
        first_dataset.time.data, dataset.time.data[0:5]
    )
    np.testing.assert_array_equal(
        second_dataset.time.data, dataset.time.data[5:10]
    )


def test_save_profiles_creates_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_profiles creates `directory` if it doesn't already exist."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)
    directory = tmp_path / "profiles" / "nested"

    save_profiles(dataset, profiles, directory, "parquet", _GEOLOCATION)

    assert directory.is_dir()


def test_save_profiles_returns_paths_under_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every returned path is inside `directory`."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)

    paths = save_profiles(dataset, profiles, tmp_path, "parquet", _GEOLOCATION)

    assert all(path.parent == tmp_path for path in paths)


def _stub_attach_geolocation(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace attach_geolocation with a call-recording, pass-through stub."""
    calls: list = []

    def fake_attach_geolocation(dataset, settings, external_dataset):
        calls.append((dataset, settings, external_dataset))
        return dataset

    monkeypatch.setattr(
        save_module, "attach_geolocation", fake_attach_geolocation
    )
    return calls


def test_save_profiles_calls_attach_geolocation_per_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """attach_geolocation is called once per profile, after subset."""
    _stub_writers(monkeypatch)
    calls = _stub_attach_geolocation(monkeypatch)
    dataset = _dataset(10)
    profiles = _profiles(10)

    save_profiles(dataset, profiles, tmp_path, "parquet", _GEOLOCATION)

    assert len(calls) == 2
    for profile_dataset, settings, external_dataset in calls:
        assert profile_dataset.length == 5
        assert settings is _GEOLOCATION
        assert external_dataset is None


def test_save_profiles_opens_external_dataset_once_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external dataset is opened once per call, not once per profile."""
    _stub_writers(monkeypatch)
    _stub_attach_geolocation(monkeypatch)
    open_calls: list = []
    close_calls: list = []

    class _FakeExternalDataset:
        def close(self) -> None:
            close_calls.append(1)

    def fake_open_dataset(path):
        open_calls.append(path)
        return _FakeExternalDataset()

    monkeypatch.setattr(save_module.xr, "open_dataset", fake_open_dataset)
    dataset = _dataset(10)
    profiles = _profiles(10)
    geolocation = GeolocationSettings(external_dataset_path="gps.nc")

    save_profiles(dataset, profiles, tmp_path, "parquet", geolocation)

    assert len(open_calls) == 1
    assert len(close_calls) == 1


def test_save_profiles_does_not_open_external_dataset_for_reference_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No external dataset is opened when a reference position is configured."""
    _stub_writers(monkeypatch)
    _stub_attach_geolocation(monkeypatch)
    open_calls: list = []
    monkeypatch.setattr(
        save_module.xr,
        "open_dataset",
        lambda path: open_calls.append(path),
    )
    dataset = _dataset(10)
    profiles = _profiles(10)

    save_profiles(dataset, profiles, tmp_path, "parquet", _GEOLOCATION)

    assert open_calls == []
