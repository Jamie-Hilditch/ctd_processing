"""Tests for ctd_processing.process.save."""

from pathlib import Path

import numpy as np
import pytest

import ctd_processing.process.save as save_module
from ctd_processing.config import GeolocationSettings, ProcessSettings
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save import (
    load_profile,
    profile_filename,
    save_profile,
)

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)
_PARQUET_SETTINGS = ProcessSettings(
    geolocation=_GEOLOCATION, profile_format="parquet"
)
_NETCDF_SETTINGS = ProcessSettings(
    geolocation=_GEOLOCATION, profile_format="netcdf"
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
    """profile_filename builds the expected stem_pIDX shape."""
    dataset = _dataset()

    filename = profile_filename(dataset, 0, "parquet")

    assert filename == "243188_20260809_0304_p0000.parquet"


def test_profile_filename_differs_by_index() -> None:
    """Two profiles from the same dataset get different filenames."""
    dataset = _dataset()

    name0 = profile_filename(dataset, 0, "nc")
    name1 = profile_filename(dataset, 1, "nc")

    assert name0 != name1
    assert name0.endswith(".nc")


def test_profile_filename_pads_index_to_four_digits() -> None:
    """Index padding is always (at least) four digits, regardless of total."""
    dataset = _dataset()

    filename = profile_filename(dataset, 7, "parquet")

    assert filename == "243188_20260809_0304_p0007.parquet"


def _stub_writers(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace write_netcdf/write_parquet with recording stubs."""
    calls: dict = {"netcdf": [], "parquet": []}

    def fake_write_netcdf(
        dataset: Dataset, path: Path, process_settings: ProcessSettings
    ) -> Path:
        calls["netcdf"].append((dataset, path, process_settings))
        return path

    def fake_write_parquet(
        dataset: Dataset, path: Path, process_settings: ProcessSettings
    ) -> Path:
        calls["parquet"].append((dataset, path, process_settings))
        return path

    monkeypatch.setattr(save_module, "write_netcdf", fake_write_netcdf)
    monkeypatch.setattr(save_module, "write_parquet", fake_write_parquet)
    return calls


def test_save_profile_dispatches_to_parquet_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """profile_format="parquet" calls write_parquet, not write_netcdf."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(
        dataset, profile_dataset, 0, tmp_path, _PARQUET_SETTINGS
    )

    assert len(calls["parquet"]) == 1
    assert not calls["netcdf"]
    assert path.suffix == ".parquet"


def test_save_profile_dispatches_to_netcdf_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """profile_format="netcdf" calls write_netcdf, not write_parquet."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(dataset, profile_dataset, 0, tmp_path, _NETCDF_SETTINGS)

    assert len(calls["netcdf"]) == 1
    assert not calls["parquet"]
    assert path.suffix == ".nc"


def test_save_profile_passes_profile_dataset_to_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer is called with the given profile_dataset, unmodified."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    save_profile(dataset, profile_dataset, 0, tmp_path, _PARQUET_SETTINGS)

    written_dataset, _, _ = calls["parquet"][0]
    assert written_dataset is profile_dataset


def test_save_profile_creates_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_profile creates the deployment subdirectory if it doesn't exist."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")
    directory = tmp_path / "profiles" / "nested"

    path = save_profile(
        dataset, profile_dataset, 0, directory, _PARQUET_SETTINGS
    )

    assert path.parent.is_dir()


def test_save_profile_writes_into_deployment_stem_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile lands under directory/<deployment_stem>/, not directory."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(
        dataset, profile_dataset, 0, tmp_path, _PARQUET_SETTINGS
    )

    assert path.parent == tmp_path / "243188_20260809_0304"
    assert path.name == "243188_20260809_0304_p0000.parquet"


def test_load_profile_dispatches_to_read_netcdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .nc path is loaded via read_netcdf, not read_parquet."""
    sentinel = _dataset()
    calls: list[Path] = []
    monkeypatch.setattr(
        save_module,
        "read_netcdf",
        lambda path: (calls.append(path), sentinel)[1],
    )

    path = tmp_path / "profile.nc"
    result = load_profile(path)

    assert result is sentinel
    assert calls == [path]


def test_load_profile_dispatches_to_read_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .parquet path is loaded via read_parquet, not read_netcdf."""
    sentinel = _dataset()
    calls: list[Path] = []
    monkeypatch.setattr(
        save_module,
        "read_parquet",
        lambda path: (calls.append(path), sentinel)[1],
    )

    path = tmp_path / "profile.parquet"
    result = load_profile(path)

    assert result is sentinel
    assert calls == [path]


def test_load_profile_rejects_unrecognized_extension(tmp_path: Path) -> None:
    """An unrecognized extension raises ValueError rather than guessing."""
    path = tmp_path / "profile.csv"

    with pytest.raises(ValueError, match=r"\.csv"):
        load_profile(path)
