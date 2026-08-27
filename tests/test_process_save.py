"""Tests for ctd_processing.process.save."""

from pathlib import Path

import numpy as np
import pytest

import ctd_processing.process.save as save_module
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save import (
    load_profile,
    profile_filename,
    save_profile,
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


def test_save_profile_dispatches_to_parquet_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format="parquet" calls write_parquet, not write_netcdf."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(dataset, profile_dataset, 0, 2, tmp_path, "parquet")

    assert len(calls["parquet"]) == 1
    assert not calls["netcdf"]
    assert path.suffix == ".parquet"


def test_save_profile_dispatches_to_netcdf_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format="netcdf" calls write_netcdf, not write_parquet."""
    calls = _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(dataset, profile_dataset, 0, 2, tmp_path, "netcdf")

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

    save_profile(dataset, profile_dataset, 0, 2, tmp_path, "parquet")

    written_dataset, _ = calls["parquet"][0]
    assert written_dataset is profile_dataset


def test_save_profile_creates_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_profile creates `directory` if it doesn't already exist."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")
    directory = tmp_path / "profiles" / "nested"

    save_profile(dataset, profile_dataset, 0, 1, directory, "parquet")

    assert directory.is_dir()


def test_save_profile_returns_path_under_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned path is inside `directory`."""
    _stub_writers(monkeypatch)
    dataset = _dataset(10)
    profile_dataset = dataset.subset(slice(0, 5), "test subset")

    path = save_profile(dataset, profile_dataset, 0, 1, tmp_path, "parquet")

    assert path.parent == tmp_path


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
