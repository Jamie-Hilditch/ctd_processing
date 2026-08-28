"""Tests for ctd_processing.bin.save."""

from pathlib import Path

import numpy as np
import xarray as xr

from ctd_processing.bin.save import binned_filename, save_binned_dataset
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset


def test_binned_filename_format() -> None:
    """Filename combines serial number and deployment stem, with suffix."""
    time = Channel(data=np.array(["2026-01-01"], dtype="datetime64[s]"))
    dataset = Dataset(
        time=time,
        metadata={
            "instrument_serial_number": 208532,
            "source_file": "/data/rsk/243188_20260809_0304.rsk",
        },
    )

    assert (
        binned_filename(dataset, "nc")
        == "208532_243188_20260809_0304_binned.nc"
    )
    assert (
        binned_filename(dataset, "zarr")
        == "208532_243188_20260809_0304_binned.zarr"
    )


def _combined_dataset() -> xr.Dataset:
    """Build a tiny, already-combined binned dataset to round-trip."""
    return xr.Dataset(
        {"sea_water_temperature": (("profile", "z"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={
            "z": ("z", [-0.5, -1.5], {"units": "m", "standard_name": "height"}),
            "time": (
                "profile",
                np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[s]"),
            ),
            "latitude": ("profile", [45.0, 45.0]),
            "longitude": ("profile", [-125.0, -125.0]),
        },
        attrs={"instrument_serial_number": 208532},
    )


def test_save_binned_dataset_netcdf_round_trips(tmp_path: Path) -> None:
    """output_format='netcdf' writes a file xarray can read back."""
    dataset = _combined_dataset()

    path = save_binned_dataset(dataset, tmp_path, "out.nc", "netcdf")

    assert path == tmp_path / "out.nc"
    with xr.open_dataset(path) as reopened:
        np.testing.assert_array_equal(
            reopened["sea_water_temperature"].values,
            dataset["sea_water_temperature"].values,
        )
        assert reopened.attrs["instrument_serial_number"] == 208532


def test_save_binned_dataset_zarr_round_trips(tmp_path: Path) -> None:
    """output_format='zarr' writes a store xarray can read back."""
    dataset = _combined_dataset()

    path = save_binned_dataset(dataset, tmp_path, "out.zarr", "zarr")

    assert path == tmp_path / "out.zarr"
    with xr.open_zarr(path) as reopened:
        np.testing.assert_array_equal(
            reopened["sea_water_temperature"].values,
            dataset["sea_water_temperature"].values,
        )


def test_save_binned_dataset_creates_missing_directory(tmp_path: Path) -> None:
    """A missing target directory is created."""
    dataset = _combined_dataset()
    directory = tmp_path / "nested" / "binned"

    save_binned_dataset(dataset, directory, "out.nc", "netcdf")

    assert directory.is_dir()
    assert (directory / "out.nc").is_file()
