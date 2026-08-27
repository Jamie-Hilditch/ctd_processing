"""Tests for ctd_processing.process.save_netcdf."""

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_netcdf import write_netcdf


def _dataset() -> Dataset:
    """Build a small Dataset covering CF attrs, history, and NaN handling."""
    time = Channel(
        data=np.array(
            [
                "2026-08-09T03:04:12",
                "2026-08-09T03:04:13",
                "2026-08-09T03:04:14",
            ],
            dtype="datetime64[ms]",
        )
    )
    dataset = Dataset(time=time)
    dataset.record("read from example.rsk")
    dataset.metadata.update(
        {
            "source_file": "example.rsk",
            "instrument_serial_number": 208532,
            "deployment_comment": None,
            "deployment_start_time": datetime.datetime(2026, 8, 9, 3, 0, 0),
        }
    )
    dataset.add_channel(
        "sea_water_temperature",
        Channel(
            data=np.array([10.0, np.nan, 12.0]),
            metadata={
                "units": "degree_C",
                "long_name": "Sea water temperature",
                "standard_name": "sea_water_temperature",
            },
            history=["removed 1 zero-order hold value(s)"],
        ),
    )
    dataset.add_channel(
        "sea_pressure",
        Channel(
            data=np.array([0.1, 5.0, 10.0]),
            metadata={"units": "dbar", "long_name": "Sea pressure"},
        ),
    )
    return dataset


def test_write_netcdf_round_trips_float_data_with_nan(tmp_path: Path) -> None:
    """Float channel data, including NaN, round-trips exactly."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        np.testing.assert_array_equal(
            ds["sea_water_temperature"].to_numpy(),
            dataset.channels["sea_water_temperature"].data,
            strict=False,
        )


def test_write_netcdf_round_trips_time_exactly(tmp_path: Path) -> None:
    """The time coordinate round-trips to the exact same datetime64 values."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert np.array_equal(
            ds["time"].to_numpy().astype("datetime64[ms]"), dataset.time.data
        )
        assert ds["time"].attrs["standard_name"] == "time"
        assert ds["time"].attrs["axis"] == "T"


def test_write_netcdf_variable_attrs_match_channel_metadata(
    tmp_path: Path,
) -> None:
    """Each variable's CF attrs match its source Channel.metadata."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        temp_attrs = ds["sea_water_temperature"].attrs
        assert temp_attrs["units"] == "degree_C"
        assert temp_attrs["long_name"] == "Sea water temperature"
        assert temp_attrs["standard_name"] == "sea_water_temperature"

        pressure_attrs = ds["sea_pressure"].attrs
        assert pressure_attrs["units"] == "dbar"
        assert "standard_name" not in pressure_attrs


def test_write_netcdf_attaches_channel_history_to_its_own_variable(
    tmp_path: Path,
) -> None:
    """A channel's history becomes an attribute on that channel's variable."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert (
            ds["sea_water_temperature"].attrs["history"]
            == "removed 1 zero-order hold value(s)"
        )
        assert "history" not in ds["sea_pressure"].attrs


def test_write_netcdf_global_history_is_dataset_level_only(
    tmp_path: Path,
) -> None:
    """The global history attr holds dataset.history, not channel history."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.attrs["history"] == "; ".join(dataset.history)
        assert "zero-order hold" not in ds.attrs["history"]


def test_write_netcdf_global_attrs_drop_none_and_format_datetimes(
    tmp_path: Path,
) -> None:
    """None-valued metadata is dropped; datetime values become ISO strings."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert "deployment_comment" not in ds.attrs
        assert ds.attrs["deployment_start_time"] == "2026-08-09T03:00:00"
        assert ds.attrs["instrument_serial_number"] == 208532


def test_write_netcdf_compresses_float_variables(tmp_path: Path) -> None:
    """Float data variables are written with zlib compression applied."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc")

    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        assert f["sea_water_temperature"]._h5ds.compression == "gzip"


def test_write_netcdf_creates_missing_parent_directory(tmp_path: Path) -> None:
    """write_netcdf creates path.parent if it doesn't already exist."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "nested" / "profile.nc")

    assert path.exists()
