"""Tests for ctd_processing.process.save_netcdf."""

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ctd_processing.config import (
    ChannelSettings,
    GeolocationSettings,
    NetcdfCompressionSettings,
    ProcessSettings,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_netcdf import (
    netcdf_compression_encoding,
    read_netcdf,
    write_netcdf,
)

_PROCESS_SETTINGS = ProcessSettings(
    geolocation=GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )
)


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
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        np.testing.assert_array_equal(
            ds["sea_water_temperature"].to_numpy(),
            dataset.channels["sea_water_temperature"].data,
            strict=False,
        )


def test_write_netcdf_round_trips_time_exactly(tmp_path: Path) -> None:
    """The time coordinate round-trips to the exact same datetime64 values."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

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
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

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
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

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
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.attrs["history"] == "; ".join(dataset.history)
        assert "zero-order hold" not in ds.attrs["history"]


def test_write_netcdf_global_attrs_drop_none_and_format_datetimes(
    tmp_path: Path,
) -> None:
    """None-valued metadata is dropped; datetime values become ISO strings."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert "deployment_comment" not in ds.attrs
        assert ds.attrs["deployment_start_time"] == "2026-08-09T03:00:00"
        assert ds.attrs["instrument_serial_number"] == 208532


def test_write_netcdf_casts_float_channels_to_default_dtype(
    tmp_path: Path,
) -> None:
    """Float channels are written in the project-wide default dtype."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["sea_water_temperature"].dtype == np.float32
        assert ds["sea_pressure"].dtype == np.float32


def test_write_netcdf_honors_per_channel_output_dtype_override(
    tmp_path: Path,
) -> None:
    """A channel's own output_dtype overrides the project-wide default."""
    settings = ProcessSettings(
        geolocation=GeolocationSettings(
            reference_latitude=0.0, reference_longitude=0.0
        ),
        channels={"sea_pressure": ChannelSettings(output_dtype="float64")},
    )
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", settings)

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["sea_water_temperature"].dtype == np.float32
        assert ds["sea_pressure"].dtype == np.float64


def test_write_netcdf_compresses_float_variables(tmp_path: Path) -> None:
    """Float data variables are written with zlib compression applied."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        assert f["sea_water_temperature"]._h5ds.compression == "gzip"


def test_write_netcdf_honors_custom_complevel_and_shuffle(
    tmp_path: Path,
) -> None:
    """A custom complevel/shuffle in netcdf_compression takes effect."""
    settings = ProcessSettings(
        geolocation=GeolocationSettings(
            reference_latitude=0.0, reference_longitude=0.0
        ),
        netcdf_compression=NetcdfCompressionSettings(
            complevel=9, shuffle=False
        ),
    )
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", settings)

    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        h5ds = f["sea_water_temperature"]._h5ds
        assert h5ds.compression_opts == 9
        assert h5ds.shuffle is False


def test_write_netcdf_disabling_compression_leaves_variables_uncompressed(
    tmp_path: Path,
) -> None:
    """enabled=False writes fully uncompressed netCDF files."""
    settings = ProcessSettings(
        geolocation=GeolocationSettings(
            reference_latitude=0.0, reference_longitude=0.0
        ),
        netcdf_compression=NetcdfCompressionSettings(enabled=False),
    )
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", settings)

    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        assert f["sea_water_temperature"]._h5ds.compression is None


def test_netcdf_compression_encoding_returns_none_for_non_float_dtype() -> None:
    """Integer/datetime dtypes are never compressed regardless of settings."""
    assert (
        netcdf_compression_encoding(
            np.dtype("int64"), NetcdfCompressionSettings()
        )
        is None
    )


def test_netcdf_compression_encoding_returns_none_when_disabled() -> None:
    """enabled=False returns None even for a floating-point dtype."""
    assert (
        netcdf_compression_encoding(
            np.dtype("float32"), NetcdfCompressionSettings(enabled=False)
        )
        is None
    )


def test_netcdf_compression_encoding_returns_expected_encoding() -> None:
    """A float dtype with enabled=True returns the h5netcdf encoding."""
    encoding = netcdf_compression_encoding(
        np.dtype("float32"),
        NetcdfCompressionSettings(complevel=7, shuffle=False),
    )
    assert encoding == {"zlib": True, "complevel": 7, "shuffle": False}


def test_write_netcdf_creates_missing_parent_directory(tmp_path: Path) -> None:
    """write_netcdf creates path.parent if it doesn't already exist."""
    dataset = _dataset()
    path = write_netcdf(
        dataset, tmp_path / "nested" / "profile.nc", _PROCESS_SETTINGS
    )

    assert path.exists()


def test_read_netcdf_round_trips_channel_data_and_metadata(
    tmp_path: Path,
) -> None:
    """A channel's data, metadata, and history round-trip via read_netcdf."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    loaded = read_netcdf(path)

    temperature = loaded.channels["sea_water_temperature"]
    np.testing.assert_array_equal(
        temperature.data,
        dataset.channels["sea_water_temperature"].data,
        strict=False,
    )
    assert (
        temperature.metadata
        == dataset.channels["sea_water_temperature"].metadata
    )
    assert (
        temperature.history == dataset.channels["sea_water_temperature"].history
    )

    pressure = loaded.channels["sea_pressure"]
    assert pressure.metadata == dataset.channels["sea_pressure"].metadata
    assert pressure.history == []


def test_read_netcdf_round_trips_time_and_adds_cf_defaults(
    tmp_path: Path,
) -> None:
    """Time round-trips exactly, carrying the CF defaults write_netcdf adds."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    loaded = read_netcdf(path)

    assert np.array_equal(
        loaded.time.data.astype("datetime64[ms]"), dataset.time.data
    )
    assert loaded.time.metadata["standard_name"] == "time"
    assert loaded.time.metadata["axis"] == "T"
    assert loaded.time.history == []


def test_read_netcdf_round_trips_dataset_metadata_and_history(
    tmp_path: Path,
) -> None:
    """Global attrs and history decode back to dataset.metadata/history."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    loaded = read_netcdf(path)

    assert loaded.metadata["source_file"] == "example.rsk"
    assert loaded.metadata["instrument_serial_number"] == 208532
    assert loaded.metadata["deployment_start_time"] == "2026-08-09T03:00:00"
    assert "deployment_comment" not in loaded.metadata
    assert loaded.history == dataset.history


def test_read_netcdf_does_not_inject_extra_history(tmp_path: Path) -> None:
    """Loading channels bypasses add_channel, so history isn't padded out."""
    dataset = _dataset()
    path = write_netcdf(dataset, tmp_path / "profile.nc", _PROCESS_SETTINGS)

    loaded = read_netcdf(path)

    assert loaded.history == dataset.history
    assert list(loaded.channels) == list(dataset.channels)
