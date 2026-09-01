"""Tests for ctd_processing.bin.save."""

from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BytesCodec
from zarr.core.array import Array
from zarr.core.metadata import ArrayV3Metadata

from ctd_processing.bin.save import load_binned_dataset, save_binned_dataset
from ctd_processing.config import (
    BinSettings,
    NetcdfCompressionSettings,
    ZarrCompressionSettings,
)


def _codecs(path: Path, variable: str) -> tuple[object, ...]:
    """Read the written zarr codecs for one array."""
    group = zarr.open_group(path, mode="r")
    array = group[variable]
    assert isinstance(array, Array)
    metadata = array.metadata
    assert isinstance(metadata, ArrayV3Metadata)
    return metadata.codecs


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

    path = save_binned_dataset(dataset, tmp_path, "out.nc", BinSettings())

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

    path = save_binned_dataset(
        dataset, tmp_path, "out.zarr", BinSettings(output_format="zarr")
    )

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

    save_binned_dataset(dataset, directory, "out.nc", BinSettings())

    assert directory.is_dir()
    assert (directory / "out.nc").is_file()


def test_save_binned_dataset_netcdf_compresses_float_variables(
    tmp_path: Path,
) -> None:
    """The default settings zlib-compress float variables, not coordinates."""
    dataset = _combined_dataset()

    path = save_binned_dataset(dataset, tmp_path, "out.nc", BinSettings())

    with h5netcdf.File(path, "r") as f:
        assert f["sea_water_temperature"]._h5ds.compression == "gzip"
        assert f["z"]._h5ds.compression is None


def test_save_binned_dataset_netcdf_honors_custom_complevel_and_shuffle(
    tmp_path: Path,
) -> None:
    """A custom complevel/shuffle in netcdf_compression takes effect."""
    dataset = _combined_dataset()
    settings = BinSettings(
        netcdf_compression=NetcdfCompressionSettings(complevel=9, shuffle=False)
    )

    path = save_binned_dataset(dataset, tmp_path, "out.nc", settings)

    with h5netcdf.File(path, "r") as f:
        h5ds = f["sea_water_temperature"]._h5ds
        assert h5ds.compression_opts == 9
        assert h5ds.shuffle is False


def test_save_binned_dataset_netcdf_disabling_compression_leaves_uncompressed(
    tmp_path: Path,
) -> None:
    """enabled=False writes fully uncompressed netCDF variables."""
    dataset = _combined_dataset()
    settings = BinSettings(
        netcdf_compression=NetcdfCompressionSettings(enabled=False)
    )

    path = save_binned_dataset(dataset, tmp_path, "out.nc", settings)

    with h5netcdf.File(path, "r") as f:
        assert f["sea_water_temperature"]._h5ds.compression is None


def test_save_binned_dataset_zarr_uses_explicit_default_compressor(
    tmp_path: Path,
) -> None:
    """The default settings attach an explicit Blosc/zstd codec, not zarr's."""
    dataset = _combined_dataset()

    path = save_binned_dataset(
        dataset, tmp_path, "out.zarr", BinSettings(output_format="zarr")
    )

    codecs = _codecs(path, "sea_water_temperature")
    blosc_codecs = [codec for codec in codecs if isinstance(codec, BloscCodec)]
    assert len(blosc_codecs) == 1
    assert blosc_codecs[0].cname == "zstd"


def test_save_binned_dataset_zarr_honors_custom_cname_and_clevel(
    tmp_path: Path,
) -> None:
    """A custom cname/clevel in BinSettings.zarr_compression takes effect."""
    dataset = _combined_dataset()
    settings = BinSettings(
        output_format="zarr",
        zarr_compression=ZarrCompressionSettings(cname="lz4", clevel=1),
    )

    path = save_binned_dataset(dataset, tmp_path, "out.zarr", settings)

    codecs = _codecs(path, "sea_water_temperature")
    blosc_codecs = [codec for codec in codecs if isinstance(codec, BloscCodec)]
    assert len(blosc_codecs) == 1
    assert blosc_codecs[0].cname == "lz4"
    assert blosc_codecs[0].clevel == 1


def test_save_binned_dataset_zarr_disabling_compression_uses_bytes_codec_only(
    tmp_path: Path,
) -> None:
    """enabled=False writes with no Blosc codec at all."""
    dataset = _combined_dataset()
    settings = BinSettings(
        output_format="zarr",
        zarr_compression=ZarrCompressionSettings(enabled=False),
    )

    path = save_binned_dataset(dataset, tmp_path, "out.zarr", settings)

    codecs = _codecs(path, "sea_water_temperature")
    assert not any(isinstance(codec, BloscCodec) for codec in codecs)
    assert any(isinstance(codec, BytesCodec) for codec in codecs)


def test_load_binned_dataset_netcdf_round_trips(tmp_path: Path) -> None:
    """A netcdf-written dataset loads back with matching data and attrs."""
    dataset = _combined_dataset()
    path = save_binned_dataset(dataset, tmp_path, "out.nc", BinSettings())

    loaded = load_binned_dataset(path, "netcdf")

    np.testing.assert_array_equal(
        loaded["sea_water_temperature"].values,
        dataset["sea_water_temperature"].values,
    )
    assert loaded.attrs["instrument_serial_number"] == 208532


def test_load_binned_dataset_zarr_round_trips(tmp_path: Path) -> None:
    """A zarr-written dataset loads back with matching data."""
    dataset = _combined_dataset()
    path = save_binned_dataset(
        dataset, tmp_path, "out.zarr", BinSettings(output_format="zarr")
    )

    loaded = load_binned_dataset(path, "zarr")

    np.testing.assert_array_equal(
        loaded["sea_water_temperature"].values,
        dataset["sea_water_temperature"].values,
    )


def test_load_binned_dataset_does_not_leave_file_open(tmp_path: Path) -> None:
    """The loaded dataset is safe to use after its source file is deleted."""
    dataset = _combined_dataset()
    path = save_binned_dataset(dataset, tmp_path, "out.nc", BinSettings())

    loaded = load_binned_dataset(path, "netcdf")
    path.unlink()

    np.testing.assert_array_equal(
        loaded["sea_water_temperature"].values,
        dataset["sea_water_temperature"].values,
    )
