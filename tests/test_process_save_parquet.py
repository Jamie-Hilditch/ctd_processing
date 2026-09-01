"""Tests for ctd_processing.process.save_parquet."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ctd_processing.config import (
    ChannelSettings,
    GeolocationSettings,
    ParquetCompressionSettings,
    ProcessSettings,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_parquet import read_parquet, write_parquet

_PROCESS_SETTINGS = ProcessSettings(
    geolocation=GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )
)


def _dataset() -> Dataset:
    """Build a small Dataset covering metadata/history and NaN handling."""
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
    return dataset


def test_write_parquet_round_trips_float_data_with_nan(tmp_path: Path) -> None:
    """Float channel data, including NaN, round-trips exactly."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    table = pq.read_table(path)
    np.testing.assert_array_equal(
        table.column("sea_water_temperature").to_numpy(),
        dataset.channels["sea_water_temperature"].data,
    )


def test_write_parquet_round_trips_time_exactly(tmp_path: Path) -> None:
    """The time column round-trips to the exact same datetime64[ms] values."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    table = pq.read_table(path)
    time_values = table.column("time").to_numpy(zero_copy_only=False)
    assert np.array_equal(
        time_values.astype("datetime64[ms]"), dataset.time.data
    )


def test_write_parquet_field_metadata_round_trips_channel_metadata_and_history(
    tmp_path: Path,
) -> None:
    """Per-field metadata JSON-decodes back to the channel metadata/history."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    schema = pq.read_schema(path)
    field = schema.field("sea_water_temperature")
    metadata = json.loads(field.metadata[b"metadata"])
    history = json.loads(field.metadata[b"history"])

    assert metadata == dataset.channels["sea_water_temperature"].metadata
    assert history == ["removed 1 zero-order hold value(s)"]


def test_write_parquet_schema_metadata_round_trips_dataset_metadata_and_history(
    tmp_path: Path,
) -> None:
    """Schema-level metadata JSON-decodes back to dataset.metadata/history."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    schema = pq.read_schema(path)
    metadata = json.loads(schema.metadata[b"ctd_processing.dataset_metadata"])
    history = json.loads(schema.metadata[b"ctd_processing.dataset_history"])

    assert metadata == dataset.metadata
    assert history == dataset.history


def test_write_parquet_casts_float_columns_to_default_dtype(
    tmp_path: Path,
) -> None:
    """Float columns are written in the project-wide default dtype."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    table = pq.read_table(path)
    assert table.column("sea_water_temperature").type == pa.float32()


def test_write_parquet_honors_per_channel_output_dtype_override(
    tmp_path: Path,
) -> None:
    """A channel's own output_dtype overrides the project-wide default."""
    settings = ProcessSettings(
        geolocation=GeolocationSettings(
            reference_latitude=0.0, reference_longitude=0.0
        ),
        channels={
            "sea_water_temperature": ChannelSettings(output_dtype="float64")
        },
    )
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "profile.parquet", settings)

    table = pq.read_table(path)
    assert table.column("sea_water_temperature").type == pa.float64()


def test_write_parquet_uses_zstd_compression(tmp_path: Path) -> None:
    """Every column is compressed with zstd."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    row_group = pq.ParquetFile(path).metadata.row_group(0)
    for i in range(row_group.num_columns):
        assert row_group.column(i).compression == "ZSTD"


def test_write_parquet_uses_byte_stream_split_for_float_columns_only(
    tmp_path: Path,
) -> None:
    """BYTE_STREAM_SPLIT is used for float columns, not the time column."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    metadata = pq.ParquetFile(path).metadata
    row_group = metadata.row_group(0)
    schema = metadata.schema
    columns_by_name = {
        schema.column(i).name: row_group.column(i)
        for i in range(row_group.num_columns)
    }

    assert (
        "BYTE_STREAM_SPLIT"
        in columns_by_name["sea_water_temperature"].encodings
    )
    assert "BYTE_STREAM_SPLIT" not in columns_by_name["time"].encodings


def test_write_parquet_honors_custom_compression_level(tmp_path: Path) -> None:
    """A lower compression_level produces a larger file than a higher one."""
    rng = np.random.default_rng(42)
    n = 200_000
    time = Channel(
        data=(
            np.datetime64("2026-08-09T00:00:00", "ms")
            + np.arange(n, dtype="timedelta64[ms]")
        )
    )
    dataset = Dataset(time=time)
    dataset.add_channel(
        "sea_water_temperature",
        Channel(data=rng.normal(size=n).astype(np.float32)),
    )

    low_path = write_parquet(
        dataset,
        tmp_path / "low.parquet",
        ProcessSettings(
            geolocation=GeolocationSettings(
                reference_latitude=0.0, reference_longitude=0.0
            ),
            parquet_compression=ParquetCompressionSettings(level=1),
        ),
    )
    high_path = write_parquet(
        dataset,
        tmp_path / "high.parquet",
        ProcessSettings(
            geolocation=GeolocationSettings(
                reference_latitude=0.0, reference_longitude=0.0
            ),
            parquet_compression=ParquetCompressionSettings(level=19),
        ),
    )

    assert high_path.stat().st_size < low_path.stat().st_size


def test_write_parquet_disabling_compression_writes_uncompressed(
    tmp_path: Path,
) -> None:
    """enabled=False writes every column uncompressed."""
    settings = ProcessSettings(
        geolocation=GeolocationSettings(
            reference_latitude=0.0, reference_longitude=0.0
        ),
        parquet_compression=ParquetCompressionSettings(enabled=False),
    )
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "profile.parquet", settings)

    metadata = pq.ParquetFile(path).metadata
    row_group = metadata.row_group(0)
    schema = metadata.schema
    columns_by_name = {
        schema.column(i).name: row_group.column(i)
        for i in range(row_group.num_columns)
    }

    for column in columns_by_name.values():
        assert column.compression == "UNCOMPRESSED"
    assert (
        "BYTE_STREAM_SPLIT"
        in columns_by_name["sea_water_temperature"].encodings
    )


def test_write_parquet_creates_missing_parent_directory(tmp_path: Path) -> None:
    """write_parquet creates path.parent if it doesn't already exist."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "nested" / "profile.parquet", _PROCESS_SETTINGS
    )

    assert path.exists()


def test_read_parquet_round_trips_channel_data_metadata_and_history(
    tmp_path: Path,
) -> None:
    """A channel's data, metadata, and history round-trip via read_parquet."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    loaded = read_parquet(path)

    temperature = loaded.channels["sea_water_temperature"]
    np.testing.assert_array_equal(
        temperature.data, dataset.channels["sea_water_temperature"].data
    )
    assert (
        temperature.metadata
        == dataset.channels["sea_water_temperature"].metadata
    )
    assert (
        temperature.history == dataset.channels["sea_water_temperature"].history
    )


def test_read_parquet_round_trips_time_exactly(tmp_path: Path) -> None:
    """The time column round-trips to the exact same datetime64[ms] values."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    loaded = read_parquet(path)

    assert np.array_equal(loaded.time.data, dataset.time.data)
    assert loaded.time.metadata == dataset.time.metadata
    assert loaded.time.history == dataset.time.history


def test_read_parquet_round_trips_dataset_metadata_and_history(
    tmp_path: Path,
) -> None:
    """Schema metadata/history decode back to dataset.metadata/history."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    loaded = read_parquet(path)

    assert loaded.metadata == dataset.metadata
    assert loaded.history == dataset.history


def test_read_parquet_does_not_inject_extra_history(tmp_path: Path) -> None:
    """Loading channels bypasses add_channel, so history isn't padded out."""
    dataset = _dataset()
    path = write_parquet(
        dataset, tmp_path / "profile.parquet", _PROCESS_SETTINGS
    )

    loaded = read_parquet(path)

    assert loaded.history == dataset.history
    assert list(loaded.channels) == list(dataset.channels)
