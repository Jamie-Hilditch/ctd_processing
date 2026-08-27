"""Tests for ctd_processing.process.save_parquet."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_parquet import write_parquet


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
    path = write_parquet(dataset, tmp_path / "profile.parquet")

    table = pq.read_table(path)
    np.testing.assert_array_equal(
        table.column("sea_water_temperature").to_numpy(),
        dataset.channels["sea_water_temperature"].data,
    )


def test_write_parquet_round_trips_time_exactly(tmp_path: Path) -> None:
    """The time column round-trips to the exact same datetime64[ms] values."""
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "profile.parquet")

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
    path = write_parquet(dataset, tmp_path / "profile.parquet")

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
    path = write_parquet(dataset, tmp_path / "profile.parquet")

    schema = pq.read_schema(path)
    metadata = json.loads(schema.metadata[b"ctd_processing.dataset_metadata"])
    history = json.loads(schema.metadata[b"ctd_processing.dataset_history"])

    assert metadata == dataset.metadata
    assert history == dataset.history


def test_write_parquet_uses_zstd_compression(tmp_path: Path) -> None:
    """Every column is compressed with zstd."""
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "profile.parquet")

    row_group = pq.ParquetFile(path).metadata.row_group(0)
    for i in range(row_group.num_columns):
        assert row_group.column(i).compression == "ZSTD"


def test_write_parquet_uses_byte_stream_split_for_float_columns_only(
    tmp_path: Path,
) -> None:
    """BYTE_STREAM_SPLIT is used for float columns, not the time column."""
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "profile.parquet")

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


def test_write_parquet_creates_missing_parent_directory(tmp_path: Path) -> None:
    """write_parquet creates path.parent if it doesn't already exist."""
    dataset = _dataset()
    path = write_parquet(dataset, tmp_path / "nested" / "profile.parquet")

    assert path.exists()
