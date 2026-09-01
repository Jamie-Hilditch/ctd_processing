"""Read/write a `Dataset` as Parquet via pyarrow, with byte-stream-split.

Parquet has no CF-equivalent convention for per-variable/global scientific
metadata, so `Channel.metadata`/`history` are preserved via per-field
key/value metadata, and `Dataset.metadata`/`history` via schema (file)-level
metadata, namespaced under ``ctd_processing.`` to avoid colliding with other
tools' own Parquet metadata keys (e.g. pandas' ``"pandas"`` key).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ctd_processing.config import ProcessSettings, resolve_output_dtype
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["read_parquet", "write_parquet"]


def write_parquet(
    dataset: Dataset, path: Path, process_settings: ProcessSettings
) -> Path:
    """Write `dataset` to `path` as a zstd-compressed Parquet file.

    `dataset.time` plus every channel in `dataset.channels` each become
    one Parquet column, with its `Channel.metadata`/`history` JSON-encoded
    into that column's field metadata. `dataset.metadata`/`history` are
    JSON-encoded into schema (file)-level metadata the same way (using
    ``default=str`` for the one non-JSON-safe value this package
    produces -- `datetime`-like deployment timestamps -- an accepted,
    documented lossy conversion: the text is recoverable, the original
    type is not). Every floating-point column is cast to its resolved
    output dtype (see `ctd_processing.config.resolve_output_dtype`)
    before being written. Every column is compressed per
    `process_settings.parquet_compression` (see
    `ctd_processing.config.ParquetCompressionSettings`); float columns
    additionally use byte-stream-split encoding, which transposes each
    value's bytes across the column before compression -- more
    compressible for floating-point data than a plain byte-level
    encoding.

    Parameters
    ----------
    dataset : Dataset
        The dataset to write.
    path : pathlib.Path
        File to write to. Its parent directory is created if it does not
        already exist.
    process_settings : ProcessSettings
        Supplies `output_dtype`/`channels` for
        `ctd_processing.config.resolve_output_dtype`.

    Returns
    -------
    pathlib.Path
        `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    arrays = []
    float_columns = []
    channels_to_write = {"time": dataset.time, **dataset.channels}
    for name, channel in channels_to_write.items():
        data = channel.data
        if np.issubdtype(data.dtype, np.floating):
            dtype = resolve_output_dtype(process_settings, name)
            data = data.astype(dtype, copy=False)
            float_columns.append(name)
        array = pa.array(data)
        arrays.append(array)
        field_metadata = {
            b"metadata": json.dumps(channel.metadata).encode(),
            b"history": json.dumps(channel.history).encode(),
        }
        fields.append(pa.field(name, array.type, metadata=field_metadata))

    schema_metadata = {
        b"ctd_processing.dataset_metadata": json.dumps(
            dataset.metadata, default=str
        ).encode(),
        b"ctd_processing.dataset_history": json.dumps(dataset.history).encode(),
    }
    table = pa.Table.from_arrays(
        arrays, schema=pa.schema(fields, metadata=schema_metadata)
    )

    compression_settings = process_settings.parquet_compression
    pq.write_table(
        table,
        path,
        compression="zstd" if compression_settings.enabled else "none",
        compression_level=(
            compression_settings.level if compression_settings.enabled else None
        ),
        use_dictionary=False,
        column_encoding={name: "BYTE_STREAM_SPLIT" for name in float_columns},
    )
    log_verbose(logger, "wrote parquet profile file: %s", path)
    return path


def read_parquet(path: Path) -> Dataset:
    """Read a `Dataset` back from a Parquet file written by `write_parquet`.

    Reverses `write_parquet`: the ``time`` column becomes `Dataset.time`,
    every other column becomes a `Channel` keyed by its column name, and
    each column's/the schema's JSON-encoded metadata becomes that
    `Channel`'s/the `Dataset`'s `metadata` and `history`. Channels are
    attached directly to `Dataset.channels`, bypassing
    `Dataset.add_channel`, so that loading a file does not itself inject
    extra "added channel" entries into `history` beyond what
    `write_parquet` actually wrote -- the same approach `Dataset.subset`
    uses to reconstruct a `Dataset` with pre-existing channels.

    `Dataset.metadata` values that were not JSON-serializable (e.g.
    `datetime`-like deployment timestamps) were written as their
    ``default=str`` text form, which is not reversed back to the
    original Python type here -- the documented, accepted lossy
    conversion described in `write_parquet`.

    Parameters
    ----------
    path : pathlib.Path
        Path to a Parquet file written by `write_parquet`.

    Returns
    -------
    Dataset
        The reconstructed dataset.
    """
    table = pq.read_table(path)
    schema = table.schema

    def _channel(name: str) -> Channel:
        field = schema.field(name)
        metadata = json.loads(field.metadata[b"metadata"])
        history = json.loads(field.metadata[b"history"])
        data = table.column(name).to_numpy(zero_copy_only=False)
        return Channel(data=data, metadata=metadata, history=history)

    dataset_metadata = json.loads(
        schema.metadata[b"ctd_processing.dataset_metadata"]
    )
    dataset_history = json.loads(
        schema.metadata[b"ctd_processing.dataset_history"]
    )

    dataset = Dataset(
        time=_channel("time"),
        metadata=dataset_metadata,
        history=dataset_history,
    )
    for name in schema.names:
        if name == "time":
            continue
        dataset.channels[name] = _channel(name)

    log_verbose(logger, "read parquet profile file: %s", path)
    return dataset
