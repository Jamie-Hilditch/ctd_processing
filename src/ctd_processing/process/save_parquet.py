"""Write a `Dataset` to a Parquet file via pyarrow, zstd + byte-stream-split.

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

from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["write_parquet"]


def write_parquet(dataset: Dataset, path: Path) -> Path:
    """Write `dataset` to `path` as a zstd-compressed Parquet file.

    Every channel in `dataset.channels` (including `time`) becomes one
    Parquet column, with its `Channel.metadata`/`history` JSON-encoded
    into that column's field metadata. `dataset.metadata`/`history` are
    JSON-encoded into schema (file)-level metadata the same way (using
    ``default=str`` for the one non-JSON-safe value this package
    produces -- `datetime`-like deployment timestamps -- an accepted,
    documented lossy conversion: the text is recoverable, the original
    type is not). Every column is zstd-compressed; float columns
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

    Returns
    -------
    pathlib.Path
        `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    arrays = []
    float_columns = []
    for name, channel in dataset.channels.items():
        array = pa.array(channel.data)
        arrays.append(array)
        field_metadata = {
            b"metadata": json.dumps(channel.metadata).encode(),
            b"history": json.dumps(channel.history).encode(),
        }
        fields.append(pa.field(name, array.type, metadata=field_metadata))
        if np.issubdtype(channel.data.dtype, np.floating):
            float_columns.append(name)

    schema_metadata = {
        b"ctd_processing.dataset_metadata": json.dumps(
            dataset.metadata, default=str
        ).encode(),
        b"ctd_processing.dataset_history": json.dumps(dataset.history).encode(),
    }
    table = pa.Table.from_arrays(
        arrays, schema=pa.schema(fields, metadata=schema_metadata)
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        column_encoding={name: "BYTE_STREAM_SPLIT" for name in float_columns},
    )
    log_verbose(logger, "wrote parquet profile file: %s", path)
    return path
