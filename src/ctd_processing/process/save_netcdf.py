"""Read/write a `Dataset` to a CF-compliant netCDF file via xarray + h5netcdf.

Compression uses h5netcdf's bundled zlib (with the shuffle filter) rather
than an HDF5 filter plugin such as blosc/zstd, so files stay readable by any
netCDF4/HDF5 reader without extra system dependencies.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ctd_processing.cf_attrs import channel_attrs, dataset_attrs, pop_history
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["read_netcdf", "write_netcdf"]


def read_netcdf(path: Path) -> Dataset:
    """Read a `Dataset` back from a netCDF file written by `write_netcdf`.

    Reverses `write_netcdf`: the ``time`` coordinate becomes `Dataset.time`,
    every other data variable becomes a `Channel` keyed by its variable
    name, and each variable's/the file's attributes become that
    `Channel`'s/the `Dataset`'s `metadata` and `history` (see
    `ctd_processing.cf_attrs.pop_history`). Channels are attached
    directly to `Dataset.channels`, bypassing `Dataset.add_channel`, so
    that loading a file does not itself inject extra "added channel"
    entries into `history` beyond what `write_netcdf` actually wrote --
    the same approach `Dataset.subset` uses to reconstruct a `Dataset`
    with pre-existing channels.

    `None`-valued metadata and the original Python type of
    `datetime`-like metadata values are not recoverable (see
    `ctd_processing.cf_attrs.sanitize_attr`): this reproduces
    exactly what `write_netcdf` wrote to `path`, not necessarily the
    `Dataset` originally passed to it.

    Parameters
    ----------
    path : pathlib.Path
        Path to a netCDF file written by `write_netcdf`.

    Returns
    -------
    Dataset
        The reconstructed dataset.
    """
    with xr.open_dataset(path, engine="h5netcdf") as xr_dataset:
        time_attrs = dict(xr_dataset["time"].attrs)
        time = Channel(
            data=xr_dataset["time"].to_numpy(),
            metadata=time_attrs,
            history=pop_history(time_attrs),
        )

        global_attrs = dict(xr_dataset.attrs)
        dataset = Dataset(
            time=time,
            metadata=global_attrs,
            history=pop_history(global_attrs),
        )

        for name, variable in xr_dataset.data_vars.items():
            variable_attrs = dict(variable.attrs)
            dataset.channels[str(name)] = Channel(
                data=variable.to_numpy(),
                metadata=variable_attrs,
                history=pop_history(variable_attrs),
            )

    log_verbose(logger, "read netCDF profile file: %s", path)
    return dataset


def write_netcdf(dataset: Dataset, path: Path) -> Path:
    """Write `dataset` to `path` as a CF-compliant netCDF file.

    Every channel in `dataset.channels` becomes a data variable along a
    single ``time`` dimension; `dataset.time` itself becomes the ``time``
    coordinate. Each channel's `Channel.metadata` (already
    CF-shaped) becomes that variable's attributes, plus a ``history``
    attribute from `Channel.history` when non-empty -- attached to the
    variable it actually describes, not merged into one global blob.
    `dataset.metadata` becomes global attributes (`None` values dropped,
    `datetime`-like values ISO-8601-formatted via
    `ctd_processing.cf_attrs.sanitize_attr`), and
    `dataset.history` becomes the global ``history`` attribute. Float data
    variables are compressed with zlib + the shuffle filter via
    `h5netcdf`; `time` is left uncompressed (a monotonic timestamp array
    compresses poorly and isn't worth the encoding complexity).

    Parameters
    ----------
    dataset : Dataset
        The dataset to write. Must have a `time` channel (every `Dataset`
        has one by construction).
    path : pathlib.Path
        File to write to. Its parent directory is created if it does not
        already exist.

    Returns
    -------
    pathlib.Path
        `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data_vars = {}
    encoding = {}
    for name, channel in dataset.channels.items():
        data_vars[name] = xr.DataArray(
            channel.data, dims=("time",), attrs=channel_attrs(channel)
        )
        if np.issubdtype(channel.data.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}

    time_attrs = channel_attrs(dataset.time)
    time_attrs.setdefault("standard_name", "time")
    time_attrs.setdefault("axis", "T")
    coords = {
        "time": xr.DataArray(
            dataset.time.data, dims=("time",), attrs=time_attrs
        )
    }

    xr_dataset = xr.Dataset(
        data_vars=data_vars, coords=coords, attrs=dataset_attrs(dataset)
    )
    xr_dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    log_verbose(logger, "wrote netCDF profile file: %s", path)
    return path
