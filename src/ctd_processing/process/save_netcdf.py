"""Read/write a `Dataset` to a CF-compliant netCDF file via xarray + h5netcdf.

Compression uses h5netcdf's bundled zlib (with the shuffle filter) rather
than an HDF5 filter plugin such as blosc/zstd, so files stay readable by any
netCDF4/HDF5 reader without extra system dependencies.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["read_netcdf", "write_netcdf"]


def _sanitize_attr(value: Any) -> Any:
    """Convert one `Dataset.metadata` value into a netCDF-safe attribute.

    Parameters
    ----------
    value : Any
        A `Dataset.metadata` value -- `str`, `int`, `float`, a
        `datetime`-like object, or anything else.

    Returns
    -------
    Any
        `value` unchanged if it is already `str`/`int`/`float`; its
        ``isoformat()`` if it has one (e.g. a `datetime`); otherwise
        `str(value)`.
    """
    if isinstance(value, str | int | float):
        return value
    isoformat = getattr(value, "isoformat", None)
    if isoformat is not None:
        return isoformat()
    return str(value)


def _global_attrs(dataset: Dataset) -> dict[str, Any]:
    """Build netCDF global attributes from `dataset.metadata`/`history`.

    Parameters
    ----------
    dataset : Dataset
        The dataset to build global attributes for.

    Returns
    -------
    dict[str, Any]
        `dataset.metadata` with `None`-valued keys dropped (netCDF/HDF5
        attributes cannot be null) and every remaining value passed
        through `_sanitize_attr`, plus a ``history`` key holding
        `dataset.history` only -- each channel's own `history` is
        attached to that channel's own variable instead (see
        `_channel_attrs`), not merged in here.
    """
    attrs = {
        key: _sanitize_attr(value)
        for key, value in dataset.metadata.items()
        if value is not None
    }
    attrs["history"] = "; ".join(dataset.history)
    return attrs


def _channel_attrs(channel: Channel) -> dict[str, Any]:
    """Build netCDF variable attributes for one `Channel`.

    Parameters
    ----------
    channel : Channel
        The channel to build variable attributes for.

    Returns
    -------
    dict[str, Any]
        `channel.metadata` with `None`-valued keys dropped (already
        CF-shaped: ``units``/``long_name``/``standard_name``/
        ``source_channel_name``), plus a ``history`` key holding
        `channel.history`, semicolon-joined, when `channel.history` is
        non-empty.
    """
    attrs = {
        key: value
        for key, value in channel.metadata.items()
        if value is not None
    }
    if channel.history:
        attrs["history"] = "; ".join(channel.history)
    return attrs


def _pop_history(attrs: dict[str, Any]) -> list[str]:
    """Pop and split a netCDF ``history`` attribute back into a list.

    Reverses the ``"; ".join(...)`` done by `_global_attrs`/
    `_channel_attrs`: the inverse operation for a `Dataset`/`Channel`
    that had no history at all (an absent or empty ``history`` attribute)
    is an empty list, not ``[""]``.

    Parameters
    ----------
    attrs : dict[str, Any]
        A variable's or dataset's attributes, as read back from a netCDF
        file. Mutated in place: ``"history"`` is removed if present.

    Returns
    -------
    list[str]
        `attrs["history"]` split on ``"; "``, or ``[]`` if `attrs` has no
        ``history`` key or its value is the empty string.
    """
    history = attrs.pop("history", "")
    return history.split("; ") if history else []


def read_netcdf(path: Path) -> Dataset:
    """Read a `Dataset` back from a netCDF file written by `write_netcdf`.

    Reverses `write_netcdf`: the ``time`` coordinate becomes `Dataset.time`,
    every other data variable becomes a `Channel` keyed by its variable
    name, and each variable's/the file's attributes become that
    `Channel`'s/the `Dataset`'s `metadata` and `history` (see
    `_pop_history`). Channels are attached directly to
    `Dataset.channels`, bypassing `Dataset.add_channel`, so that loading a
    file does not itself inject extra "added channel" entries into
    `history` beyond what `write_netcdf` actually wrote -- the same
    approach `Dataset.subset` uses to reconstruct a `Dataset` with
    pre-existing channels.

    `None`-valued metadata and the original Python type of
    `datetime`-like metadata values are not recoverable (see
    `_sanitize_attr`): this reproduces exactly what `write_netcdf` wrote
    to `path`, not necessarily the `Dataset` originally passed to it.

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
            history=_pop_history(time_attrs),
        )

        global_attrs = dict(xr_dataset.attrs)
        dataset = Dataset(
            time=time,
            metadata=global_attrs,
            history=_pop_history(global_attrs),
        )

        for name, variable in xr_dataset.data_vars.items():
            channel_attrs = dict(variable.attrs)
            dataset.channels[str(name)] = Channel(
                data=variable.to_numpy(),
                metadata=channel_attrs,
                history=_pop_history(channel_attrs),
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
    `datetime`-like values ISO-8601-formatted via `_sanitize_attr`), and
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
            channel.data, dims=("time",), attrs=_channel_attrs(channel)
        )
        if np.issubdtype(channel.data.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}

    time_attrs = _channel_attrs(dataset.time)
    time_attrs.setdefault("standard_name", "time")
    time_attrs.setdefault("axis", "T")
    coords = {
        "time": xr.DataArray(
            dataset.time.data, dims=("time",), attrs=time_attrs
        )
    }

    xr_dataset = xr.Dataset(
        data_vars=data_vars, coords=coords, attrs=_global_attrs(dataset)
    )
    xr_dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    log_verbose(logger, "wrote netCDF profile file: %s", path)
    return path
