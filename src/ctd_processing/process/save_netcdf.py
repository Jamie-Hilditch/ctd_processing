"""Write a `Dataset` to a CF-compliant netCDF file via xarray + h5netcdf.

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

__all__ = ["write_netcdf"]


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


def write_netcdf(dataset: Dataset, path: Path) -> Path:
    """Write `dataset` to `path` as a CF-compliant netCDF file.

    Every channel in `dataset.channels` other than `time` becomes a data
    variable along a single ``time`` dimension; `time` itself becomes the
    ``time`` coordinate. Each channel's `Channel.metadata` (already
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
        if name == "time":
            continue
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
