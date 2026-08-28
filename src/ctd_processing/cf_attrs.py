"""Build/parse CF-style attribute dicts shared across output writers.

Extracted from `ctd_processing.process.save_netcdf` so that
`ctd_processing.bin.binning` (which builds `xarray.Dataset` objects
directly, rather than going through a `Dataset`/`Channel` round trip) can
reuse the same attribute conventions without duplicating them. Lives at
the package top level (like `ctd_processing.config`/`ctd_processing.
logging_utils`) since it is shared by both `ctd_processing.process` and
`ctd_processing.bin`, not specific to either.

`Channel`/`Dataset` are imported only under `TYPE_CHECKING`: this module is
imported by `ctd_processing.process.save_netcdf`, which sits underneath
`ctd_processing.process`'s package `__init__`, so a real runtime import of
`ctd_processing.process.channel`/`ctd_processing.process.dataset` here
would re-enter that partially-initialized package and fail. Neither
function below needs more than duck-typed `.metadata`/`.history` access,
so the type-only import is enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ctd_processing.process.channel import Channel
    from ctd_processing.process.dataset import Dataset

__all__ = ["channel_attrs", "dataset_attrs", "pop_history", "sanitize_attr"]


def sanitize_attr(value: Any) -> Any:
    """Convert one `Dataset.metadata`/`Channel.metadata` value into a safe attr.

    Parameters
    ----------
    value : Any
        A metadata value -- `str`, `int`, `float`, a numpy scalar (e.g.
        `numpy.int64`, as pyrsktools/xarray often hand back), a
        `datetime`-like object, or anything else.

    Returns
    -------
    Any
        `value` unchanged if it is already `str`/`int`/`float`;
        `value.item()` if it is a numpy scalar (`numpy.generic`) --
        `numpy.int64` in particular is not a Python `int`, so without this
        it would otherwise fall through to `str(value)` below, and two
        datasets whose only difference is one going through a netCDF
        round trip (which hands integer attributes back as numpy scalars)
        would wrongly look like a metadata conflict when combined (see
        `ctd_processing.bin.binning.combine_binned_profiles`); its
        ``isoformat()`` if it has one (e.g. a `datetime`); otherwise
        `str(value)`.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str | int | float):
        return value
    isoformat = getattr(value, "isoformat", None)
    if isoformat is not None:
        return isoformat()
    return str(value)


def dataset_attrs(dataset: Dataset) -> dict[str, Any]:
    """Build netCDF/xarray global attributes from `dataset.metadata`/`history`.

    Parameters
    ----------
    dataset : Dataset
        The dataset to build global attributes for.

    Returns
    -------
    dict[str, Any]
        `dataset.metadata` with `None`-valued keys dropped (netCDF/HDF5
        attributes cannot be null) and every remaining value passed
        through `sanitize_attr`, plus a ``history`` key holding
        `dataset.history` only -- each channel's own `history` is
        attached to that channel's own variable instead (see
        `channel_attrs`), not merged in here.
    """
    attrs = {
        key: sanitize_attr(value)
        for key, value in dataset.metadata.items()
        if value is not None
    }
    attrs["history"] = "; ".join(dataset.history)
    return attrs


def channel_attrs(channel: Channel) -> dict[str, Any]:
    """Build netCDF/xarray variable attributes for one `Channel`.

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


def pop_history(attrs: dict[str, Any]) -> list[str]:
    """Pop and split a netCDF/xarray ``history`` attribute back into a list.

    Reverses the ``"; ".join(...)`` done by `dataset_attrs`/`channel_attrs`:
    the inverse operation for a `Dataset`/`Channel` that had no history at
    all (an absent or empty ``history`` attribute) is an empty list, not
    ``[""]``.

    Parameters
    ----------
    attrs : dict[str, Any]
        A variable's or dataset's attributes, as read back from a file.
        Mutated in place: ``"history"`` is removed if present.

    Returns
    -------
    list[str]
        `attrs["history"]` split on ``"; "``, or ``[]`` if `attrs` has no
        ``history`` key or its value is the empty string.
    """
    history = attrs.pop("history", "")
    return history.split("; ") if history else []
