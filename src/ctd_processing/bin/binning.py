"""Bin one profile, and combine binned profiles, per `BinSettings`.

See `ctd_processing.config.BinSettings`. `bin_profile` bins a single
already-extracted, already-processed profile `Dataset` (see
`ctd_processing.process.process_profile`) along one of its channels
(`BinSettings.channel`); `combine_binned_profiles` stacks the resulting
per-profile `xarray.Dataset` objects into one combined dataset along a new
``profile`` dimension. See `ctd_processing.bin.bin_deployment` for the
orchestration that ties these together for a whole deployment's profiles.
"""

import logging

import numpy as np
import numpy.typing as npt
import xarray as xr

from ctd_processing.cf_attrs import channel_attrs, sanitize_attr
from ctd_processing.config import BinSettings
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["bin_profile", "combine_binned_profiles", "compute_bin_edges"]

# `Dataset.metadata` keys that describe one specific profile rather than the
# whole deployment (see
# `ctd_processing.process.geolocation.attach_geolocation`). Moved to
# profile-indexed coordinates rather than dataset-level attrs, since every
# profile has its own value for each of these.
_PROFILE_METADATA_KEYS = (
    "profile_start_time",
    "profile_end_time",
    "latitude",
    "longitude",
)

# Of `_PROFILE_METADATA_KEYS`, the ones that are timestamps rather than
# plain numbers.
_TIME_METADATA_KEYS = frozenset({"profile_start_time", "profile_end_time"})


def _coerce_profile_metadata(key: str, value: object) -> object:
    """Parse a time-valued profile metadata entry back into `numpy.datetime64`.

    `dataset.metadata["profile_start_time"]`/`["profile_end_time"]` are
    `numpy.datetime64` in memory, but a save/load round trip through
    either output format stringifies them (`numpy.datetime64` has no
    `.isoformat()`, and isn't JSON-serializable -- see
    `ctd_processing.cf_attrs.sanitize_attr` and
    `ctd_processing.process.save_parquet.write_parquet`'s docstring). Since
    `bin` always operates on saved-and-reloaded profiles, this parses that
    string back so the ``time``/``profile_end_time`` coordinates stay
    properly time-typed rather than becoming plain-string variables.

    Parameters
    ----------
    key : str
        One of `_PROFILE_METADATA_KEYS`.
    value : object
        That key's value, as read from `Dataset.metadata`.

    Returns
    -------
    object
        `value` unchanged, except for `_TIME_METADATA_KEYS`, where a
        `str` value is parsed via `numpy.datetime64`.

    Raises
    ------
    TypeError
        If `key` is in `_TIME_METADATA_KEYS` and `value` is neither
        ``None``, a `str`, nor a `numpy.datetime64`.
    """
    if key not in _TIME_METADATA_KEYS:
        return value
    if value is None or isinstance(value, np.datetime64):
        return value
    if isinstance(value, str):
        return np.datetime64(value)
    raise TypeError(
        f"Cannot parse {key!r} metadata value {value!r} as a time: "
        "expected None, str, or numpy.datetime64."
    )


def compute_bin_edges(
    values: list[npt.NDArray[np.floating]], settings: BinSettings
) -> npt.NDArray[np.float64]:
    """Generate bin edges for `BinSettings.channel`, resolving auto bounds.

    Parameters
    ----------
    values : list[numpy.typing.NDArray[numpy.floating]]
        Every profile's `BinSettings.channel` data, used to resolve
        `settings.first`/`settings.last` if either is unset. Ignored
        entirely if both are set.
    settings : BinSettings
        Supplies `step` and, optionally, `first`/`last`.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Bin edges: ``first, first + step, first + 2 * step, ...``,
        stopping at the first edge that reaches or passes `last` (so,
        unlike `numpy.arange`, the final edge may slightly overshoot
        `last`). Increasing if `settings.step` is positive, decreasing if
        negative.

    Raises
    ------
    ValueError
        If `settings.first`/`settings.last` are both unset and `values`
        contains no finite data to compute them from.
    """
    first = settings.first
    last = settings.last
    if first is None or last is None:
        all_values = np.concatenate(values) if values else np.array([])
        finite = all_values[np.isfinite(all_values)]
        if finite.size == 0:
            raise ValueError(
                "Cannot auto-compute bin edges: no finite "
                f"{settings.channel!r} data was given."
            )
        data_min = float(finite.min())
        data_max = float(finite.max())
        if first is None:
            first = data_min if settings.step > 0 else data_max
        if last is None:
            last = data_max if settings.step > 0 else data_min

    n_bins = int(np.ceil((last - first) / settings.step))
    n_bins = max(n_bins, 1)
    return first + settings.step * np.arange(n_bins + 1, dtype=np.float64)


def bin_profile(
    dataset: Dataset, channel: str, edges: npt.NDArray
) -> xr.Dataset:
    """Bin one profile's channels onto `edges`, averaging within each bin.

    Every channel except `channel` itself is averaged (NaN-aware) within
    each bin via `xarray.Dataset.groupby_bins` (accelerated by `flox` when
    installed), producing a data variable along a dimension named after
    `channel`, whose coordinate is that bin's center. `channel` itself is
    not included as a data variable -- the bin coordinate replaces it.

    `dataset.metadata`'s per-profile entries (`profile_start_time`,
    `profile_end_time`, `latitude`, `longitude` -- see
    `ctd_processing.process.geolocation.attach_geolocation`) become
    coordinates along a new, length-1 ``profile`` dimension (so they stack
    correctly once `combine_binned_profiles` concatenates every profile
    along it, even when their values happen to be identical, e.g. a fixed
    reference position); `profile_start_time` is additionally exposed
    under the CF-conventional name ``time``. Every other `dataset.metadata`
    entry, plus `dataset.history`, becomes a dataset (global) attribute,
    matching `ctd_processing.cf_attrs.dataset_attrs`.

    Parameters
    ----------
    dataset : Dataset
        One already-extracted, already-processed profile. Not mutated.
    channel : str
        The channel key to bin by (see `BinSettings.channel`). Must be
        present in `dataset.channels`.
    edges : numpy.typing.NDArray
        Bin edges, as returned by `compute_bin_edges`. May be increasing
        or decreasing; sorted ascending internally since
        `xarray.Dataset.groupby_bins` requires monotonically increasing
        bins. The resulting bin coordinate is always presented in
        ascending numeric order -- the configured direction only affects
        how `compute_bin_edges` derives `edges` from `step`/`first`/`last`,
        not how bins are laid out in the output.

    Returns
    -------
    xarray.Dataset
        The binned profile, with dimension/coordinate `channel`.

    Raises
    ------
    ValueError
        If `channel` is not present in `dataset.channels`.
    """
    if channel not in dataset.channels:
        raise ValueError(
            f"Cannot bin by {channel!r}: dataset has no such channel."
        )

    sorted_edges = np.sort(edges)
    centers = (sorted_edges[:-1] + sorted_edges[1:]) / 2

    bin_channel = dataset.channels[channel]
    data_vars = {
        name: xr.DataArray(
            other.data, dims=("obs",), attrs=channel_attrs(other)
        )
        for name, other in dataset.channels.items()
        if name != channel
    }
    obs_dataset = xr.Dataset(
        data_vars=data_vars, coords={channel: ("obs", bin_channel.data)}
    )

    grouped = obs_dataset.groupby_bins(
        channel,
        bins=sorted_edges,
        labels=centers,
        right=True,
        include_lowest=True,
    )
    binned = grouped.mean(skipna=True)
    binned = binned.rename({f"{channel}_bins": channel})
    binned = binned.reindex({channel: centers})
    binned[channel].attrs = dict(bin_channel.metadata)

    metadata = dict(dataset.metadata)
    coords = {
        key: (
            "profile",
            [_coerce_profile_metadata(key, metadata.pop(key, None))],
        )
        for key in _PROFILE_METADATA_KEYS
    }
    coords["time"] = coords.pop("profile_start_time")
    binned = binned.expand_dims("profile").assign_coords(coords)
    binned["time"].attrs = {"standard_name": "time", "axis": "T"}
    binned["latitude"].attrs = {
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    binned["longitude"].attrs = {
        "standard_name": "longitude",
        "units": "degrees_east",
    }

    attrs = {
        key: sanitize_attr(value)
        for key, value in metadata.items()
        if value is not None
    }
    attrs["history"] = "; ".join(dataset.history)
    binned.attrs = attrs

    return binned


def combine_binned_profiles(binned: list[xr.Dataset]) -> xr.Dataset:
    """Stack per-profile binned datasets along a new ``profile`` dimension.

    Parameters
    ----------
    binned : list[xarray.Dataset]
        Per-profile datasets, as returned by `bin_profile`, all sharing
        the same bin coordinate.

    Returns
    -------
    xarray.Dataset
        The combined dataset, concatenated with
        ``combine_attrs="drop_conflicts"``: an attribute (dataset-level or
        per-variable) is kept only if every input dataset has it with the
        same value. Any attribute dropped this way is reported in one
        `logging.Logger.warning` call.
    """
    combined = xr.concat(
        binned,
        dim="profile",
        data_vars="all",
        combine_attrs="drop_conflicts",
    )

    dropped_global = sorted(
        set().union(*(set(ds.attrs) for ds in binned)) - set(combined.attrs)
    )
    dropped_variables: dict[str, list[str]] = {}
    for name in combined.variables:
        source_attrs = [
            set(ds.variables[name].attrs)
            for ds in binned
            if name in ds.variables
        ]
        if not source_attrs:
            continue
        dropped = sorted(
            set().union(*source_attrs) - set(combined.variables[name].attrs)
        )
        if dropped:
            dropped_variables[str(name)] = dropped

    if dropped_global:
        logger.warning(
            "Dropped conflicting global attribute(s) when combining "
            "profiles: %s",
            ", ".join(dropped_global),
        )
    for name, dropped in dropped_variables.items():
        logger.warning(
            "Dropped conflicting attribute(s) of %r when combining "
            "profiles: %s",
            name,
            ", ".join(dropped),
        )

    return combined
