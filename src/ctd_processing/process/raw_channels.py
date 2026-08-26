"""Per-raw-channel processing steps, configured via `process.raw_channels`.

See `ctd_processing.config.RawChannelSettings`.
"""

import logging
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from ctd_processing.config import ProcessSettings, RawChannelSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "add_offset",
    "process_raw_channel",
    "process_raw_channels",
    "remove_holds",
    "shift_time",
]


@numba.njit(cache=True)
def _remove_holds_inplace(data: npt.NDArray[Any]) -> int:
    """Replace zero-order hold repeats in `data` with NaN, in place.

    Single forward pass tracking the last known-valid value, rather than
    comparing each element to `data[i - 1]` directly -- the latter would
    under-count runs longer than 2, since earlier repeats in the same run
    get overwritten to NaN before later ones are compared. No new arrays
    are allocated; `count` is a scalar.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to correct, mutated in place.

    Returns
    -------
    int
        The number of samples replaced with NaN.
    """
    count = 0
    if data.shape[0] == 0:
        return count
    last_valid = data[0]
    for i in range(1, data.shape[0]):
        current = data[i]
        if current == last_valid:
            data[i] = np.nan
            count += 1
        else:
            last_valid = current
    return count


def remove_holds(channel: Channel) -> Channel:
    """Replace zero-order hold repeats in `channel` with NaN, in place.

    RBR instruments' analogue-to-digital converters must periodically
    recalibrate; if a sample is due before recalibration finishes, the
    firmware fills it in by repeating the previous sample verbatim (a
    "zero-order hold") rather than measuring a new value. This is
    detectable as an exact repeat of the immediately preceding value, with
    no minimum run length -- a single repeated pair already counts, and
    for a longer run, every repeat past the first is held rather than
    measured. This reproduces `pyrsktools`'s `RSK.correcthold` detection
    rule (exact equality, not a distance/tolerance check) but as a single
    in-place pass instead of `correcthold`'s per-channel, per-profile
    Python-level looping.

    A pre-existing NaN in `channel.data` breaks the run: the sample after
    it is compared against the NaN (never equal), not against the last
    real value before it, matching `correcthold`'s own behavior.

    Parameters
    ----------
    channel : Channel
        The channel to correct. `channel.data` is mutated in place; this
        is not a copying operation.

    Returns
    -------
    Channel
        `channel` itself (not a copy), with a new entry appended to
        `channel.history` recording how many samples were replaced.

    Raises
    ------
    ValueError
        If `channel.data` is not a floating-point array (`np.nan` cannot
        be assigned into an integer array).
    """
    if not np.issubdtype(channel.data.dtype, np.floating):
        raise ValueError(
            "remove_holds requires floating-point data; got "
            f"{channel.data.dtype}."
        )

    count = _remove_holds_inplace(channel.data)  # ty: ignore
    channel.record(f"removed {count} zero-order hold value(s)")
    log_verbose(logger, "removed %d zero-order hold value(s)", count)
    return channel


@numba.njit(cache=True)
def _shift_inplace(data: npt.NDArray[Any], shift: int) -> None:
    """Shift `data` by `shift` samples in place, pandas `.shift()`-style.

    ``output[i] = input[i - shift]`` wherever that index is valid, else
    NaN. A single pass, but the iteration direction depends on the sign
    of `shift` to avoid overwriting a source value before it's read:
    positive `shift` (destination `i` always greater than source
    `i - shift`) must go high-to-low; negative `shift` (destination `i`
    always less than source `i + abs(shift)`) must go low-to-high. No new
    arrays are allocated.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to shift, mutated in place.
    shift : int
        Number of samples to shift by. Positive delays (shifts toward
        higher indices' source, i.e. pulls earlier values forward);
        negative advances. Zero is a no-op.
    """
    n = data.shape[0]
    if shift == 0:
        return
    if shift > 0:
        for i in range(n - 1, -1, -1):
            data[i] = data[i - shift] if i >= shift else np.nan
    else:
        m = -shift
        for i in range(n):
            data[i] = data[i + m] if i < n - m else np.nan


def shift_time(channel: Channel, shift: int) -> Channel:
    """Shift `channel`'s data by `shift` samples in place, pandas-style.

    Follows pandas' `Series.shift(periods=shift)` convention: a positive
    `shift` delays the channel (each sample takes the value from `shift`
    samples earlier, leaving the first `shift` samples as NaN); a negative
    `shift` advances it (each sample takes the value from ``abs(shift)``
    samples later, leaving the last ``abs(shift)`` samples as NaN).
    Intended for correcting a known sensor response lag relative to the
    other channels. Should run after `remove_holds` and before
    `add_offset`.

    Parameters
    ----------
    channel : Channel
        The channel to shift. `channel.data` is mutated in place; this is
        not a copying operation.
    shift : int
        Number of samples to shift by. See above for sign convention.
        ``0`` is a no-op.

    Returns
    -------
    Channel
        `channel` itself (not a copy), with a new entry appended to
        `channel.history` recording the shift applied.

    Raises
    ------
    ValueError
        If `channel.data` is not a floating-point array.
    """
    if not np.issubdtype(channel.data.dtype, np.floating):
        raise ValueError(
            "shift_time requires floating-point data; got "
            f"{channel.data.dtype}."
        )

    _shift_inplace(channel.data, shift)  # ty: ignore
    channel.record(f"shifted by {shift} sample(s)")
    log_verbose(logger, "shifted by %d sample(s)", shift)
    return channel


def add_offset(channel: Channel, offset: float) -> Channel:
    """Add a fixed offset to `channel`'s data, in place.

    Intended for correcting a known, fixed calibration bias. Should run
    after `remove_holds`, so held values are already NaN by the time any
    offset is added.

    Parameters
    ----------
    channel : Channel
        The channel to offset. `channel.data` is mutated in place; this
        is not a copying operation.
    offset : float
        The value to add to every element of `channel.data`.

    Returns
    -------
    Channel
        `channel` itself (not a copy), with a new entry appended to
        `channel.history` recording the offset applied.

    Raises
    ------
    ValueError
        If `channel.data` is not a floating-point array.
    """
    if not np.issubdtype(channel.data.dtype, np.floating):
        raise ValueError(
            "add_offset requires floating-point data; got "
            f"{channel.data.dtype}."
        )

    channel.data += offset
    channel.record(f"added offset {offset}")
    log_verbose(logger, "added offset %s", offset)
    return channel


def process_raw_channel(
    channel: Channel, settings: RawChannelSettings
) -> Channel:
    """Apply configured raw-channel processing steps to `channel`, in place.

    Steps run in a fixed order: `remove_holds` (if enabled), then
    `shift_time` (if `settings.shift` is set), then `add_offset` (if
    `settings.offset` is set) -- matching `RawChannelSettings`' documented
    step ordering.

    Parameters
    ----------
    channel : Channel
        The channel to process. Mutated in place; this is not a copying
        operation.
    settings : RawChannelSettings
        Which steps to apply and their parameters.

    Returns
    -------
    Channel
        `channel` itself (not a copy).
    """
    if settings.remove_holds:
        remove_holds(channel)
    if settings.shift is not None:
        shift_time(channel, settings.shift)
    if settings.offset is not None:
        add_offset(channel, settings.offset)
    return channel


def process_raw_channels(
    dataset: Dataset, settings: ProcessSettings
) -> Dataset:
    """Apply configured per-channel processing to every raw channel.

    Looks up each channel's `RawChannelSettings` by its name in
    `dataset.channels` (the same key `settings.raw_channels` is keyed
    by); a channel with no matching entry uses `RawChannelSettings()`
    defaults. `"time"` is skipped -- it isn't a raw measurement channel
    this config applies to.

    This function only depends on `Dataset`/`ProcessSettings`, not on how
    `dataset` was built -- it can run at any point after the dataset
    exists, independent of `pyrsktools`.

    Parameters
    ----------
    dataset : Dataset
        The dataset to process. Every non-``"time"`` channel is mutated
        in place via `process_raw_channel`.
    settings : ProcessSettings
        Settings providing `raw_channels`.

    Returns
    -------
    Dataset
        `dataset` itself (not a copy).
    """
    for name, channel in dataset.channels.items():
        if name == "time":
            continue
        raw_channel_settings = settings.raw_channels.get(
            name, RawChannelSettings()
        )
        process_raw_channel(channel, raw_channel_settings)
    return dataset
