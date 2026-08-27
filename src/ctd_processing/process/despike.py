"""Despiking, configured via `process.despike`/`process.despike_channels`.

See `ctd_processing.config.DespikeSettings` and
`ctd_processing.config.resolve_despike_settings`. This mirrors
`pyrsktools.RSK.despike`'s algorithm -- smooth with a rolling median
filter to get a "reference" series, form the residual against it, and
flag points whose residual exceeds ``threshold`` standard deviations --
with two deliberate differences:

- Flagged points are always replaced with NaN. `pyrsktools.RSK.despike`
  also supports replacing with the reference value or a linear
  interpolation; neither is offered here.
- The whole detect-and-replace step runs iteratively, up to
  `DespikeSettings.max_iterations` times, stopping early the first pass
  that flags nothing new -- removing large spikes can reveal or unmask
  smaller ones the previous pass's median/std missed. `pyrsktools.RSK.
  despike` only ever runs one pass.

Like `ct_lag.py`'s `_moving_nanmean`, the rolling median filter here uses
an edge-clipped window (shrinks near the array boundary) instead of
`pyrsktools.utils.runmed`'s mirror-padding -- the same deliberate,
documented divergence from pyrsktools at the edges.
"""

import logging
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from ctd_processing.config import DespikeSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel

logger = logging.getLogger(__name__)

__all__ = ["despike_array", "despike_channel"]


@numba.njit(cache=True)
def _rolling_nanmedian(
    data: npt.NDArray[Any], window_length: int
) -> npt.NDArray[Any]:
    """Center a `window_length`-wide median filter on every sample of `data`.

    For each sample, takes the median of the finite values in the window
    ``[i - half, i + half]``, clipped to the array's bounds -- a shorter
    window right at the edges, rather than `pyrsktools.utils.runmed`'s
    mirror-padding (see module docstring). A window with no finite values
    yields NaN.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to filter. Not mutated.
    window_length : int
        Width of the filter window, in samples. Must be odd.

    Returns
    -------
    numpy.typing.NDArray[Any]
        The filtered array, the same length as `data`.
    """
    n = data.shape[0]
    half = (window_length - 1) // 2
    out = np.empty(n, dtype=np.float64)
    buffer = np.empty(window_length, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        count = 0
        for j in range(lo, hi):
            value = data[j]
            if not np.isnan(value):
                buffer[count] = value
                count += 1
        if count == 0:
            out[i] = np.nan
        else:
            window = buffer[:count]
            window.sort()
            mid = count // 2
            if count % 2 == 1:
                out[i] = window[mid]
            else:
                out[i] = (window[mid - 1] + window[mid]) / 2.0
    return out


def despike_array(
    data: npt.NDArray[Any], settings: DespikeSettings
) -> tuple[npt.NDArray[Any], int]:
    """Replace spikes in `data` with NaN, up to `max_iterations` passes.

    Each pass: smooths `data` with `_rolling_nanmedian` to get a
    reference series, forms the residual (`data - reference`), and flags
    points whose residual magnitude exceeds
    ``settings.threshold * np.nanstd(residual)`` (ignoring non-finite
    residuals). Flagged points are set to NaN. Stops as soon as a pass
    flags nothing new, or after `settings.max_iterations` passes,
    whichever comes first.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The data to despike. Not mutated; a new array is returned.
    settings : DespikeSettings
        Configures the filter window, spike threshold, and iteration
        limit.

    Returns
    -------
    tuple[numpy.typing.NDArray[Any], int]
        The despiked data, and the total number of points replaced
        across every pass.
    """
    working = data.astype(np.float64, copy=True)
    total = 0
    for _ in range(settings.max_iterations):
        reference = _rolling_nanmedian(working, settings.window_length)  # ty: ignore
        residual = working - reference
        sd = np.nanstd(residual)
        if not np.isfinite(sd) or sd == 0:
            break

        finite = np.isfinite(residual)
        spike_mask = np.zeros(residual.shape, dtype=bool)
        spike_mask[finite] = np.abs(residual[finite]) > settings.threshold * sd
        indices = np.flatnonzero(spike_mask)
        if indices.size == 0:
            break

        working[indices] = np.nan
        total += indices.size

    return working, total


def despike_channel(channel: Channel, settings: DespikeSettings) -> Channel:
    """Replace spikes in `channel`'s data with NaN, in place.

    Parameters
    ----------
    channel : Channel
        The channel to despike. `channel.data` is mutated in place; this
        is not a copying operation.
    settings : DespikeSettings
        Forwarded to `despike_array`.

    Returns
    -------
    Channel
        `channel` itself (not a copy), with a new entry appended to
        `channel.history` recording how many points were replaced, if
        any were.

    Raises
    ------
    ValueError
        If `channel.data` is not a floating-point array.
    """
    if not np.issubdtype(channel.data.dtype, np.floating):
        raise ValueError(
            "despike_channel requires floating-point data; got "
            f"{channel.data.dtype}."
        )

    despiked, count = despike_array(channel.data, settings)
    channel.data[:] = despiked
    if count:
        description = f"despiked {count} point(s)"
        channel.record(description)
        log_verbose(logger, description)
    return channel
