"""Despiking, configured via `process.despiking` and `process.channels`.

See `ctd_processing.config.DespikeSettings` and
`ctd_processing.config.resolve_despike_settings`. Smooths a channel with a
rolling median filter to get a "reference" series, forms the residual
against it, and flags points whose residual magnitude exceeds a rejection
threshold -- resolved per `DespikeSettings.rejection_method` -- replacing
them with NaN. Loosely based on `pyrsktools.RSK.despike`'s
reference-then-threshold shape, with deliberate differences:

- The default rejection rule (`rejection_method="gpd_percentile"`) fits a
  Generalized Pareto Distribution to the residual's tail via
  peaks-over-threshold, rather than assuming the residual is Gaussian (an
  "N x MAD"/"N x std" rule). Real residuals (e.g. `practical_salinity`
  against a rolling median) are often heavy-tailed / power-law-like, not
  Gaussian, and a fixed-multiplier rule miscalibrated for that shape either
  misses real spikes or floods the output with false positives. The
  Pickands-Balkema-de Haan theorem guarantees the distribution of
  exceedances above a high-enough threshold converges to a GPD regardless
  of the parent distribution's shape, which is what makes this a
  principled default rather than another fixed-shape assumption.
  `rejection_method` also offers two escape hatches (`"raw_percentile"`,
  `"absolute_threshold"`) for users manually tuning despiking who want to
  skip the GPD fit.
- `despike_arrays` can despike several related arrays (e.g. one per cast in
  a deployment) together, fitting a single rejection threshold pooled
  across all of them -- a GPD fit is more stable with more data, which is
  why `ctd_processing.process.derived_variables` pools every cast in a
  deployment. `despike_array` is the single-array case (used for raw
  channels, already deployment-wide as a single array).
- There is no `iterations`/repeat-pass option. The old MAD-based design
  could safely re-fit after masking, since MAD isn't inflated by the
  spikes it measures. A GPD fit is different: it depends on the most
  extreme points in the tail to estimate `xi`/`sigma`. Masking those out
  and refitting would remove exactly the evidence the fit needs, biasing
  it the same way fitting to a truncated/windowed tail does (see
  `ctd_processing.config.DespikeSettings.rejection_method`'s docstring).
  Despiking is a single detect-and-replace pass.
- Flagged points are always replaced with NaN. `pyrsktools.RSK.despike`
  also supports replacing with the reference value or a linear
  interpolation; neither is offered here.

Like `ct_lag.py`'s `_moving_nanmean`, the rolling median filter here uses an
edge-clipped window (shrinks near the array boundary) instead of
`pyrsktools.utils.runmed`'s mirror-padding -- the same deliberate,
documented divergence from pyrsktools at the edges.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
from scipy import stats

from ctd_processing.config import DespikeSettings, RejectionMethod
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel

logger = logging.getLogger(__name__)

__all__ = [
    "RejectionThresholdResult",
    "describe_rejection_threshold",
    "despike_array",
    "despike_arrays",
    "despike_channel",
    "resolve_rejection_threshold",
]


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


@dataclass(frozen=True)
class RejectionThresholdResult:
    """The resolved rejection threshold for one despike pass, and how.

    Returned by `resolve_rejection_threshold`; carried through
    `despike_arrays`/`despike_array` so callers can log/record what
    actually happened, not just the outcome.

    Attributes
    ----------
    threshold : float
        Points whose residual magnitude exceeds this are flagged as
        spikes. ``numpy.nan`` if no threshold could be resolved (e.g. no
        finite residuals at all under `"gpd_percentile"`/`"raw_percentile"`).
    method : ctd_processing.config.RejectionMethod
        Which `DespikeSettings.rejection_method` produced `threshold`.
    used_fallback : bool
        Only meaningful when `method` is ``"gpd_percentile"``: whether the
        GPD fit was skipped in favor of a plain empirical percentile,
        either because too few points exceeded
        `DespikeSettings.fit_percentile` (see
        `DespikeSettings.min_exceedances`) or because there was no finite
        data at all.
    shape : float or None
        The fitted GPD shape parameter (``xi``). Only set when `method` is
        ``"gpd_percentile"`` and `used_fallback` is `False`.
    scale : float or None
        The fitted GPD scale parameter (``sigma``). Only set when `method`
        is ``"gpd_percentile"`` and `used_fallback` is `False`.
    n : int or None
        Number of finite ``|residual|`` values `threshold` was resolved
        from. ``None`` for `"absolute_threshold"`, which uses no data.
    n_exceedances : int or None
        Number of points exceeding `DespikeSettings.fit_percentile`'s
        threshold. Only meaningful when `method` is ``"gpd_percentile"``.
    """

    threshold: float
    method: RejectionMethod
    used_fallback: bool
    shape: float | None
    scale: float | None
    n: int | None
    n_exceedances: int | None


def _fit_gpd_threshold(
    abs_residual: npt.NDArray[Any], settings: DespikeSettings
) -> RejectionThresholdResult:
    """Fit a GPD to `abs_residual`'s tail via peaks-over-threshold.

    Solves for the value ``x`` such that
    ``P(|residual| > x) = 1 - settings.rejection_percentile / 100`` under
    the fitted tail. Falls back to a plain empirical percentile (no
    extrapolation) if there are fewer than `settings.min_exceedances`
    points above the `settings.fit_percentile` threshold, or no finite
    data at all -- the GPD's extrapolation is only trustworthy with enough
    exceedances to fit `xi`/`sigma` stably.

    Parameters
    ----------
    abs_residual : numpy.typing.NDArray[Any]
        Finite ``|residual|`` values, pooled across however many arrays
        this fit spans. Must contain no NaN/inf.
    settings : DespikeSettings
        Supplies `fit_percentile`, `rejection_percentile`, and
        `min_exceedances`.

    Returns
    -------
    RejectionThresholdResult
        `method` is always ``"gpd_percentile"``.
    """
    n = abs_residual.size
    if n == 0:
        return RejectionThresholdResult(
            threshold=np.nan,
            method="gpd_percentile",
            used_fallback=True,
            shape=None,
            scale=None,
            n=0,
            n_exceedances=0,
        )

    u = np.percentile(abs_residual, settings.fit_percentile)
    exceedances = abs_residual[abs_residual > u] - u
    n_exceedances = exceedances.size
    if n_exceedances < settings.min_exceedances:
        threshold = float(
            np.percentile(abs_residual, settings.rejection_percentile)
        )
        return RejectionThresholdResult(
            threshold=threshold,
            method="gpd_percentile",
            used_fallback=True,
            shape=None,
            scale=None,
            n=n,
            n_exceedances=n_exceedances,
        )

    shape, _, scale = stats.genpareto.fit(exceedances, floc=0)
    exceed_prob = (1 - settings.rejection_percentile / 100) * n / n_exceedances
    if abs(shape) < 1e-8:
        threshold = float(u + scale * (-np.log(exceed_prob)))
    else:
        threshold = float(u + (scale / shape) * (exceed_prob ** (-shape) - 1))
    return RejectionThresholdResult(
        threshold=threshold,
        method="gpd_percentile",
        used_fallback=False,
        shape=float(shape),
        scale=float(scale),
        n=n,
        n_exceedances=n_exceedances,
    )


def resolve_rejection_threshold(
    residual: npt.NDArray[Any], settings: DespikeSettings
) -> RejectionThresholdResult:
    """Resolve a rejection threshold for `residual` per `settings`.

    Dispatches on `settings.rejection_method`:

    - ``"absolute_threshold"``: returns `settings.rejection_threshold`
      verbatim; `residual` is not touched.
    - ``"raw_percentile"``: ``numpy.percentile(|residual|,
      settings.rejection_percentile)``, no tail extrapolation.
    - ``"gpd_percentile"`` (default): see `_fit_gpd_threshold`.

    Parameters
    ----------
    residual : numpy.typing.NDArray[Any]
        ``data - reference``, possibly pooled across several arrays (e.g.
        every cast in a deployment). May contain NaN/inf (ignored).
    settings : DespikeSettings
        Selects and configures the rejection method.

    Returns
    -------
    RejectionThresholdResult
    """
    if settings.rejection_method == "absolute_threshold":
        assert settings.rejection_threshold is not None  # validated
        return RejectionThresholdResult(
            threshold=settings.rejection_threshold,
            method="absolute_threshold",
            used_fallback=False,
            shape=None,
            scale=None,
            n=None,
            n_exceedances=None,
        )

    abs_residual = np.abs(residual[np.isfinite(residual)])
    if settings.rejection_method == "raw_percentile":
        threshold = (
            float(np.percentile(abs_residual, settings.rejection_percentile))
            if abs_residual.size
            else np.nan
        )
        return RejectionThresholdResult(
            threshold=threshold,
            method="raw_percentile",
            used_fallback=False,
            shape=None,
            scale=None,
            n=abs_residual.size,
            n_exceedances=None,
        )

    return _fit_gpd_threshold(abs_residual, settings)


def despike_arrays(
    data_list: list[npt.NDArray[Any]], settings: DespikeSettings
) -> tuple[list[npt.NDArray[Any]], list[int], RejectionThresholdResult]:
    """Replace spikes with NaN across several related arrays at once.

    Each array is smoothed with `_rolling_nanmedian` over
    `settings.reference_window_length` independently to get its own
    reference/residual -- the reference never crosses array boundaries.
    Every array's finite residuals are then pooled and fed to one shared
    `resolve_rejection_threshold` call, and each array is flagged/replaced
    independently against that one threshold. A single, non-iterated pass
    (see module docstring for why).

    Parameters
    ----------
    data_list : list[numpy.typing.NDArray[Any]]
        The arrays to despike, e.g. one per cast in a deployment. Not
        mutated; new arrays are returned. May be empty.
    settings : DespikeSettings
        Configures the reference window and rejection method.

    Returns
    -------
    tuple[list[numpy.typing.NDArray[Any]], list[int], RejectionThresholdResult]
        The despiked arrays (same order/lengths as `data_list`), the
        number of points replaced in each, and the threshold resolution
        shared across all of them.
    """
    working = [d.astype(np.float64, copy=True) for d in data_list]
    residuals = [
        w - _rolling_nanmedian(w, settings.reference_window_length)  # ty: ignore
        for w in working
    ]
    pooled = (
        np.concatenate(residuals)
        if residuals
        else np.array([], dtype=np.float64)
    )
    result = resolve_rejection_threshold(pooled, settings)

    counts = [0] * len(working)
    if np.isfinite(result.threshold):
        for i, residual in enumerate(residuals):
            valid = np.isfinite(residual)
            spike_mask = np.zeros(residual.shape, dtype=bool)
            spike_mask[valid] = np.abs(residual[valid]) > result.threshold
            indices = np.flatnonzero(spike_mask)
            if indices.size:
                working[i][indices] = np.nan
                counts[i] = indices.size

    return working, counts, result


def despike_array(
    data: npt.NDArray[Any], settings: DespikeSettings
) -> tuple[npt.NDArray[Any], int, RejectionThresholdResult]:
    """Replace spikes in `data` with NaN.

    Single-array case of `despike_arrays`, for a channel that is already
    despiked as one whole array (e.g. a raw channel, despiked deployment-
    wide before any cast splitting).

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The data to despike. Not mutated; a new array is returned.
    settings : DespikeSettings
        Configures the reference window and rejection method.

    Returns
    -------
    tuple[numpy.typing.NDArray[Any], int, RejectionThresholdResult]
        The despiked data, the number of points replaced, and the
        threshold resolution.
    """
    despiked, counts, result = despike_arrays([data], settings)
    return despiked[0], counts[0], result


def describe_rejection_threshold(
    settings: DespikeSettings, result: RejectionThresholdResult
) -> str:
    """Describe how a rejection threshold was resolved, for logging/history.

    Shared by `despike_channel` and
    `ctd_processing.process.derived_variables` so every recorded/logged
    despike description carries the same threshold detail regardless of
    call site -- each caller wraps this in its own ``"despiked N point(s)
    (...)"``-shaped message, since how many points that threshold actually
    flagged is call-site-specific (e.g. one shared pooled fit, many
    per-profile counts).

    Parameters
    ----------
    settings : DespikeSettings
        The settings the threshold in `result` was resolved under.
    result : RejectionThresholdResult
        The resolved threshold, as returned by `despike_array`/
        `despike_arrays`.

    Returns
    -------
    str
        E.g. ``"GPD fit: threshold=0.041, xi=0.52, sigma=0.0031, n=1200,
        n_exceedances=120"``, with no surrounding parentheses.
    """
    if result.method == "absolute_threshold":
        return f"absolute threshold={result.threshold:g}"
    if result.method == "raw_percentile":
        return (
            f"raw percentile: threshold={result.threshold:g}, "
            f"percentile={settings.rejection_percentile:g}, n={result.n}"
        )
    if result.used_fallback:
        return (
            "empirical percentile fallback: "
            f"threshold={result.threshold:g}, n={result.n}, "
            f"n_exceedances={result.n_exceedances} < "
            f"min_exceedances={settings.min_exceedances}"
        )
    return (
        f"GPD fit: threshold={result.threshold:g}, xi={result.shape:g}, "
        f"sigma={result.scale:g}, n={result.n}, "
        f"n_exceedances={result.n_exceedances}"
    )


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
        `channel.history` recording how many points were replaced and how
        the rejection threshold was resolved, if any points were replaced.
        The outcome is logged at `VERBOSE` regardless -- including a
        replaced count of ``0`` -- so a run that despiked nothing is still
        visible in the log, not just a silent no-op.

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

    despiked, count, result = despike_array(channel.data, settings)
    channel.data[:] = despiked
    detail = describe_rejection_threshold(settings, result)
    description = f"despiked {count} point(s) ({detail})"
    if count:
        channel.record(description)
    log_verbose(logger, description)
    return channel
