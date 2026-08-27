"""Conductivity/temperature (CT) lag correction, configured via `ct_lag`.

See `ctd_processing.config.CTLagSettings`. This mirrors pyrsktools'
``RSK.calculateCTlag``/``RSK.alignchannel`` -- grid-searching conductivity
shifts and scoring each by the standard deviation of a high-pass-filtered
derived salinity, then applying the best one -- with two deliberate
differences:

- A single, deployment-wide shift is computed from every profile's
  residuals pooled together, rather than one shift per profile.
- The high-pass filter is a numba-jitted moving average instead of
  pyrsktools' ``utils.runavg``, a pure-Python generator that calls
  ``np.nanmean`` sample by sample; that, run once per lag trial per
  profile, is the main reason pyrsktools' version is slow.

This intentionally does not reproduce pyrsktools' tie-break
(``np.min(np.abs(lags[...]))``, which discards the sign of the winning
lag whenever it's the unique minimum); ties here are broken by the
smallest-magnitude *signed* lag instead.
"""

import logging
from typing import Any

import gsw
import numba
import numpy as np
import numpy.typing as npt

from ctd_processing.config import CTLagSettings
from ctd_processing.process._shift import shift_array
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.profiles import Profile
from ctd_processing.process.raw_channels import shift_time

logger = logging.getLogger(__name__)

__all__ = ["calculate_ct_lag", "process_ct_lag"]


@numba.njit(cache=True)
def _moving_nanmean(
    data: npt.NDArray[Any], window_length: int
) -> npt.NDArray[Any]:
    """Center a `window_length`-wide nanmean filter on every sample of `data`.

    For each sample, averages the finite values in the window
    ``[i - half, i + half]``, clipped to the array's bounds -- equivalent
    to pyrsktools' ``utils.runavg`` with NaN edge padding (a shorter
    window right at the edges, since out-of-range positions would be NaN
    and `nanmean` excludes them), but computed as a single compiled pass
    instead of a Python-level generator recomputing `np.nanmean` per
    sample.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to filter. Not mutated.
    window_length : int
        Width of the averaging window, in samples. Must be odd.

    Returns
    -------
    numpy.typing.NDArray[Any]
        The filtered array, the same length as `data`. A sample is NaN
        only if every value in its window is NaN.
    """
    n = data.shape[0]
    half = (window_length - 1) // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        total = 0.0
        count = 0
        for j in range(lo, hi):
            value = data[j]
            if not np.isnan(value):
                total += value
                count += 1
        out[i] = total / count if count > 0 else np.nan
    return out


def calculate_ct_lag(
    dataset: Dataset, profiles: list[Profile], settings: CTLagSettings
) -> int:
    """Calculate a single, deployment-wide conductivity/temperature lag.

    For every candidate lag in ``[settings.min_lag, settings.max_lag]``,
    each profile's `sea_water_electrical_conductivity` is trial-shifted by
    that lag, a practical salinity is derived from it via
    `gsw.SP_from_C` (TEOS-10), and the residual between that salinity and
    its `_moving_nanmean`-smoothed version is computed -- the same
    salinity-spiking score pyrsktools' ``calculateCTlag`` uses. Every
    profile's finite residuals for a given lag are pooled into one array
    before taking its standard deviation, so the result is the single lag
    that minimizes spiking across the whole deployment at once, not the
    best lag for any one profile.

    Parameters
    ----------
    dataset : Dataset
        The dataset to calculate a lag for. Must have
        `sea_water_electrical_conductivity`, `sea_water_temperature`, and
        `sea_pressure` channels.
    profiles : list[Profile]
        Profile boundaries within `dataset` (see
        `ctd_processing.process.profiles.find_profiles`). Each profile's
        full cast span (``down_start`` through ``up_end``) is used.
    settings : CTLagSettings
        `sea_pressure_min`/`sea_pressure_max` restrict which samples of
        each profile feed the search; `window_length`, `min_lag`, and
        `max_lag` configure it. `settings.enabled` is not consulted here
        -- that gate belongs to the caller (`process_ct_lag`).

    Returns
    -------
    int
        The lag, in samples, that minimizes pooled residual salinity
        spiking across every profile. Ties are broken by the
        smallest-magnitude signed lag.

    Raises
    ------
    ValueError
        If `dataset` is missing any of the required channels, or if no
        candidate lag produces any finite residual at all (e.g. `profiles`
        is empty, or `sea_pressure_min`/`sea_pressure_max` excludes every
        sample).
    """
    for name in (
        "sea_water_electrical_conductivity",
        "sea_water_temperature",
        "sea_pressure",
    ):
        if name not in dataset.channels:
            raise ValueError(
                f"Cannot calculate CT lag: dataset has no {name} channel."
            )

    conductivity = dataset.channels["sea_water_electrical_conductivity"].data
    temperature = dataset.channels["sea_water_temperature"].data
    sea_pressure = dataset.channels["sea_pressure"].data

    pressure_min = (
        settings.sea_pressure_min
        if settings.sea_pressure_min is not None
        else -np.inf
    )
    pressure_max = (
        settings.sea_pressure_max
        if settings.sea_pressure_max is not None
        else np.inf
    )
    spans = [slice(profile.down_start, profile.up_end) for profile in profiles]

    lags = list(range(settings.min_lag, settings.max_lag + 1))
    scores = np.full(len(lags), np.inf)
    for lag_index, lag in enumerate(lags):
        residuals = []
        for span in spans:
            c = conductivity[span]
            t = temperature[span]
            sp = sea_pressure[span]
            in_range = (sp >= pressure_min) & (sp <= pressure_max)
            if not in_range.all():
                c, t, sp = c[in_range], t[in_range], sp[in_range]
            if c.size == 0:
                continue

            shifted_c = shift_array(c, lag)
            salinity = gsw.SP_from_C(shifted_c, t, sp)
            smoothed = _moving_nanmean(salinity, settings.window_length)  # ty: ignore
            residual = salinity - smoothed
            finite_residual = residual[np.isfinite(residual)]
            if finite_residual.size:
                residuals.append(finite_residual)

        if residuals:
            scores[lag_index] = np.std(np.concatenate(residuals))

    if not np.isfinite(scores).any():
        raise ValueError(
            "Cannot calculate CT lag: no finite salinity residuals in any "
            "profile (empty profiles, or sea_pressure_min/sea_pressure_max "
            "excluded every sample)."
        )

    best_score = np.min(scores)
    candidate_lags = np.array(lags)[scores == best_score]
    best_lag = int(candidate_lags[np.argmin(np.abs(candidate_lags))])

    logger.info("Calculated CT lag: %d sample(s)", best_lag)
    return best_lag


def process_ct_lag(
    dataset: Dataset, profiles: list[Profile], settings: CTLagSettings
) -> Dataset:
    """Compute and apply the configured CT lag correction to `dataset`.

    If `settings.enabled` is ``False``, `dataset` is returned unchanged.
    Otherwise, calculates the lag via `calculate_ct_lag` and applies it to
    the `sea_water_electrical_conductivity` channel via
    `ctd_processing.process.raw_channels.shift_time`, which already
    records the shift in the channel's history and logs it at `VERBOSE`
    -- no additional logging of the mutation is needed here.

    Parameters
    ----------
    dataset : Dataset
        The dataset to correct. Mutated in place when `settings.enabled`
        is ``True``; see `calculate_ct_lag` for the required channels.
    profiles : list[Profile]
        Profile boundaries within `dataset`, forwarded to
        `calculate_ct_lag`.
    settings : CTLagSettings
        Whether and how to calculate and apply the lag.

    Returns
    -------
    Dataset
        `dataset` itself (not a copy).
    """
    if not settings.enabled:
        logger.info("Skipping CT lag correction (not enabled).")
        return dataset

    lag = calculate_ct_lag(dataset, profiles, settings)
    shift_time(dataset.channels["sea_water_electrical_conductivity"], lag)
    return dataset
