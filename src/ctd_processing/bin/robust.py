"""Robust bin-averaging reducers, dispatched per `BinSettings.method`.

See `ctd_processing.config.BinSettings`/`ctd_processing.config.BinMethod`
for the configurable per-channel bin-averaging method, and
`ctd_processing.config.resolve_bin_method` for resolving one channel's
effective method and settings. `reduce_bin` is the single entry point
`ctd_processing.bin.binning.bin_profile` calls once per channel, per bin.

Every reducer here operates on a finite-only 1-D array and returns a
single scalar location estimate. Plain NumPy throughout -- unlike
`ctd_processing.process.despike`'s `numba`-jitted rolling filters, these
run over small arrays (one bin's worth of samples, typically ~10-15)
called many times per profile (once per channel, per bin -- hundreds of
calls), a scale where numba's per-call JIT-dispatch overhead outweighs
any per-call speedup.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from ctd_processing.config import (
    BinMethod,
    BiweightSettings,
    HuberSettings,
    TrimmedMeanSettings,
    WinsorizedMeanSettings,
)

__all__ = [
    "trimmed_mean",
    "winsorized_mean",
    "huber_location",
    "biweight_location",
    "reduce_bin",
]


def trimmed_mean(
    x: npt.NDArray[np.floating], proportion_to_cut: float
) -> float:
    """Mean of `x` after dropping the most extreme values at each tail.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        Finite-only sample values.
    proportion_to_cut : float
        Fraction of points dropped from each tail (by rank) before
        averaging the rest (see `ctd_processing.config.
        TrimmedMeanSettings.proportion_to_cut`).

    Returns
    -------
    float
        The mean of `x` with ``floor(len(x) * proportion_to_cut)``
        values dropped from each end (by sorted rank). Falls back to
        the plain mean if `x` has fewer than 3 elements, or if trimming
        would leave fewer than one point.
    """
    if x.size < 3:
        return float(np.mean(x))
    cut = int(np.floor(x.size * proportion_to_cut))
    if 2 * cut >= x.size:
        return float(np.mean(x))
    sorted_x = np.sort(x)
    return float(np.mean(sorted_x[cut : x.size - cut]))


def winsorized_mean(x: npt.NDArray[np.floating], limits: float) -> float:
    """Mean of `x` after clipping the most extreme values at each tail.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        Finite-only sample values.
    limits : float
        Fraction of points clipped to the nearest retained value at
        each tail (by rank) before averaging (see
        `ctd_processing.config.WinsorizedMeanSettings.limits`).

    Returns
    -------
    float
        The mean of `x` with the most extreme
        ``floor(len(x) * limits)`` values at each tail replaced by the
        nearest retained value (by sorted rank). Falls back to the
        plain mean if `x` has fewer than 3 elements, or if clipping
        would leave fewer than one unclipped point.
    """
    if x.size < 3:
        return float(np.mean(x))
    cut = int(np.floor(x.size * limits))
    if 2 * cut >= x.size:
        return float(np.mean(x))
    sorted_x = np.sort(x)
    if cut > 0:
        sorted_x[:cut] = sorted_x[cut]
        sorted_x[x.size - cut :] = sorted_x[x.size - cut - 1]
    return float(np.mean(sorted_x))


def _irls_location(
    x: npt.NDArray[np.floating],
    weight_fn: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    max_iter: int,
    tol: float,
) -> float:
    """Compute an iteratively reweighted robust location estimate.

    Shared scaffold for `huber_location` and `biweight_location`:
    starting from the median, jointly re-estimates a location and a
    MAD-based scale each pass, weighting each point via `weight_fn` of
    its residual scaled by the current spread estimate.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        Finite-only sample values.
    weight_fn : Callable[[NDArray[floating]], NDArray[floating]]
        Maps scaled residuals ``u = (x - location) / scale`` to
        per-point weights.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance: iteration stops once the location's
        change between passes drops below ``tol * scale``.

    Returns
    -------
    float
        The converged (or `max_iter`-truncated) location estimate.
        Falls back to the plain mean if `x` has fewer than 3 elements.
        If the scale estimate hits ``0`` (all-identical values, or the
        location has converged onto a value shared by at least half the
        sample) or every point loses all weight, returns the location
        as of that iteration rather than dividing by zero or looping
        forever.
    """
    if x.size < 3:
        return float(np.mean(x))

    location = float(np.median(x))
    scale = 1.4826 * float(np.median(np.abs(x - location)))
    for _ in range(max_iter):
        if scale == 0.0:
            return location
        weights = weight_fn((x - location) / scale)
        weight_sum = float(np.sum(weights))
        if weight_sum == 0.0:
            return location
        new_location = float(np.sum(weights * x) / weight_sum)
        new_scale = 1.4826 * float(np.median(np.abs(x - new_location)))
        converged = new_scale == 0.0 or (
            abs(new_location - location) < tol * new_scale
        )
        location, scale = new_location, new_scale
        if converged:
            return location
    return location


def huber_location(
    x: npt.NDArray[np.floating],
    k: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Huber's M-estimator of location.

    See `ctd_processing.config.HuberSettings` and `_irls_location`.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        Finite-only sample values.
    k : float, optional
        Huber's tuning constant, in MAD-based scale units: residuals
        within `k` keep full weight, residuals beyond it are
        downweighted proportionally to ``1 / |residual|``. Defaults to
        ``1.345``.
    max_iter : int, optional
        Maximum number of IRLS iterations. Defaults to ``100``.
    tol : float, optional
        Convergence tolerance. Defaults to ``1e-8``.

    Returns
    -------
    float
        The Huber location estimate. See `_irls_location` for
        degenerate-case behavior.
    """

    def weight_fn(u: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        abs_u = np.abs(u)
        safe_abs_u = np.where(abs_u == 0.0, 1.0, abs_u)
        return np.where(abs_u <= k, 1.0, k / safe_abs_u)

    return _irls_location(x, weight_fn, max_iter, tol)


def biweight_location(
    x: npt.NDArray[np.floating],
    c: float = 4.685,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Tukey's biweight (bisquare) M-estimator of location.

    See `ctd_processing.config.BiweightSettings` and `_irls_location`.
    Unlike `huber_location`, this weight *redescends* to exactly ``0``
    past its cutoff, rather than asymptotically shrinking -- points far
    enough from the current location are fully excluded, not merely
    downweighted.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        Finite-only sample values.
    c : float, optional
        Tukey's biweight tuning constant, in MAD-based scale units:
        residuals beyond `c` get exactly zero weight. Defaults to
        ``4.685``.
    max_iter : int, optional
        Maximum number of IRLS iterations. Defaults to ``100``.
    tol : float, optional
        Convergence tolerance. Defaults to ``1e-8``.

    Returns
    -------
    float
        The biweight location estimate. See `_irls_location` for
        degenerate-case behavior.
    """

    def weight_fn(u: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        abs_u = np.abs(u)
        inside = abs_u <= c
        weights = np.zeros_like(u)
        weights[inside] = (1.0 - (u[inside] / c) ** 2) ** 2
        return weights

    return _irls_location(x, weight_fn, max_iter, tol)


def reduce_bin(
    x: npt.NDArray[np.floating], method: BinMethod, settings: BaseModel | None
) -> float:
    """Reduce one bin's values to a scalar via `method`.

    The dispatcher `ctd_processing.bin.binning.bin_profile` calls once
    per channel, per bin, after resolving that channel's method and
    settings via `ctd_processing.config.resolve_bin_method`.

    Parameters
    ----------
    x : numpy.typing.NDArray[numpy.floating]
        One bin's sample values for one channel. May contain non-finite
        values (e.g. ``numpy.nan``), which are excluded before
        reducing.
    method : BinMethod
        Which reducer to use -- see `ctd_processing.config.BinMethod`.
    settings : pydantic.BaseModel or None
        That method's resolved settings, matching `method` (e.g.
        `ctd_processing.config.HuberSettings` when `method` is
        ``"huber"``) -- ``None`` for ``"mean"``/``"median"``, which take
        no parameters.

    Returns
    -------
    float
        The bin's reduced value, or ``numpy.nan`` if every value in
        `x` is non-finite.

    Raises
    ------
    ValueError
        If `method` is not a recognized `BinMethod`.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float(np.nan)

    if method == "mean":
        return float(np.mean(finite))
    if method == "median":
        return float(np.median(finite))
    if method == "trimmed_mean":
        trimmed_settings = cast(TrimmedMeanSettings, settings)
        return trimmed_mean(finite, trimmed_settings.proportion_to_cut)
    if method == "winsorized_mean":
        winsorized_settings = cast(WinsorizedMeanSettings, settings)
        return winsorized_mean(finite, winsorized_settings.limits)
    if method == "huber":
        huber_settings = cast(HuberSettings, settings)
        return huber_location(
            finite,
            k=huber_settings.k,
            max_iter=huber_settings.max_iter,
            tol=huber_settings.tol,
        )
    if method == "biweight":
        biweight_settings = cast(BiweightSettings, settings)
        return biweight_location(
            finite,
            c=biweight_settings.c,
            max_iter=biweight_settings.max_iter,
            tol=biweight_settings.tol,
        )
    raise ValueError(f"Unknown bin method {method!r}.")
