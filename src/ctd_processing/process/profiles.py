"""Identify individual profiles (casts) via `profinder`.

Configured via `process.profiles` -- see
`ctd_processing.config.ProfileSettings`. This module only identifies
turnaround-cycle index boundaries; it does not split `dataset` apart. Some
later corrections (e.g. conductivity/temperature lag alignment) need to
know the full cycle's boundaries and run on the full, still-continuous
`Dataset` first, before any extraction -- see `find_profiles`.
`resolve_cast_slices` then decides which cast(s) of a cycle are actually
worth extracting as separate profiles; actual extraction via
`Dataset.subset` is a separate, later step still.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import profinder

from ctd_processing.config import ProfileSettings
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["Profile", "find_profiles", "resolve_cast_slices"]


@dataclass(frozen=True)
class Profile:
    """Index boundaries of one identified turnaround cycle (a down/up pair).

    `profinder` always identifies both the downcast and upcast of a
    cycle, regardless of `ctd_processing.config.ProfileSettings.direction`
    -- that setting only affects which of them `resolve_cast_slices`
    later selects for extraction.

    Attributes
    ----------
    down_start : int
        Index where the downcast begins.
    down_end : int
        Index where the downcast ends.
    up_start : int
        Index where the upcast begins. Equal to `up_end`'s peak index
        (and to `down_end`) unless a speed threshold was applied.
    up_end : int
        Index where the upcast ends.
    """

    down_start: int
    down_end: int
    up_start: int
    up_end: int


def find_profiles(dataset: Dataset, settings: ProfileSettings) -> list[Profile]:
    """Identify turnaround cycles in `dataset` from its `sea_pressure` channel.

    Does not mutate `dataset`; it only computes index boundaries. See the
    module docstring for why extraction is deferred to a later step.
    `settings.direction` is not consulted here -- every cycle's downcast
    and upcast are always both identified; `settings.direction` only
    matters later, to `resolve_cast_slices`.

    Parameters
    ----------
    dataset : Dataset
        The dataset to identify profiles in. Must have a `sea_pressure`
        channel (see `ctd_processing.process.sea_pressure`).
    settings : ProfileSettings
        Parameters forwarded to `profinder.find_profiles`.
        `settings.speed_threshold_direction` (not `settings.direction`)
        is forwarded as `profinder.find_profiles`'s own ``direction``
        argument.

    Returns
    -------
    list[Profile]
        One `Profile` per identified turnaround cycle, in the order
        `profinder` returns them.

    Raises
    ------
    ValueError
        If `dataset` has no `sea_pressure` channel.
    """
    if "sea_pressure" not in dataset.channels:
        raise ValueError(
            "Cannot find profiles: dataset has no sea_pressure channel."
        )

    pressure = dataset.channels["sea_pressure"].data

    time_seconds = None
    if settings.apply_speed_threshold:
        elapsed = dataset.time.data - dataset.time.data[0]
        time_seconds = elapsed / np.timedelta64(1, "s")

    # profinder 0.2.5's missing="drop" path raises UnboundLocalError if
    # `pressure` has no non-finite values to drop (it only initializes its
    # index-remapping array inside the "contains NaN" branch). "raise" and
    # "drop" behave identically when there's nothing to drop, so fall back
    # to "raise" in that case to sidestep the bug without changing behavior.
    missing = settings.missing
    if missing == "drop" and np.isfinite(pressure).all():
        missing = "raise"

    raw_profiles = profinder.find_profiles(
        pressure,
        apply_smoothing=settings.apply_smoothing,
        window_length=settings.window_length,
        polyorder=settings.polyorder,
        min_pressure=settings.min_pressure,
        peaks_kwargs={
            "height": settings.peak_height,
            "distance": settings.peak_distance,
            "width": settings.peak_width,
            "prominence": settings.peak_prominence,
        },
        troughs_kwargs={
            "prominence": settings.trough_prominence,
            "distance": settings.trough_distance,
            "width": settings.trough_width,
        },
        run_length=settings.run_length,
        min_pressure_change=settings.min_pressure_change,
        apply_speed_threshold=settings.apply_speed_threshold,
        time=time_seconds,
        min_speed=settings.min_speed,
        direction=settings.speed_threshold_direction,
        missing=missing,
    )

    profiles = [Profile(*profile) for profile in raw_profiles]
    logger.info("found %d profile(s)", len(profiles))
    return profiles


def resolve_cast_slices(
    profile: Profile, direction: Literal["up", "down", "both"]
) -> list[slice]:
    """Resolve the cast segment(s) of `profile` to extract as profiles.

    Never includes the dwell between `profile.down_end` and
    `profile.up_start` (e.g. time spent at the bottom of a cast) in any
    returned slice, regardless of `direction`.

    Parameters
    ----------
    profile : Profile
        One identified turnaround cycle (see `find_profiles`).
    direction : {"up", "down", "both"}
        Which cast direction(s) to extract, e.g.
        `ctd_processing.config.ProfileSettings.direction`.

    Returns
    -------
    list[slice]
        ``[slice(profile.down_start, profile.down_end)]`` for
        ``"down"``; ``[slice(profile.up_start, profile.up_end)]`` for
        ``"up"``; both, downcast first, for ``"both"``.
    """
    slices = []
    if direction in ("down", "both"):
        slices.append(slice(profile.down_start, profile.down_end))
    if direction in ("up", "both"):
        slices.append(slice(profile.up_start, profile.up_end))
    return slices
