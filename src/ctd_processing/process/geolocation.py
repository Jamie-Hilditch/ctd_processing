"""Attach a canonical time and position to one extracted profile.

Configured via `process.geolocation` (see
`ctd_processing.config.GeolocationSettings`). Every profile is given a
canonical time (its start time) and a position for that time, resolved
either by interpolating an external netCDF position time series or from a
fixed reference position -- see `attach_geolocation`.
"""

import logging

import numpy as np
import numpy.typing as npt
import xarray as xr

from ctd_processing.config import GeolocationSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["attach_geolocation"]


def _interpolate_position(
    canonical_time: npt.NDArray[np.datetime64],
    settings: GeolocationSettings,
    external_dataset: xr.Dataset,
) -> tuple[float, float]:
    """Interpolate a position from `external_dataset` onto `canonical_time`.

    Parameters
    ----------
    canonical_time : numpy.datetime64
        The time to interpolate a position onto.
    settings : GeolocationSettings
        Supplies the `latitude_variable`/`longitude_variable`/
        `time_variable` names to read from `external_dataset`.
    external_dataset : xarray.Dataset
        The already-opened external position dataset (see
        `GeolocationSettings.external_dataset_path`).

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` linearly interpolated onto
        `canonical_time`.

    Raises
    ------
    ValueError
        If `canonical_time` falls outside `external_dataset`'s time
        coverage.
    """
    time = external_dataset[settings.time_variable]
    time_min = time.min().values
    time_max = time.max().values
    if canonical_time < time_min or canonical_time > time_max:
        raise ValueError(
            f"Cannot interpolate position: canonical time {canonical_time} "
            "is outside the external dataset's time coverage "
            f"({time_min} to {time_max})."
        )

    coords = {settings.time_variable: canonical_time}
    latitude = float(
        external_dataset[settings.latitude_variable].interp(coords).item()
    )
    longitude = float(
        external_dataset[settings.longitude_variable].interp(coords).item()
    )
    return latitude, longitude


def attach_geolocation(
    dataset: Dataset,
    settings: GeolocationSettings,
    external_dataset: xr.Dataset | None,
) -> Dataset:
    """Attach a canonical time and position to one extracted profile `Dataset`.

    Records `profile_start_time`/`profile_end_time` (`dataset.time.data`'s
    first/last values) in `dataset.metadata`; the start time is the
    profile's canonical time. Resolves a position for that time --
    interpolated from `external_dataset` if
    `settings.external_dataset_path` is set (see `_interpolate_position`),
    otherwise `settings.reference_latitude`/`reference_longitude` -- and
    records it (`latitude`, `longitude`, `position_source`) in
    `dataset.metadata`.

    Parameters
    ----------
    dataset : Dataset
        One already-extracted profile `Dataset` (see
        `ctd_processing.process.save.save_profiles`). Mutated in place.
    settings : GeolocationSettings
        Configures which position source to use.
    external_dataset : xarray.Dataset or None
        The already-opened external position dataset, when
        `settings.external_dataset_path` is set; otherwise ``None``. The
        caller (`ctd_processing.process.save.save_profiles`) opens this
        once per deployment rather than once per profile.

    Returns
    -------
    Dataset
        `dataset` itself (not a copy).

    Raises
    ------
    ValueError
        If interpolating from `external_dataset` and the profile's
        canonical time falls outside its time coverage.
    """
    start_time = dataset.time.data[0]
    end_time = dataset.time.data[-1]
    dataset.metadata["profile_start_time"] = start_time
    dataset.metadata["profile_end_time"] = end_time

    if settings.external_dataset_path is not None:
        assert external_dataset is not None
        latitude, longitude = _interpolate_position(
            start_time, settings, external_dataset
        )
        source = f"interpolated from {settings.external_dataset_path}"
    else:
        latitude = settings.reference_latitude
        longitude = settings.reference_longitude
        source = "reference position"

    dataset.metadata["latitude"] = latitude
    dataset.metadata["longitude"] = longitude
    dataset.metadata["position_source"] = source

    description = f"attached position ({source})"
    dataset.record(description)
    log_verbose(logger, description)
    return dataset
