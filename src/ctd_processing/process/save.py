"""Extract and save profiles, configured via `process.profile_format`.

See `ctd_processing.config.ProcessSettings.profile_format`. This slices each
identified `Profile` out of the full deployment `Dataset` (see
`ctd_processing.process.profiles.find_profiles`) and writes it out via
`ctd_processing.process.save_netcdf.write_netcdf` or
`ctd_processing.process.save_parquet.write_parquet`.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from ctd_processing.config import GeolocationSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.geolocation import attach_geolocation
from ctd_processing.process.profiles import Profile
from ctd_processing.process.save_netcdf import write_netcdf
from ctd_processing.process.save_parquet import write_parquet

logger = logging.getLogger(__name__)

__all__ = ["profile_filename", "save_profiles"]


def profile_filename(
    dataset: Dataset,
    profile_dataset: Dataset,
    index: int,
    total: int,
    extension: str,
) -> str:
    """Build a unique, informative filename for one extracted profile.

    Parameters
    ----------
    dataset : Dataset
        The full deployment dataset the profile was extracted from,
        supplying ``instrument_serial_number`` and ``source_file`` from
        its `Dataset.metadata`.
    profile_dataset : Dataset
        The extracted profile itself, supplying its first timestamp.
    index : int
        The profile's 0-based position within `dataset`.
    total : int
        The number of profiles identified in `dataset`. Only used to
        size `index`'s zero-padding so it never truncates.
    extension : str
        The filename extension to use, without a leading dot (e.g.
        ``"parquet"`` or ``"nc"``).

    Returns
    -------
    str
        E.g. ``"208532_243188_20260809_0304_p000_20260809T030412.parquet"``.
    """
    serial_number = dataset.metadata["instrument_serial_number"]
    deployment_stem = Path(dataset.metadata["source_file"]).stem
    width = max(3, len(str(total)))
    start = np.datetime_as_string(profile_dataset.time.data[0], unit="s")
    start = start.replace("-", "").replace(":", "")
    return (
        f"{serial_number}_{deployment_stem}_p{index:0{width}d}_{start}"
        f".{extension}"
    )


def save_profiles(
    dataset: Dataset,
    profiles: list[Profile],
    directory: Path,
    format: Literal["netcdf", "parquet"],
    geolocation: GeolocationSettings,
) -> list[Path]:
    """Extract every profile from `dataset` and save it to `directory`.

    Each `Profile`'s full cast span (``down_start`` through ``up_end`` --
    the same span `ctd_processing.process.ct_lag.calculate_ct_lag` uses) is
    extracted via `Dataset.subset`, given a position via
    `ctd_processing.process.geolocation.attach_geolocation`, named via
    `profile_filename`, and written out in `format`. If
    `geolocation.external_dataset_path` is set, that dataset is opened once
    for this call and reused across every profile, rather than once per
    profile.

    Parameters
    ----------
    dataset : Dataset
        The full deployment dataset to extract profiles from.
    profiles : list[Profile]
        Profile boundaries within `dataset` (see
        `ctd_processing.process.profiles.find_profiles`).
    directory : pathlib.Path
        Directory to write profile files into. Created (including any
        missing parents) if it does not already exist.
    format : {"netcdf", "parquet"}
        File format to write each profile as (see
        `ctd_processing.config.ProcessSettings.profile_format`).
    geolocation : GeolocationSettings
        Configures the position attached to each profile (see
        `ctd_processing.process.geolocation.attach_geolocation`).

    Returns
    -------
    list[pathlib.Path]
        The path each profile was written to, in `profiles` order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    extension = "nc" if format == "netcdf" else "parquet"

    total = len(profiles)
    paths = []
    external_dataset = None
    if geolocation.external_dataset_path is not None:
        external_dataset = xr.open_dataset(geolocation.external_dataset_path)
    try:
        for index, profile in enumerate(profiles):
            description = (
                f"extracted profile {index + 1} of {total} "
                f"(samples {profile.down_start}:{profile.up_end})"
            )
            profile_dataset = dataset.subset(
                slice(profile.down_start, profile.up_end), description
            )
            log_verbose(logger, description)
            profile_dataset = attach_geolocation(
                profile_dataset, geolocation, external_dataset
            )

            filename = profile_filename(
                dataset, profile_dataset, index, total, extension
            )
            path = directory / filename
            if format == "netcdf":
                write_netcdf(profile_dataset, path)
            else:
                write_parquet(profile_dataset, path)
            paths.append(path)
    finally:
        if external_dataset is not None:
            external_dataset.close()

    return paths
