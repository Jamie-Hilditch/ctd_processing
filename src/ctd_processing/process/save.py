"""Save and load one profile file, configured via `process.profile_format`.

See `ctd_processing.config.ProcessSettings.profile_format`. `save_profile`
writes one already-extracted, already-processed profile `Dataset` out via
`ctd_processing.process.save_netcdf.write_netcdf` or
`ctd_processing.process.save_parquet.write_parquet`. `load_profile` reverses
that: it reads a single saved profile file back into a `Dataset`. Extracting
a profile out of a full deployment `Dataset` (`Dataset.subset`) and
processing it (see `ctd_processing.process.process_profile`) both happen
before `save_profile` is called -- see
`ctd_processing.process.process_deployment`.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_netcdf import read_netcdf, write_netcdf
from ctd_processing.process.save_parquet import read_parquet, write_parquet

logger = logging.getLogger(__name__)

__all__ = ["load_profile", "profile_filename", "save_profile"]


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


def save_profile(
    dataset: Dataset,
    profile_dataset: Dataset,
    index: int,
    total: int,
    directory: Path,
    format: Literal["netcdf", "parquet"],
) -> Path:
    """Write one already-extracted, already-processed profile to `directory`.

    Purely responsible for naming and writing `profile_dataset` -- by the
    time this is called, `profile_dataset` has already been extracted from
    the full deployment `Dataset` (`Dataset.subset`) and processed (see
    `ctd_processing.process.process_profile`), typically by
    `ctd_processing.process.process_deployment`.

    Parameters
    ----------
    dataset : Dataset
        The full deployment dataset `profile_dataset` was extracted from,
        forwarded to `profile_filename`.
    profile_dataset : Dataset
        The profile to write.
    index : int
        The profile's 0-based position within `dataset`, forwarded to
        `profile_filename`.
    total : int
        The number of profiles identified in `dataset`, forwarded to
        `profile_filename`.
    directory : pathlib.Path
        Directory to write the profile file into. Created (including any
        missing parents) if it does not already exist.
    format : {"netcdf", "parquet"}
        File format to write the profile as (see
        `ctd_processing.config.ProcessSettings.profile_format`).

    Returns
    -------
    pathlib.Path
        The path the profile was written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    extension = "nc" if format == "netcdf" else "parquet"

    filename = profile_filename(
        dataset, profile_dataset, index, total, extension
    )
    path = directory / filename
    if format == "netcdf":
        write_netcdf(profile_dataset, path)
    else:
        write_parquet(profile_dataset, path)
    return path


def load_profile(path: Path) -> Dataset:
    """Load one profile file, written by `save_profile`, into a `Dataset`.

    Dispatches on `path`'s suffix to
    `ctd_processing.process.save_netcdf.read_netcdf` (``.nc``) or
    `ctd_processing.process.save_parquet.read_parquet` (``.parquet``) --
    the inverse of `save_profile`'s own dispatch on `format`.

    Parameters
    ----------
    path : pathlib.Path
        Path to a profile file written by `save_profile`.

    Returns
    -------
    Dataset
        The reconstructed profile dataset.

    Raises
    ------
    ValueError
        If `path`'s suffix is neither ``.nc`` nor ``.parquet``.
    """
    if path.suffix == ".nc":
        return read_netcdf(path)
    if path.suffix == ".parquet":
        return read_parquet(path)
    raise ValueError(
        f"Unrecognized profile file extension {path.suffix!r} for {path}; "
        "expected '.nc' or '.parquet'."
    )
