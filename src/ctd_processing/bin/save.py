"""Name and write the combined, binned dataset for one deployment.

Mirrors `ctd_processing.process.save`'s naming/writing split, one stage
later in the pipeline: `binned_filename` names the single combined output
file for a deployment (as opposed to one file per profile), and
`save_binned_dataset` writes it via `xarray`, configured by
`ctd_processing.config.BinSettings.output_format`.
"""

import logging
from pathlib import Path
from typing import Literal

import xarray as xr

from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["binned_filename", "save_binned_dataset"]


def binned_filename(dataset: Dataset, extension: str) -> str:
    """Build a filename for the combined, binned dataset of one deployment.

    Parameters
    ----------
    dataset : Dataset
        Any one of the deployment's profiles, supplying
        ``instrument_serial_number`` and ``source_file`` from its
        `Dataset.metadata`.
    extension : str
        The filename extension to use, without a leading dot (e.g.
        ``"nc"`` or ``"zarr"``).

    Returns
    -------
    str
        E.g. ``"208532_243188_20260809_0304_binned.nc"``.
    """
    serial_number = dataset.metadata["instrument_serial_number"]
    deployment_stem = Path(dataset.metadata["source_file"]).stem
    return f"{serial_number}_{deployment_stem}_binned.{extension}"


def save_binned_dataset(
    dataset: xr.Dataset,
    directory: Path,
    filename: str,
    output_format: Literal["netcdf", "zarr"],
) -> Path:
    """Write the combined, binned dataset to `directory`.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to write, as returned by
        `ctd_processing.bin.binning.combine_binned_profiles`.
    directory : pathlib.Path
        Directory to write into. Created (including any missing parents)
        if it does not already exist.
    filename : str
        Filename to write to, as returned by `binned_filename`.
    output_format : {"netcdf", "zarr"}
        ``"netcdf"`` writes via `xarray.Dataset.to_netcdf` with the
        ``h5netcdf`` engine (matching
        `ctd_processing.process.save_netcdf.write_netcdf`); ``"zarr"``
        writes via `xarray.Dataset.to_zarr`.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if output_format == "netcdf":
        dataset.to_netcdf(path, engine="h5netcdf")
    else:
        dataset.to_zarr(path, mode="w")
    log_verbose(logger, "wrote %s binned dataset: %s", output_format, path)
    return path
