"""Write and load the combined, binned dataset for one deployment.

One stage later in the pipeline than `ctd_processing.process.save`:
`save_binned_dataset` writes the single combined output file for a
deployment (as opposed to one file per profile) via `xarray`, configured
by `ctd_processing.config.BinSettings.output_format` and compressed per
`ctd_processing.config.BinSettings.netcdf_compression`/`zarr_compression`.
Naming is the caller's responsibility (see
`ctd_processing.cli.bin.bin_command`, which names it after the deployment
stem). `load_binned_dataset` reverses that, for
`ctd_processing.cli.concatenate.concatenate_command` to read every
deployment's binned file back in before concatenating them.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

from ctd_processing.config import BinSettings, ZarrCompressionSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.save_netcdf import netcdf_compression_encoding

logger = logging.getLogger(__name__)

__all__ = ["load_binned_dataset", "save_binned_dataset"]


def _zarr_compression_encoding(
    dataset: xr.Dataset, settings: ZarrCompressionSettings
) -> dict[str, dict[str, list[BloscCodec]]]:
    """Build `xarray.Dataset.to_zarr`'s encoding for every float variable.

    Every coordinate is left out of the returned mapping entirely, so
    `xarray.Dataset.to_zarr` falls back to zarr's own default codec for
    it -- matching `ctd_processing.process.save_netcdf.
    netcdf_compression_encoding`'s treatment of ``time``.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset about to be written; only `dataset.data_vars` is
        considered.
    settings : ctd_processing.config.ZarrCompressionSettings
        `enabled`/`cname`/`clevel`/`shuffle` to encode.

    Returns
    -------
    dict[str, dict[str, list]]
        The `encoding` mapping for `xarray.Dataset.to_zarr`. Each
        floating-point data variable maps to an explicit `BloscCodec`
        when `settings.enabled` is ``True``, or to an explicit empty
        ``compressors`` list (zarr's raw bytes codec only, i.e.
        uncompressed) when ``False``.
    """
    encoding: dict[str, dict[str, list[BloscCodec]]] = {}
    for name, variable in dataset.data_vars.items():
        if not np.issubdtype(variable.dtype, np.floating):
            continue
        if settings.enabled:
            encoding[str(name)] = {
                "compressors": [
                    BloscCodec(
                        cname=settings.cname,
                        clevel=settings.clevel,
                        shuffle=settings.shuffle,
                    )
                ]
            }
        else:
            encoding[str(name)] = {"compressors": []}
    return encoding


def save_binned_dataset(
    dataset: xr.Dataset,
    directory: Path,
    filename: str,
    bin_settings: BinSettings,
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
        Filename to write to, e.g. ``f"{deployment_stem}.{extension}"``.
    bin_settings : ctd_processing.config.BinSettings
        Supplies `output_format` -- ``"netcdf"`` writes via
        `xarray.Dataset.to_netcdf` with the ``h5netcdf`` engine
        (matching `ctd_processing.process.save_netcdf.write_netcdf`),
        compressed per `netcdf_compression`; ``"zarr"`` writes via
        `xarray.Dataset.to_zarr`, compressed per `zarr_compression`.
        Every floating-point data variable is compressed; every
        coordinate is always left uncompressed.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if bin_settings.output_format == "netcdf":
        encoding: dict[str, dict[str, object]] = {}
        for name, variable in dataset.data_vars.items():
            variable_encoding = netcdf_compression_encoding(
                variable.dtype, bin_settings.netcdf_compression
            )
            if variable_encoding is not None:
                encoding[str(name)] = variable_encoding
        dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    else:
        dataset.to_zarr(
            path,
            mode="w",
            encoding=_zarr_compression_encoding(
                dataset, bin_settings.zarr_compression
            ),
        )
    log_verbose(
        logger, "wrote %s binned dataset: %s", bin_settings.output_format, path
    )
    return path


def load_binned_dataset(
    path: Path, output_format: Literal["netcdf", "zarr"]
) -> xr.Dataset:
    """Load a combined, binned dataset written by `save_binned_dataset`.

    Parameters
    ----------
    path : pathlib.Path
        Path to load from.
    output_format : {"netcdf", "zarr"}
        Must match the format `path` was actually written in -- the
        same setting `save_binned_dataset` was originally called with
        (see `ctd_processing.config.BinSettings.output_format`). Unlike
        `save_binned_dataset`, no compression settings are needed here:
        both `h5netcdf` and `zarr` decompress transparently regardless
        of which codec/level a file was written with.

    Returns
    -------
    xarray.Dataset
        The loaded dataset, fully read into memory rather than lazily
        backed by the file/store, so `path` is not left open once this
        returns.
    """
    if output_format == "netcdf":
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            return dataset.load()
    with xr.open_zarr(path) as dataset:
        return dataset.load()
