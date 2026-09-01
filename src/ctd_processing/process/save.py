"""Save and load one profile file, configured via `process.profile_format`.

See `ctd_processing.config.ProcessSettings.profile_format`. `save_profile`
writes one already-extracted, already-processed profile `Dataset` out via
`ctd_processing.process.save_netcdf.write_netcdf` or
`ctd_processing.process.save_parquet.write_parquet`, into a per-deployment
subdirectory of the given profiles directory. `load_profile` reverses that:
it reads a single saved profile file back into a `Dataset`. Extracting a
profile out of a full deployment `Dataset` (`Dataset.subset`) and processing
it (see `ctd_processing.process.process_profile`) both happen before
`save_profile` is called -- see `ctd_processing.process.process_deployment`.
"""

import logging
from pathlib import Path

from ctd_processing.config import ProcessSettings
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.save_netcdf import read_netcdf, write_netcdf
from ctd_processing.process.save_parquet import read_parquet, write_parquet

logger = logging.getLogger(__name__)

__all__ = ["load_profile", "profile_filename", "save_profile"]


def profile_filename(dataset: Dataset, index: int, extension: str) -> str:
    """Build the filename for one extracted profile.

    Parameters
    ----------
    dataset : Dataset
        The full deployment dataset the profile was extracted from,
        supplying its deployment stem via ``source_file`` in
        `Dataset.metadata`.
    index : int
        The profile's 0-based position within `dataset`.
    extension : str
        The filename extension to use, without a leading dot (e.g.
        ``"parquet"`` or ``"nc"``).

    Returns
    -------
    str
        E.g. ``"243188_20260809_0304_p0000.parquet"``.
    """
    deployment_stem = Path(dataset.metadata["source_file"]).stem
    return f"{deployment_stem}_p{index:04d}.{extension}"


def save_profile(
    dataset: Dataset,
    profile_dataset: Dataset,
    index: int,
    directory: Path,
    process_settings: ProcessSettings,
) -> Path:
    """Write one already-extracted, already-processed profile under `directory`.

    Purely responsible for naming and writing `profile_dataset` -- by the
    time this is called, `profile_dataset` has already been extracted from
    the full deployment `Dataset` (`Dataset.subset`) and processed (see
    `ctd_processing.process.process_profile`), typically by
    `ctd_processing.process.process_deployment`. Written into a
    subdirectory of `directory` named after the deployment's ``.rsk`` stem
    (see `profile_filename`), so every profile from one deployment lands
    together -- the shape `ctd_processing.cli.bin.bin_command` expects for
    its own `input_path`.

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
    directory : pathlib.Path
        Base profiles directory. The profile is actually written into
        ``directory / deployment_stem``, created (including any missing
        parents) if it does not already exist.
    process_settings : ProcessSettings
        Supplies `process_settings.profile_format` (the file format to
        write the profile as) and, forwarded to `write_netcdf`/
        `write_parquet`, `process_settings.output_dtype`/`channels` (each
        channel's output dtype -- see
        `ctd_processing.config.resolve_output_dtype`).

    Returns
    -------
    pathlib.Path
        The path the profile was written to.
    """
    deployment_stem = Path(dataset.metadata["source_file"]).stem
    deployment_directory = directory / deployment_stem
    deployment_directory.mkdir(parents=True, exist_ok=True)
    format = process_settings.profile_format
    extension = "nc" if format == "netcdf" else "parquet"

    filename = profile_filename(dataset, index, extension)
    path = deployment_directory / filename
    if format == "netcdf":
        write_netcdf(profile_dataset, path, process_settings)
    else:
        write_parquet(profile_dataset, path, process_settings)
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
