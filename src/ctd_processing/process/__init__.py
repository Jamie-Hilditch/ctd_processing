"""Processing pipeline: turn raw ``.rsk`` deployments into extracted profiles.

Sister package to :mod:`ctd_processing.cli`. :func:`process_deployment_files`
is the entry point that :mod:`ctd_processing.cli.process` calls with the
full batch of resolved ``.rsk`` deployment files; :func:`process_deployment`
is the per-deployment worker it dispatches to. The rest of this package
holds the supporting implementation (reading deployments, identifying and
extracting profiles, computing TEOS-10 derived variables with `gsw`, and
writing CF-compliant output) that :func:`process_deployment` will grow to
use.
"""

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ctd_processing.config import Settings, resolve_process_settings
from ctd_processing.process.build import build_dataset
from ctd_processing.process.profiles import find_profiles
from ctd_processing.process.raw_channels import process_raw_channels
from ctd_processing.process.read import read_rsk
from ctd_processing.process.sea_pressure import compute_sea_pressure

logger = logging.getLogger(__name__)

__all__ = ["process_deployment", "process_deployment_files"]


def process_deployment(
    file: Path,
    profiles_directory: Path,
    settings: Settings,
) -> None:
    """Process one ``.rsk`` deployment into extracted profile files.

    Reads the deployment and builds a `Dataset` from it, then resolves
    the effective `ProcessSettings` for it -- an instrument's serial
    number is only known once its data has actually been read (never
    inferred from a filename), so settings resolution happens here,
    after `build_dataset`, rather than before the file is read (see
    :func:`ctd_processing.config.resolve_process_settings`). It then
    applies configured raw-channel processing (`raw_channels`) to every
    channel, ensures a `sea_pressure` channel exists (trusting one
    already in the dataset by default, or recomputing it from
    `absolute_pressure` if `atmospheric_pressure` is set), and identifies
    profiles from it using `profiles`. Profile extraction, TEOS-10
    derived variables, and CF-compliant output are not yet implemented.
    Once the `Dataset` is built, the underlying `pyrsktools.RSK` object
    is no longer referenced and is free to be garbage collected -- every
    later step operates purely on the `Dataset`. `settings.project`
    metadata (e.g. `name`) is intended to be attached to every output
    file's metadata once implemented.

    Parameters
    ----------
    file : pathlib.Path
        The ``.rsk`` deployment file to process. Should be a private
        copy (see :func:`process_deployment_files`), since later steps
        may run write-capable `pyrsktools.RSK` methods against it. Its
        filename stem is used to look up a matching
        ``settings.deployments`` override.
    profiles_directory : pathlib.Path
        Directory to write extracted profile files into.
    settings : Settings
        The project's full settings, used to resolve this deployment's
        effective `ProcessSettings` (see
        :func:`ctd_processing.config.resolve_process_settings`) and to
        supply `settings.project` metadata.
    """
    logger.info("Reading deployment: %s", file)
    rsk = read_rsk(file)
    dataset = build_dataset(rsk, file, settings.project)
    process_settings = resolve_process_settings(
        settings,
        serial_number=str(dataset.metadata["instrument_serial_number"]),
        stem=file.stem,
    )
    dataset = process_raw_channels(dataset, process_settings)
    dataset = compute_sea_pressure(
        dataset, process_settings.atmospheric_pressure
    )
    profiles = find_profiles(dataset, process_settings.profiles)
    logger.info("Identified %d profile(s) in %s", len(profiles), file)
    logger.debug("Built dataset: %s", dataset)


def _copy_deployment(source: Path, destination: Path) -> Path:
    """Copy `source` to `destination`, creating parent directories as needed.

    Parameters
    ----------
    source : pathlib.Path
        File to copy.
    destination : pathlib.Path
        Destination path for the copy. Its parent directory is created
        if it does not already exist.

    Returns
    -------
    pathlib.Path
        `destination`.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(source, destination))


def process_deployment_files(
    deployment_files: list[Path],
    profiles_directory: Path,
    settings: Settings,
) -> None:
    """Copy deployments into a private temp directory and process concurrently.

    Each deployment file is copied into its own subdirectory of a
    shared temporary directory (preserving its original filename), and
    :func:`process_deployment` is dispatched on that copy as soon as
    its copy completes, without waiting for the other copies to
    finish. Copying runs on a thread pool since it is I/O-bound, so
    copying one deployment overlaps with processing another. The
    temporary directory and all copies are removed once every
    deployment has been attempted.

    Parameters
    ----------
    deployment_files : list[pathlib.Path]
        The ``.rsk`` deployment files to process.
    profiles_directory : pathlib.Path
        Directory to write extracted profile files into.
    settings : Settings
        The project's full settings, forwarded to :func:`process_deployment`
        so it can resolve each deployment's effective `ProcessSettings`
        (instrument/deployment overrides can only be resolved per file,
        once that file's instrument serial number is known).

    Raises
    ------
    ExceptionGroup
        If one or more deployments failed to copy or process. Every
        deployment is still attempted regardless of earlier failures;
        the failures are collected and raised together once all
        deployments have been attempted.
    """
    errors: list[Exception] = []

    with tempfile.TemporaryDirectory(
        prefix="ctd_processing_", ignore_cleanup_errors=True
    ) as tmp_dir:
        tmp_root = Path(tmp_dir)
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(
                    _copy_deployment, file, tmp_root / str(index) / file.name
                ): file
                for index, file in enumerate(deployment_files)
            }
            for future in as_completed(future_to_file):
                deployment_file = future_to_file[future]
                try:
                    copy_path = future.result()
                    process_deployment(copy_path, profiles_directory, settings)
                except Exception as exc:
                    logger.exception(
                        "Failed to process deployment: %s", deployment_file
                    )
                    exc.add_note(
                        f"while processing deployment: {deployment_file}"
                    )
                    errors.append(exc)

    if errors:
        raise ExceptionGroup(
            f"Failed to process {len(errors)} of {len(deployment_files)} "
            "deployment(s).",
            errors,
        )
