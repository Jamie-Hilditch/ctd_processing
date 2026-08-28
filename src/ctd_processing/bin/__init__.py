"""Binning pipeline: combine one deployment's profiles onto a common grid.

Sister package to :mod:`ctd_processing.process`. :func:`bin_deployment` is
the entry point that :mod:`ctd_processing.cli.bin` calls with every profile
`Dataset` (see `ctd_processing.process.dataset.Dataset`) belonging to one
deployment (loaded back via `ctd_processing.process.save.load_profile`).
The rest of this package holds the supporting implementation: binning and
averaging one profile (`ctd_processing.bin.binning`) and naming/writing the
combined result (`ctd_processing.bin.save`).
"""

import logging

import xarray as xr

from ctd_processing.bin.binning import (
    bin_profile,
    combine_binned_profiles,
    compute_bin_edges,
)
from ctd_processing.config import BinSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["bin_deployment"]


def _validate_same_deployment(profiles: list[Dataset]) -> None:
    """Require every profile to share one deployment identity.

    `bin_deployment` combines its input profiles under the assumption
    (stated in `ctd_processing.config.BinSettings`'s docstring) that they
    all come from the same deployment, sharing metadata/history closely
    enough for `combine_binned_profiles`'s ``combine_attrs="drop_conflicts"``
    to behave sensibly. This checks that assumption explicitly rather than
    silently combining profiles from different deployments.

    Parameters
    ----------
    profiles : list[Dataset]
        The profiles to check.

    Raises
    ------
    ValueError
        If `profiles` is empty, or if more than one distinct
        ``(instrument_serial_number, source_file)`` pair is present.
    """
    if not profiles:
        raise ValueError("Cannot bin: no profiles were given.")

    identities = {
        (
            dataset.metadata.get("instrument_serial_number"),
            dataset.metadata.get("source_file"),
        )
        for dataset in profiles
    }
    if len(identities) > 1:
        raise ValueError(
            "Cannot bin: input profiles span multiple deployments "
            f"{sorted(identities, key=repr)!r}; pass profiles from a "
            "single deployment."
        )


def bin_deployment(
    profiles: list[Dataset], settings: BinSettings
) -> xr.Dataset:
    """Bin every profile of one deployment onto a common grid and combine.

    Sorts `profiles` by their canonical start time, computes bin edges
    once from `settings` and every profile's `settings.channel` data (see
    `ctd_processing.bin.binning.compute_bin_edges`), bins each profile onto
    those edges (`ctd_processing.bin.binning.bin_profile`), and stacks the
    results along a new ``profile`` dimension
    (`ctd_processing.bin.binning.combine_binned_profiles`).

    Parameters
    ----------
    profiles : list[Dataset]
        Every already-extracted, already-processed profile belonging to
        one deployment (see `ctd_processing.process.process_profile`).
    settings : BinSettings
        Configures which channel to bin by and the bin grid.

    Returns
    -------
    xarray.Dataset
        The combined, binned dataset for the whole deployment.

    Raises
    ------
    ValueError
        If `profiles` is empty, spans more than one deployment (see
        `_validate_same_deployment`), or if `settings.channel` is missing
        from a profile (see
        `ctd_processing.bin.binning.bin_profile`).
    """
    _validate_same_deployment(profiles)
    ordered = sorted(
        profiles, key=lambda dataset: dataset.metadata["profile_start_time"]
    )

    edges = compute_bin_edges(
        [dataset.channels[settings.channel].data for dataset in ordered],
        settings,
    )

    total = len(ordered)
    binned = []
    for index, dataset in enumerate(ordered):
        binned.append(bin_profile(dataset, settings.channel, edges))
        log_verbose(logger, "binned profile %d of %d", index + 1, total)

    combined = combine_binned_profiles(binned)
    logger.info("Binned %d profile(s) by %r", total, settings.channel)
    return combined
