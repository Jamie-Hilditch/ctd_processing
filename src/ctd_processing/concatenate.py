"""Concatenate multiple deployments' binned datasets into one, in time.

See `ctd_processing.cli.concatenate`. `concatenate_deployments` merges
already-binned, per-deployment datasets (see
`ctd_processing.bin.bin_deployment`/
`ctd_processing.bin.save.save_binned_dataset`) along their shared
``profile`` dimension. It also guards against a common RBR gotcha:
forgetting to wipe an instrument's onboard memory between deployments, so
the next deployment's raw data -- and hence its binned profiles -- repeats
the tail of the previous one. Any profile sharing an exact ``time``
coordinate value with another is a duplicate and is dropped, and the
result is sorted by ``time`` ascending.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = ["concatenate_deployments"]


def concatenate_deployments(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate binned deployment datasets, deduplicated and time-sorted.

    Concatenates `datasets` along their shared ``profile`` dimension
    (``combine_attrs="drop_conflicts"``, matching
    `ctd_processing.bin.binning.combine_binned_profiles` -- a global or
    per-variable attribute is kept only if every input dataset agrees on
    it, since deployment-specific attributes like ``source_file`` or
    ``instrument_serial_number`` rarely do once more than one deployment
    is involved). Every profile's ``time`` coordinate is then used to
    both deduplicate (profiles sharing an exact `time` with an
    earlier-sorted one are dropped) and sort the result ascending, in a
    single pass.

    Parameters
    ----------
    datasets : list[xarray.Dataset]
        Each deployment's combined, binned dataset (see
        `ctd_processing.bin.save.load_binned_dataset`), sharing a
        ``profile`` dimension and a ``time`` coordinate along it.

    Returns
    -------
    xarray.Dataset
        The concatenated dataset: one profile per distinct ``time``
        value across every input, sorted by ``time`` ascending.

    Raises
    ------
    ValueError
        If `datasets` is empty.
    """
    if not datasets:
        raise ValueError("Cannot concatenate: no datasets were given.")

    combined = xr.concat(
        datasets,
        dim="profile",
        data_vars="all",
        combine_attrs="drop_conflicts",
    )

    dropped_global = sorted(
        set().union(*(set(ds.attrs) for ds in datasets)) - set(combined.attrs)
    )
    if dropped_global:
        logger.warning(
            "Dropped conflicting global attribute(s) when concatenating "
            "deployments: %s",
            ", ".join(dropped_global),
        )

    # np.unique sorts its input, and (with return_index=True) reports the
    # first occurrence of each unique value -- so this both drops profiles
    # sharing an exact time with an earlier one and sorts by time, in one
    # pass, rather than deduplicating and sorting separately.
    _, first_indices = np.unique(combined["time"].values, return_index=True)
    duplicate_count = combined.sizes["profile"] - first_indices.size
    if duplicate_count:
        logger.warning(
            "Dropped %d duplicate profile(s) sharing a time value with "
            "another profile.",
            duplicate_count,
        )

    return combined.isel(profile=first_indices)
