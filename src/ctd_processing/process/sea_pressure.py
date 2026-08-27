"""Use or compute `sea_pressure`, configured via `process.atmospheric_pressure`.

See `ctd_processing.config.ProcessSettings.atmospheric_pressure`.
"""

import logging

from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["compute_sea_pressure"]


def compute_sea_pressure(
    dataset: Dataset, atmospheric_pressure: float | None
) -> Dataset:
    """Ensure `dataset` has a `sea_pressure` channel.

    RBR's Ruskin software commonly derives `sea_pressure` itself from
    `absolute_pressure` and a per-deployment atmospheric reference before
    the ``.rsk`` file is even read (see
    `ctd_processing.config.ProcessSettings.atmospheric_pressure`), so by
    default this trusts that channel as-is rather than recomputing it.
    Pass `atmospheric_pressure` to force a specific, explicit constant
    instead, e.g. when no such channel exists or a different reference is
    wanted.

    Parameters
    ----------
    dataset : Dataset
        The dataset to ensure a `sea_pressure` channel on.
    atmospheric_pressure : float or None
        If ``None`` (default), the dataset's existing `sea_pressure`
        channel is used as-is; it is an error for one not to exist. If a
        float, `sea_pressure` is (re)computed as
        ``absolute_pressure - atmospheric_pressure``, in dbar,
        overwriting any `sea_pressure` channel already present via
        `Dataset.remove_channel` + `Dataset.add_channel`.

    Returns
    -------
    Dataset
        `dataset` itself (not a copy).

    Raises
    ------
    ValueError
        If `atmospheric_pressure` is ``None`` and `dataset` has no
        `sea_pressure` channel, or if `atmospheric_pressure` is a float
        and `dataset` has no `absolute_pressure` channel.
    """
    if atmospheric_pressure is None:
        if "sea_pressure" not in dataset.channels:
            raise ValueError(
                "Cannot use sea_pressure: dataset has no sea_pressure "
                "channel and no atmospheric_pressure was configured to "
                "compute one."
            )
        logger.info(
            "Using the sea_pressure channel already present in the "
            "dataset (no atmospheric_pressure configured)."
        )
        return dataset

    if "absolute_pressure" not in dataset.channels:
        raise ValueError(
            "Cannot compute sea_pressure: dataset has no absolute_pressure "
            "channel."
        )

    absolute_pressure = dataset.channels["absolute_pressure"]
    sea_pressure = Channel(
        data=absolute_pressure.data - atmospheric_pressure,
        metadata={
            "units": absolute_pressure.metadata.get("units"),
            "long_name": "Sea pressure",
            "standard_name": "sea_water_pressure_due_to_sea_water",
        },
    )

    if "sea_pressure" in dataset.channels:
        dataset.remove_channel("sea_pressure")
        log_verbose(logger, "removed existing sea_pressure channel")

    dataset.add_channel("sea_pressure", sea_pressure)
    log_verbose(
        logger,
        "computed sea pressure using atmospheric pressure %s dbar",
        atmospheric_pressure,
    )
    return dataset
