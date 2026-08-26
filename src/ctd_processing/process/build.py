"""Build a `Dataset` out of an opened `pyrsktools.RSK`."""

import logging
from pathlib import Path

import pyrsktools

from ctd_processing.config import ProjectSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.cf_channels import (
    cf_metadata_for_longname,
    channel_key_for_longname,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = ["build_dataset"]


def _sampling_period_seconds(rsk: pyrsktools.RSK) -> float | None:
    """Best-effort sampling period, in seconds, from `rsk.scheduleInfos`.

    Parameters
    ----------
    rsk : pyrsktools.RSK
        The opened dataset to inspect.

    Returns
    -------
    float or None
        The first schedule's sampling period in seconds, or ``None`` if
        `rsk.scheduleInfos` is empty or its first entry doesn't expose a
        `samplingperiod` method (e.g. non-continuous logging modes).
    """
    if not rsk.scheduleInfos:
        return None
    samplingperiod = getattr(rsk.scheduleInfos[0], "samplingperiod", None)
    if samplingperiod is None:
        return None
    return samplingperiod()


def build_dataset(
    rsk: pyrsktools.RSK, file: Path, project: ProjectSettings
) -> Dataset:
    """Build a `Dataset` from an opened, read `pyrsktools.RSK`.

    Every channel in `rsk.channels` is added under
    `ctd_processing.process.cf_channels.channel_key_for_longname`'s result
    for it -- a short, stable identifier, deliberately not the (often
    long and qualifier-laden) CF `standard_name`, which is still recorded
    in the channel's own `metadata` when known.

    Parameters
    ----------
    rsk : pyrsktools.RSK
        The dataset to read, as returned by
        `ctd_processing.process.read.read_rsk`. Must already have had
        `readdata` called (as `read_rsk` does), so that `instrument`,
        `epoch`, and `deployment` are populated.
    file : pathlib.Path
        The `.rsk` file `rsk` was read from, recorded in `metadata` for
        provenance.
    project : ProjectSettings
        Project metadata to attach to the returned Dataset.

    Returns
    -------
    Dataset
        A Dataset whose `time` channel is `rsk.data["timestamp"]`, with
        every measured/derived channel pyrsktools reports added under its
        `channel_key_for_longname` key (see above), and `metadata`
        populated with deployment/instrument provenance.

    Raises
    ------
    ValueError
        If `rsk.instrument`, `rsk.epoch`, or `rsk.deployment` is ``None``
        (i.e. `rsk` was not read via `read_rsk` first).
    """
    if rsk.instrument is None or rsk.epoch is None or rsk.deployment is None:
        raise ValueError(
            "rsk is missing instrument/epoch/deployment metadata; "
            "read it with read_rsk first."
        )

    time = Channel(data=rsk.data["timestamp"])
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {
            "source_file": str(file),
            "instrument_model": rsk.instrument.model,
            "instrument_serial_number": rsk.instrument.serialID,
            "instrument_firmware_version": rsk.instrument.firmwareVersion,
            "deployment_start_time": rsk.epoch.startTime,
            "deployment_end_time": rsk.epoch.endTime,
            "deployment_comment": rsk.deployment.comment,
            "project_name": project.name,
            "sampling_period_seconds": _sampling_period_seconds(rsk),
        }
    )
    dataset.record(f"read from {file}")
    log_verbose(logger, "read from %s", file)

    data_field_names = rsk.data.dtype.names or ()
    for rsk_channel in rsk.channels:
        if rsk_channel.longName not in data_field_names:
            logger.warning(
                "Skipping channel %r: not present in this schedule's data.",
                rsk_channel.longName,
            )
            continue

        cf = cf_metadata_for_longname(rsk_channel.longName)
        key = channel_key_for_longname(rsk_channel.longName)
        if key in dataset.channels:
            logger.warning(
                "Channel name %r already used in this dataset; adding "
                "channel %r under its own name instead.",
                key,
                rsk_channel.longName,
            )
            key = rsk_channel.longName
        if cf.standard_name is None:
            logger.info(
                "No CF standard_name for channel %r.", rsk_channel.longName
            )

        metadata = {
            "units": rsk_channel.unitsPlainText or rsk_channel.units,
            "long_name": cf.long_name,
            "source_channel_name": rsk_channel.longName,
        }
        if cf.standard_name is not None:
            metadata["standard_name"] = cf.standard_name

        channel = Channel(
            data=rsk.data[rsk_channel.longName], metadata=metadata
        )
        dataset.add_channel(key, channel)
        log_verbose(logger, "added channel %r", key)

    return dataset
