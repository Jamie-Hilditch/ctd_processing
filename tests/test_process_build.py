"""Tests for ctd_processing.process.build."""

from pathlib import Path

import numpy as np
import pytest

from ctd_processing.config import ProjectSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.build import build_dataset
from ctd_processing.process.read import read_rsk


@pytest.mark.requires_example_data
def test_build_dataset_time_channel_matches_rsk(
    example_rsk_path: Path,
) -> None:
    """The Dataset's time channel matches rsk.data['timestamp']."""
    rsk = read_rsk(example_rsk_path)

    dataset = build_dataset(rsk, example_rsk_path, ProjectSettings())

    assert np.array_equal(dataset.time.data, rsk.data["timestamp"])
    assert dataset.length == rsk.data.size


@pytest.mark.requires_example_data
def test_build_dataset_metadata_from_instrument_and_project(
    example_rsk_path: Path,
) -> None:
    """Dataset metadata pulls instrument/deployment/project info."""
    rsk = read_rsk(example_rsk_path)
    assert rsk.instrument is not None
    assert rsk.epoch is not None
    project = ProjectSettings(name="test-cruise")

    dataset = build_dataset(rsk, example_rsk_path, project)

    assert dataset.metadata["source_file"] == str(example_rsk_path)
    assert dataset.metadata["instrument_model"] == rsk.instrument.model
    assert (
        dataset.metadata["instrument_serial_number"] == rsk.instrument.serialID
    )
    assert (
        dataset.metadata["instrument_firmware_version"]
        == rsk.instrument.firmwareVersion
    )
    assert dataset.metadata["deployment_start_time"] == rsk.epoch.startTime
    assert dataset.metadata["deployment_end_time"] == rsk.epoch.endTime
    assert dataset.metadata["project_name"] == "test-cruise"


@pytest.mark.requires_example_data
def test_build_dataset_keys_channels_by_long_name_slug(
    example_rsk_path: Path,
) -> None:
    """Channels are keyed by a slug of long_name, not by standard_name.

    Temperature's slug happens to coincide with its standard_name, but
    pressure's doesn't ("absolute_pressure" vs. "sea_water_pressure") --
    checking both demonstrates the key really comes from long_name.
    """
    rsk = read_rsk(example_rsk_path)

    dataset = build_dataset(rsk, example_rsk_path, ProjectSettings())

    temperature = dataset.channels["sea_water_temperature"]
    assert temperature.metadata["standard_name"] == "sea_water_temperature"
    assert temperature.metadata["long_name"] == "Sea water temperature"
    assert temperature.metadata["source_channel_name"] == "temperature"
    assert np.array_equal(temperature.data, rsk.data["temperature"])

    rsk_temperature_channel = next(
        c for c in rsk.channels if c.longName == "temperature"
    )
    expected_units = (
        rsk_temperature_channel.unitsPlainText or rsk_temperature_channel.units
    )
    assert temperature.metadata["units"] == expected_units

    pressure = dataset.channels["absolute_pressure"]
    assert pressure.metadata["standard_name"] == "sea_water_pressure"
    assert pressure.metadata["long_name"] == "Absolute pressure"


@pytest.mark.requires_example_data
def test_build_dataset_covers_every_data_channel(
    example_rsk_path: Path,
) -> None:
    """Every rsk.channels entry with a matching data column is included."""
    rsk = read_rsk(example_rsk_path)
    data_field_names = rsk.data.dtype.names or ()

    dataset = build_dataset(rsk, example_rsk_path, ProjectSettings())

    source_channel_names = {
        channel.metadata["source_channel_name"]
        for name, channel in dataset.channels.items()
        if name != "time"
    }
    expected = {
        c.longName for c in rsk.channels if c.longName in data_field_names
    }
    assert source_channel_names == expected


@pytest.mark.requires_example_data
def test_build_dataset_logs_at_verbose_level(
    example_rsk_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """build_dataset logs the read and each add_channel at VERBOSE."""
    rsk = read_rsk(example_rsk_path)
    caplog.set_level(VERBOSE, logger="ctd_processing.process.build")

    build_dataset(rsk, example_rsk_path, ProjectSettings())

    verbose_messages = {
        record.getMessage()
        for record in caplog.records
        if record.levelno == VERBOSE
    }
    assert f"read from {example_rsk_path}" in verbose_messages
    assert "added channel 'sea_water_temperature'" in verbose_messages
