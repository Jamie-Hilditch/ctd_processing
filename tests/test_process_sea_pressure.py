"""Tests for ctd_processing.process.sea_pressure."""

import numpy as np
import pytest

from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.sea_pressure import compute_sea_pressure


def _dataset_with_absolute_pressure() -> Dataset:
    dataset = Dataset(time=Channel(data=np.array([0.0, 1.0, 2.0])))
    dataset.add_channel(
        "absolute_pressure",
        Channel(
            data=np.array([10.0, 15.0, 20.0]),
            metadata={"units": "dbar", "long_name": "Absolute pressure"},
        ),
    )
    return dataset


def test_none_uses_existing_sea_pressure_channel_unchanged() -> None:
    """atmospheric_pressure=None trusts an existing sea_pressure channel."""
    dataset = _dataset_with_absolute_pressure()
    existing = Channel(data=np.array([1.0, 2.0, 3.0]))
    dataset.add_channel("sea_pressure", existing)

    result = compute_sea_pressure(dataset, atmospheric_pressure=None)

    assert result.channels["sea_pressure"] is existing
    assert np.array_equal(result.channels["sea_pressure"].data, [1.0, 2.0, 3.0])


def test_none_raises_without_existing_sea_pressure_channel() -> None:
    """atmospheric_pressure=None with no sea_pressure channel raises."""
    dataset = _dataset_with_absolute_pressure()

    with pytest.raises(ValueError, match="sea_pressure"):
        compute_sea_pressure(dataset, atmospheric_pressure=None)


def test_none_logs_at_info_level(caplog: pytest.LogCaptureFixture) -> None:
    """atmospheric_pressure=None logs at INFO that the channel is reused."""
    dataset = _dataset_with_absolute_pressure()
    dataset.add_channel("sea_pressure", Channel(data=np.array([1.0, 2.0, 3.0])))
    caplog.set_level("INFO", logger="ctd_processing.process.sea_pressure")

    compute_sea_pressure(dataset, atmospheric_pressure=None)

    messages = [r.getMessage() for r in caplog.records]
    assert any("sea_pressure channel already present" in m for m in messages)


def test_value_computes_sea_pressure_when_absent() -> None:
    """A float atmospheric_pressure computes sea_pressure = absolute - atm."""
    dataset = _dataset_with_absolute_pressure()

    result = compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    assert np.allclose(
        result.channels["sea_pressure"].data, [-0.1325, 4.8675, 9.8675]
    )


def test_value_sets_cf_metadata() -> None:
    """The derived channel carries units, long_name, and standard_name."""
    dataset = _dataset_with_absolute_pressure()

    result = compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    metadata = result.channels["sea_pressure"].metadata
    assert metadata["units"] == "dbar"
    assert metadata["long_name"] == "Sea pressure"
    assert metadata["standard_name"] == "sea_water_pressure_due_to_sea_water"


def test_value_overwrites_existing_sea_pressure_channel() -> None:
    """A float atmospheric_pressure overwrites an existing sea_pressure."""
    dataset = _dataset_with_absolute_pressure()
    dataset.add_channel("sea_pressure", Channel(data=np.array([1.0, 2.0, 3.0])))

    result = compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    assert np.allclose(
        result.channels["sea_pressure"].data, [-0.1325, 4.8675, 9.8675]
    )


def test_value_returns_same_dataset() -> None:
    """compute_sea_pressure mutates in place and returns `dataset`."""
    dataset = _dataset_with_absolute_pressure()

    result = compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    assert result is dataset


def test_value_raises_without_absolute_pressure() -> None:
    """A float atmospheric_pressure with no absolute_pressure raises."""
    dataset = Dataset(time=Channel(data=np.array([0.0, 1.0, 2.0])))

    with pytest.raises(ValueError, match="absolute_pressure"):
        compute_sea_pressure(dataset, atmospheric_pressure=10.1325)


def test_value_logs_at_verbose_level_when_computing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Computing sea_pressure logs the atmospheric pressure used at VERBOSE."""
    dataset = _dataset_with_absolute_pressure()
    caplog.set_level(VERBOSE, logger="ctd_processing.process.sea_pressure")

    compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    verbose_records = [r for r in caplog.records if r.levelno == VERBOSE]
    assert len(verbose_records) == 1
    assert "10.1325" in verbose_records[0].getMessage()


def test_value_logs_removal_at_verbose_when_overwriting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overwriting sea_pressure logs both the removal and the compute."""
    dataset = _dataset_with_absolute_pressure()
    dataset.add_channel("sea_pressure", Channel(data=np.array([1.0, 2.0, 3.0])))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.sea_pressure")

    compute_sea_pressure(dataset, atmospheric_pressure=10.1325)

    verbose_messages = [
        r.getMessage() for r in caplog.records if r.levelno == VERBOSE
    ]
    assert any(
        "removed existing sea_pressure channel" in m for m in verbose_messages
    )
    assert any("computed sea pressure" in m for m in verbose_messages)
