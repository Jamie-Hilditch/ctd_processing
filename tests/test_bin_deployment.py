"""Tests for ctd_processing.bin.bin_deployment."""

import logging

import numpy as np
import pytest

from ctd_processing.bin import bin_deployment
from ctd_processing.config import BinSettings
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset


def _profile(
    z: np.ndarray,
    temperature: np.ndarray,
    *,
    start: str,
    serial: object = 208532,
    source_file: str = "243188_20260809_0304.rsk",
) -> Dataset:
    """Build a small, already-geolocated profile Dataset for bin tests."""
    n = z.size
    time = Channel(
        data=np.datetime64(start) + np.arange(n) * np.timedelta64(1, "s")
    )
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {"instrument_serial_number": serial, "source_file": source_file}
    )
    dataset.add_channel("z", Channel(data=z, metadata={"units": "m"}))
    dataset.add_channel(
        "sea_water_temperature",
        Channel(data=temperature, metadata={"units": "degree_C"}),
    )
    dataset.metadata.update(
        {
            "profile_start_time": time.data[0],
            "profile_end_time": time.data[-1],
            "latitude": 45.0,
            "longitude": -125.0,
        }
    )
    return dataset


def test_bin_deployment_raises_on_empty_input() -> None:
    """No profiles at all raises ValueError."""
    with pytest.raises(ValueError, match="no profiles"):
        bin_deployment([], BinSettings())


def test_bin_deployment_raises_on_mixed_deployments() -> None:
    """Profiles from different (serial, source_file) pairs are rejected."""
    p1 = _profile(
        np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
    )
    p2 = _profile(
        np.array([-0.1, -0.6]),
        np.array([3.0, 4.0]),
        start="2026-01-02",
        serial=999999,
    )

    with pytest.raises(ValueError, match="multiple deployments"):
        bin_deployment([p1, p2], BinSettings())


def test_bin_deployment_error_message_survives_mixed_identity_types() -> None:
    """Mismatched identity value types don't crash the error path itself."""
    p1 = _profile(
        np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
    )
    p2 = _profile(
        np.array([-0.1, -0.6]),
        np.array([3.0, 4.0]),
        start="2026-01-02",
        serial="not-a-number",
    )

    with pytest.raises(ValueError, match="multiple deployments"):
        bin_deployment([p1, p2], BinSettings())


def test_bin_deployment_combines_profiles_sorted_by_start_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Out-of-order input profiles are sorted by profile_start_time."""
    caplog.set_level(logging.INFO, logger="ctd_processing.bin")
    later = _profile(
        np.array([-0.1, -0.6]), np.array([100.0, 200.0]), start="2026-01-02"
    )
    earlier = _profile(
        np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
    )

    combined = bin_deployment(
        [later, earlier], BinSettings(channel="z", step=-0.5)
    )

    assert combined.sizes["profile"] == 2
    times = combined["time"].values
    assert times[0] < times[1]
    messages = [record.getMessage() for record in caplog.records]
    assert any("Binned 2 profile(s)" in m for m in messages)
