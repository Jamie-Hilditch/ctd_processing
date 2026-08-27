"""Tests for ctd_processing.process.geolocation."""

import numpy as np
import pytest
import xarray as xr

from ctd_processing.config import GeolocationSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.geolocation import attach_geolocation


def _profile_dataset(start: str, count: int = 3) -> Dataset:
    """Build a small profile Dataset with a datetime64[ms] time channel."""
    time = Channel(
        data=(
            np.datetime64(start, "ms")
            + np.arange(count) * np.timedelta64(1, "s")
        )
    )
    return Dataset(time=time)


def _external_dataset() -> xr.Dataset:
    """Build a synthetic position time series, linear in time."""
    time = np.array(
        ["2026-01-01T00:00:00", "2026-01-01T01:00:00", "2026-01-01T02:00:00"],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "latitude": ("time", [0.0, 10.0, 20.0]),
            "longitude": ("time", [-50.0, -40.0, -30.0]),
        },
        coords={"time": time},
    )


def test_attach_geolocation_reference_position_sets_metadata() -> None:
    """A reference position is recorded verbatim, with its source noted."""
    dataset = _profile_dataset("2026-08-09T03:04:12")
    settings = GeolocationSettings(
        reference_latitude=45.0, reference_longitude=-125.0
    )

    result = attach_geolocation(dataset, settings, None)

    assert result is dataset
    assert dataset.metadata["latitude"] == 45.0
    assert dataset.metadata["longitude"] == -125.0
    assert dataset.metadata["position_source"] == "reference position"


def test_attach_geolocation_records_start_and_end_time() -> None:
    """profile_start_time/profile_end_time come from time.data's first/last."""
    dataset = _profile_dataset("2026-08-09T03:04:12", count=5)
    settings = GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )

    attach_geolocation(dataset, settings, None)

    assert dataset.metadata["profile_start_time"] == dataset.time.data[0]
    assert dataset.metadata["profile_end_time"] == dataset.time.data[-1]
    assert (
        dataset.metadata["profile_start_time"]
        != dataset.metadata["profile_end_time"]
    )


def test_attach_geolocation_records_history() -> None:
    """attach_geolocation records a history entry describing what it did."""
    dataset = _profile_dataset("2026-08-09T03:04:12")
    settings = GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )

    attach_geolocation(dataset, settings, None)

    assert any("reference position" in entry for entry in dataset.history)


def test_attach_geolocation_logs_at_verbose(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """attach_geolocation logs its action at VERBOSE."""
    caplog.set_level(VERBOSE, logger="ctd_processing.process.geolocation")
    dataset = _profile_dataset("2026-08-09T03:04:12")
    settings = GeolocationSettings(
        reference_latitude=0.0, reference_longitude=0.0
    )

    attach_geolocation(dataset, settings, None)

    verbose_records = [r for r in caplog.records if r.levelno == VERBOSE]
    assert any("reference position" in r.getMessage() for r in verbose_records)


def test_attach_geolocation_interpolates_from_external_dataset() -> None:
    """Position is linearly interpolated onto the profile's canonical time."""
    dataset = _profile_dataset("2026-01-01T00:30:00")
    settings = GeolocationSettings(external_dataset_path="gps.nc")
    external_dataset = _external_dataset()

    attach_geolocation(dataset, settings, external_dataset)

    assert dataset.metadata["latitude"] == pytest.approx(5.0)
    assert dataset.metadata["longitude"] == pytest.approx(-45.0)
    assert dataset.metadata["position_source"] == "interpolated from gps.nc"


def test_attach_geolocation_respects_custom_variable_names() -> None:
    """Custom latitude/longitude/time variable names are honored."""
    dataset = _profile_dataset("2026-01-01T00:30:00")
    settings = GeolocationSettings(
        external_dataset_path="gps.nc",
        latitude_variable="lat",
        longitude_variable="lon",
        time_variable="t",
    )
    external_dataset = _external_dataset().rename(
        {"latitude": "lat", "longitude": "lon", "time": "t"}
    )

    attach_geolocation(dataset, settings, external_dataset)

    assert dataset.metadata["latitude"] == pytest.approx(5.0)
    assert dataset.metadata["longitude"] == pytest.approx(-45.0)


def test_attach_geolocation_raises_when_canonical_time_before_coverage() -> (
    None
):
    """A canonical time before the external dataset's coverage raises."""
    dataset = _profile_dataset("2025-12-31T23:00:00")
    settings = GeolocationSettings(external_dataset_path="gps.nc")
    external_dataset = _external_dataset()

    with pytest.raises(ValueError, match="outside the external dataset"):
        attach_geolocation(dataset, settings, external_dataset)


def test_attach_geolocation_raises_when_canonical_time_after_coverage() -> None:
    """A canonical time after the external dataset's coverage raises."""
    dataset = _profile_dataset("2026-01-01T03:00:00")
    settings = GeolocationSettings(external_dataset_path="gps.nc")
    external_dataset = _external_dataset()

    with pytest.raises(ValueError, match="outside the external dataset"):
        attach_geolocation(dataset, settings, external_dataset)
