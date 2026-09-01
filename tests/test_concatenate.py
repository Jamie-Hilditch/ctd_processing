"""Tests for ctd_processing.concatenate."""

import logging

import numpy as np
import pytest
import xarray as xr

from ctd_processing.concatenate import concatenate_deployments


def _dataset(times: list[str], **attrs: str) -> xr.Dataset:
    """Build a minimal binned-shaped dataset for concatenate tests."""
    n = len(times)
    return xr.Dataset(
        {"temperature": (("profile", "z"), np.zeros((n, 2)))},
        coords={
            "z": ("z", [-0.5, -1.5]),
            "time": ("profile", np.array(times, dtype="datetime64[s]")),
            "latitude": ("profile", np.full(n, 45.0)),
            "longitude": ("profile", np.full(n, -125.0)),
        },
        attrs=attrs,
    )


def test_concatenate_deployments_raises_when_empty() -> None:
    """An empty input list is an error, not an empty result."""
    with pytest.raises(ValueError, match="no datasets"):
        concatenate_deployments([])


def test_concatenate_deployments_sorts_by_time() -> None:
    """Datasets given out of time order come back sorted ascending."""
    later = _dataset(["2026-08-10T00:00:00"])
    earlier = _dataset(["2026-08-09T00:00:00"])

    combined = concatenate_deployments([later, earlier])

    times = list(combined["time"].values)
    assert times == sorted(times)
    assert combined.sizes["profile"] == 2


def test_concatenate_deployments_drops_exact_duplicate_time() -> None:
    """A profile sharing an exact time with another is dropped, once."""
    first = _dataset(["2026-08-09T00:00:00", "2026-08-09T01:00:00"])
    second = _dataset(["2026-08-09T01:00:00", "2026-08-09T02:00:00"])

    combined = concatenate_deployments([first, second])

    times = list(combined["time"].values)
    assert combined.sizes["profile"] == 3
    assert len(set(times)) == 3
    assert times == sorted(times)


def test_concatenate_deployments_logs_dropped_duplicate_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The number of dropped duplicate profiles is logged at WARNING."""
    first = _dataset(["2026-08-09T00:00:00"])
    second = _dataset(["2026-08-09T00:00:00"])
    caplog.set_level(logging.WARNING, logger="ctd_processing.concatenate")

    concatenate_deployments([first, second])

    messages = [record.getMessage() for record in caplog.records]
    assert any("Dropped 1 duplicate profile" in m for m in messages)


def test_concatenate_deployments_no_duplicates_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning is logged when there is nothing to drop."""
    first = _dataset(["2026-08-09T00:00:00"])
    second = _dataset(["2026-08-10T00:00:00"])
    caplog.set_level(logging.WARNING, logger="ctd_processing.concatenate")

    concatenate_deployments([first, second])

    assert caplog.records == []


def test_concatenate_deployments_drops_conflicting_global_attrs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A global attribute that disagrees across deployments is dropped."""
    first = _dataset(
        ["2026-08-09T00:00:00"], project_name="cruise", source_file="a.rsk"
    )
    second = _dataset(
        ["2026-08-10T00:00:00"], project_name="cruise", source_file="b.rsk"
    )
    caplog.set_level(logging.WARNING, logger="ctd_processing.concatenate")

    combined = concatenate_deployments([first, second])

    assert combined.attrs["project_name"] == "cruise"
    assert "source_file" not in combined.attrs
    messages = [record.getMessage() for record in caplog.records]
    assert any("source_file" in m for m in messages)
