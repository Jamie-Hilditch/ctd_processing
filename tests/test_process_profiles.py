"""Tests for ctd_processing.process.profiles."""

import numpy as np
import pytest
from profinder import synthetic_glider_pressure

from ctd_processing.config import ProfileSettings
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.profiles import (
    Profile,
    find_profiles,
    resolve_cast_slices,
)


def _dataset_with_sea_pressure(pressure: np.ndarray) -> Dataset:
    n = pressure.size
    time = np.datetime64("2024-01-01") + np.arange(n) * np.timedelta64(1, "s")
    dataset = Dataset(time=Channel(data=time))
    dataset.add_channel("sea_pressure", Channel(data=pressure))
    return dataset


def test_find_profiles_identifies_profiles_in_synthetic_data() -> None:
    """A synthetic multi-cycle glider pressure record yields profiles."""
    pressure = synthetic_glider_pressure(
        n_points=2000, max_p=500.0, intermediate_p=200.0, n_cycles=3
    )
    dataset = _dataset_with_sea_pressure(pressure)

    profiles = find_profiles(dataset, ProfileSettings())

    assert len(profiles) > 0
    for profile in profiles:
        assert isinstance(profile, Profile)
        assert 0 <= profile.down_start <= profile.down_end
        assert profile.down_end <= profile.up_start
        assert profile.up_start <= profile.up_end < pressure.size


def test_find_profiles_raises_without_sea_pressure() -> None:
    """A dataset with no sea_pressure channel raises ValueError."""
    dataset = Dataset(time=Channel(data=np.array([0.0, 1.0, 2.0])))

    with pytest.raises(ValueError, match="sea_pressure"):
        find_profiles(dataset, ProfileSettings())


def test_find_profiles_does_not_mutate_dataset() -> None:
    """find_profiles is a pure read/analyze step; the dataset is untouched."""
    pressure = synthetic_glider_pressure(n_points=2000, n_cycles=2)
    dataset = _dataset_with_sea_pressure(pressure)
    original_history = list(dataset.history)
    original_channels = set(dataset.channels)

    find_profiles(dataset, ProfileSettings())

    assert dataset.history == original_history
    assert set(dataset.channels) == original_channels


def test_find_profiles_with_speed_threshold_runs_without_error() -> None:
    """apply_speed_threshold=True computes elapsed time and succeeds."""
    pressure = synthetic_glider_pressure(n_points=2000, n_cycles=2)
    dataset = _dataset_with_sea_pressure(pressure)
    settings = ProfileSettings(apply_speed_threshold=True)

    profiles = find_profiles(dataset, settings)

    assert isinstance(profiles, list)


_PROFILE = Profile(down_start=0, down_end=4, up_start=5, up_end=9)


def test_resolve_cast_slices_down_returns_only_downcast() -> None:
    """direction="down" returns just the downcast slice."""
    assert resolve_cast_slices(_PROFILE, "down") == [slice(0, 4)]


def test_resolve_cast_slices_up_returns_only_upcast() -> None:
    """direction="up" returns just the upcast slice."""
    assert resolve_cast_slices(_PROFILE, "up") == [slice(5, 9)]


def test_resolve_cast_slices_both_returns_downcast_then_upcast() -> None:
    """direction="both" returns the downcast and upcast, in that order."""
    assert resolve_cast_slices(_PROFILE, "both") == [slice(0, 4), slice(5, 9)]


def test_resolve_cast_slices_never_includes_the_dwell() -> None:
    """The dwell between down_end and up_start (index 4) is never returned."""
    for direction in ("down", "up", "both"):
        for cast_slice in resolve_cast_slices(_PROFILE, direction):
            assert (
                cast_slice.stop <= _PROFILE.down_end
                or cast_slice.start >= _PROFILE.up_start
            )
