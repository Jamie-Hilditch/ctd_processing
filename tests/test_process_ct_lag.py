"""Tests for ctd_processing.process.ct_lag."""

import logging

import numpy as np
import pytest

import ctd_processing.process.ct_lag as ct_lag_module
from ctd_processing.config import CTLagSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process._shift import shift_array
from ctd_processing.process.channel import Channel
from ctd_processing.process.ct_lag import calculate_ct_lag, process_ct_lag
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.profiles import Profile


def _dataset(
    conductivity: np.ndarray,
    temperature: np.ndarray,
    sea_pressure: np.ndarray,
) -> Dataset:
    """Build a Dataset with the three channels calculate_ct_lag requires."""
    n = len(conductivity)
    dataset = Dataset(time=Channel(data=np.arange(float(n))))
    dataset.add_channel(
        "sea_water_electrical_conductivity", Channel(data=conductivity.copy())
    )
    dataset.add_channel(
        "sea_water_temperature", Channel(data=temperature.copy())
    )
    dataset.add_channel("sea_pressure", Channel(data=sea_pressure.copy()))
    return dataset


def _stub_salinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace gsw.SP_from_C with `c - t`.

    Isolates the lag-search/pooling/tie-break logic from real TEOS-10
    physics, which isn't needed to test that logic: any function
    combining conductivity and temperature so that only the correct
    alignment cancels a rough (here, random noise) component works
    equally well as a test fixture.
    """
    monkeypatch.setattr(ct_lag_module.gsw, "SP_from_C", lambda c, t, p: c - t)


def _injected_shift_dataset(
    seed: int, true_shift: int, n: int = 80
) -> tuple[Dataset, np.ndarray]:
    """Build a dataset with conductivity `true_shift` samples off temperature.

    `sea_water_temperature` is set to an aperiodic noise sequence and the
    "true" (aligned) conductivity to a smooth trend plus that same noise,
    so that only shifting the measured conductivity by exactly
    `-true_shift` cancels the noise and recovers the smooth trend (near
    zero high-pass residual); any other candidate lag leaves mismatched,
    independent noise samples (large high-pass residual). Returns the
    dataset and the unmutated measured-conductivity array (for computing
    an expected post-correction value in tests).
    """
    rng = np.random.default_rng(seed)
    trend = np.arange(n) * 0.01
    noise = rng.normal(0.0, 1.0, size=n)
    conductivity_true = trend + noise
    conductivity_measured = shift_array(conductivity_true, true_shift)
    sea_pressure = np.arange(float(n))
    dataset = _dataset(conductivity_measured, noise, sea_pressure)
    return dataset, conductivity_measured


def test_calculate_ct_lag_recovers_injected_shift_pooled_across_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chosen lag reflects residuals pooled across every profile at once."""
    _stub_salinity(monkeypatch)
    profile_length = 80
    n = profile_length * 2
    true_shift = 4
    rng = np.random.default_rng(2)
    trend = np.tile(np.arange(profile_length) * 0.01, 2)
    noise = rng.normal(0.0, 1.0, size=n)
    conductivity_measured = shift_array(trend + noise, true_shift)
    dataset = _dataset(
        conductivity_measured,
        noise,
        np.tile(np.arange(float(profile_length)), 2),
    )
    profiles = [
        Profile(
            down_start=0,
            down_end=profile_length,
            up_start=profile_length,
            up_end=profile_length,
        ),
        Profile(down_start=profile_length, down_end=n, up_start=n, up_end=n),
    ]
    settings = CTLagSettings(min_lag=-10, max_lag=10, window_length=5)

    lag = calculate_ct_lag(dataset, profiles, settings)

    assert lag == -true_shift


def test_calculate_ct_lag_respects_sea_pressure_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restricting sea_pressure still recovers the injected shift."""
    _stub_salinity(monkeypatch)
    dataset, _ = _injected_shift_dataset(seed=1, true_shift=-2)
    profile = Profile(down_start=0, down_end=80, up_start=80, up_end=80)
    settings = CTLagSettings(
        min_lag=-10,
        max_lag=10,
        window_length=5,
        sea_pressure_min=10.0,
        sea_pressure_max=70.0,
    )

    lag = calculate_ct_lag(dataset, [profile], settings)

    assert lag == 2


@pytest.mark.parametrize(
    "missing_channel",
    [
        "sea_water_electrical_conductivity",
        "sea_water_temperature",
        "sea_pressure",
    ],
)
def test_calculate_ct_lag_missing_channel_raises(missing_channel: str) -> None:
    """A missing required channel raises ValueError naming it."""
    n = 5
    channels = {
        "sea_water_electrical_conductivity": np.arange(float(n)),
        "sea_water_temperature": np.arange(float(n)),
        "sea_pressure": np.arange(float(n)),
    }
    del channels[missing_channel]
    dataset = Dataset(time=Channel(data=np.arange(float(n))))
    for name, data in channels.items():
        dataset.add_channel(name, Channel(data=data))

    with pytest.raises(ValueError, match=missing_channel):
        calculate_ct_lag(dataset, [], CTLagSettings())


def test_calculate_ct_lag_raises_when_no_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty profiles list leaves nothing to score."""
    _stub_salinity(monkeypatch)
    n = 10
    dataset = _dataset(np.arange(float(n)), np.zeros(n), np.arange(float(n)))

    with pytest.raises(ValueError, match="no finite salinity residuals"):
        calculate_ct_lag(dataset, [], CTLagSettings())


def test_calculate_ct_lag_raises_when_sea_pressure_range_excludes_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sea_pressure_min/max excluding every sample raises the same error."""
    _stub_salinity(monkeypatch)
    n = 10
    dataset = _dataset(np.arange(float(n)), np.zeros(n), np.arange(float(n)))
    profile = Profile(down_start=0, down_end=n, up_start=n, up_end=n)
    settings = CTLagSettings(sea_pressure_min=100.0)

    with pytest.raises(ValueError, match="no finite salinity residuals"):
        calculate_ct_lag(dataset, [profile], settings)


def test_calculate_ct_lag_tie_break_prefers_smallest_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tie across the whole search range is broken toward lag 0.

    Unlike pyrsktools' calculateCTlag (which takes abs() of the winning
    lag itself, silently flipping the sign of a negative unique winner),
    this picks the signed, smallest-magnitude lag among ties.
    """
    _stub_salinity(monkeypatch)
    n = 40
    dataset = _dataset(np.full(n, 35.0), np.full(n, 10.0), np.arange(float(n)))
    profile = Profile(down_start=0, down_end=n, up_start=n, up_end=n)
    settings = CTLagSettings(min_lag=-10, max_lag=10, window_length=5)

    lag = calculate_ct_lag(dataset, [profile], settings)

    assert lag == 0


def test_process_ct_lag_disabled_skips_and_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """settings.enabled=False leaves the dataset untouched."""
    dataset = _dataset(np.arange(5.0), np.arange(5.0), np.arange(5.0))
    original = dataset.channels["sea_water_electrical_conductivity"].data.copy()
    caplog.set_level(logging.INFO, logger="ctd_processing.process.ct_lag")

    result = process_ct_lag(dataset, [], CTLagSettings(enabled=False))

    assert result is dataset
    assert np.array_equal(
        result.channels["sea_water_electrical_conductivity"].data, original
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any("not enabled" in m for m in messages)


def test_process_ct_lag_enabled_applies_calculated_shift(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """settings.enabled=True shifts conductivity by the calculated lag."""
    _stub_salinity(monkeypatch)
    true_shift = 2
    dataset, conductivity_measured = _injected_shift_dataset(
        seed=3, true_shift=true_shift
    )
    profile = Profile(down_start=0, down_end=80, up_start=80, up_end=80)
    settings = CTLagSettings(
        enabled=True, min_lag=-10, max_lag=10, window_length=5
    )
    # Set the more permissive (lower) level last: caplog.set_level shares
    # one handler across calls, so calling it for INFO after VERBOSE would
    # raise the shared threshold back up and filter out the VERBOSE record
    # checked below.
    caplog.set_level(logging.INFO, logger="ctd_processing.process.ct_lag")
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    result = process_ct_lag(dataset, [profile], settings)

    assert result is dataset
    expected = shift_array(conductivity_measured, -true_shift)
    actual = result.channels["sea_water_electrical_conductivity"].data
    np.testing.assert_array_equal(actual[20:-20], expected[20:-20])

    history = result.channels["sea_water_electrical_conductivity"].history
    assert any("shifted by" in entry for entry in history)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Calculated CT lag" in m for m in messages)
    assert any(record.levelno == VERBOSE for record in caplog.records)
