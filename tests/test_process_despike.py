"""Tests for ctd_processing.process.despike."""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from ctd_processing.config import DespikeSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.despike import (
    RejectionThresholdResult,
    describe_rejection_threshold,
    despike_array,
    despike_arrays,
    despike_channel,
    resolve_rejection_threshold,
)


def _absolute_settings(reference_window_length: int = 3) -> DespikeSettings:
    """Build a deterministic DespikeSettings that skips the GPD fit entirely."""
    return DespikeSettings(
        reference_window_length=reference_window_length,
        rejection_method="absolute_threshold",
        rejection_threshold=1.0,
    )


def test_despike_array_replaces_single_spike_with_nan() -> None:
    """A single large spike is detected and replaced with NaN."""
    data = np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    settings = _absolute_settings()

    result, count, threshold_result = despike_array(data, settings)

    assert count == 1
    assert np.isnan(result[4])
    assert np.array_equal(np.delete(result, 4), np.delete(data, 4))
    assert threshold_result.threshold == 1.0


def test_despike_array_flags_multiple_spikes_in_one_pass() -> None:
    """A single, non-iterated pass already flags every spike at once.

    There is no repeat/iterate option (see the module docstring): the GPD
    fit depends on the tail's most extreme points, so a second pass after
    masking would remove exactly the evidence it needs.
    """
    data = np.array([1.0, 1.02, 0.98, 1000.0, 1.03, 0.97, 1.01, 5.0, 0.99, 1.0])
    settings = _absolute_settings()

    result, count, _ = despike_array(data, settings)

    assert count == 2
    assert np.isnan(result[3])
    assert np.isnan(result[7])


def test_despike_array_no_spikes_returns_unchanged_data() -> None:
    """Constant data has no spikes; nothing is replaced."""
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result, count, _ = despike_array(data, DespikeSettings())

    assert count == 0
    assert np.array_equal(result, data)


def test_despike_array_does_not_mutate_input() -> None:
    """despike_array returns a new array; the input is untouched."""
    data = np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    original = data.copy()
    settings = _absolute_settings()

    despike_array(data, settings)

    assert np.array_equal(data, original)


def test_despike_arrays_pools_threshold_across_arrays() -> None:
    """One shared threshold applies to every array; flagging stays per-array.

    The spike sits in the middle of `a`, away from the array boundary, so
    the edge-clipped rolling median doesn't also drag a neighboring
    sample's residual over threshold (see
    `test_process_raw_channel_applies_despike_last` for that edge case).
    """
    settings = _absolute_settings()
    a = np.array([1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    despiked, counts, result = despike_arrays([a, b], settings)

    assert counts == [1, 0]
    assert np.isnan(despiked[0][3])
    assert np.array_equal(despiked[1], b)
    assert result.threshold == 1.0


def test_despike_arrays_pools_residuals_across_all_arrays() -> None:
    """The threshold is resolved from every array's combined residuals."""
    settings = DespikeSettings(
        reference_window_length=3,
        fit_percentile=10.0,
        rejection_method="raw_percentile",
        rejection_percentile=50.0,
    )
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])

    _, _, result = despike_arrays([a, b], settings)

    assert result.n == 6


def test_despike_arrays_empty_list_is_a_noop() -> None:
    """An empty list of arrays despikes to an empty list, no crash."""
    despiked, counts, result = despike_arrays([], DespikeSettings())

    assert despiked == []
    assert counts == []
    assert np.isnan(result.threshold)


def test_resolve_rejection_threshold_absolute_threshold_ignores_data() -> None:
    """`absolute_threshold` returns the configured value, untouched by data."""
    residual = np.array([1.0, 2.0, 1000.0])
    settings = DespikeSettings(
        rejection_method="absolute_threshold", rejection_threshold=5.0
    )

    result = resolve_rejection_threshold(residual, settings)

    assert result == RejectionThresholdResult(
        threshold=5.0,
        method="absolute_threshold",
        used_fallback=False,
        shape=None,
        scale=None,
        n=None,
        n_exceedances=None,
    )


def test_resolve_rejection_threshold_raw_percentile_matches_numpy() -> None:
    """`raw_percentile` is a plain `numpy.percentile` call, no GPD fit."""
    residual = np.array([-1.0, 2.0, -3.0, 4.0, -5.0, 100.0])
    settings = DespikeSettings(
        fit_percentile=10.0,
        rejection_method="raw_percentile",
        rejection_percentile=90.0,
    )

    result = resolve_rejection_threshold(residual, settings)

    assert result.method == "raw_percentile"
    assert result.threshold == pytest.approx(
        np.percentile(np.abs(residual), 90.0)
    )
    assert result.shape is None
    assert result.scale is None
    assert result.n == residual.size


def test_resolve_rejection_threshold_gpd_falls_back_below_min_exceedances() -> (
    None
):
    """Too few exceedances above `fit_percentile` falls back to a percentile."""
    residual = np.concatenate([np.zeros(50), np.array([10.0])])
    settings = DespikeSettings(
        fit_percentile=50.0, rejection_percentile=99.0, min_exceedances=200
    )

    result = resolve_rejection_threshold(residual, settings)

    assert result.method == "gpd_percentile"
    assert result.used_fallback
    assert result.shape is None
    assert result.scale is None
    assert result.threshold == pytest.approx(
        np.percentile(np.abs(residual), 99.0)
    )


def test_resolve_rejection_threshold_gpd_fit_recovers_known_shape() -> None:
    """A synthetic GPD-tailed sample's fitted shape lands near the true value.

    Peaks-over-threshold exceedances of a Generalized Pareto Distribution
    are themselves Generalized Pareto with the same shape parameter, so
    fitting above `fit_percentile` should recover something close to the
    true `xi` used to generate the sample.
    """
    rng = np.random.default_rng(0)
    true_shape, true_scale = 0.4, 1.5
    residual = stats.genpareto.rvs(
        true_shape, scale=true_scale, size=5000, random_state=rng
    )
    settings = DespikeSettings(
        fit_percentile=90.0, rejection_percentile=99.9, min_exceedances=50
    )

    result = resolve_rejection_threshold(residual, settings)

    assert result.method == "gpd_percentile"
    assert not result.used_fallback
    assert result.shape == pytest.approx(true_shape, abs=0.25)
    assert result.n == 5000
    assert result.n_exceedances == pytest.approx(500, rel=0.2)


def test_describe_rejection_threshold_formats_gpd_fit() -> None:
    """A successful GPD fit's description names threshold, xi, sigma, n."""
    settings = DespikeSettings()
    result = RejectionThresholdResult(
        threshold=0.5,
        method="gpd_percentile",
        used_fallback=False,
        shape=0.3,
        scale=0.1,
        n=1000,
        n_exceedances=100,
    )

    assert describe_rejection_threshold(settings, result) == (
        "GPD fit: threshold=0.5, xi=0.3, sigma=0.1, n=1000, n_exceedances=100"
    )


def test_describe_rejection_threshold_formats_fallback() -> None:
    """A GPD fallback's description names the unmet min_exceedances."""
    settings = DespikeSettings(min_exceedances=200)
    result = RejectionThresholdResult(
        threshold=0.2,
        method="gpd_percentile",
        used_fallback=True,
        shape=None,
        scale=None,
        n=50,
        n_exceedances=5,
    )

    assert describe_rejection_threshold(settings, result) == (
        "empirical percentile fallback: threshold=0.2, n=50, n_exceedances=5 "
        "< min_exceedances=200"
    )


def test_describe_rejection_threshold_formats_raw_percentile() -> None:
    """A raw-percentile description names the percentile used, no xi/sigma."""
    settings = DespikeSettings(
        rejection_method="raw_percentile", rejection_percentile=99.0
    )
    result = RejectionThresholdResult(
        threshold=0.3,
        method="raw_percentile",
        used_fallback=False,
        shape=None,
        scale=None,
        n=500,
        n_exceedances=None,
    )

    assert describe_rejection_threshold(settings, result) == (
        "raw percentile: threshold=0.3, percentile=99, n=500"
    )


def test_describe_rejection_threshold_formats_absolute_threshold() -> None:
    """An absolute-threshold description names only the configured value."""
    settings = DespikeSettings(
        rejection_method="absolute_threshold", rejection_threshold=5.0
    )
    result = RejectionThresholdResult(
        threshold=5.0,
        method="absolute_threshold",
        used_fallback=False,
        shape=None,
        scale=None,
        n=None,
        n_exceedances=None,
    )

    assert (
        describe_rejection_threshold(settings, result) == "absolute threshold=5"
    )


def test_despike_settings_rejects_even_reference_window_length() -> None:
    """An even reference_window_length fails validation."""
    with pytest.raises(ValidationError):
        DespikeSettings(reference_window_length=4)


def test_despike_channel_rejects_non_floating_dtype() -> None:
    """An integer-dtype channel raises ValueError instead of corrupting."""
    channel = Channel(data=np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="floating-point"):
        despike_channel(channel, DespikeSettings())


def test_despike_channel_mutates_in_place_and_returns_same_object() -> None:
    """despike_channel mutates channel.data in place and returns `channel`."""
    channel = Channel(
        data=np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    )
    original_data = channel.data
    settings = _absolute_settings()

    result = despike_channel(channel, settings)

    assert result is channel
    assert result.data is original_data
    assert np.isnan(result.data[4])


def test_despike_channel_records_count_in_history() -> None:
    """Exactly one history entry is added, naming the count and threshold."""
    channel = Channel(
        data=np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0]),
        history=["loaded"],
    )
    settings = _absolute_settings()

    result = despike_channel(channel, settings)

    assert result.history == [
        "loaded",
        "despiked 1 point(s) (absolute threshold=1)",
    ]


def test_despike_channel_no_spikes_adds_no_history() -> None:
    """No history entry is added when nothing was despiked."""
    channel = Channel(
        data=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), history=["loaded"]
    )
    settings = _absolute_settings()

    result = despike_channel(channel, settings)

    assert result.history == ["loaded"]


def test_despike_channel_logs_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """despike_channel logs the replaced count and threshold at VERBOSE."""
    channel = Channel(
        data=np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    )
    settings = _absolute_settings()
    caplog.set_level(VERBOSE, logger="ctd_processing.process.despike")

    despike_channel(channel, settings)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "despiked 1 point(s) (absolute threshold=1)"


def test_despike_channel_no_spikes_still_logs_zero_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 0-count VERBOSE record is still emitted when nothing was despiked."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    settings = _absolute_settings()
    caplog.set_level(VERBOSE, logger="ctd_processing.process.despike")

    despike_channel(channel, settings)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "despiked 0 point(s) (absolute threshold=1)"
