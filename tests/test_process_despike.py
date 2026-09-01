"""Tests for ctd_processing.process.despike."""

import numpy as np
import pytest
from pydantic import ValidationError

from ctd_processing.config import DespikeSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.despike import despike_array, despike_channel


def test_despike_array_replaces_single_spike_with_nan() -> None:
    """A single large spike is detected and replaced with NaN."""
    data = np.array([1.0, 1.0, 1.0, 50.0, 1.0])

    result, count = despike_array(
        data, DespikeSettings(threshold=2.0, window_length=3)
    )

    assert count == 1
    assert np.isnan(result[3])
    assert np.array_equal(np.delete(result, 3), np.delete(data, 3))


def test_despike_array_no_spikes_returns_unchanged_data() -> None:
    """Constant data has no spikes; nothing is replaced."""
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result, count = despike_array(data, DespikeSettings())

    assert count == 0
    assert np.array_equal(result, data)


def test_despike_array_zero_residual_variability_is_a_noop() -> None:
    """No variability in the residual (sd == 0) means nothing to flag."""
    data = np.full(5, 3.0)

    result, count = despike_array(data, DespikeSettings())

    assert count == 0
    assert np.array_equal(result, data)


def test_despike_array_does_not_mutate_input() -> None:
    """despike_array returns a new array; the input is untouched."""
    data = np.array([1.0, 1.0, 1.0, 50.0, 1.0])
    original = data.copy()

    despike_array(data, DespikeSettings(threshold=2.0, window_length=3))

    assert np.array_equal(data, original)


def test_despike_array_iteration_reveals_a_second_spike() -> None:
    """A second, smaller spike is only caught once the first is removed.

    The first pass's residual std is dominated by the huge spike at
    index 3, so the smaller spike at index 7 doesn't cross the
    threshold until index 3 is NaN'd out and std drops.
    """
    data = np.array([1.0, 1.0, 1.0, 1000.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
    settings = DespikeSettings(threshold=2.0, window_length=3, max_iterations=5)

    result, count = despike_array(data, settings)

    assert count == 2
    assert np.isnan(result[3])
    assert np.isnan(result[7])


def test_despike_array_single_pass_misses_the_masked_spike() -> None:
    """max_iterations=1 only catches the spike a single pass can see."""
    data = np.array([1.0, 1.0, 1.0, 1000.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
    settings = DespikeSettings(threshold=2.0, window_length=3, max_iterations=1)

    result, count = despike_array(data, settings)

    assert count == 1
    assert np.isnan(result[3])
    assert result[7] == 5.0


def test_despike_array_bails_out_early_once_converged() -> None:
    """Extra iterations beyond convergence change nothing further."""
    data = np.array([1.0, 1.0, 1.0, 1000.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0])

    few = despike_array(
        data, DespikeSettings(threshold=2.0, window_length=3, max_iterations=2)
    )
    many = despike_array(
        data, DespikeSettings(threshold=2.0, window_length=3, max_iterations=50)
    )

    assert few[1] == many[1]
    assert np.array_equal(few[0], many[0], equal_nan=True)


def test_despike_settings_rejects_even_window_length() -> None:
    """An even window_length fails validation."""
    with pytest.raises(ValidationError):
        DespikeSettings(window_length=4)


def test_despike_channel_rejects_non_floating_dtype() -> None:
    """An integer-dtype channel raises ValueError instead of corrupting."""
    channel = Channel(data=np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="floating-point"):
        despike_channel(channel, DespikeSettings())


def test_despike_channel_mutates_in_place_and_returns_same_object() -> None:
    """despike_channel mutates channel.data in place and returns `channel`."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0, 50.0, 1.0]))
    original_data = channel.data

    result = despike_channel(
        channel, DespikeSettings(threshold=2.0, window_length=3)
    )

    assert result is channel
    assert result.data is original_data
    assert np.isnan(result.data[3])


def test_despike_channel_records_count_in_history() -> None:
    """Exactly one history entry is added, naming the count replaced."""
    channel = Channel(
        data=np.array([1.0, 1.0, 1.0, 50.0, 1.0]), history=["loaded"]
    )

    result = despike_channel(
        channel, DespikeSettings(threshold=2.0, window_length=3)
    )

    assert result.history == ["loaded", "despiked 1 point(s)"]


def test_despike_channel_no_spikes_adds_no_history() -> None:
    """No history entry is added when nothing was despiked."""
    channel = Channel(
        data=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), history=["loaded"]
    )

    result = despike_channel(channel, DespikeSettings())

    assert result.history == ["loaded"]


def test_despike_channel_logs_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """despike_channel logs the replaced count at VERBOSE."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0, 50.0, 1.0]))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.despike")

    despike_channel(channel, DespikeSettings(threshold=2.0, window_length=3))

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "despiked 1 point(s)"


def test_despike_channel_no_spikes_still_logs_zero_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 0-count VERBOSE record is still emitted when nothing was despiked."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.despike")

    despike_channel(channel, DespikeSettings())

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "despiked 0 point(s)"
