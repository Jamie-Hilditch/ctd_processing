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
    data = np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])

    result, count = despike_array(data, DespikeSettings())

    assert count == 1
    assert np.isnan(result[4])
    assert np.array_equal(np.delete(result, 4), np.delete(data, 4))


def test_despike_array_local_scale_window_beats_a_global_one() -> None:
    """A narrow scale_window_length localizes the spread; a wide one doesn't.

    The first 10 samples are a quiet, low-noise region with one genuine
    spike at index 5; the remaining 20 are a naturally noisier (but
    spike-free) wandering region. With `scale_window_length` set almost
    as wide as the whole array (mimicking a single whole-array MAD), the
    noisy region's larger natural spread pulls the one shared scale to a
    value that's simultaneously too tight for the noisy region (several
    of its ordinary points cross threshold) and coincidentally still
    catches the real spike. A narrow, local `scale_window_length`
    resolves each region's own local spread instead: it flags only the
    genuine spike, and leaves the entire noisy region alone.
    """
    data = np.array(
        [
            0.102,
            -0.1278,
            0.0209,
            -0.0284,
            -0.0226,
            1.0,
            -0.101,
            -0.0116,
            -0.0433,
            0.1661,
            0.4446,
            0.0919,
            -0.1893,
            -0.8574,
            -1.9125,
            -2.3033,
            -1.8214,
            -2.0599,
            -1.1022,
            -1.302,
            -1.2777,
            0.2681,
            0.8132,
            0.308,
            0.1251,
            0.6657,
            2.6007,
            2.3311,
            2.0876,
            3.0899,
        ]
    )

    wide, wide_count = despike_array(
        data, DespikeSettings(scale_window_length=29, iterations=1)
    )
    narrow, narrow_count = despike_array(
        data, DespikeSettings(scale_window_length=7, iterations=1)
    )

    assert wide_count > 1
    assert np.isnan(wide[5])
    assert np.isnan(wide[1])

    assert narrow_count == 1
    assert np.isnan(narrow[5])
    assert np.array_equal(np.delete(narrow, 5), np.delete(data, 5))


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
    data = np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    original = data.copy()

    despike_array(data, DespikeSettings())

    assert np.array_equal(data, original)


def test_despike_array_single_pass_catches_both_spikes() -> None:
    """A single pass already catches a big spike and a nearby smaller one.

    Unlike a std-based scale -- which the huge spike at index 3 would
    inflate, hiding the smaller spike at index 7 until a later pass --
    the MAD-based scale isn't inflated by the spikes it's measuring, so
    the default `iterations=1` already flags both.
    """
    data = np.array([1.0, 1.02, 0.98, 1000.0, 1.03, 0.97, 1.01, 5.0, 0.99, 1.0])
    settings = DespikeSettings(iterations=1)

    result, count = despike_array(data, settings)

    assert count == 2
    assert np.isnan(result[3])
    assert np.isnan(result[7])


def test_despike_array_extra_iterations_are_a_noop() -> None:
    """Extra iterations beyond the first, converged pass change nothing."""
    data = np.array([1.0, 1.02, 0.98, 1000.0, 1.03, 0.97, 1.01, 5.0, 0.99, 1.0])

    few = despike_array(data, DespikeSettings(iterations=1))
    many = despike_array(data, DespikeSettings(iterations=50))

    assert few[1] == many[1]
    assert np.array_equal(few[0], many[0], equal_nan=True)


def test_despike_settings_rejects_even_reference_window_length() -> None:
    """An even reference_window_length fails validation."""
    with pytest.raises(ValidationError):
        DespikeSettings(reference_window_length=4)


def test_despike_settings_rejects_even_scale_window_length() -> None:
    """An even scale_window_length fails validation."""
    with pytest.raises(ValidationError):
        DespikeSettings(scale_window_length=4)


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

    result = despike_channel(channel, DespikeSettings())

    assert result is channel
    assert result.data is original_data
    assert np.isnan(result.data[4])


def test_despike_channel_records_count_in_history() -> None:
    """Exactly one history entry is added, naming the count replaced."""
    channel = Channel(
        data=np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0]),
        history=["loaded"],
    )

    result = despike_channel(channel, DespikeSettings())

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
    channel = Channel(
        data=np.array([1.0, 1.02, 0.98, 1.01, 50.0, 0.99, 1.03, 0.97, 1.0])
    )
    caplog.set_level(VERBOSE, logger="ctd_processing.process.despike")

    despike_channel(channel, DespikeSettings())

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
