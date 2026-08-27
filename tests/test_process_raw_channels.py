"""Tests for ctd_processing.process.raw_channels."""

import numpy as np
import pytest

from ctd_processing.config import (
    DespikeSettings,
    GeolocationSettings,
    ProcessSettings,
    RawChannelSettings,
)
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.raw_channels import (
    add_offset,
    process_raw_channel,
    process_raw_channels,
    remove_holds,
    shift_time,
)

_GEOLOCATION = GeolocationSettings(
    reference_latitude=0.0, reference_longitude=0.0
)


def test_remove_holds_single_repeated_pair() -> None:
    """A single repeat is replaced with NaN; the run's first value is not."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))

    result = remove_holds(channel)

    assert np.array_equal(result.data, [1.0, np.nan, 2.0], equal_nan=True)


def test_remove_holds_long_run_keeps_only_first() -> None:
    """A run of N identical values keeps only the first; the rest are NaN.

    This is the case that distinguishes tracking the last known-valid
    value from naively comparing each element to its immediate
    predecessor (which would under-count runs longer than 2).
    """
    channel = Channel(data=np.array([5.0, 5.0, 5.0, 5.0]))

    result = remove_holds(channel)

    assert np.array_equal(
        result.data, [5.0, np.nan, np.nan, np.nan], equal_nan=True
    )


def test_remove_holds_alternating_repeats() -> None:
    """Matches pyrsktools' correcthold behavior on alternating repeats."""
    channel = Channel(data=np.array([5.0, 5.0, 6.0, 6.0, 5.0, 5.0]))

    result = remove_holds(channel)

    assert np.array_equal(
        result.data,
        [5.0, np.nan, 6.0, np.nan, 5.0, np.nan],
        equal_nan=True,
    )


def test_remove_holds_no_holds_present() -> None:
    """Strictly-changing data is left untouched."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]))

    result = remove_holds(channel)

    assert np.array_equal(result.data, [1.0, 2.0, 3.0])
    assert result.history == ["removed 0 zero-order hold value(s)"]


@pytest.mark.parametrize("values", [[], [1.0]])
def test_remove_holds_empty_and_single_element_are_no_ops(
    values: list[float],
) -> None:
    """Empty and single-element arrays don't error and have no holds."""
    channel = Channel(data=np.array(values))

    result = remove_holds(channel)

    assert np.array_equal(result.data, values)
    assert result.history == ["removed 0 zero-order hold value(s)"]


def test_remove_holds_preexisting_nan_breaks_the_run() -> None:
    """A pre-existing NaN resets the last-valid tracker, like correcthold.

    Index 1 repeats index 0, so it's flagged. Index 3 repeats the value
    from before the NaN, but since NaN != anything, it is NOT flagged.
    """
    channel = Channel(data=np.array([1.0, 1.0, np.nan, 1.0]))

    result = remove_holds(channel)

    assert np.array_equal(
        result.data, [1.0, np.nan, np.nan, 1.0], equal_nan=True
    )
    assert result.history == ["removed 1 zero-order hold value(s)"]


def test_remove_holds_mutates_in_place_and_returns_same_object() -> None:
    """remove_holds mutates channel.data in place and returns `channel`."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))
    original_data = channel.data

    result = remove_holds(channel)

    assert result is channel
    assert result.data is original_data


def test_remove_holds_records_count_in_history() -> None:
    """Exactly one history entry is added, naming the count removed."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0]), history=["loaded"])

    result = remove_holds(channel)

    assert result.history == ["loaded", "removed 2 zero-order hold value(s)"]


def test_remove_holds_rejects_non_floating_dtype() -> None:
    """An integer-dtype channel raises ValueError instead of corrupting."""
    channel = Channel(data=np.array([1, 1, 2], dtype=np.int64))

    with pytest.raises(ValueError, match="floating-point"):
        remove_holds(channel)


def test_remove_holds_logs_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """remove_holds logs the removed count at VERBOSE."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0]))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    remove_holds(channel)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "removed 2 zero-order hold value(s)"


@pytest.mark.parametrize(
    ("offset", "expected"), [(1.5, [2.5, 3.5]), (-1.0, [0.0, 1.0])]
)
def test_add_offset_adds_value(offset: float, expected: list[float]) -> None:
    """add_offset adds a positive or negative offset to every element."""
    channel = Channel(data=np.array([1.0, 2.0]))

    result = add_offset(channel, offset)

    assert np.array_equal(result.data, expected)


def test_add_offset_preserves_nan() -> None:
    """Adding an offset to NaN leaves it as NaN (composes with remove_holds)."""
    channel = Channel(data=np.array([1.0, np.nan, 3.0]))

    result = add_offset(channel, 1.0)

    assert np.array_equal(result.data, [2.0, np.nan, 4.0], equal_nan=True)


def test_add_offset_mutates_in_place_and_returns_same_object() -> None:
    """add_offset mutates channel.data in place and returns `channel`."""
    channel = Channel(data=np.array([1.0, 2.0]))
    original_data = channel.data

    result = add_offset(channel, 1.0)

    assert result is channel
    assert result.data is original_data


def test_add_offset_records_value_in_history() -> None:
    """Exactly one history entry is added, naming the offset."""
    channel = Channel(data=np.array([1.0, 2.0]), history=["loaded"])

    result = add_offset(channel, 1.5)

    assert result.history == ["loaded", "added offset 1.5"]


def test_add_offset_rejects_non_floating_dtype() -> None:
    """An integer-dtype channel raises ValueError instead of corrupting."""
    channel = Channel(data=np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="floating-point"):
        add_offset(channel, 1.0)


def test_add_offset_logs_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """add_offset logs the offset at VERBOSE."""
    channel = Channel(data=np.array([1.0, 2.0]))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    add_offset(channel, 1.5)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "added offset 1.5"


def test_shift_time_positive_shift_delays_and_pads_start() -> None:
    """A positive shift matches pandas .shift(N): NaN at the start."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0, 4.0]))

    result = shift_time(channel, 2)

    assert np.array_equal(
        result.data, [np.nan, np.nan, 1.0, 2.0], equal_nan=True
    )


def test_shift_time_negative_shift_advances_and_pads_end() -> None:
    """A negative shift matches pandas .shift(-N): NaN at the end."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0, 4.0]))

    result = shift_time(channel, -2)

    assert np.array_equal(
        result.data, [3.0, 4.0, np.nan, np.nan], equal_nan=True
    )


def test_shift_time_zero_is_a_noop() -> None:
    """shift=0 leaves the data unchanged."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0, 4.0]))

    result = shift_time(channel, 0)

    assert np.array_equal(result.data, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("shift", [10, -10])
def test_shift_time_magnitude_exceeding_length_yields_all_nan(
    shift: int,
) -> None:
    """A shift with |shift| >= len(data) turns everything into NaN."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0, 4.0]))

    result = shift_time(channel, shift)

    assert np.all(np.isnan(result.data))


def test_shift_time_rejects_non_floating_dtype() -> None:
    """An integer-dtype channel raises ValueError instead of corrupting."""
    channel = Channel(data=np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="floating-point"):
        shift_time(channel, 1)


def test_shift_time_mutates_in_place_and_returns_same_object() -> None:
    """shift_time mutates channel.data in place and returns `channel`."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0, 4.0]))
    original_data = channel.data

    result = shift_time(channel, 1)

    assert result is channel
    assert result.data is original_data


def test_shift_time_records_value_in_history() -> None:
    """Exactly one history entry is added, naming the shift."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]), history=["loaded"])

    result = shift_time(channel, -1)

    assert result.history == ["loaded", "shifted by -1 sample(s)"]


def test_shift_time_logs_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """shift_time logs the shift at VERBOSE."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]))
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    shift_time(channel, 1)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "shifted by 1 sample(s)"


def test_process_raw_channel_default_settings_removes_holds_only() -> None:
    """Default RawChannelSettings applies remove_holds but not add_offset."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))

    result = process_raw_channel(channel, RawChannelSettings())

    assert np.array_equal(result.data, [1.0, np.nan, 2.0], equal_nan=True)
    assert result.history == ["removed 1 zero-order hold value(s)"]


def test_process_raw_channel_all_disabled_is_a_noop() -> None:
    """remove_holds=False, shift=None, offset=None leaves data untouched."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))

    result = process_raw_channel(
        channel,
        RawChannelSettings(remove_holds=False, shift=None, offset=None),
    )

    assert np.array_equal(result.data, [1.0, 1.0, 2.0])
    assert result.history == []


def test_process_raw_channel_applies_all_three_in_order() -> None:
    """remove_holds runs, then shift_time, then add_offset -- in that order.

    Starting from [1.0, 1.0, 2.0, 3.0]: remove_holds NaNs index 1 ->
    [1.0, nan, 2.0, 3.0]. shift_time(1) (pandas-style) then gives
    [nan, 1.0, nan, 2.0]. add_offset(10.0) gives [nan, 11.0, nan, 12.0].
    A different step order would produce a different result, so this
    pins down the order actually executed.
    """
    channel = Channel(data=np.array([1.0, 1.0, 2.0, 3.0]))

    result = process_raw_channel(
        channel,
        RawChannelSettings(remove_holds=True, shift=1, offset=10.0),
    )

    assert np.array_equal(
        result.data, [np.nan, 11.0, np.nan, 12.0], equal_nan=True
    )
    assert result.history == [
        "removed 1 zero-order hold value(s)",
        "shifted by 1 sample(s)",
        "added offset 10.0",
    ]


def test_process_raw_channel_offset_only() -> None:
    """remove_holds=False, shift=None with an offset applies just that."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))

    result = process_raw_channel(
        channel,
        RawChannelSettings(remove_holds=False, shift=None, offset=10.0),
    )

    assert np.array_equal(result.data, [11.0, 11.0, 12.0])
    assert result.history == ["added offset 10.0"]


def test_process_raw_channel_shift_only() -> None:
    """remove_holds=False, offset=None with a shift applies just that."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]))

    result = process_raw_channel(
        channel,
        RawChannelSettings(remove_holds=False, shift=1, offset=None),
    )

    assert np.array_equal(result.data, [np.nan, 1.0, 2.0], equal_nan=True)
    assert result.history == ["shifted by 1 sample(s)"]


def test_process_raw_channel_mutates_in_place() -> None:
    """process_raw_channel mutates in place and returns `channel`."""
    channel = Channel(data=np.array([1.0, 1.0, 2.0]))
    original_data = channel.data

    result = process_raw_channel(channel, RawChannelSettings())

    assert result is channel
    assert result.data is original_data


def test_process_raw_channel_despike_is_a_noop_by_default() -> None:
    """With no despike argument, a spike is left untouched."""
    channel = Channel(data=np.array([1.0, 1.0, 1.0, 50.0, 1.0]))

    result = process_raw_channel(
        channel, RawChannelSettings(remove_holds=False)
    )

    assert result.data[3] == 50.0
    assert result.history == []


def test_process_raw_channel_applies_despike_last() -> None:
    """Despiking runs after remove_holds/shift/offset, on the corrected signal.

    remove_holds+shift(1) turns [1, 1, 2, 3, 50, 3, 3] into
    [nan, 1, nan, 2, 3, 50, 3] -- the spike ends up at (post-shift) index
    5. Despiking must find it there, proving it ran on the already
    remove_holds+shift-corrected array, not the raw input.
    """
    channel = Channel(data=np.array([1.0, 1.0, 2.0, 3.0, 50.0, 3.0, 3.0]))

    result = process_raw_channel(
        channel,
        RawChannelSettings(remove_holds=True, shift=1),
        DespikeSettings(threshold=2.0, window_length=3),
    )

    assert np.isnan(result.data[5])
    assert result.history == [
        "removed 2 zero-order hold value(s)",
        "shifted by 1 sample(s)",
        "despiked 1 point(s)",
    ]


def _dataset_with_channels() -> Dataset:
    dataset = Dataset(time=Channel(data=np.array([0.0, 1.0, 2.0])))
    dataset.add_channel(
        "sea_water_temperature", Channel(data=np.array([1.0, 1.0, 2.0]))
    )
    dataset.add_channel(
        "sea_water_practical_salinity", Channel(data=np.array([5.0, 5.0, 5.0]))
    )
    return dataset


def test_process_raw_channels_applies_configured_settings_by_name() -> None:
    """A channel with a matching raw_channels entry uses its settings."""
    dataset = _dataset_with_channels()
    settings = ProcessSettings(
        raw_channels={
            "sea_water_temperature": RawChannelSettings(
                remove_holds=False, offset=1.0
            )
        },
        geolocation=_GEOLOCATION,
    )

    result = process_raw_channels(dataset, settings)

    assert np.array_equal(
        result.channels["sea_water_temperature"].data, [2.0, 2.0, 3.0]
    )


def test_process_raw_channels_unconfigured_channel_uses_defaults() -> None:
    """A channel with no raw_channels entry still gets remove_holds=True."""
    dataset = _dataset_with_channels()

    result = process_raw_channels(
        dataset, ProcessSettings(geolocation=_GEOLOCATION)
    )

    assert np.array_equal(
        result.channels["sea_water_practical_salinity"].data,
        [5.0, np.nan, np.nan],
        equal_nan=True,
    )


def test_process_raw_channels_skips_time() -> None:
    """The 'time' channel is never processed, even if it looks like a hold."""
    dataset = _dataset_with_channels()

    result = process_raw_channels(
        dataset, ProcessSettings(geolocation=_GEOLOCATION)
    )

    assert np.array_equal(result.time.data, [0.0, 1.0, 2.0])
    assert result.time.history == []


def test_process_raw_channels_returns_same_dataset() -> None:
    """process_raw_channels mutates in place and returns `dataset`."""
    dataset = _dataset_with_channels()

    result = process_raw_channels(
        dataset, ProcessSettings(geolocation=_GEOLOCATION)
    )

    assert result is dataset


def test_process_raw_channels_despikes_only_configured_channels() -> None:
    """Only the channel named in `despike` is despiked."""
    dataset = Dataset(time=Channel(data=np.array([0.0, 1.0, 2.0, 3.0, 4.0])))
    dataset.add_channel(
        "sea_water_temperature",
        Channel(data=np.array([1.0, 1.0, 1.0, 50.0, 1.0])),
    )
    dataset.add_channel(
        "sea_water_electrical_conductivity",
        Channel(data=np.array([1.0, 1.0, 1.0, 50.0, 1.0])),
    )

    result = process_raw_channels(
        dataset,
        ProcessSettings(
            geolocation=_GEOLOCATION,
            raw_channels={
                "sea_water_temperature": RawChannelSettings(remove_holds=False),
                "sea_water_electrical_conductivity": RawChannelSettings(
                    remove_holds=False
                ),
            },
        ),
        despike={
            "sea_water_temperature": DespikeSettings(
                threshold=2.0, window_length=3
            )
        },
    )

    assert np.isnan(result.channels["sea_water_temperature"].data[3])
    assert result.channels["sea_water_electrical_conductivity"].data[3] == 50.0
