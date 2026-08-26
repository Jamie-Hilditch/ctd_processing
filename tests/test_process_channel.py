"""Tests for ctd_processing.process.channel."""

import numpy as np
import pytest

from ctd_processing.process.channel import Channel


def test_channel_accepts_1d_array() -> None:
    """A 1D array is accepted, with empty metadata/history by default."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]))

    assert np.array_equal(channel.data, [1.0, 2.0, 3.0])
    assert channel.metadata == {}
    assert channel.history == []


def test_channel_rejects_2d_array() -> None:
    """A non-1D array raises ValueError."""
    with pytest.raises(ValueError, match="1D"):
        Channel(data=np.zeros((2, 2)))


def test_channel_default_metadata_and_history_are_independent() -> None:
    """Mutable defaults are not shared between Channel instances."""
    a = Channel(data=np.array([1.0]))
    b = Channel(data=np.array([2.0]))

    a.metadata["units"] = "dbar"
    a.history.append("step")

    assert b.metadata == {}
    assert b.history == []


def test_record_appends_to_history() -> None:
    """record() appends the given description to history."""
    channel = Channel(data=np.array([1.0, 2.0]))

    channel.record("first step")
    channel.record("second step")

    assert channel.history == ["first step", "second step"]


def test_subset_with_boolean_mask() -> None:
    """subset() with a boolean mask keeps the selected elements."""
    channel = Channel(data=np.array([10.0, 20.0, 30.0, 40.0]))

    result = channel.subset(
        np.array([True, False, True, False]), "boolean mask subset"
    )

    assert np.array_equal(result.data, [10.0, 30.0])


def test_subset_with_integer_array() -> None:
    """subset() with an integer index array keeps the selected elements."""
    channel = Channel(data=np.array([10.0, 20.0, 30.0, 40.0]))

    result = channel.subset(np.array([3, 1]), "integer array subset")

    assert np.array_equal(result.data, [40.0, 20.0])


def test_subset_with_slice() -> None:
    """subset() with a slice keeps the selected elements."""
    channel = Channel(data=np.array([10.0, 20.0, 30.0, 40.0]))

    result = channel.subset(slice(1, 3), "slice subset")

    assert np.array_equal(result.data, [20.0, 30.0])


def test_subset_data_is_independent_of_source() -> None:
    """Mutating subset data does not affect the source, and vice versa."""
    source = Channel(data=np.array([1.0, 2.0, 3.0]))

    result = source.subset(slice(None), "full copy")
    result.data[0] = 999.0

    assert source.data[0] == 1.0

    source.data[1] = -1.0

    assert result.data[1] == 2.0


def test_subset_history_appends_without_mutating_source() -> None:
    """subset() appends description on top of a copy of the source history."""
    source = Channel(data=np.array([1.0, 2.0, 3.0]), history=["loaded"])

    result = source.subset(slice(None), "extracted profile")

    assert result.history == ["loaded", "extracted profile"]
    assert source.history == ["loaded"]

    result.history.append("further step")

    assert source.history == ["loaded"]


def test_subset_metadata_is_copied_without_mutating_source() -> None:
    """subset() copies metadata; mutating the copy leaves the source intact."""
    source = Channel(data=np.array([1.0, 2.0, 3.0]), metadata={"units": "dbar"})

    result = source.subset(slice(None), "extracted profile")
    result.metadata["units"] = "m"

    assert source.metadata == {"units": "dbar"}
    assert result.metadata == {"units": "m"}


@pytest.mark.parametrize(
    ("values", "increasing", "decreasing", "monotonic"),
    [
        ([1.0, 2.0, 3.0], True, False, True),
        ([3.0, 2.0, 1.0], False, True, True),
        ([1.0, 3.0, 2.0], False, False, False),
        ([1.0, 1.0, 1.0], False, False, False),
        ([], True, True, True),
        ([1.0], True, True, True),
    ],
)
def test_monotonicity_methods_on_float_data(
    values: list[float], increasing: bool, decreasing: bool, monotonic: bool
) -> None:
    """is_increasing/is_decreasing/is_monotonic match expected results."""
    channel = Channel(data=np.array(values, dtype=float))

    assert channel.is_increasing() is increasing
    assert channel.is_decreasing() is decreasing
    assert channel.is_monotonic() is monotonic


@pytest.mark.parametrize(
    ("timestamps", "increasing", "decreasing"),
    [
        (
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            True,
            False,
        ),
        (
            ["2020-01-03", "2020-01-02", "2020-01-01"],
            False,
            True,
        ),
        (
            ["2020-01-01", "2020-01-01", "2020-01-02"],
            False,
            False,
        ),
    ],
)
def test_monotonicity_methods_on_datetime64_data(
    timestamps: list[str], increasing: bool, decreasing: bool
) -> None:
    """Monotonicity checks work on datetime64 data via the int64 view."""
    channel = Channel(data=np.array(timestamps, dtype="datetime64[ms]"))

    assert channel.is_increasing() is increasing
    assert channel.is_decreasing() is decreasing


def test_repr_summarizes_data_without_dumping_it() -> None:
    """repr() shows dtype/size instead of the raw array contents."""
    channel = Channel(data=np.arange(1000, dtype=np.float64))
    channel.record("loaded")

    text = repr(channel)

    assert "float64[1000]" in text
    assert "loaded" in text
    assert "1.0" not in text  # raw array contents not dumped


def test_str_gives_concise_human_readable_summary() -> None:
    """str() gives a short summary of sample count, dtype, and history."""
    channel = Channel(data=np.array([1.0, 2.0, 3.0]))
    channel.record("loaded")

    text = str(channel)

    assert "3 samples" in text
    assert "float64" in text
    assert "1 processing step" in text
