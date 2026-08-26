"""Tests for ctd_processing.process.dataset."""

import numpy as np
import pytest

from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset


def _time_channel(values: list[float]) -> Channel:
    return Channel(data=np.array(values, dtype=float))


def test_dataset_seeds_channels_and_length_from_time() -> None:
    """Constructing with just `time` populates channels and length."""
    time = _time_channel([0.0, 1.0, 2.0])

    dataset = Dataset(time=time)

    assert dataset.channels == {"time": time}
    assert dataset.length == 3
    assert dataset.metadata == {}
    assert dataset.history == []


def test_dataset_rejects_non_increasing_time() -> None:
    """A flat or decreasing time channel raises ValueError."""
    with pytest.raises(ValueError, match="increasing"):
        Dataset(time=_time_channel([0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="increasing"):
        Dataset(time=_time_channel([2.0, 1.0, 0.0]))


def test_dataset_metadata_and_history_are_independent() -> None:
    """Mutable defaults are not shared between Dataset instances."""
    a = Dataset(time=_time_channel([0.0, 1.0]))
    b = Dataset(time=_time_channel([0.0, 1.0]))

    a.metadata["project"] = "cruise-1"
    a.history.append("step")

    assert b.metadata == {}
    assert b.history == []


def test_channels_and_length_are_not_constructor_arguments() -> None:
    """channels/length can only be set by __post_init__, not the caller."""
    with pytest.raises(TypeError):
        Dataset(time=_time_channel([0.0, 1.0]), channels={})  # ty: ignore

    with pytest.raises(TypeError):
        Dataset(time=_time_channel([0.0, 1.0]), length=5)  # ty: ignore


def test_record_appends_to_history() -> None:
    """record() appends the given description to history."""
    dataset = Dataset(time=_time_channel([0.0, 1.0]))

    dataset.record("first step")
    dataset.record("second step")

    assert dataset.history == ["first step", "second step"]


def test_add_channel_adds_and_records_history() -> None:
    """add_channel stores the channel and logs it in history."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    temperature = Channel(data=np.array([10.0, 11.0, 12.0]))

    dataset.add_channel("temperature", temperature)

    assert dataset.channels["temperature"] is temperature
    assert dataset.history == ["added channel 'temperature'"]


def test_add_channel_rejects_duplicate_name() -> None:
    """Adding a channel under an already-used name raises ValueError."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    dataset.add_channel("temperature", Channel(data=np.array([1.0, 2.0, 3.0])))

    with pytest.raises(ValueError, match="temperature"):
        dataset.add_channel(
            "temperature", Channel(data=np.array([4.0, 5.0, 6.0]))
        )

    with pytest.raises(ValueError, match="time"):
        dataset.add_channel("time", Channel(data=np.array([1.0, 2.0, 3.0])))


def test_add_channel_rejects_mismatched_length() -> None:
    """A channel whose length doesn't match the dataset's raises ValueError."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="length"):
        dataset.add_channel("temperature", Channel(data=np.array([1.0, 2.0])))


def test_remove_channel_pops_and_records_history() -> None:
    """remove_channel returns and removes the channel, logging it."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    temperature = Channel(data=np.array([10.0, 11.0, 12.0]))
    dataset.add_channel("temperature", temperature)

    removed = dataset.remove_channel("temperature")

    assert removed is temperature
    assert "temperature" not in dataset.channels
    assert dataset.history[-1] == "removed channel 'temperature'"


def test_remove_channel_missing_name_raises_keyerror() -> None:
    """Removing an unknown channel name raises KeyError."""
    dataset = Dataset(time=_time_channel([0.0, 1.0]))

    with pytest.raises(KeyError):
        dataset.remove_channel("does_not_exist")


def test_remove_channel_time_raises_valueerror() -> None:
    """Removing the 'time' channel is rejected."""
    dataset = Dataset(time=_time_channel([0.0, 1.0]))

    with pytest.raises(ValueError, match="time"):
        dataset.remove_channel("time")


def test_subset_applies_same_indices_to_every_channel() -> None:
    """subset() slices time and every added channel identically."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0, 3.0]))
    dataset.add_channel(
        "temperature", Channel(data=np.array([10.0, 11.0, 12.0, 13.0]))
    )

    result = dataset.subset(slice(1, 3), "extracted profile")

    assert np.array_equal(result.time.data, [1.0, 2.0])
    assert np.array_equal(result.channels["temperature"].data, [11.0, 12.0])
    assert result.length == 2


def test_subset_returned_dataset_is_independent_of_source() -> None:
    """Mutating the subset result does not affect the source dataset."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    dataset.add_channel(
        "temperature", Channel(data=np.array([10.0, 11.0, 12.0]))
    )

    result = dataset.subset(slice(None), "full copy")
    result.channels["temperature"].data[0] = 999.0
    result.time.data[0] = 999.0

    assert dataset.channels["temperature"].data[0] == 10.0
    assert dataset.time.data[0] == 0.0


def test_subset_history_and_metadata_independently_copied() -> None:
    """subset() appends to a copy of history/metadata, not the source's."""
    dataset = Dataset(
        time=_time_channel([0.0, 1.0, 2.0]),
        metadata={"project": "cruise-1"},
        history=["loaded"],
    )

    result = dataset.subset(slice(None), "extracted profile")
    result.metadata["project"] = "cruise-2"
    result.history.append("further step")

    assert result.history == ["loaded", "extracted profile", "further step"]
    assert dataset.history == ["loaded"]
    assert dataset.metadata == {"project": "cruise-1"}
    assert result.metadata == {"project": "cruise-2"}


def test_repr_names_channels_without_dumping_them() -> None:
    """repr() lists channel names instead of each channel's own repr."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    dataset.add_channel(
        "temperature", Channel(data=np.array([10.0, 11.0, 12.0]))
    )

    text = repr(dataset)

    assert "'time'" in text
    assert "'temperature'" in text
    assert "length=3" in text


def test_str_gives_concise_human_readable_summary() -> None:
    """str() gives a short summary of sample count and channel names."""
    dataset = Dataset(time=_time_channel([0.0, 1.0, 2.0]))
    dataset.add_channel(
        "temperature", Channel(data=np.array([10.0, 11.0, 12.0]))
    )

    text = str(dataset)

    assert "3 samples" in text
    assert "2 channel(s)" in text
    assert "temperature" in text
