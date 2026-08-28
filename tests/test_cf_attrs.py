"""Tests for ctd_processing.cf_attrs."""

import datetime

import numpy as np

from ctd_processing.cf_attrs import (
    channel_attrs,
    dataset_attrs,
    pop_history,
    sanitize_attr,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset


def test_sanitize_attr_passes_through_str_int_float() -> None:
    """str/int/float values are returned unchanged."""
    assert sanitize_attr("text") == "text"
    assert sanitize_attr(3) == 3
    assert sanitize_attr(3.5) == 3.5


def test_sanitize_attr_unwraps_numpy_scalars() -> None:
    """Numpy scalar types (e.g. int64, not a Python int subclass) unwrap.

    Without this, `numpy.int64` would fall through to `str(value)` below,
    silently turning e.g. an instrument serial number into a string --
    which then looks like a real conflict (int vs. str) to
    `ctd_processing.bin.binning.combine_binned_profiles`'s
    ``combine_attrs="drop_conflicts"`` even when every profile actually
    agrees on the value.
    """
    result = sanitize_attr(np.int64(208532))

    assert result == 208532
    assert isinstance(result, int)
    assert not isinstance(result, np.generic)


def test_sanitize_attr_uses_isoformat_for_datetime() -> None:
    """A datetime-like value with isoformat() is converted via it."""
    value = datetime.datetime(2026, 8, 9, 3, 0, 0)

    assert sanitize_attr(value) == value.isoformat()


def test_sanitize_attr_falls_back_to_str() -> None:
    """A value with no isoformat() and not str/int/float becomes str(value)."""
    assert sanitize_attr([1, 2, 3]) == "[1, 2, 3]"


def test_dataset_attrs_drops_none_and_adds_history() -> None:
    """None-valued metadata is dropped; history is joined with '; '."""
    time = Channel(data=np.array(["2026-08-09"], dtype="datetime64[s]"))
    dataset = Dataset(time=time, metadata={"a": 1, "b": None})
    dataset.record("step one")
    dataset.record("step two")

    attrs = dataset_attrs(dataset)

    assert attrs == {"a": 1, "history": "step one; step two"}


def test_dataset_attrs_empty_history_is_empty_string() -> None:
    """No history entries produces an empty-string history attr."""
    time = Channel(data=np.array(["2026-08-09"], dtype="datetime64[s]"))
    dataset = Dataset(time=time)

    assert dataset_attrs(dataset)["history"] == ""


def test_channel_attrs_drops_none_and_adds_history_if_present() -> None:
    """None-valued metadata dropped; history attr only added if non-empty."""
    channel = Channel(
        data=np.array([1.0, 2.0]),
        metadata={"units": "degree_C", "source_channel_name": None},
        history=["did a thing"],
    )

    attrs = channel_attrs(channel)

    assert attrs == {"units": "degree_C", "history": "did a thing"}


def test_channel_attrs_omits_history_key_when_empty() -> None:
    """No history entries means no 'history' key at all (not an empty one)."""
    channel = Channel(data=np.array([1.0]), metadata={"units": "m"})

    assert "history" not in channel_attrs(channel)


def test_pop_history_splits_and_removes_key() -> None:
    """pop_history splits on '; ' and removes the key from attrs."""
    attrs = {"history": "step one; step two", "units": "m"}

    history = pop_history(attrs)

    assert history == ["step one", "step two"]
    assert "history" not in attrs
    assert attrs == {"units": "m"}


def test_pop_history_empty_string_is_empty_list() -> None:
    """An empty or absent history attr becomes [] rather than ['']."""
    assert pop_history({"history": ""}) == []
    assert pop_history({}) == []
