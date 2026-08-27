"""Tests for ctd_processing.process._shift."""

import numpy as np

from ctd_processing.process._shift import shift_array, shift_inplace


def test_shift_array_matches_shift_inplace_positive_shift() -> None:
    """shift_array's result equals shift_inplace applied to a copy."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = data.copy()
    shift_inplace(expected, 2)  # ty: ignore

    result = shift_array(data, 2)

    assert np.array_equal(result, expected, equal_nan=True)


def test_shift_array_matches_shift_inplace_negative_shift() -> None:
    """shift_array's result equals shift_inplace applied to a copy."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = data.copy()
    shift_inplace(expected, -2)  # ty: ignore

    result = shift_array(data, -2)

    assert np.array_equal(result, expected, equal_nan=True)


def test_shift_array_does_not_mutate_input() -> None:
    """shift_array leaves its input untouched."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = data.copy()

    shift_array(data, 1)

    assert np.array_equal(data, original)


def test_shift_array_zero_is_a_noop_copy() -> None:
    """shift_array with shift=0 returns an equal but distinct array."""
    data = np.array([1.0, 2.0, 3.0])

    result = shift_array(data, 0)

    assert np.array_equal(result, data)
    assert result is not data
