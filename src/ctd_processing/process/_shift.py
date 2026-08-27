"""Shared pandas `.shift()`-style array shifting, used by more than one module.

`raw_channels.py`'s `shift_time` and `ct_lag.py`'s lag search both need the
same "shift by N samples, NaN-fill whatever falls off the end" operation --
the former in place on a `Channel`, the latter as many non-mutating trial
shifts of the same source array. Both get it from here rather than each
keeping their own copy.
"""

from typing import Any

import numba
import numpy as np
import numpy.typing as npt

__all__ = ["shift_array", "shift_inplace"]


@numba.njit(cache=True)
def shift_inplace(data: npt.NDArray[Any], shift: int) -> None:
    """Shift `data` by `shift` samples in place, pandas `.shift()`-style.

    ``output[i] = input[i - shift]`` wherever that index is valid, else
    NaN. A single pass, but the iteration direction depends on the sign
    of `shift` to avoid overwriting a source value before it's read:
    positive `shift` (destination `i` always greater than source
    `i - shift`) must go high-to-low; negative `shift` (destination `i`
    always less than source `i + abs(shift)`) must go low-to-high. No new
    arrays are allocated.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to shift, mutated in place.
    shift : int
        Number of samples to shift by. Positive delays (shifts toward
        higher indices' source, i.e. pulls earlier values forward);
        negative advances. Zero is a no-op.
    """
    n = data.shape[0]
    if shift == 0:
        return
    if shift > 0:
        for i in range(n - 1, -1, -1):
            data[i] = data[i - shift] if i >= shift else np.nan
    else:
        m = -shift
        for i in range(n):
            data[i] = data[i + m] if i < n - m else np.nan


def shift_array(data: npt.NDArray[Any], shift: int) -> npt.NDArray[Any]:
    """Return a copy of `data` shifted by `shift` samples, pandas-style.

    Non-mutating counterpart to `shift_inplace`, for callers that need
    many trial shifts of the same source array (e.g. a lag search)
    without touching the original.

    Parameters
    ----------
    data : numpy.typing.NDArray[Any]
        The array to shift. Not mutated.
    shift : int
        Number of samples to shift by. See `shift_inplace` for the sign
        convention. ``0`` is a no-op.

    Returns
    -------
    numpy.typing.NDArray[Any]
        A new array with the shift applied.
    """
    result = data.copy()
    shift_inplace(result, shift)  # ty: ignore
    return result
