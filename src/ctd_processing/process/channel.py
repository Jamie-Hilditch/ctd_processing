"""Per-channel data container used throughout processing.

Note that this is unrelated to `pyrsktools.datatypes.Channel`, which is a
metadata-only record (channel id, units, label, ...) with no data array.
:class:`Channel` here is the data-bearing counterpart used once a
deployment's channels have been read into memory.
"""

from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

__all__ = ["Channel"]


@numba.njit(cache=True)
def _is_strictly_increasing(data: npt.NDArray[Any]) -> bool:
    """Scan `data` once, returning False as soon as it fails to increase."""
    for i in range(1, data.shape[0]):
        if data[i] <= data[i - 1]:
            return False
    return True


@numba.njit(cache=True)
def _is_strictly_decreasing(data: npt.NDArray[Any]) -> bool:
    """Scan `data` once, returning False as soon as it fails to decrease."""
    for i in range(1, data.shape[0]):
        if data[i] >= data[i - 1]:
            return False
    return True


@dataclass(eq=False, repr=False)
class Channel:
    """A single channel's data, metadata, and processing history.

    `eq` is disabled because the dataclass-generated equality check would
    compare `data` with ``==``, which returns an array rather than a bool
    and raises ``ValueError`` when Python tries to interpret it as one.
    `repr` is disabled in favor of the explicit `__repr__` below, since
    the auto-generated one would print the full `data` array.

    Attributes
    ----------
    data : numpy.typing.NDArray[Any]
        The channel's data, as a 1D array.
    metadata : dict[str, Any]
        Open, schema-less metadata describing this channel (e.g. `units`,
        `standard_name`), populated incrementally as processing steps
        learn more about the data. Defaults to an empty dict.
    history : list[str]
        Ordered log of processing steps that have been applied to this
        channel, in the order they were applied. Defaults to an empty
        list.

    Raises
    ------
    ValueError
        If `data` is not 1-dimensional.
    """

    data: npt.NDArray[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that `data` is 1-dimensional."""
        if self.data.ndim != 1:
            raise ValueError(
                "Channel data must be 1D; got shape "
                f"{self.data.shape} (ndim={self.data.ndim})"
            )

    def __repr__(self) -> str:
        """Unambiguous representation, summarizing `data` instead of dumping it.

        Returns
        -------
        str
            E.g. ``"Channel(data=<float64[3]>, metadata={}, history=[])"``.
        """
        summary = f"<{self.data.dtype}[{self.data.size}]>"
        return (
            f"{type(self).__name__}(data={summary}, "
            f"metadata={self.metadata!r}, history={self.history!r})"
        )

    def __str__(self) -> str:
        """Concise, human-readable summary of this channel.

        Returns
        -------
        str
            E.g. ``"Channel: 3 samples (float64), 1 processing step(s)"``.
        """
        return (
            f"{type(self).__name__}: {self.data.size} samples "
            f"({self.data.dtype}), {len(self.history)} processing step(s)"
        )

    def record(self, description: str) -> None:
        """Append a processing step to `history`.

        Parameters
        ----------
        description : str
            Description of the processing step that was applied.
        """
        self.history.append(description)

    def subset(
        self,
        indices: slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer],
        description: str,
    ) -> "Channel":
        """Return a new Channel restricted to a subset of `data`.

        This is the mechanism by which individual profiles are extracted
        from a full deployment channel.

        Parameters
        ----------
        indices : slice or npt.NDArray[np.bool_] or npt.NDArray[np.integer]
            A boolean mask, integer index array, or slice selecting
            the elements of `data` to keep.
        description : str
            Description of the subset operation, appended to the returned
            Channel's `history`.

        Returns
        -------
        Channel
            A new Channel wrapping ``data[indices]``, always copied so it
            never aliases this Channel's underlying array. Its `metadata`
            is a shallow copy of this Channel's `metadata`, and its
            `history` is this Channel's `history` plus `description`;
            neither this Channel's `metadata` nor its `history` is
            mutated.
        """
        new_data = np.asarray(self.data[indices]).copy()
        new_metadata = dict(self.metadata)
        new_history = [*self.history, description]
        return Channel(
            data=new_data, metadata=new_metadata, history=new_history
        )

    def _numeric_view(self) -> npt.NDArray[Any]:
        """Return `data`, viewed as int64 if it is a date/time dtype.

        numba's nopython mode does not support `datetime64`/`timedelta64`
        dtypes directly. Both are already stored as an integer count of
        some time unit since an epoch/zero point, so viewing (not
        copying) as `int64` preserves ordering while giving `numba` a
        dtype it can compile against.

        Returns
        -------
        numpy.typing.NDArray[Any]
            `data` unchanged, or an `int64` view of it for `datetime64`/
            `timedelta64` data.
        """
        if self.data.dtype.kind in "Mm":
            return self.data.view(np.int64)
        return self.data

    def is_increasing(self) -> bool:
        """Whether `data` is strictly increasing.

        Checks ``data[i] < data[i + 1]`` for every `i` in a single,
        early-exiting pass over `data` that allocates no new arrays.
        Vacuously ``True`` for arrays of length 0 or 1.

        Returns
        -------
        bool
            Whether `data` is strictly increasing.
        """
        view = self._numeric_view()
        return bool(_is_strictly_increasing(view))  # ty: ignore

    def is_decreasing(self) -> bool:
        """Whether `data` is strictly decreasing.

        Checks ``data[i] > data[i + 1]`` for every `i` in a single,
        early-exiting pass over `data` that allocates no new arrays.
        Vacuously ``True`` for arrays of length 0 or 1.

        Returns
        -------
        bool
            Whether `data` is strictly decreasing.
        """
        view = self._numeric_view()
        return bool(_is_strictly_decreasing(view))  # ty: ignore

    def is_monotonic(self) -> bool:
        """Whether `data` is strictly increasing or strictly decreasing.

        Returns
        -------
        bool
            ``self.is_increasing() or self.is_decreasing()``. Short
            circuits, so at most two single-pass scans of `data`.
        """
        return self.is_increasing() or self.is_decreasing()
