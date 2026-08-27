"""A collection of `Channel`s sharing a common time base."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ctd_processing.process.channel import Channel

__all__ = ["Dataset"]


@dataclass(eq=False, repr=False)
class Dataset:
    """A collection of `Channel`s sharing a common time base.

    A `Dataset` is created with its `time` channel, which fixes `length`
    and must be strictly increasing. Every other channel is added
    afterward via `add_channel`, which validates its length against
    `length`.

    `eq` is disabled for the same reason as `Channel`: the
    dataclass-generated equality check would need to compare `Channel`
    values, which themselves disable equality. `repr` is disabled in
    favor of the explicit `__repr__` below, since the auto-generated one
    would recursively print every channel in full.

    Attributes
    ----------
    time : Channel
        The dataset's time channel. Must be strictly increasing.
    metadata : dict[str, Any]
        Open, schema-less dataset-level metadata (e.g. project name),
        separate from each channel's own `metadata`. Defaults to an empty
        dict.
    history : list[str]
        Ordered log of processing steps applied to this dataset as a
        whole. Defaults to an empty list.
    channels : dict[str, Channel]
        Every channel in the dataset except `time` itself (see `time`),
        keyed by name. Empty at construction; every entry is added via
        `add_channel`. Not a constructor argument.
    length : int
        Number of samples in the dataset, taken from `len(time.data)` at
        construction. Every channel added via `add_channel` must have
        this length. Not a constructor argument.

    Raises
    ------
    ValueError
        If `time` is not strictly increasing.
    """

    time: Channel
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    channels: dict[str, Channel] = field(init=False, default_factory=dict)
    length: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Seed `length` from `time`; validate it; `channels` starts empty."""
        if not self.time.is_increasing():
            raise ValueError("Dataset's time channel must be increasing.")
        self.length = len(self.time.data)
        self.channels = {}

    def __repr__(self) -> str:
        """Unambiguous representation, naming channels instead of dumping them.

        Returns
        -------
        str
            E.g. ``"Dataset(channels=['time', 'temperature'], length=3, ..."``.
        """
        return (
            f"{type(self).__name__}(channels={['time', *self.channels]!r}, "
            f"length={self.length}, metadata={self.metadata!r}, "
            f"history={self.history!r})"
        )

    def __str__(self) -> str:
        """Concise, human-readable summary of this dataset.

        Returns
        -------
        str
            E.g. ``"Dataset: 3 samples across 2 channel(s): time,
            temperature"``.
        """
        names = ", ".join(["time", *self.channels])
        return (
            f"{type(self).__name__}: {self.length} samples across "
            f"{len(self.channels) + 1} channel(s): {names}"
        )

    def record(self, description: str) -> None:
        """Append a processing step to `history`.

        Parameters
        ----------
        description : str
            Description of the processing step that was applied.
        """
        self.history.append(description)

    def add_channel(self, name: str, channel: Channel) -> None:
        """Add a channel to the dataset. The primary way to add channels.

        Parameters
        ----------
        name : str
            The name to store `channel` under.
        channel : Channel
            The channel to add. Its data length must match `length`.

        Raises
        ------
        ValueError
            If `name` is ``"time"`` (reserved for `Dataset.time`), if
            `name` is already present in `channels`, or if `channel`'s
            data length does not equal `length`.
        """
        if name == "time":
            raise ValueError(
                "'time' is reserved for Dataset.time; it cannot be used "
                "as a channel name."
            )
        if name in self.channels:
            raise ValueError(
                f"Channel {name!r} is already present in this dataset."
            )
        if len(channel.data) != self.length:
            raise ValueError(
                f"Channel {name!r} has length {len(channel.data)}, "
                f"expected {self.length} to match this dataset."
            )
        self.channels[name] = channel
        self.record(f"added channel {name!r}")

    def remove_channel(self, name: str) -> Channel:
        """Remove and return a channel from the dataset.

        `"time"` is never a valid `name` here -- it was never added to
        `channels` in the first place (see `Dataset.time`), so removing
        it raises `KeyError` like any other absent name.

        Parameters
        ----------
        name : str
            The name of the channel to remove.

        Returns
        -------
        Channel
            The removed channel.

        Raises
        ------
        KeyError
            If `name` is not present in `channels`.
        """
        channel = self.channels.pop(name)
        self.record(f"removed channel {name!r}")
        return channel

    def subset(
        self,
        indices: slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer],
        description: str,
    ) -> "Dataset":
        """Return a new Dataset restricted to a subset of every channel.

        This is the mechanism by which individual profiles are extracted
        from a full deployment dataset.

        Parameters
        ----------
        indices : slice or npt.NDArray[np.bool_] or npt.NDArray[np.integer]
            A boolean mask, integer index array, or slice selecting
            the elements to keep, applied identically to every channel.
        description : str
            Description of the subset operation, passed to every
            channel's own `subset` and appended to the returned
            Dataset's `history`.

        Returns
        -------
        Dataset
            A new Dataset built from ``time.subset(indices, description)``
            (so its `length` and increasing-time check are re-derived
            normally), with every other channel subset the same way. Its
            `metadata` is a shallow copy of this Dataset's `metadata`, and
            its `history` is this Dataset's `history` plus `description`;
            neither this Dataset's `metadata` nor its `history` is
            mutated.
        """
        new_time = self.time.subset(indices, description)
        new_metadata = dict(self.metadata)
        new_history = [*self.history, description]
        new_dataset = Dataset(
            time=new_time, metadata=new_metadata, history=new_history
        )
        for name, channel in self.channels.items():
            new_dataset.channels[name] = channel.subset(indices, description)
        return new_dataset
