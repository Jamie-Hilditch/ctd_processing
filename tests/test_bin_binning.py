"""Tests for ctd_processing.bin.binning."""

import logging

import numpy as np
import pytest

from ctd_processing.bin.binning import (
    bin_profile,
    combine_binned_profiles,
    compute_bin_edges,
)
from ctd_processing.config import BinSettings
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset


def _profile(
    z: np.ndarray,
    temperature: np.ndarray,
    *,
    start: str,
    serial: object = 208532,
    source_file: str = "243188_20260809_0304.rsk",
    latitude: float = 45.0,
    longitude: float = -125.0,
    extra_history: str | None = None,
) -> Dataset:
    """Build a small, already-geolocated profile Dataset for bin tests."""
    n = z.size
    time = Channel(
        data=np.datetime64(start) + np.arange(n) * np.timedelta64(1, "s")
    )
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {"instrument_serial_number": serial, "source_file": source_file}
    )
    dataset.add_channel(
        "z",
        Channel(
            data=z,
            metadata={
                "units": "m",
                "long_name": "Height",
                "standard_name": "height",
            },
        ),
    )
    dataset.add_channel(
        "sea_water_temperature",
        Channel(
            data=temperature,
            metadata={
                "units": "degree_C",
                "long_name": "Sea water temperature",
                "standard_name": "sea_water_temperature",
            },
        ),
    )
    dataset.metadata.update(
        {
            "profile_start_time": time.data[0],
            "profile_end_time": time.data[-1],
            "latitude": latitude,
            "longitude": longitude,
            "position_source": "reference position",
        }
    )
    if extra_history is not None:
        dataset.record(extra_history)
    return dataset


class TestComputeBinEdges:
    """Tests for compute_bin_edges."""

    def test_ascending_step_generates_increasing_edges(self) -> None:
        """Positive step from 0 to 10 in steps of 2 produces 6 edges."""
        settings = BinSettings(channel="z", step=2.0, first=0.0, last=10.0)

        edges = compute_bin_edges([], settings)

        np.testing.assert_array_equal(edges, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_descending_step_generates_decreasing_edges(self) -> None:
        """Negative step from 0 to -10 produces decreasing edges."""
        settings = BinSettings(channel="z", step=-2.0, first=0.0, last=-10.0)

        edges = compute_bin_edges([], settings)

        np.testing.assert_array_equal(
            edges, [0.0, -2.0, -4.0, -6.0, -8.0, -10.0]
        )

    def test_overshoots_last_rather_than_stopping_short(self) -> None:
        """Unlike numpy.arange, the final edge may pass last."""
        settings = BinSettings(channel="z", step=3.0, first=0.0, last=10.0)

        edges = compute_bin_edges([], settings)

        np.testing.assert_array_equal(edges, [0.0, 3.0, 6.0, 9.0, 12.0])

    def test_auto_computes_first_and_last_from_data(self) -> None:
        """With first/last unset, they're the data's min/max for step > 0."""
        settings = BinSettings(channel="z", step=1.0)
        values = [np.array([-3.0, -1.0]), np.array([0.5, 5.0])]

        edges = compute_bin_edges(values, settings)

        assert edges[0] == -3.0
        assert edges[-1] >= 5.0

    def test_auto_computes_reversed_for_negative_step(self) -> None:
        """With a negative step, first defaults to max, last to min."""
        settings = BinSettings(channel="z", step=-1.0)
        values = [np.array([-10.0, -2.0, 3.0])]

        edges = compute_bin_edges(values, settings)

        assert edges[0] == 3.0
        assert edges[-1] <= -10.0

    def test_ignores_nan_when_auto_computing(self) -> None:
        """NaN values in the data don't poison the auto-computed range."""
        settings = BinSettings(channel="z", step=1.0)
        values = [np.array([np.nan, -3.0, 5.0, np.nan])]

        edges = compute_bin_edges(values, settings)

        assert edges[0] == -3.0
        assert edges[-1] >= 5.0

    def test_raises_if_no_finite_data_and_bounds_unset(self) -> None:
        """All-NaN data with unset first/last raises rather than crashing."""
        settings = BinSettings(channel="z", step=1.0)

        with pytest.raises(ValueError, match="no finite"):
            compute_bin_edges([np.array([np.nan, np.nan])], settings)


class TestBinProfile:
    """Tests for bin_profile."""

    def test_raises_if_channel_missing(self) -> None:
        """Binning by a channel absent from the profile raises ValueError."""
        dataset = _profile(
            np.array([-1.0, -2.0]), np.array([1.0, 2.0]), start="2026-01-01"
        )

        with pytest.raises(ValueError, match="sea_pressure"):
            bin_profile(dataset, "sea_pressure", np.array([0.0, -5.0]))

    def test_averages_within_each_bin(self) -> None:
        """Samples in the same bin are NaN-aware averaged."""
        z = np.array([-0.1, -0.4, -0.6, -1.4])
        temperature = np.array([10.0, 12.0, 20.0, np.nan])
        dataset = _profile(z, temperature, start="2026-01-01")
        edges = np.array([0.0, -0.5, -1.0, -1.5])

        result = bin_profile(dataset, "z", edges)

        # Bins are always presented in ascending numeric order regardless
        # of the configured edges' direction (see bin_profile's docstring),
        # so centers -1.25/-0.75/-0.25 come out in that order.
        temp = result["sea_water_temperature"].squeeze("profile")
        np.testing.assert_allclose(temp.values, [np.nan, 20.0, 11.0])

    def test_excludes_bin_channel_from_data_variables(self) -> None:
        """The channel binned by is not itself a data variable in the result."""
        dataset = _profile(
            np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
        )

        result = bin_profile(dataset, "z", np.array([0.0, -0.5, -1.0]))

        assert "z" not in result.data_vars
        assert "z" in result.coords

    def test_bin_coordinate_reuses_channel_metadata(self) -> None:
        """The bin coordinate's attrs come from the binned channel."""
        dataset = _profile(
            np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
        )

        result = bin_profile(dataset, "z", np.array([0.0, -0.5, -1.0]))

        assert result["z"].attrs["standard_name"] == "height"
        assert result["z"].attrs["units"] == "m"

    def test_profile_metadata_becomes_coordinates(self) -> None:
        """time/latitude/longitude become scalar (profile-dim) coordinates."""
        dataset = _profile(
            np.array([-0.1, -0.6]),
            np.array([1.0, 2.0]),
            start="2026-08-09T03:04:00",
            latitude=12.5,
            longitude=-45.0,
        )

        result = bin_profile(dataset, "z", np.array([0.0, -0.5, -1.0]))

        assert result["time"].values.item() == np.datetime64(
            "2026-08-09T03:04:00"
        )
        assert result["latitude"].item() == 12.5
        assert result["longitude"].item() == -45.0
        assert result["latitude"].attrs["standard_name"] == "latitude"

    def test_parses_stringified_time_metadata(self) -> None:
        """profile_start_time/profile_end_time stored as str parse back.

        A save/load round trip through either profile file format
        stringifies these (see
        `ctd_processing.process.save_parquet.write_parquet`'s docstring),
        so `bin_profile` must accept that shape too.
        """
        dataset = _profile(
            np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
        )
        dataset.metadata["profile_start_time"] = "2026-01-01T00:00:00"
        dataset.metadata["profile_end_time"] = "2026-01-01T00:00:01"

        result = bin_profile(dataset, "z", np.array([0.0, -0.5, -1.0]))

        assert result["time"].dtype.kind == "M"
        assert result["profile_end_time"].dtype.kind == "M"

    def test_other_metadata_becomes_global_attrs(self) -> None:
        """Non-per-profile metadata (plus history) becomes dataset attrs."""
        dataset = _profile(
            np.array([-0.1, -0.6]),
            np.array([1.0, 2.0]),
            start="2026-01-01",
            extra_history="did a thing",
        )

        result = bin_profile(dataset, "z", np.array([0.0, -0.5, -1.0]))

        assert result.attrs["instrument_serial_number"] == 208532
        assert result.attrs["source_file"] == "243188_20260809_0304.rsk"
        assert "did a thing" in result.attrs["history"]
        assert "latitude" not in result.attrs


class TestCombineBinnedProfiles:
    """Tests for combine_binned_profiles."""

    def test_stacks_along_new_profile_dimension(self) -> None:
        """Two profiles combine into one dataset with a profile dimension."""
        edges = np.array([0.0, -0.5, -1.0])
        p1 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
            ),
            "z",
            edges,
        )
        p2 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]), np.array([3.0, 4.0]), start="2026-01-02"
            ),
            "z",
            edges,
        )

        combined = combine_binned_profiles([p1, p2])

        assert combined.sizes["profile"] == 2
        assert combined["sea_water_temperature"].dims == ("profile", "z")

    def test_identical_coordinates_still_stack_along_profile(self) -> None:
        """latitude/longitude stay profile-indexed even if values coincide.

        A fixed reference position (the common case) gives every profile
        in a deployment the exact same latitude/longitude -- these must
        not silently collapse back to scalars.
        """
        edges = np.array([0.0, -0.5, -1.0])
        p1 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]),
                np.array([1.0, 2.0]),
                start="2026-01-01",
                latitude=45.0,
                longitude=-125.0,
            ),
            "z",
            edges,
        )
        p2 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]),
                np.array([3.0, 4.0]),
                start="2026-01-02",
                latitude=45.0,
                longitude=-125.0,
            ),
            "z",
            edges,
        )

        combined = combine_binned_profiles([p1, p2])

        assert combined["latitude"].dims == ("profile",)
        assert combined["longitude"].dims == ("profile",)
        np.testing.assert_array_equal(combined["latitude"].values, [45.0, 45.0])

    def test_warns_and_drops_conflicting_history(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Differing per-profile history is dropped, with a warning logged."""
        caplog.set_level(logging.WARNING, logger="ctd_processing.bin.binning")
        edges = np.array([0.0, -0.5, -1.0])
        p1 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
            ),
            "z",
            edges,
        )
        p2 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]),
                np.array([3.0, 4.0]),
                start="2026-01-02",
                extra_history="an extra step",
            ),
            "z",
            edges,
        )

        combined = combine_binned_profiles([p1, p2])

        assert "history" not in combined.attrs
        messages = [record.getMessage() for record in caplog.records]
        assert any("history" in m for m in messages)

    def test_agreeing_global_attrs_are_kept(self) -> None:
        """Attributes that agree across every profile are kept, not dropped."""
        edges = np.array([0.0, -0.5, -1.0])
        p1 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]), np.array([1.0, 2.0]), start="2026-01-01"
            ),
            "z",
            edges,
        )
        p2 = bin_profile(
            _profile(
                np.array([-0.1, -0.6]), np.array([3.0, 4.0]), start="2026-01-02"
            ),
            "z",
            edges,
        )

        combined = combine_binned_profiles([p1, p2])

        assert combined.attrs["instrument_serial_number"] == 208532
        assert combined.attrs["source_file"] == "243188_20260809_0304.rsk"
