"""Tests for ctd_processing.bin.robust."""

import numpy as np
import pytest

from ctd_processing.bin.binning import bin_profile
from ctd_processing.bin.robust import (
    biweight_location,
    huber_location,
    reduce_bin,
    trimmed_mean,
    winsorized_mean,
)
from ctd_processing.config import (
    BinMethod,
    BinSettings,
    BiweightSettings,
    HuberSettings,
    TrimmedMeanSettings,
    WinsorizedMeanSettings,
)
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset

# A real, validated example (found investigating a genuine practical
# salinity outlier): a ~33.6-33.72 cluster of otherwise-agreeing values,
# plus one non-physical 32.78 point that a 0%-breakdown plain mean can't
# resist.
_CLUSTER = np.array(
    [33.60, 33.62, 33.63, 33.64, 33.65, 33.66, 33.68, 33.69, 33.70, 33.72]
)
_CLUSTER_MEAN = float(np.mean(_CLUSTER))
_OUTLIER_BIN = np.append(_CLUSTER, 32.78)

_ROBUST_ESTIMATORS = [
    lambda x: trimmed_mean(x, 0.2),
    lambda x: winsorized_mean(x, 0.2),
    huber_location,
    biweight_location,
]


def _clean_gaussian_sample() -> np.ndarray:
    """Build a clean, outlier-free sample every estimator should agree on."""
    rng = np.random.default_rng(12345)
    return 33.65 + 0.02 * rng.standard_normal(200)


class TestTrimmedMean:
    """Tests for trimmed_mean."""

    def test_matches_manual_calculation(self) -> None:
        """Sorted, drop floor(n*p) from each end, mean the remainder."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

        # floor(5 * 0.2) = 1 dropped from each end -> [2, 3, 4] -> mean 3.
        assert trimmed_mean(x, proportion_to_cut=0.2) == pytest.approx(3.0)

    def test_falls_back_to_plain_mean_below_three_points(self) -> None:
        """Fewer than 3 points always falls back to the plain mean."""
        x = np.array([1.0, 100.0])

        assert trimmed_mean(x, 0.2) == pytest.approx(np.mean(x))


class TestWinsorizedMean:
    """Tests for winsorized_mean."""

    def test_matches_manual_calculation(self) -> None:
        """Clip floor(n*limits) from each end to the nearest retained value."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

        # floor(5 * 0.2) = 1 clipped from each end -> [2, 2, 3, 4, 4] -> 3.
        assert winsorized_mean(x, limits=0.2) == pytest.approx(3.0)

    def test_falls_back_to_plain_mean_below_three_points(self) -> None:
        """Fewer than 3 points always falls back to the plain mean."""
        x = np.array([1.0, 100.0])

        assert winsorized_mean(x, 0.2) == pytest.approx(np.mean(x))


class TestIrlsLocation:
    """Tests for huber_location/biweight_location's shared IRLS behavior."""

    @pytest.mark.parametrize("estimator", [huber_location, biweight_location])
    def test_falls_back_to_plain_mean_below_three_points(
        self, estimator
    ) -> None:
        """Fewer than 3 points always falls back to the plain mean."""
        x = np.array([1.0, 5.0])

        assert estimator(x) == pytest.approx(np.mean(x))

    @pytest.mark.parametrize("estimator", [huber_location, biweight_location])
    def test_handles_constant_input_without_dividing_by_zero(
        self, estimator
    ) -> None:
        """All-identical values return that constant, not NaN/inf/an error."""
        x = np.full(10, 7.5)

        assert estimator(x) == pytest.approx(7.5)

    @pytest.mark.parametrize("estimator", [huber_location, biweight_location])
    def test_handles_majority_tied_values(self, estimator) -> None:
        """A majority-tied sample converges without dividing by zero.

        Once the location estimate lands exactly on a value shared by at
        least half the sample, the MAD-based scale hits 0 mid-iteration --
        iteration must stop there rather than dividing by it.
        """
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0])

        result = estimator(x)

        assert np.isfinite(result)
        assert result == pytest.approx(5.0)


class TestRobustEstimatorsAgainstOutliers:
    """Cross-cutting behavior shared by every robust estimator."""

    @pytest.mark.parametrize("estimator", _ROBUST_ESTIMATORS)
    def test_matches_plain_mean_on_clean_gaussian_data(self, estimator) -> None:
        """Each estimator loses little efficiency when nothing is wrong."""
        x = _clean_gaussian_sample()

        assert estimator(x) == pytest.approx(float(np.mean(x)), abs=0.01)

    @pytest.mark.parametrize(
        "estimator", [*_ROBUST_ESTIMATORS, lambda x: float(np.median(x))]
    )
    def test_resists_a_real_validated_outlier(self, estimator) -> None:
        """Each estimator lands closer to the cluster than the plain mean.

        An 11-point bin with one non-physical 32.78 value among a
        ~33.6-33.72 cluster -- a real example found investigating why the
        plain mean's 0% breakdown point let one bad point bias a whole
        bin.
        """
        plain_mean = float(np.mean(_OUTLIER_BIN))

        estimate = estimator(_OUTLIER_BIN)

        assert abs(estimate - _CLUSTER_MEAN) < abs(plain_mean - _CLUSTER_MEAN)


class TestReduceBin:
    """Tests for the reduce_bin dispatcher."""

    def test_ignores_non_finite_values(self) -> None:
        """NaN/inf values are excluded before reducing."""
        x = np.array([1.0, 2.0, np.nan, 3.0, np.inf])

        assert reduce_bin(x, "mean", None) == pytest.approx(2.0)

    def test_returns_nan_for_all_non_finite_input(self) -> None:
        """A bin with no finite values reduces to NaN, not an error."""
        result = reduce_bin(np.array([np.nan, np.nan]), "mean", None)

        assert np.isnan(result)

    def test_dispatches_every_method(self) -> None:
        """Every BinMethod dispatches to its matching reducer."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert reduce_bin(x, "mean", None) == pytest.approx(3.0)
        assert reduce_bin(x, "median", None) == pytest.approx(3.0)
        assert reduce_bin(
            x, "trimmed_mean", TrimmedMeanSettings(proportion_to_cut=0.2)
        ) == pytest.approx(3.0)
        assert reduce_bin(
            x, "winsorized_mean", WinsorizedMeanSettings(limits=0.2)
        ) == pytest.approx(3.0)
        assert reduce_bin(x, "huber", HuberSettings()) == pytest.approx(3.0)
        assert reduce_bin(x, "biweight", BiweightSettings()) == pytest.approx(
            3.0
        )

    def test_raises_on_unknown_method(self) -> None:
        """An unrecognized method raises rather than silently doing nothing."""
        bogus_method: BinMethod = "bogus"  # ty: ignore
        with pytest.raises(ValueError, match="Unknown bin method"):
            reduce_bin(np.array([1.0, 2.0, 3.0]), bogus_method, None)


def _outlier_profile() -> Dataset:
    """Build a profile with one bin containing the validated outlier."""
    n = _OUTLIER_BIN.size
    z = np.full(n, -5.0)
    time = Channel(
        data=np.datetime64("2026-01-01") + np.arange(n) * np.timedelta64(1, "s")
    )
    dataset = Dataset(time=time)
    dataset.metadata.update(
        {"instrument_serial_number": 1, "source_file": "outlier.rsk"}
    )
    dataset.add_channel("z", Channel(data=z, metadata={"units": "m"}))
    dataset.add_channel(
        "practical_salinity",
        Channel(data=_OUTLIER_BIN, metadata={"units": "1"}),
    )
    dataset.metadata.update(
        {
            "profile_start_time": time.data[0],
            "profile_end_time": time.data[-1],
            "latitude": 45.0,
            "longitude": -125.0,
        }
    )
    return dataset


@pytest.mark.parametrize(
    "method", ["median", "trimmed_mean", "winsorized_mean", "huber", "biweight"]
)
def test_bin_profile_every_robust_method_resists_the_outlier(method) -> None:
    """Every non-mean method lands closer to the cluster than the mean does."""
    dataset = _outlier_profile()
    edges = np.array([0.0, -10.0])

    mean_result = bin_profile(
        dataset, BinSettings(channel="z", method="mean"), edges
    )
    robust_result = bin_profile(
        dataset, BinSettings(channel="z", method=method), edges
    )

    mean_value = mean_result["practical_salinity"].squeeze().item()
    robust_value = robust_result["practical_salinity"].squeeze().item()
    assert abs(robust_value - _CLUSTER_MEAN) < abs(mean_value - _CLUSTER_MEAN)
