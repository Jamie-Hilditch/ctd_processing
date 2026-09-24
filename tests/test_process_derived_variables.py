"""Tests for ctd_processing.process.derived_variables."""

import numpy as np
import pytest

import ctd_processing.process.derived_variables as derived_variables_module
from ctd_processing.config import DerivedVariablesSettings, DespikeSettings
from ctd_processing.logging_utils import VERBOSE
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.derived_variables import compute_derived_variables

_DERIVED_KEYS = (
    "z",
    "practical_salinity",
    "absolute_salinity",
    "conservative_temperature",
    "density_anomaly",
    "potential_temperature",
    "speed_of_sound",
    "sea_water_density",
    "spiciness",
    "freezing_point",
    "thermal_expansion_coefficient",
    "haline_contraction_coefficient",
    "oxygen_concentration_from_saturation",
)

_NONE_SETTINGS = DerivedVariablesSettings(
    z=False,
    practical_salinity=False,
    absolute_salinity=False,
    conservative_temperature=False,
    potential_density=False,
    potential_temperature=False,
    sound_speed=False,
    density=False,
    spiciness=False,
    freezing_point=False,
    thermal_expansion=False,
    haline_contraction=False,
    oxygen_concentration=False,
)


def _profile(
    n: int = 5, latitude: float | None = 45.0, longitude: float | None = -125.0
) -> Dataset:
    """Build a profile Dataset with the channels/metadata this module needs."""
    dataset = Dataset(time=Channel(data=np.arange(float(n))))
    dataset.add_channel(
        "electrical_conductivity", Channel(data=np.full(n, 35.0))
    )
    dataset.add_channel("temperature", Channel(data=np.full(n, 10.0)))
    dataset.add_channel("sea_pressure", Channel(data=np.arange(float(n))))
    if latitude is not None:
        dataset.metadata["latitude"] = latitude
    if longitude is not None:
        dataset.metadata["longitude"] = longitude
    return dataset


def _stub_gsw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every `gsw` function used with simple, deterministic stand-ins.

    Isolates the wiring (which channels get added/overwritten under which
    keys, gated by which settings field) from real TEOS-10 physics, which
    isn't needed to test that wiring.
    """
    gsw = derived_variables_module.gsw
    monkeypatch.setattr(gsw, "SP_from_C", lambda c, t, p: c - t)
    monkeypatch.setattr(gsw, "SA_from_SP", lambda sp, p, lon, lat: sp + 1.0)
    monkeypatch.setattr(gsw, "CT_from_t", lambda sa, t, p: t + 1.0)
    monkeypatch.setattr(gsw, "sigma0", lambda sa, ct: sa + ct)
    monkeypatch.setattr(gsw, "z_from_p", lambda p, lat: -p)
    monkeypatch.setattr(gsw, "pt0_from_t", lambda sa, t, p: t - 1.0)
    monkeypatch.setattr(gsw, "sound_speed", lambda sa, ct, p: sa * ct)
    monkeypatch.setattr(gsw, "rho", lambda sa, ct, p: sa * 2.0)
    monkeypatch.setattr(gsw, "spiciness0", lambda sa, ct: sa - ct)
    monkeypatch.setattr(gsw, "CT_freezing", lambda sa, p, sat: sa - p)
    monkeypatch.setattr(gsw, "alpha", lambda sa, ct, p: sa / ct)
    monkeypatch.setattr(gsw, "beta", lambda sa, ct, p: ct / sa)
    monkeypatch.setattr(
        gsw, "O2sol", lambda sa, ct, p, lon, lat: sa + ct + p + lon + lat
    )


def test_compute_derived_variables_adds_core_five_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default settings add z, practical/absolute salinity, CT, sigma0."""
    _stub_gsw(monkeypatch)
    profiles = [_profile()]

    result = compute_derived_variables(profiles, DerivedVariablesSettings())

    assert result is profiles
    for key in (
        "z",
        "practical_salinity",
        "absolute_salinity",
        "conservative_temperature",
        "density_anomaly",
    ):
        assert key in result[0].channels
    for key in (
        "potential_temperature",
        "speed_of_sound",
        "sea_water_density",
        "spiciness",
        "freezing_point",
        "thermal_expansion_coefficient",
        "haline_contraction_coefficient",
        "oxygen_concentration_from_saturation",
    ):
        assert key not in result[0].channels


@pytest.mark.parametrize(
    ("flag", "key"),
    [
        ("z", "z"),
        ("practical_salinity", "practical_salinity"),
        ("absolute_salinity", "absolute_salinity"),
        ("conservative_temperature", "conservative_temperature"),
        ("potential_density", "density_anomaly"),
        ("potential_temperature", "potential_temperature"),
        ("sound_speed", "speed_of_sound"),
        ("density", "sea_water_density"),
        ("spiciness", "spiciness"),
        ("freezing_point", "freezing_point"),
        ("thermal_expansion", "thermal_expansion_coefficient"),
        ("haline_contraction", "haline_contraction_coefficient"),
    ],
)
def test_compute_derived_variables_flag_gates_its_channel(
    monkeypatch: pytest.MonkeyPatch, flag: str, key: str
) -> None:
    """Each flag independently gates whether its channel is added.

    Enabling only this one flag still succeeds even for outputs (e.g.
    `potential_density`) that depend on practical/absolute salinity or
    conservative temperature -- those are always computed internally,
    just not added as their own channels unless separately enabled.
    """
    _stub_gsw(monkeypatch)
    profiles = [_profile()]
    settings = _NONE_SETTINGS.model_copy(update={flag: True})

    result = compute_derived_variables(profiles, settings)

    assert key in result[0].channels
    other_keys = set(_DERIVED_KEYS) - {key}
    assert not other_keys & set(result[0].channels)


def test_compute_derived_variables_overwrites_existing_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing channel under a derived-variable key is overwritten."""
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5)]
    profiles[0].add_channel(
        "practical_salinity",
        Channel(data=np.zeros(5), metadata={"source": "onboard"}),
    )

    result = compute_derived_variables(profiles, DerivedVariablesSettings())

    salinity = result[0].channels["practical_salinity"]
    assert "source" not in salinity.metadata
    assert salinity.metadata["standard_name"] == "sea_water_practical_salinity"
    expected = (
        result[0].channels["electrical_conductivity"].data
        - result[0].channels["temperature"].data
    )
    np.testing.assert_array_equal(salinity.data, expected)


@pytest.mark.parametrize(
    "missing_channel",
    [
        "electrical_conductivity",
        "temperature",
        "sea_pressure",
    ],
)
def test_compute_derived_variables_missing_channel_raises(
    missing_channel: str,
) -> None:
    """A missing required channel raises ValueError naming it."""
    profiles = [_profile()]
    profiles[0].remove_channel(missing_channel)

    with pytest.raises(ValueError, match=missing_channel):
        compute_derived_variables(profiles, DerivedVariablesSettings())


def test_compute_derived_variables_missing_position_raises() -> None:
    """No latitude/longitude in a profile's metadata raises ValueError."""
    profiles = [_profile(latitude=None, longitude=None)]

    with pytest.raises(ValueError, match="latitude/longitude"):
        compute_derived_variables(profiles, DerivedVariablesSettings())


def test_compute_derived_variables_oxygen_concentration_requires_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oxygen_concentration raises if there's no saturation channel.

    Unlike every other flag, oxygen_concentration's input isn't present
    on every dataset, so a missing channel is an error, not a silent
    skip.
    """
    _stub_gsw(monkeypatch)
    profiles = [_profile()]
    settings = _NONE_SETTINGS.model_copy(update={"oxygen_concentration": True})

    with pytest.raises(ValueError, match="dissolved_oxygen_saturation"):
        compute_derived_variables(profiles, settings)


def test_compute_derived_variables_oxygen_concentration_from_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oxygen_concentration derives concentration from saturation and O2sol."""
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5)]
    saturation = np.full(5, 50.0)
    profiles[0].add_channel(
        "dissolved_oxygen_saturation", Channel(data=saturation.copy())
    )
    settings = _NONE_SETTINGS.model_copy(update={"oxygen_concentration": True})

    result = compute_derived_variables(profiles, settings)

    oxygen = result[0].channels["oxygen_concentration_from_saturation"]
    assert (
        oxygen.metadata["standard_name"]
        == "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
    )
    sa = (
        result[0].channels["electrical_conductivity"].data
        - result[0].channels["temperature"].data
        + 1.0
    )
    ct = result[0].channels["temperature"].data + 1.0
    p = result[0].channels["sea_pressure"].data
    solubility = (
        sa
        + ct
        + p
        + result[0].metadata["longitude"]
        + result[0].metadata["latitude"]
    )
    expected = solubility * (saturation / 100.0)
    np.testing.assert_allclose(oxygen.data, expected)


def test_compute_derived_variables_despikes_before_deriving_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A despiked practical_salinity doesn't leak its spike downstream.

    A conductivity outlier spikes practical_salinity (SP = c - t); with
    despike configured for practical_salinity, that spike is replaced with
    NaN *before* absolute_salinity is derived from it, so the NaN -- not
    the raw outlier -- propagates into absolute_salinity too. Uses
    `rejection_method="absolute_threshold"` for a deterministic threshold,
    since a single 5-sample profile is far too small for the default GPD
    fit to do anything but fall back to a plain percentile.
    """
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5)]
    profiles[0].channels["electrical_conductivity"].data[:] = [
        35.0,
        35.02,
        350.0,
        34.99,
        35.01,
    ]
    despike = {
        "practical_salinity": DespikeSettings(
            reference_window_length=3,
            rejection_method="absolute_threshold",
            rejection_threshold=10.0,
        )
    }

    result = compute_derived_variables(
        profiles, DerivedVariablesSettings(), despike
    )

    assert np.isnan(result[0].channels["practical_salinity"].data[2])
    assert np.isnan(result[0].channels["absolute_salinity"].data[2])


def test_compute_derived_variables_without_despike_spike_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without despike configured, the same spike reaches absolute_salinity.

    Contrasts with the despiked case above: this confirms the spike
    would otherwise have propagated, so despiking is actually doing
    something.
    """
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5)]
    profiles[0].channels["electrical_conductivity"].data[2] = 350.0

    result = compute_derived_variables(profiles, DerivedVariablesSettings())

    assert result[0].channels["practical_salinity"].data[2] == 340.0
    assert result[0].channels["absolute_salinity"].data[2] == 341.0


def test_compute_derived_variables_despike_pools_across_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rejection threshold is shared across profiles; counts are not.

    Only the first profile has a spike. Both profiles are despiked under
    the same (pooled) threshold, but only the first one's own count is
    nonzero, and only it gets a history entry -- `_maybe_despike_profiles`
    records conditionally on each profile's *own* replaced count.
    """
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5), _profile(n=5)]
    profiles[0].channels["electrical_conductivity"].data[:] = [
        35.0,
        35.02,
        350.0,
        34.99,
        35.01,
    ]
    despike = {
        "practical_salinity": DespikeSettings(
            reference_window_length=3,
            rejection_method="absolute_threshold",
            rejection_threshold=10.0,
        )
    }

    result = compute_derived_variables(
        profiles, DerivedVariablesSettings(), despike
    )

    assert np.isnan(result[0].channels["practical_salinity"].data[2])
    assert not np.isnan(result[1].channels["practical_salinity"].data).any()
    assert any(
        "despiked practical_salinity: 1 point(s)" in entry
        for entry in result[0].history
    )
    assert not any(
        "despiked practical_salinity" in entry for entry in result[1].history
    )


def test_compute_derived_variables_logs_zero_count_when_nothing_despiked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0-count VERBOSE record is logged even when nothing was despiked."""
    _stub_gsw(monkeypatch)
    profiles = [_profile(n=5)]
    despike = {
        "practical_salinity": DespikeSettings(
            reference_window_length=3,
            rejection_method="absolute_threshold",
            rejection_threshold=1000.0,
        )
    }
    caplog.set_level(VERBOSE, logger="ctd_processing.process.derived_variables")

    compute_derived_variables(profiles, DerivedVariablesSettings(), despike)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        m.startswith("despiked practical_salinity: 0 point(s)")
        for m in messages
    )


def test_compute_derived_variables_records_history_with_added_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each profile's history gains one entry naming the variables added."""
    _stub_gsw(monkeypatch)
    profiles = [_profile()]

    compute_derived_variables(profiles, DerivedVariablesSettings())

    assert any(
        "computed derived variables" in entry and "z" in entry
        for entry in profiles[0].history
    )


def test_compute_derived_variables_records_nothing_when_all_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No history entry is added when every flag is disabled."""
    _stub_gsw(monkeypatch)
    profiles = [_profile()]
    history_before = list(profiles[0].history)

    compute_derived_variables(profiles, _NONE_SETTINGS)

    assert profiles[0].history == history_before


def test_compute_derived_variables_logs_verbose_on_add(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Adding a channel logs at VERBOSE."""
    _stub_gsw(monkeypatch)
    caplog.set_level(VERBOSE, logger="ctd_processing.process.derived_variables")
    profiles = [_profile()]

    compute_derived_variables(profiles, DerivedVariablesSettings())

    messages = [record.getMessage() for record in caplog.records]
    assert any("added channel 'z'" in m for m in messages)
