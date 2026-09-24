"""TEOS-10 derived variables, configured via `process.derived_variables`.

See `ctd_processing.config.DerivedVariablesSettings`. Applied to a
deployment's already-geolocated profile `Dataset`s (see
`ctd_processing.process.geolocation.attach_geolocation_to_profiles` and
`ctd_processing.process.process_deployment`), since `absolute_salinity` and
`z` need a position (`dataset.metadata["latitude"]`/`["longitude"]`) that is
only attached once a profile has been extracted from the full deployment
`Dataset` and geolocated.

If `despike` (see `ctd_processing.config.resolve_despike_settings`) has an
entry for a computed quantity's channel key, that quantity is despiked
immediately after it's computed -- across every profile in the deployment
at once, pooling the rejection-threshold fit across all of them (see
`ctd_processing.process.despike.despike_arrays`) -- and before it's used
any further. E.g. `practical_salinity` is despiked before it feeds
`absolute_salinity`, which is itself despiked before it feeds
`conservative_temperature`/`sigma0`/etc. This is why this module proceeds
variable-by-variable across every profile (compute that variable for every
profile, despike it, then move to the next variable), rather than
profile-by-profile through every variable: a pooled fit needs every
profile's value for one variable at the same time. It's also why every
quantity below is computed unconditionally (even ones whose own output
flag is disabled) rather than only inside its own `if settings.X:` block --
despiking an intermediate needs to happen whether or not that intermediate
is ever exposed as its own channel.
"""

import logging

import gsw
import numpy.typing as npt

from ctd_processing.config import DerivedVariablesSettings, DespikeSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.despike import (
    describe_rejection_threshold,
    despike_arrays,
)

logger = logging.getLogger(__name__)

__all__ = ["compute_derived_variables"]

_REQUIRED_CHANNELS = (
    "electrical_conductivity",
    "temperature",
    "sea_pressure",
)


def _add_channel(dataset: Dataset, name: str, channel: Channel) -> None:
    """Add `channel` under `name`, overwriting any channel already there.

    Parameters
    ----------
    dataset : Dataset
        The dataset to add `channel` to. Mutated in place.
    name : str
        The channel key to add `channel` under.
    channel : Channel
        The channel to add.
    """
    if name in dataset.channels:
        dataset.remove_channel(name)
        log_verbose(logger, "removed existing %r channel", name)
    dataset.add_channel(name, channel)
    log_verbose(logger, "added channel %r", name)


def _maybe_despike_profiles(
    profiles: list[Dataset],
    key: str,
    data_list: list[npt.NDArray],
    despike: dict[str, DespikeSettings],
) -> list[npt.NDArray]:
    """Despike `data_list` (one array per profile) if `key` is in `despike`.

    Fits one rejection threshold pooled across every profile in
    `data_list` (see `ctd_processing.process.despike.despike_arrays`), then
    records/logs each profile's own outcome against that shared fit -- so
    every profile's own output file ends up self-describing: it carries
    the full pooled-fit diagnostics that produced its despiked data, not
    just its own replaced count.

    There's no `Channel` object yet for a not-(currently)-output
    intermediate like `practical_salinity` when
    `settings.practical_salinity` is `False`, so despiking is recorded on
    each profile `Dataset` itself rather than on a channel.

    Parameters
    ----------
    profiles : list[Dataset]
        One deployment's profile `Dataset`s, in the same order as
        `data_list`. A profile is mutated (via `Dataset.record`) only if
        its own replaced count is nonzero.
    key : str
        The channel key `data_list` will end up under (whether or not
        it's actually added as its own channel).
    data_list : list[numpy.typing.NDArray]
        One array per profile in `profiles`, same order.
    despike : dict[str, DespikeSettings]
        Resolved despike settings, keyed by channel name (see
        `ctd_processing.config.resolve_despike_settings`).

    Returns
    -------
    list[numpy.typing.NDArray]
        `data_list` unchanged if `key` has no entry in `despike`;
        otherwise each profile's despiked data, in the same order.
    """
    settings = despike.get(key)
    if settings is None:
        return data_list
    despiked, counts, result = despike_arrays(data_list, settings)
    detail = describe_rejection_threshold(settings, result)
    for profile, count in zip(profiles, counts, strict=True):
        description = f"despiked {key}: {count} point(s) ({detail})"
        if count:
            profile.record(description)
        log_verbose(logger, description)
    return despiked


def compute_derived_variables(
    profiles: list[Dataset],
    settings: DerivedVariablesSettings,
    despike: dict[str, DespikeSettings] | None = None,
) -> list[Dataset]:
    """Compute and attach TEOS-10 derived variables to every profile.

    Practical salinity, absolute salinity, and conservative temperature
    are always computed internally, in that order, for every profile in
    `profiles`, regardless of which `settings` fields are enabled, since
    `settings.potential_density`, `settings.potential_temperature`,
    `settings.sound_speed`, `settings.density`, and `settings.spiciness`
    each depend on absolute salinity and/or conservative temperature. Each
    enabled output is added to its profile via `Dataset.add_channel`,
    overwriting (via `Dataset.remove_channel` first) any channel already
    present under that key -- e.g. an instrument's own onboard-computed
    `practical_salinity`/`absolute_salinity`/`density_anomaly`, since the
    recomputed value here incorporates whatever upstream raw-channel/
    CT-lag corrections were applied, which an onboard-computed value would
    not.

    See the module docstring for how `despike` interacts with this: a
    configured quantity is despiked immediately after it's computed for
    every profile, pooling the rejection-threshold fit across all of
    them, even if it's one of the always-computed intermediates whose own
    output flag is disabled.

    Parameters
    ----------
    profiles : list[Dataset]
        One deployment's already-geolocated profile `Dataset`s (each must
        have `electrical_conductivity`, `temperature`, and `sea_pressure`
        channels, and `latitude`/`longitude` in its `metadata` -- see
        `ctd_processing.process.geolocation.attach_geolocation_to_profiles`).
        Each is mutated in place.
    settings : DerivedVariablesSettings
        Which derived variables to attach.
    despike : dict[str, DespikeSettings] or None, optional
        Resolved despike settings, keyed by channel name (see
        `ctd_processing.config.resolve_despike_settings`). Optional;
        defaults to ``None``, meaning no quantity is despiked.

    Returns
    -------
    list[Dataset]
        `profiles` itself (not a copy).

    Raises
    ------
    ValueError
        If any profile in `profiles` is missing a required input channel,
        or has no `latitude`/`longitude` in its metadata. Also raised if
        `settings.oxygen_concentration` is enabled and any profile has no
        `dissolved_oxygen_saturation` channel -- unlike every other field,
        that one input isn't present on every dataset, so a missing
        channel is an error here rather than a silent skip.
    """
    despike = despike or {}

    for profile in profiles:
        for name in _REQUIRED_CHANNELS:
            if name not in profile.channels:
                raise ValueError(
                    "Cannot compute derived variables: dataset has no "
                    f"{name} channel."
                )
        if (
            profile.metadata.get("latitude") is None
            or profile.metadata.get("longitude") is None
        ):
            raise ValueError(
                "Cannot compute derived variables: dataset.metadata has no "
                "latitude/longitude (attach geolocation first)."
            )

    conductivity = [
        p.channels["electrical_conductivity"].data for p in profiles
    ]
    temperature = [p.channels["temperature"].data for p in profiles]
    sea_pressure = [p.channels["sea_pressure"].data for p in profiles]
    latitude = [p.metadata["latitude"] for p in profiles]
    longitude = [p.metadata["longitude"] for p in profiles]

    practical_salinity = [
        gsw.SP_from_C(c, t, sp)
        for c, t, sp in zip(
            conductivity, temperature, sea_pressure, strict=True
        )
    ]
    practical_salinity = _maybe_despike_profiles(
        profiles, "practical_salinity", practical_salinity, despike
    )
    absolute_salinity = [
        gsw.SA_from_SP(sp_value, sp, lon, lat)
        for sp_value, sp, lon, lat in zip(
            practical_salinity, sea_pressure, longitude, latitude, strict=True
        )
    ]
    absolute_salinity = _maybe_despike_profiles(
        profiles, "absolute_salinity", absolute_salinity, despike
    )
    conservative_temperature = [
        gsw.CT_from_t(sa, t, sp)
        for sa, t, sp in zip(
            absolute_salinity, temperature, sea_pressure, strict=True
        )
    ]
    conservative_temperature = _maybe_despike_profiles(
        profiles, "conservative_temperature", conservative_temperature, despike
    )

    added: list[str] = []

    if settings.z:
        z_data = _maybe_despike_profiles(
            profiles,
            "z",
            [
                gsw.z_from_p(sp, lat)
                for sp, lat in zip(sea_pressure, latitude, strict=True)
            ],
            despike,
        )
        for profile, data in zip(profiles, z_data, strict=True):
            _add_channel(
                profile,
                "z",
                Channel(
                    data=data,
                    metadata={
                        "units": "m",
                        "long_name": "Height",
                        "standard_name": "height",
                    },
                ),
            )
        added.append("z")

    if settings.practical_salinity:
        for profile, data in zip(profiles, practical_salinity, strict=True):
            _add_channel(
                profile,
                "practical_salinity",
                Channel(
                    data=data,
                    metadata={
                        "units": "1e-3",
                        "long_name": "Practical salinity",
                        "standard_name": "sea_water_practical_salinity",
                    },
                ),
            )
        added.append("practical_salinity")

    if settings.absolute_salinity:
        for profile, data in zip(profiles, absolute_salinity, strict=True):
            _add_channel(
                profile,
                "absolute_salinity",
                Channel(
                    data=data,
                    metadata={
                        "units": "g kg-1",
                        "long_name": "Absolute salinity",
                        "standard_name": "sea_water_absolute_salinity",
                    },
                ),
            )
        added.append("absolute_salinity")

    if settings.conservative_temperature:
        for profile, data in zip(
            profiles, conservative_temperature, strict=True
        ):
            _add_channel(
                profile,
                "conservative_temperature",
                Channel(
                    data=data,
                    metadata={
                        "units": "degree_C",
                        "long_name": "Conservative temperature",
                        "standard_name": "sea_water_conservative_temperature",
                    },
                ),
            )
        added.append("conservative_temperature")

    if settings.potential_density:
        density_anomaly_data = _maybe_despike_profiles(
            profiles,
            "density_anomaly",
            [
                gsw.sigma0(sa, ct)
                for sa, ct in zip(
                    absolute_salinity, conservative_temperature, strict=True
                )
            ],
            despike,
        )
        for profile, data in zip(profiles, density_anomaly_data, strict=True):
            _add_channel(
                profile,
                "density_anomaly",
                Channel(
                    data=data,
                    metadata={
                        "units": "kg m-3",
                        "long_name": "Potential density anomaly (sigma0)",
                        "standard_name": "sea_water_sigma_theta",
                    },
                ),
            )
        added.append("density_anomaly")

    if settings.potential_temperature:
        potential_temperature_data = _maybe_despike_profiles(
            profiles,
            "potential_temperature",
            [
                gsw.pt0_from_t(sa, t, sp)
                for sa, t, sp in zip(
                    absolute_salinity, temperature, sea_pressure, strict=True
                )
            ],
            despike,
        )
        for profile, data in zip(
            profiles, potential_temperature_data, strict=True
        ):
            _add_channel(
                profile,
                "potential_temperature",
                Channel(
                    data=data,
                    metadata={
                        "units": "degree_C",
                        "long_name": "Potential temperature",
                        "standard_name": "sea_water_potential_temperature",
                    },
                ),
            )
        added.append("potential_temperature")

    if settings.sound_speed:
        sound_speed_data = _maybe_despike_profiles(
            profiles,
            "speed_of_sound",
            [
                gsw.sound_speed(sa, ct, sp)
                for sa, ct, sp in zip(
                    absolute_salinity,
                    conservative_temperature,
                    sea_pressure,
                    strict=True,
                )
            ],
            despike,
        )
        for profile, data in zip(profiles, sound_speed_data, strict=True):
            _add_channel(
                profile,
                "speed_of_sound",
                Channel(
                    data=data,
                    metadata={
                        "units": "m s-1",
                        "long_name": "Speed of sound in sea water",
                        "standard_name": "speed_of_sound_in_sea_water",
                    },
                ),
            )
        added.append("speed_of_sound")

    if settings.density:
        density_data = _maybe_despike_profiles(
            profiles,
            "sea_water_density",
            [
                gsw.rho(sa, ct, sp)
                for sa, ct, sp in zip(
                    absolute_salinity,
                    conservative_temperature,
                    sea_pressure,
                    strict=True,
                )
            ],
            despike,
        )
        for profile, data in zip(profiles, density_data, strict=True):
            _add_channel(
                profile,
                "sea_water_density",
                Channel(
                    data=data,
                    metadata={
                        "units": "kg m-3",
                        "long_name": "In-situ density",
                        "standard_name": "sea_water_density",
                    },
                ),
            )
        added.append("sea_water_density")

    if settings.spiciness:
        spiciness_data = _maybe_despike_profiles(
            profiles,
            "spiciness",
            [
                gsw.spiciness0(sa, ct)
                for sa, ct in zip(
                    absolute_salinity, conservative_temperature, strict=True
                )
            ],
            despike,
        )
        for profile, data in zip(profiles, spiciness_data, strict=True):
            _add_channel(
                profile,
                "spiciness",
                Channel(
                    data=data,
                    metadata={
                        "units": "kg m-3",
                        "long_name": "Spiciness (referenced to 0 dbar)",
                    },
                ),
            )
        added.append("spiciness")

    if settings.freezing_point:
        freezing_point_data = _maybe_despike_profiles(
            profiles,
            "freezing_point",
            [
                gsw.CT_freezing(sa, sp, 0)
                for sa, sp in zip(absolute_salinity, sea_pressure, strict=True)
            ],
            despike,
        )
        for profile, data in zip(profiles, freezing_point_data, strict=True):
            _add_channel(
                profile,
                "freezing_point",
                Channel(
                    data=data,
                    metadata={
                        "units": "degree_C",
                        "long_name": (
                            "Freezing point (conservative temperature)"
                        ),
                    },
                ),
            )
        added.append("freezing_point")

    if settings.thermal_expansion:
        thermal_expansion_data = _maybe_despike_profiles(
            profiles,
            "thermal_expansion_coefficient",
            [
                gsw.alpha(sa, ct, sp)
                for sa, ct, sp in zip(
                    absolute_salinity,
                    conservative_temperature,
                    sea_pressure,
                    strict=True,
                )
            ],
            despike,
        )
        for profile, data in zip(profiles, thermal_expansion_data, strict=True):
            _add_channel(
                profile,
                "thermal_expansion_coefficient",
                Channel(
                    data=data,
                    metadata={
                        "units": "K-1",
                        "long_name": "Thermal expansion coefficient",
                    },
                ),
            )
        added.append("thermal_expansion_coefficient")

    if settings.haline_contraction:
        haline_contraction_data = _maybe_despike_profiles(
            profiles,
            "haline_contraction_coefficient",
            [
                gsw.beta(sa, ct, sp)
                for sa, ct, sp in zip(
                    absolute_salinity,
                    conservative_temperature,
                    sea_pressure,
                    strict=True,
                )
            ],
            despike,
        )
        for profile, data in zip(
            profiles, haline_contraction_data, strict=True
        ):
            _add_channel(
                profile,
                "haline_contraction_coefficient",
                Channel(
                    data=data,
                    metadata={
                        "units": "kg g-1",
                        "long_name": "Haline contraction coefficient",
                    },
                ),
            )
        added.append("haline_contraction_coefficient")

    if settings.oxygen_concentration:
        for profile in profiles:
            if "dissolved_oxygen_saturation" not in profile.channels:
                raise ValueError(
                    "Cannot compute oxygen_concentration: dataset has no "
                    "dissolved_oxygen_saturation channel."
                )
        saturation = [
            p.channels["dissolved_oxygen_saturation"].data for p in profiles
        ]
        solubility = [
            gsw.O2sol(sa, ct, sp, lon, lat)
            for sa, ct, sp, lon, lat in zip(
                absolute_salinity,
                conservative_temperature,
                sea_pressure,
                longitude,
                latitude,
                strict=True,
            )
        ]
        oxygen_concentration_data = _maybe_despike_profiles(
            profiles,
            "oxygen_concentration_from_saturation",
            [
                sol * (sat / 100.0)
                for sol, sat in zip(solubility, saturation, strict=True)
            ],
            despike,
        )
        for profile, data in zip(
            profiles, oxygen_concentration_data, strict=True
        ):
            _add_channel(
                profile,
                "oxygen_concentration_from_saturation",
                Channel(
                    data=data,
                    metadata={
                        "units": "umol kg-1",
                        "long_name": (
                            "Dissolved oxygen concentration "
                            "(derived from saturation)"
                        ),
                        "standard_name": (
                            "mole_concentration_of_dissolved_molecular_oxygen_"
                            "in_sea_water"
                        ),
                    },
                ),
            )
        added.append("oxygen_concentration_from_saturation")

    if added:
        description = f"computed derived variables: {', '.join(added)}"
        for profile in profiles:
            profile.record(description)
        log_verbose(logger, description)

    return profiles
