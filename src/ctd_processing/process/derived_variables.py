"""TEOS-10 derived variables, configured via `process.derived_variables`.

See `ctd_processing.config.DerivedVariablesSettings`. Applied to one
already-geolocated profile `Dataset` (see
`ctd_processing.process.geolocation.attach_geolocation` and
`ctd_processing.process.process_profile`), since `absolute_salinity` and
`z` need a position (`dataset.metadata["latitude"]`/`["longitude"]`) that is
only attached once a profile has been extracted from the full deployment
`Dataset` and geolocated.

If `despike` (see `ctd_processing.config.resolve_despike_settings`) has an
entry for a computed quantity's channel key, that quantity is despiked
immediately after it's computed and before it's used any further -- e.g.
`practical_salinity` is despiked before it feeds `absolute_salinity`,
which is itself despiked before it feeds `conservative_temperature`/
`sigma0`/etc. This is why every quantity below is computed unconditionally
(even ones whose own output flag is disabled) rather than only inside its
own `if settings.X:` block: despiking an intermediate needs to happen
whether or not that intermediate is ever exposed as its own channel.
"""

import logging

import gsw
import numpy.typing as npt

from ctd_processing.config import DerivedVariablesSettings, DespikeSettings
from ctd_processing.logging_utils import log_verbose
from ctd_processing.process.channel import Channel
from ctd_processing.process.dataset import Dataset
from ctd_processing.process.despike import despike_array

logger = logging.getLogger(__name__)

__all__ = ["compute_derived_variables"]

_REQUIRED_CHANNELS = (
    "sea_water_electrical_conductivity",
    "sea_water_temperature",
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


def _maybe_despike(
    dataset: Dataset,
    key: str,
    data: npt.NDArray,
    despike: dict[str, DespikeSettings],
) -> npt.NDArray:
    """Despike `data` if `key` has an entry in `despike`.

    There's no `Channel` object yet for a not-(currently)-output
    intermediate like `practical_salinity` when
    `settings.practical_salinity` is `False`, so a despike here is
    recorded on `dataset` itself rather than on a channel.

    Parameters
    ----------
    dataset : Dataset
        The dataset `data` belongs to. Mutated (via `dataset.record`) only
        if `key` is configured for despiking and at least one point is
        actually replaced.
    key : str
        The channel key `data` will end up under (whether or not it's
        actually added as its own channel).
    data : numpy.typing.NDArray
        The data to despike.
    despike : dict[str, DespikeSettings]
        Resolved despike settings, keyed by channel name (see
        `ctd_processing.config.resolve_despike_settings`).

    Returns
    -------
    numpy.typing.NDArray
        `data` unchanged if `key` has no entry in `despike`; otherwise the
        despiked data.
    """
    settings = despike.get(key)
    if settings is None:
        return data
    despiked, count = despike_array(data, settings)
    if count:
        description = f"despiked {key}: {count} point(s)"
        dataset.record(description)
        log_verbose(logger, description)
    return despiked


def compute_derived_variables(
    dataset: Dataset,
    settings: DerivedVariablesSettings,
    despike: dict[str, DespikeSettings] | None = None,
) -> Dataset:
    """Compute and attach TEOS-10 derived variables to `dataset`.

    Practical salinity, absolute salinity, and conservative temperature
    are always computed internally, in that order, regardless of which
    `settings` fields are enabled, since `settings.potential_density`,
    `settings.potential_temperature`, `settings.sound_speed`,
    `settings.density`, and `settings.spiciness` each depend on absolute
    salinity and/or conservative temperature. Each enabled output is
    added to `dataset` via `Dataset.add_channel`, overwriting (via
    `Dataset.remove_channel` first) any channel already present under
    that key -- e.g. an instrument's own onboard-computed
    `practical_salinity`/`absolute_salinity`/`density_anomaly`, since the
    recomputed value here incorporates whatever upstream raw-channel/
    CT-lag corrections were applied, which an onboard-computed value
    would not.

    See the module docstring for how `despike` interacts with this: a
    configured quantity is despiked immediately after it's computed, even
    if it's one of the always-computed intermediates whose own output
    flag is disabled.

    Parameters
    ----------
    dataset : Dataset
        One profile's already-geolocated `Dataset` (must have
        `sea_water_electrical_conductivity`, `sea_water_temperature`, and
        `sea_pressure` channels, and `latitude`/`longitude` in
        `dataset.metadata` -- see
        `ctd_processing.process.geolocation.attach_geolocation`).
        Mutated in place.
    settings : DerivedVariablesSettings
        Which derived variables to attach.
    despike : dict[str, DespikeSettings] or None, optional
        Resolved despike settings, keyed by channel name (see
        `ctd_processing.config.resolve_despike_settings`). Optional;
        defaults to ``None``, meaning no quantity is despiked.

    Returns
    -------
    Dataset
        `dataset` itself (not a copy).

    Raises
    ------
    ValueError
        If `dataset` is missing a required input channel, or has no
        `latitude`/`longitude` in `dataset.metadata`. Also raised if
        `settings.oxygen_concentration` is enabled and `dataset` has no
        `dissolved_oxygen_saturation` channel -- unlike every other
        field, that one input isn't present on every dataset, so a
        missing channel is an error here rather than a silent skip.
    """
    despike = despike or {}

    for name in _REQUIRED_CHANNELS:
        if name not in dataset.channels:
            raise ValueError(
                "Cannot compute derived variables: dataset has no "
                f"{name} channel."
            )

    latitude = dataset.metadata.get("latitude")
    longitude = dataset.metadata.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError(
            "Cannot compute derived variables: dataset.metadata has no "
            "latitude/longitude (attach geolocation first)."
        )

    conductivity = dataset.channels["sea_water_electrical_conductivity"].data
    temperature = dataset.channels["sea_water_temperature"].data
    sea_pressure = dataset.channels["sea_pressure"].data

    practical_salinity = gsw.SP_from_C(conductivity, temperature, sea_pressure)
    practical_salinity = _maybe_despike(
        dataset, "practical_salinity", practical_salinity, despike
    )
    absolute_salinity = gsw.SA_from_SP(
        practical_salinity, sea_pressure, longitude, latitude
    )
    absolute_salinity = _maybe_despike(
        dataset, "absolute_salinity", absolute_salinity, despike
    )
    conservative_temperature = gsw.CT_from_t(
        absolute_salinity, temperature, sea_pressure
    )
    conservative_temperature = _maybe_despike(
        dataset, "conservative_temperature", conservative_temperature, despike
    )

    added: list[str] = []

    if settings.z:
        z_data = _maybe_despike(
            dataset, "z", gsw.z_from_p(sea_pressure, latitude), despike
        )
        _add_channel(
            dataset,
            "z",
            Channel(
                data=z_data,
                metadata={
                    "units": "m",
                    "long_name": "Height",
                    "standard_name": "height",
                },
            ),
        )
        added.append("z")

    if settings.practical_salinity:
        _add_channel(
            dataset,
            "practical_salinity",
            Channel(
                data=practical_salinity,
                metadata={
                    "units": "1e-3",
                    "long_name": "Practical salinity",
                    "standard_name": "sea_water_practical_salinity",
                },
            ),
        )
        added.append("practical_salinity")

    if settings.absolute_salinity:
        _add_channel(
            dataset,
            "absolute_salinity",
            Channel(
                data=absolute_salinity,
                metadata={
                    "units": "g kg-1",
                    "long_name": "Absolute salinity",
                    "standard_name": "sea_water_absolute_salinity",
                },
            ),
        )
        added.append("absolute_salinity")

    if settings.conservative_temperature:
        _add_channel(
            dataset,
            "conservative_temperature",
            Channel(
                data=conservative_temperature,
                metadata={
                    "units": "degree_C",
                    "long_name": "Conservative temperature",
                    "standard_name": "sea_water_conservative_temperature",
                },
            ),
        )
        added.append("conservative_temperature")

    if settings.potential_density:
        density_anomaly_data = _maybe_despike(
            dataset,
            "density_anomaly",
            gsw.sigma0(absolute_salinity, conservative_temperature),
            despike,
        )
        _add_channel(
            dataset,
            "density_anomaly",
            Channel(
                data=density_anomaly_data,
                metadata={
                    "units": "kg m-3",
                    "long_name": "Potential density anomaly (sigma0)",
                    "standard_name": "sea_water_sigma_theta",
                },
            ),
        )
        added.append("density_anomaly")

    if settings.potential_temperature:
        potential_temperature_data = _maybe_despike(
            dataset,
            "potential_temperature",
            gsw.pt0_from_t(absolute_salinity, temperature, sea_pressure),
            despike,
        )
        _add_channel(
            dataset,
            "potential_temperature",
            Channel(
                data=potential_temperature_data,
                metadata={
                    "units": "degree_C",
                    "long_name": "Potential temperature",
                    "standard_name": "sea_water_potential_temperature",
                },
            ),
        )
        added.append("potential_temperature")

    if settings.sound_speed:
        sound_speed_data = _maybe_despike(
            dataset,
            "speed_of_sound_in_sea_water",
            gsw.sound_speed(
                absolute_salinity, conservative_temperature, sea_pressure
            ),
            despike,
        )
        _add_channel(
            dataset,
            "speed_of_sound_in_sea_water",
            Channel(
                data=sound_speed_data,
                metadata={
                    "units": "m s-1",
                    "long_name": "Speed of sound in sea water",
                    "standard_name": "speed_of_sound_in_sea_water",
                },
            ),
        )
        added.append("speed_of_sound_in_sea_water")

    if settings.density:
        density_data = _maybe_despike(
            dataset,
            "sea_water_density",
            gsw.rho(absolute_salinity, conservative_temperature, sea_pressure),
            despike,
        )
        _add_channel(
            dataset,
            "sea_water_density",
            Channel(
                data=density_data,
                metadata={
                    "units": "kg m-3",
                    "long_name": "In-situ density",
                    "standard_name": "sea_water_density",
                },
            ),
        )
        added.append("sea_water_density")

    if settings.spiciness:
        spiciness_data = _maybe_despike(
            dataset,
            "spiciness",
            gsw.spiciness0(absolute_salinity, conservative_temperature),
            despike,
        )
        _add_channel(
            dataset,
            "spiciness",
            Channel(
                data=spiciness_data,
                metadata={
                    "units": "kg m-3",
                    "long_name": "Spiciness (referenced to 0 dbar)",
                },
            ),
        )
        added.append("spiciness")

    if settings.freezing_point:
        freezing_point_data = _maybe_despike(
            dataset,
            "freezing_point",
            gsw.CT_freezing(absolute_salinity, sea_pressure, 0),
            despike,
        )
        _add_channel(
            dataset,
            "freezing_point",
            Channel(
                data=freezing_point_data,
                metadata={
                    "units": "degree_C",
                    "long_name": "Freezing point (conservative temperature)",
                },
            ),
        )
        added.append("freezing_point")

    if settings.thermal_expansion:
        thermal_expansion_data = _maybe_despike(
            dataset,
            "thermal_expansion_coefficient",
            gsw.alpha(
                absolute_salinity, conservative_temperature, sea_pressure
            ),
            despike,
        )
        _add_channel(
            dataset,
            "thermal_expansion_coefficient",
            Channel(
                data=thermal_expansion_data,
                metadata={
                    "units": "K-1",
                    "long_name": "Thermal expansion coefficient",
                },
            ),
        )
        added.append("thermal_expansion_coefficient")

    if settings.haline_contraction:
        haline_contraction_data = _maybe_despike(
            dataset,
            "haline_contraction_coefficient",
            gsw.beta(absolute_salinity, conservative_temperature, sea_pressure),
            despike,
        )
        _add_channel(
            dataset,
            "haline_contraction_coefficient",
            Channel(
                data=haline_contraction_data,
                metadata={
                    "units": "kg g-1",
                    "long_name": "Haline contraction coefficient",
                },
            ),
        )
        added.append("haline_contraction_coefficient")

    if settings.oxygen_concentration:
        if "dissolved_oxygen_saturation" not in dataset.channels:
            raise ValueError(
                "Cannot compute oxygen_concentration: dataset has no "
                "dissolved_oxygen_saturation channel."
            )
        saturation = dataset.channels["dissolved_oxygen_saturation"].data
        solubility = gsw.O2sol(
            absolute_salinity,
            conservative_temperature,
            sea_pressure,
            longitude,
            latitude,
        )
        oxygen_concentration_data = _maybe_despike(
            dataset,
            "oxygen_concentration_from_saturation",
            solubility * (saturation / 100.0),
            despike,
        )
        _add_channel(
            dataset,
            "oxygen_concentration_from_saturation",
            Channel(
                data=oxygen_concentration_data,
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
        dataset.record(description)
        log_verbose(logger, description)

    return dataset
