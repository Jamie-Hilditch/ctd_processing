"""CF `standard_name`/`long_name` metadata for pyrsktools channels.

`pyrsktools` identifies a channel by `Channel.longName`
(`.venv/Lib/site-packages/pyrsktools/readers/reader.py:268-270`), a
lowercased, underscore-joined identifier it computes itself (e.g.
``"temperature"``, ``"dissolved_o2_concentration"``) — not a CF `long_name`
or `standard_name`, and not the human-readable name RBR's own Ruskin
software stores in the ``.rsk`` database (that survives only in the private
`Channel._dbName`). This module maps that pyrsktools identifier to correct
CF metadata for every channel pyrsktools recognizes for its own derivations
(the full set defined in `.venv/Lib/site-packages/pyrsktools/channels.py`).

A CF `standard_name` cannot be invented, so it is left unset here for
channels with no real CF equivalent (raw fluorometer/turbidity counts,
accelerometer axes, instrument housekeeping temperatures, etc.).
"""

import re
from dataclasses import dataclass

__all__ = [
    "ChannelCFMetadata",
    "cf_metadata_for_longname",
    "channel_key_for_longname",
]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ChannelCFMetadata:
    """CF metadata for one pyrsktools channel identifier.

    Attributes
    ----------
    long_name : str
        A CF `long_name`-style, human-readable description of the channel.
    standard_name : str or None
        The CF Standard Name for this channel's physical quantity, or
        ``None`` if no CF Standard Name applies (e.g. an uncalibrated raw
        instrument signal, or an instrument-housekeeping value rather than
        an environmental measurement).
    """

    long_name: str
    standard_name: str | None


# Keyed by `pyrsktools.datatypes.Channel.longName` as computed by pyrsktools
# itself (see module docstring), i.e. the same string used both as the
# `RSK.data` field name and in `pyrsktools/channels.py`'s own constants.
#
# Confidence notes (see the plan this was implemented from for detail):
# - High confidence: temperature, conductivity, pressure, salinity,
#   absolute_salinity, depth, potential_temperature, speed_of_sound,
#   barometer_pressure/atmospheric_pressure (air_pressure spot-checked
#   against the live CF Standard Name Table).
# - Verified against the official CF Standard Name Table (definitions
#   supplied by project maintainer): `pressure` -> sea_water_pressure
#   ("includes the pressure due to overlying sea water, sea ice, air and
#   any other medium that may be present" -- i.e. absolute pressure);
#   `sea_pressure` -> sea_water_pressure_due_to_sea_water ("excludes the
#   pressure due to sea ice, air and any other medium"); `bpr_pressure`/
#   `bpr_corrected_pressure` -> sea_water_pressure_at_sea_floor, since a
#   bottom pressure recorder specifically measures pressure at the sea
#   floor, which CF has a distinct, more precise standard name for.
#   `dissolved_o2_concentration`/`_corrected` ->
#   mole_concentration_of_dissolved_molecular_oxygen_in_sea_water: also
#   confirmed against the table -- RBR reports this channel in µmol/L, a
#   molar (per-volume) unit dimensionally compatible with CF's canonical
#   `mol m-3` (a factor-of-1000 conversion, not a different quantity).
#   Ruled out: mass_concentration_of_oxygen_in_sea_water (mass-based, not
#   molar); moles_of_oxygen_per_unit_mass_in_sea_water (molality, i.e.
#   per unit mass, not per unit volume); the `_at_saturation`,
#   `_at_shallowest_local_minimum_in_vertical_profile`, and `preformed_`
#   variants (each a distinct derived/diagnostic quantity, not a plain
#   measured-or-corrected concentration).
#   `dissolved_o2_saturation`/`_corrected` ->
#   fractional_saturation_of_oxygen_in_sea_water: confirmed against the
#   table -- canonical unit `1` (dimensionless ratio), compatible with
#   RBR's `%` (factor-of-100 scaling of the same ratio quantity), and
#   distinct in kind from the concentration-valued oxygen standard names
#   above ("ratio of some measure of concentration to the saturated
#   value of the same quantity" vs. a concentration itself).
#   `buoyancy_frequency_squared` ->
#   square_of_brunt_vaisala_frequency_in_sea_water: confirmed against the
#   table -- Brunt-Vaisala frequency is explicitly defined there as "also
#   sometimes called 'buoyancy frequency'", and the canonical unit `s-2`
#   matches pyrsktools' `1/s²` exactly.
# - density_anomaly: pyrsktools computes this via
#   `gsw.sigma0(absoluteSalinity, conservativeTemperature)`
#   (`.venv/Lib/site-packages/pyrsktools/_rsk/calculators.py:1032`), i.e.
#   potential density anomaly referenced to 0 dbar, which is
#   `sea_water_sigma_theta` (not `sea_water_sigma_t`, which would be
#   referenced to in-situ pressure).
_CF_CHANNEL_METADATA: dict[str, ChannelCFMetadata] = {
    "temperature": ChannelCFMetadata(
        "Sea water temperature", "sea_water_temperature"
    ),
    "temperature_corrected": ChannelCFMetadata(
        "Sea water temperature (thermal-lag corrected)",
        "sea_water_temperature",
    ),
    "conductivity": ChannelCFMetadata(
        "Sea water electrical conductivity",
        "sea_water_electrical_conductivity",
    ),
    "pressure": ChannelCFMetadata("Absolute pressure", "sea_water_pressure"),
    "sea_pressure": ChannelCFMetadata(
        "Sea pressure", "sea_water_pressure_due_to_sea_water"
    ),
    "pressure_drift": ChannelCFMetadata(
        "Pressure sensor drift correction", None
    ),
    "bpr_pressure": ChannelCFMetadata(
        "Bottom pressure recorder pressure", "sea_water_pressure_at_sea_floor"
    ),
    "bpr_corrected_pressure": ChannelCFMetadata(
        "Bottom pressure recorder pressure (drift corrected)",
        "sea_water_pressure_at_sea_floor",
    ),
    "barometer_pressure": ChannelCFMetadata(
        "Barometer atmospheric pressure", "air_pressure"
    ),
    "bpr_temperature": ChannelCFMetadata(
        "Bottom pressure recorder internal temperature", None
    ),
    "barometer_temperature": ChannelCFMetadata(
        "Barometer internal temperature", None
    ),
    "atmospheric_pressure": ChannelCFMetadata(
        "Atmospheric pressure (for sea pressure correction)", "air_pressure"
    ),
    "salinity": ChannelCFMetadata(
        "Practical salinity", "sea_water_practical_salinity"
    ),
    "depth": ChannelCFMetadata("Depth", "depth"),
    "velocity": ChannelCFMetadata("Profiling velocity", None),
    "specific_conductivity": ChannelCFMetadata(
        "Specific conductivity (referenced to 25°C)", None
    ),
    "dissolved_o2_concentration": ChannelCFMetadata(
        "Dissolved oxygen concentration",
        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    ),
    "dissolved_o2_saturation": ChannelCFMetadata(
        "Dissolved oxygen saturation",
        "fractional_saturation_of_oxygen_in_sea_water",
    ),
    "dissolved_o2_concentration_corrected": ChannelCFMetadata(
        "Dissolved oxygen concentration (corrected)",
        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    ),
    "dissolved_o2_saturation_corrected": ChannelCFMetadata(
        "Dissolved oxygen saturation (corrected)",
        "fractional_saturation_of_oxygen_in_sea_water",
    ),
    "buoyancy_frequency_squared": ChannelCFMetadata(
        "Buoyancy frequency squared",
        "square_of_brunt_vaisala_frequency_in_sea_water",
    ),
    "stability": ChannelCFMetadata("Water column stability", None),
    "density_anomaly": ChannelCFMetadata(
        "Density anomaly", "sea_water_sigma_theta"
    ),
    "absolute_salinity": ChannelCFMetadata(
        "Absolute salinity", "sea_water_absolute_salinity"
    ),
    "potential_temperature": ChannelCFMetadata(
        "Potential temperature", "sea_water_potential_temperature"
    ),
    "speed_of_sound": ChannelCFMetadata(
        "Speed of sound in sea water", "speed_of_sound_in_sea_water"
    ),
    "chlorophyll": ChannelCFMetadata(
        "Chlorophyll fluorescence (raw counts)", None
    ),
    "backscatter": ChannelCFMetadata("Optical backscatter (raw counts)", None),
    "phycoerythrin": ChannelCFMetadata(
        "Phycoerythrin fluorescence (raw counts)", None
    ),
    "x_axis_acceleration": ChannelCFMetadata("X-axis acceleration", None),
    "y_axis_acceleration": ChannelCFMetadata("Y-axis acceleration", None),
    "z_axis_acceleration": ChannelCFMetadata("Z-axis acceleration", None),
    "accelerometer_temperature": ChannelCFMetadata(
        "Accelerometer internal temperature", None
    ),
}


def cf_metadata_for_longname(long_name: str) -> ChannelCFMetadata:
    """Look up CF metadata for a pyrsktools channel `longName`.

    Parameters
    ----------
    long_name : str
        A `pyrsktools.datatypes.Channel.longName` value (see module
        docstring for what this actually is).

    Returns
    -------
    ChannelCFMetadata
        The matching table entry, or, for a `long_name` this module
        doesn't recognize (e.g. a sensor type pyrsktools itself has no
        constant for, such as pH or turbidity), a fallback
        `ChannelCFMetadata` that reuses `long_name` verbatim as `long_name`
        and leaves `standard_name` as ``None``.
    """
    return _CF_CHANNEL_METADATA.get(
        long_name, ChannelCFMetadata(long_name=long_name, standard_name=None)
    )


def _slugify(text: str) -> str:
    """Lowercase `text` and join runs of non-alphanumeric characters with `_`.

    Parameters
    ----------
    text : str
        Text to slugify.

    Returns
    -------
    str
        E.g. ``"Sea water temperature"`` -> ``"sea_water_temperature"``.
    """
    return _NON_ALNUM.sub("_", text.lower()).strip("_")


def channel_key_for_longname(long_name: str) -> str:
    """Return the short, stable identifier to store a channel under.

    This is deliberately **not** the CF `standard_name` (see
    `cf_metadata_for_longname`) -- `standard_name`s can be long and
    cluttered with qualifiers needed to disambiguate related quantities
    (e.g. ``mole_concentration_of_dissolved_molecular_oxygen_in_sea_water``),
    which makes them a poor fit for a dict key, a config-section name, or
    anything else meant to be typed by a person. Instead this slugifies
    the channel's CF-style `long_name`, which this module already defines
    to be short and unique per channel.

    `Dataset.channels` (`ctd_processing.process.build.build_dataset`) and
    `process.raw_channels` config sections
    (`ctd_processing.config.ProcessSettings`) are both keyed by this
    function's return value.

    Parameters
    ----------
    long_name : str
        A `pyrsktools.datatypes.Channel.longName` value (see module
        docstring for what this actually is) -- the same input
        `cf_metadata_for_longname` takes.

    Returns
    -------
    str
        E.g. ``"temperature"`` -> ``"sea_water_temperature"``,
        ``"dissolved_o2_concentration"`` ->
        ``"dissolved_oxygen_concentration"``. For a `long_name` this
        module doesn't recognize, slugifies pyrsktools' own (already
        snake_case) identifier, which is usually a no-op.
    """
    return _slugify(cf_metadata_for_longname(long_name).long_name)
