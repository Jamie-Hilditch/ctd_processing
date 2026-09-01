"""Tests for ctd_processing.process.cf_channels."""

from ctd_processing.process.cf_channels import (
    ChannelCFMetadata,
    cf_metadata_for_longname,
    channel_key_for_longname,
)


def test_known_channel_resolves_to_documented_metadata() -> None:
    """A recognized longName resolves to its table entry."""
    result = cf_metadata_for_longname("temperature")

    assert result == ChannelCFMetadata(
        "Sea water temperature", "sea_water_temperature"
    )


def test_known_channel_without_standard_name() -> None:
    """A recognized but non-CF-mappable longName has standard_name=None."""
    result = cf_metadata_for_longname("chlorophyll")

    assert result.long_name == "Chlorophyll fluorescence (raw counts)"
    assert result.standard_name is None


def test_unrecognized_channel_falls_back_to_input() -> None:
    """An unrecognized longName falls back to itself, with no standard_name."""
    result = cf_metadata_for_longname("ph")

    assert result == ChannelCFMetadata(long_name="ph", standard_name=None)


def test_channel_key_slugifies_long_name_not_standard_name() -> None:
    """The key comes from long_name, which may differ from standard_name."""
    assert channel_key_for_longname("temperature") == "temperature"
    assert channel_key_for_longname("pressure") == "absolute_pressure"
    assert channel_key_for_longname("sea_pressure") == "sea_pressure"
    assert channel_key_for_longname("salinity") == "practical_salinity"
    assert (
        channel_key_for_longname("dissolved_o2_concentration")
        == "dissolved_oxygen_concentration"
    )


def test_channel_key_drops_redundant_sea_water_qualifier() -> None:
    """A "sea water" qualifier is dropped from the key; standard_name is not."""
    assert channel_key_for_longname("conductivity") == "electrical_conductivity"
    assert (
        cf_metadata_for_longname("conductivity").standard_name
        == "sea_water_electrical_conductivity"
    )


def test_channel_key_drops_leading_in_before_sea_water() -> None:
    """A "... in sea water" qualifier drops both "in" and "sea_water"."""
    assert channel_key_for_longname("speed_of_sound") == "speed_of_sound"
    assert (
        cf_metadata_for_longname("speed_of_sound").standard_name
        == "speed_of_sound_in_sea_water"
    )


def test_channel_key_bare_sea_water_is_not_emptied() -> None:
    """Dropping "sea_water" never leaves an empty key."""
    assert channel_key_for_longname("sea_water") == "sea_water"


def test_channel_key_for_unrecognized_channel_is_usually_a_no_op() -> None:
    """An unrecognized, already-snake_case longName round-trips unchanged."""
    assert channel_key_for_longname("ph") == "ph"
