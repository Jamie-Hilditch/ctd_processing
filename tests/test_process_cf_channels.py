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
    assert channel_key_for_longname("temperature") == "sea_water_temperature"
    assert channel_key_for_longname("pressure") == "absolute_pressure"
    assert channel_key_for_longname("sea_pressure") == "sea_pressure"
    assert channel_key_for_longname("salinity") == "practical_salinity"
    assert (
        channel_key_for_longname("dissolved_o2_concentration")
        == "dissolved_oxygen_concentration"
    )


def test_channel_key_for_unrecognized_channel_is_usually_a_no_op() -> None:
    """An unrecognized, already-snake_case longName round-trips unchanged."""
    assert channel_key_for_longname("ph") == "ph"
