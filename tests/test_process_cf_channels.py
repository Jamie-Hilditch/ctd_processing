"""Tests for ctd_processing.process.cf_channels."""

from ctd_processing.process.cf_channels import (
    ChannelCFMetadata,
    cf_metadata_for_longname,
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
