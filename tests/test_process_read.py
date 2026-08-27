"""Tests for ctd_processing.process.read."""

from pathlib import Path

import pyrsktools
import pytest

from ctd_processing.process.read import read_rsk


@pytest.mark.requires_example_data
def test_read_rsk_returns_opened_rsk_with_data(
    example_rsk_path: Path,
) -> None:
    """read_rsk returns an RSK instance with data already populated."""
    rsk = read_rsk(example_rsk_path)

    assert isinstance(rsk, pyrsktools.RSK)
    assert rsk.data.size > 0
    assert rsk.channelNames


@pytest.mark.requires_example_data
def test_read_rsk_returns_opened_rsk_with_data_for_fluorometer_instrument(
    example_rsk_path_fluorometer: Path,
) -> None:
    """read_rsk also works against a second, differently-configured device."""
    rsk = read_rsk(example_rsk_path_fluorometer)

    assert isinstance(rsk, pyrsktools.RSK)
    assert rsk.data.size > 0
    assert rsk.channelNames


@pytest.mark.requires_example_data
def test_read_rsk_returns_opened_rsk_with_data_for_oxygen_instrument(
    example_rsk_path_oxygen: Path,
) -> None:
    """read_rsk also works against a third instrument, carrying oxygen data."""
    rsk = read_rsk(example_rsk_path_oxygen)

    assert isinstance(rsk, pyrsktools.RSK)
    assert rsk.data.size > 0
    assert rsk.channelNames
