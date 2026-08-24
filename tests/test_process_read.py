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
