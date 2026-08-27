"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import pytest

_EXAMPLE_RSK_PATH = (
    Path(__file__).parent / "example_data" / "243188_20260823_0120.rsk"
)
_EXAMPLE_RSK_PATH_FLUOROMETER = (
    Path(__file__).parent / "example_data" / "066064_20220704_0913.rsk"
)
_EXAMPLE_RSK_PATH_OXYGEN = (
    Path(__file__).parent / "example_data" / "DO_065798_20210622_0729.rsk"
)


@pytest.fixture(scope="session")
def example_rsk_path() -> Path:
    """Path to a real RBR ``.rsk`` deployment file.

    Skips the requesting test if ``tests/example_data/`` is not present,
    since it holds real instrument data that is not checked into git.

    Returns
    -------
    Path
        Path to the example ``.rsk`` file.
    """
    if not _EXAMPLE_RSK_PATH.exists():
        pytest.skip(f"example data not available: {_EXAMPLE_RSK_PATH}")
    return _EXAMPLE_RSK_PATH


@pytest.fixture(scope="session")
def example_rsk_path_fluorometer() -> Path:
    """Path to a real ``.rsk`` file from a second, different instrument.

    Channels: ``conductivity``/``temperature``/``pressure`` plus
    ``backscatter``/``chlorophyll``/``cdom``. No ``sea_pressure`` channel
    and no oxygen channel.

    Skips the requesting test if ``tests/example_data/`` is not present,
    since it holds real instrument data that is not checked into git.

    Returns
    -------
    Path
        Path to the example ``.rsk`` file.
    """
    if not _EXAMPLE_RSK_PATH_FLUOROMETER.exists():
        pytest.skip(
            f"example data not available: {_EXAMPLE_RSK_PATH_FLUOROMETER}"
        )
    return _EXAMPLE_RSK_PATH_FLUOROMETER


@pytest.fixture(scope="session")
def example_rsk_path_oxygen() -> Path:
    """Path to a real ``.rsk`` file from a third instrument, with oxygen data.

    Channels: ``conductivity``/``temperature``/``pressure`` plus
    ``dissolved_o2_saturation``/``backscatter``/``chlorophyll``/
    ``phycoerythrin``. No ``sea_pressure`` channel.

    Skips the requesting test if ``tests/example_data/`` is not present,
    since it holds real instrument data that is not checked into git.

    Returns
    -------
    Path
        Path to the example ``.rsk`` file.
    """
    if not _EXAMPLE_RSK_PATH_OXYGEN.exists():
        pytest.skip(f"example data not available: {_EXAMPLE_RSK_PATH_OXYGEN}")
    return _EXAMPLE_RSK_PATH_OXYGEN
