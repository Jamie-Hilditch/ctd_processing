"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import pytest

_EXAMPLE_RSK_PATH = (
    Path(__file__).parent / "example_data" / "243188_20260823_0120.rsk"
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
