"""Smoke test verifying the package is importable."""

import ctd_processing


def test_import() -> None:
    """Package should import without error."""
    assert ctd_processing is not None
