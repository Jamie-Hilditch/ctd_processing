"""Tests for ctd_processing.process."""

from pathlib import Path

import pytest

from ctd_processing.config import ProcessSettings, ProjectSettings
from ctd_processing.process import process_deployment


@pytest.mark.requires_example_data
def test_process_deployment_reads_and_returns_none(
    tmp_path: Path, example_rsk_path: Path
) -> None:
    """process_deployment reads the deployment (step 1) and returns None.

    Profile extraction is not yet implemented, so a real, readable
    ``.rsk`` deployment currently produces no further effect.
    """
    result = process_deployment(
        example_rsk_path,
        tmp_path / "profiles",
        ProcessSettings(),
        ProjectSettings(),
    )

    assert result is None
