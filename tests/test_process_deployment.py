"""Tests for ctd_processing.process."""

from pathlib import Path

import pytest

from ctd_processing.config import ProcessSettings, ProjectSettings
from ctd_processing.logging_utils import VERBOSE
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


@pytest.mark.requires_example_data
def test_process_deployment_applies_raw_channel_processing(
    tmp_path: Path,
    example_rsk_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """process_deployment's dataset has had remove_holds applied to it.

    The example file's temperature channel has known zero-order holds
    (verified in earlier work), so a VERBOSE record from `remove_holds`
    appearing here proves the full read_rsk -> build_dataset ->
    process_raw_channels chain actually ran.
    """
    caplog.set_level(VERBOSE, logger="ctd_processing.process.raw_channels")

    process_deployment(
        example_rsk_path,
        tmp_path / "profiles",
        ProcessSettings(),
        ProjectSettings(),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("zero-order hold value(s)" in m for m in messages)
