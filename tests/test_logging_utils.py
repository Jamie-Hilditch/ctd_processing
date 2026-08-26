"""Tests for ctd_processing.logging_utils."""

import logging

import pytest

from ctd_processing.logging_utils import VERBOSE, log_verbose


def test_verbose_is_between_debug_and_info() -> None:
    """VERBOSE sits strictly between DEBUG (10) and INFO (20)."""
    assert logging.DEBUG < VERBOSE < logging.INFO


def test_verbose_level_name_is_registered() -> None:
    """The VERBOSE level number is registered under the name 'VERBOSE'."""
    assert logging.getLevelName(VERBOSE) == "VERBOSE"


def test_log_verbose_emits_record_at_verbose_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_verbose logs at VERBOSE with lazy %-style formatting."""
    logger = logging.getLogger("ctd_processing.test_logging_utils")
    caplog.set_level(VERBOSE, logger=logger.name)

    log_verbose(logger, "removed %d value(s)", 3)

    [record] = caplog.records
    assert record.levelno == VERBOSE
    assert record.getMessage() == "removed 3 value(s)"


def test_log_verbose_suppressed_below_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A VERBOSE record is not captured when the threshold is INFO."""
    logger = logging.getLogger("ctd_processing.test_logging_utils")
    caplog.set_level(logging.INFO, logger=logger.name)

    log_verbose(logger, "should not appear")

    assert caplog.records == []
