"""Logging configuration for the ctd_processing CLI.

Every submodule gets its own logger via ``logging.getLogger(__name__)``;
this module is the single place that attaches handlers, following the
standard practice that libraries only obtain loggers while the application
(here, the CLI) configures them.
"""

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER_NAME = "ctd_processing"

_STDOUT_HANDLER_NAME = "ctd_processing.stdout"
_LOG_FILE_HANDLER_NAME = "ctd_processing.log_file"
_ERROR_LOG_FILE_HANDLER_NAME = "ctd_processing.error_log_file"

_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

__all__ = ["configure_stdout_logging", "configure_file_logging"]


class _MaxLevelFilter(logging.Filter):
    """Reject log records at or above a given level.

    Parameters
    ----------
    max_level : int
        Records with `levelno` greater than or equal to this value are
        rejected.
    """

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether `record` is below `max_level`.

        Parameters
        ----------
        record : logging.LogRecord
            The record to test.

        Returns
        -------
        bool
            ``True`` if `record.levelno` is below `max_level`.
        """
        return record.levelno < self.max_level


def _remove_handler_named(logger: logging.Logger, name: str) -> None:
    """Remove and close a previously-installed handler, if present.

    Parameters
    ----------
    logger : logging.Logger
        Logger to remove the handler from.
    name : str
        `logging.Handler.name` of the handler to remove.
    """
    for handler in list(logger.handlers):
        if handler.name == name:
            logger.removeHandler(handler)
            handler.close()


def configure_stdout_logging(level: int, enable_stdout: bool) -> None:
    """Configure the ``ctd_processing`` package logger's stdout handler.

    Parameters
    ----------
    level : int
        Minimum level (e.g. `logging.INFO`) the package logger will
        process; records below this are dropped before reaching any
        handler.
    enable_stdout : bool
        If ``True``, attach a `logging.StreamHandler` writing every
        emitted record to `sys.stdout`. If ``False``, no stdout handler
        is attached (file handlers configured separately via
        :func:`configure_file_logging` are unaffected).

    Notes
    -----
    Any stdout handler previously installed by this function is removed
    and closed first, so repeated calls (e.g. once per CLI invocation in
    tests) don't accumulate duplicate handlers. The handler is built at
    call time so it captures the current `sys.stdout` (important under
    `typer.testing.CliRunner`, which redirects `sys.stdout` per
    invocation).
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    _remove_handler_named(logger, _STDOUT_HANDLER_NAME)

    if enable_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.name = _STDOUT_HANDLER_NAME
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)


def configure_file_logging(
    log_file: Path | None, error_log_file: Path | None
) -> None:
    """Configure the ``ctd_processing`` package logger's file handlers.

    Parameters
    ----------
    log_file : pathlib.Path or None
        If given, records below ``ERROR`` level are appended here. Its
        parent directory is created if it does not already exist.
    error_log_file : pathlib.Path or None
        If given, records at ``ERROR`` level and above are appended
        here. Its parent directory is created if it does not already
        exist.

    Notes
    -----
    Any file handlers previously installed by this function are removed
    and closed first (releasing their file handles), regardless of
    whether `log_file`/`error_log_file` are given this time, so a
    previous invocation's file logging doesn't leak into one that
    doesn't configure it.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _remove_handler_named(logger, _LOG_FILE_HANDLER_NAME)
    _remove_handler_named(logger, _ERROR_LOG_FILE_HANDLER_NAME)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.name = _LOG_FILE_HANDLER_NAME
        handler.addFilter(_MaxLevelFilter(logging.ERROR))
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)

    if error_log_file is not None:
        error_log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(error_log_file, encoding="utf-8")
        handler.name = _ERROR_LOG_FILE_HANDLER_NAME
        handler.setLevel(logging.ERROR)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
