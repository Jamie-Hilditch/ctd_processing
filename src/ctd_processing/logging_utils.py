"""Shared logging utilities: a VERBOSE level between INFO and DEBUG.

`VERBOSE` is for processing-history events (see `Channel`/`Dataset`'s
`record`/`add_channel`/`remove_channel`) -- detailed enough to be noisy at
the default `INFO` level, but not as voluminous as full `DEBUG` output.
"""

import logging

__all__ = ["VERBOSE", "log_verbose"]

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


def log_verbose(logger: logging.Logger, message: str, *args: object) -> None:
    """Log `message` at the `VERBOSE` level (between `INFO` and `DEBUG`).

    Parameters
    ----------
    logger : logging.Logger
        The logger to emit the record on.
    message : str
        The log message, using ``%``-style placeholders for `args` (lazily
        formatted, matching the rest of the codebase's logging calls).
    *args : object
        Values to substitute into `message`'s placeholders.
    """
    logger.log(VERBOSE, message, *args)
