"""Shared Typer option definitions reused across ctd_processing commands."""

from enum import StrEnum
from typing import Annotated

import typer

# Repeatable ``--set key=value`` option shared by every ctd_processing
# command. Defaults to None (not a mutable []); commands normalize with
# `set_ or []`.
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help="Override a config option, e.g. --set section.key=value. "
        "May be repeated.",
    ),
]


class LogLevel(StrEnum):
    """Log levels selectable via ``--log-level``."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Global ``--log-level`` option, defined once on the app's callback so its
# behavior is identical across every command.
LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        "--log-level",
        help="Minimum log level to emit.",
        case_sensitive=False,
    ),
]

# Global ``--no-stdout-log`` option, defined once on the app's callback so
# its behavior is identical across every command.
NoStdoutLogOption = Annotated[
    bool,
    typer.Option(
        "--no-stdout-log",
        help="Disable writing log records to stdout.",
    ),
]
