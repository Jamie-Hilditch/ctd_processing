"""Shared Typer option definitions reused across ctd_processing commands."""

from enum import StrEnum
from pathlib import Path
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

# Shared ``--config``/``-c`` option, added to every command that loads a
# `ctd_processing.config.Settings` (``process``, ``bin``, ``concatenate``).
# Defaults to ``Path("config.toml")`` in the current directory; `exists=True`
# means Click validates that path even when it is left at its default, so
# an invocation with no config.toml in the current directory and no
# explicit `--config` fails fast with a clear message.
ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help='Path to a TOML configuration file. Defaults to "config.toml" '
        "in the current directory.",
    ),
]


class LogLevel(StrEnum):
    """Log levels selectable via ``--log-level``."""

    DEBUG = "DEBUG"
    VERBOSE = "VERBOSE"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Shared ``--log-level`` option, added to every ctd_processing command
# (via `ctd_processing.cli._logging.configure_cli_logging`) so its
# behavior is identical across all of them.
LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        "--log-level",
        help="Minimum log level to emit.",
        case_sensitive=False,
    ),
]

# Shared ``--no-stdout-log`` option, added to every ctd_processing command
# so its behavior is identical across all of them.
NoStdoutLogOption = Annotated[
    bool,
    typer.Option(
        "--no-stdout-log",
        help="Disable writing log records to stdout.",
    ),
]

# Shared ``--verbose``/``--debug`` shortcuts, added to every ctd_processing
# command. Both override ``--log-level`` when given, regardless of what
# ``--log-level`` is set to; ``--debug`` wins if both are given (it implies
# ``--verbose``, since DEBUG is a lower/more detailed level than VERBOSE).
VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        help="Enable VERBOSE-level logging. Overridden by --debug.",
    ),
]

DebugOption = Annotated[
    bool,
    typer.Option(
        "--debug",
        help="Enable DEBUG-level logging (implies --verbose).",
    ),
]
