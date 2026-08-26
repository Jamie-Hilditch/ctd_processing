"""Command line interface for ctd_processing."""

import logging

import typer

from ctd_processing.cli._logging import configure_stdout_logging
from ctd_processing.cli._options import (
    DebugOption,
    LogLevel,
    LogLevelOption,
    NoStdoutLogOption,
    VerboseOption,
)
from ctd_processing.cli.bin import bin_command
from ctd_processing.cli.concatenate import concatenate_command
from ctd_processing.cli.init import init_command
from ctd_processing.cli.process import process_command
from ctd_processing.logging_utils import VERBOSE

app = typer.Typer(
    name="ctd-processing", help="Process RBR CTD (.rsk) data files."
)


@app.callback()
def main(
    log_level: LogLevelOption = LogLevel.INFO,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
    no_stdout_log: NoStdoutLogOption = False,
) -> None:
    """Process RBR CTD (.rsk) data files.

    Parameters
    ----------
    log_level : LogLevel, optional
        Minimum log level to emit, consistently applied across every
        command. Overridden by `verbose`/`debug` when either is given.
    verbose : bool, optional
        If given, emit VERBOSE-level (and above) log records regardless
        of `log_level`. Overridden by `debug`.
    debug : bool, optional
        If given, emit DEBUG-level (and above) log records regardless of
        `log_level`/`verbose` -- DEBUG is more detailed than VERBOSE, so
        this includes every VERBOSE record too.
    no_stdout_log : bool, optional
        If given, log records are not written to stdout. Independent
        of any ``paths.log_file``/``paths.error_log_file`` configured
        for the command being run.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = VERBOSE
    else:
        level = logging.getLevelNamesMapping()[log_level.value]
    configure_stdout_logging(level=level, enable_stdout=not no_stdout_log)


app.command(name="init")(init_command)
app.command(name="process")(process_command)
app.command(name="bin")(bin_command)
app.command(name="concatenate")(concatenate_command)

__all__ = ["app"]
