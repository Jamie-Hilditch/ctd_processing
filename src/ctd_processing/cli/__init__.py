"""Command line interface for ctd_processing."""

import logging

import typer

from ctd_processing.cli._logging import configure_stdout_logging
from ctd_processing.cli._options import (
    LogLevel,
    LogLevelOption,
    NoStdoutLogOption,
)
from ctd_processing.cli.bin import bin_command
from ctd_processing.cli.concatenate import concatenate_command
from ctd_processing.cli.init import init_command
from ctd_processing.cli.process import process_command

app = typer.Typer(
    name="ctd-processing", help="Process RBR CTD (.rsk) data files."
)


@app.callback()
def main(
    log_level: LogLevelOption = LogLevel.INFO,
    no_stdout_log: NoStdoutLogOption = False,
) -> None:
    """Process RBR CTD (.rsk) data files.

    Parameters
    ----------
    log_level : LogLevel, optional
        Minimum log level to emit, consistently applied across every
        command.
    no_stdout_log : bool, optional
        If given, log records are not written to stdout. Independent
        of any ``paths.log_file``/``paths.error_log_file`` configured
        for the command being run.
    """
    configure_stdout_logging(
        level=logging.getLevelNamesMapping()[log_level.value],
        enable_stdout=not no_stdout_log,
    )


app.command(name="init")(init_command)
app.command(name="process")(process_command)
app.command(name="bin")(bin_command)
app.command(name="concatenate")(concatenate_command)

__all__ = ["app"]
