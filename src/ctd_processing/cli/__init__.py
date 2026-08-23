"""Command line interface for ctd_processing."""

import typer

from ctd_processing.cli.bin import bin_command
from ctd_processing.cli.concatenate import concatenate_command
from ctd_processing.cli.init import init_command
from ctd_processing.cli.process import process_command

app = typer.Typer(
    name="ctd-processing", help="Process RBR CTD (.rsk) data files."
)

app.command(name="init")(init_command)
app.command(name="process")(process_command)
app.command(name="bin")(bin_command)
app.command(name="concatenate")(concatenate_command)

__all__ = ["app"]
