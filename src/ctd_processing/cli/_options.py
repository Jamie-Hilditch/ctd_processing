"""Shared Typer option definitions reused across ctd_processing commands."""

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
