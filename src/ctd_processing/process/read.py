"""Reading raw RBR ``.rsk`` deployment files with pyrsktools."""

import logging
from pathlib import Path

import pyrsktools

logger = logging.getLogger(__name__)

__all__ = ["read_rsk"]


def read_rsk(file: Path) -> pyrsktools.RSK:
    """Open an RBR ``.rsk`` file and read its data.

    Parameters
    ----------
    file : pathlib.Path
        Path to the ``.rsk`` file to read. `pyrsktools.RSK` opens this
        file strictly read-only, so it is safe to point directly at a
        deployment file; callers that later run write-capable
        `pyrsktools.RSK` methods should still operate on a private copy.

    Returns
    -------
    pyrsktools.RSK
        The opened dataset, with `readdata` already called so that
        `.data`/`.dataArrays`/`.channelNames` are populated. The
        underlying database connection has been closed by the time
        this returns; only the in-memory data is used downstream.
    """
    logger.debug("Opening RSK file: %s", file)
    with pyrsktools.RSK(str(file)) as rsk:
        rsk.readdata()
    return rsk
