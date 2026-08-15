"""Central logging configuration.

Global engineering standard: all diagnostic output goes to **stderr**, never
stdout. Nothing in this package may use ``print()`` on a production path.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Configure root logging to stderr. Call once, from the CLI entrypoint."""
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format=_LOG_FORMAT,
        force=force,
    )
    logger.debug("Logging configured at level %s", level)
