from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Create or return a module logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
