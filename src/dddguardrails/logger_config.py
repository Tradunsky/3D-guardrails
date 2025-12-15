"""Logging utilities with a compact console-friendly formatter."""

from __future__ import annotations

import logging
from logging.config import dictConfig

LOG_FORMAT = "%(levelname).1s %(asctime)s %(name)s:%(lineno)d | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging using a dictConfig with a compact console formatter.

    Minimal output for small terminals: single-letter level, short timestamp,
    module name with line number, and the message.
    """
    if getattr(configure_logging, "_configured", False):
        # Allow callers to bump the level without reconfiguring handlers.
        logging.getLogger().setLevel(level)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).setLevel(level)
        return

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "compact": {
                    "format": LOG_FORMAT,
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "compact",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
            "loggers": {
                "uvicorn": {"level": level, "propagate": True},
                "uvicorn.error": {"level": level, "propagate": True},
                "uvicorn.access": {"level": level, "propagate": True},
                "dddguardrails": {"level": level, "propagate": True},
            },
        }
    )

    configure_logging._configured = True  # type: ignore[attr-defined]

