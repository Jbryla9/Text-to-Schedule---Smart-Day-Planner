"""Small local observability layer: console logs, rotating logs, and JSONL events."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "smart_scheduler"


def configure_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return

    level_name = os.getenv("SMART_SCHEDULER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    rotating = RotatingFileHandler(
        log_dir / "application.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(console)
    logger.addHandler(rotating)
    logger.propagate = False


def record_event(event_type: str, payload: dict[str, Any]) -> None:
    """Append compact operational metadata without storing the user's text."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logging.getLogger(LOGGER_NAME).exception("Could not write monitoring event")


def _log_dir() -> Path:
    return Path(os.getenv("SMART_SCHEDULER_LOG_DIR", "logs"))

