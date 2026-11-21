"""Logging setup for the command line: plain text or JSON lines, on stderr.

stdlib ``logging`` only. Library code just calls ``logging.getLogger(__name__)``;
this module is the one place that decides where records go.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

LOGGER_NAME = "calico"
_NOISY_LOGGERS = ("huggingface_hub", "urllib3")
_STANDARD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per record: ts, level, logger, msg, plus any ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StderrHandler(logging.Handler):
    """Writes to whatever ``sys.stderr`` is *now*, so a replaced stream never goes stale."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_logging(verbosity: int, json_lines: bool) -> None:
    """Warnings by default; ``-v`` adds INFO, ``-vv`` DEBUG. Safe to call repeatedly."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    handler = _StderrHandler()
    handler.setFormatter(
        JsonFormatter()
        if json_lines
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    # Ours is the only handler: a library that calls logging.basicConfig() gives the
    # root logger one too, and every line would be printed twice.
    logger.propagate = False
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR if verbosity < 2 else logging.NOTSET)
