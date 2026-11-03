"""Structured JSON logging.

Enabled with LOG_FORMAT=json (or ``configure_logging(json_format=True)``).
Output is one JSON object per line: ts, level, logger, msg, plus any
``extra=`` fields and exception info. Plain text remains the default.

Note: core.logging.get_logger() configures logging with defaults at import
time, which makes a later setup_logging(level, file) a silent no-op. This
module therefore (re)configures the ``stockbot`` logger itself, idempotently.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}
_SECRET_HINTS = ("key", "secret", "token", "password", "authorization")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            payload[k] = "***" if any(h in k.lower() for h in _SECRET_HINTS) else v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


_TEXT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    level: str = "INFO", log_file: Optional[str] = None, json_format: Optional[bool] = None
) -> None:
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "text").lower() == "json"
    root = logging.getLogger("stockbot")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(level.upper())
    root.propagate = False
    fmt = JsonFormatter() if json_format else logging.Formatter(_TEXT_FMT, "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError as exc:  # e.g. read-only volume; stdout still works
            sys.stderr.write(f"could not open log file {log_file}: {exc}\n")
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    # Tell core.logging it no longer needs to self-configure.
    import core.logging as core_logging
    core_logging._CONFIGURED = True
