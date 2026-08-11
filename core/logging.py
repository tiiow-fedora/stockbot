"""Structured logging setup shared by every component.

Every log line goes through the standard library `logging` module so it
works the same in a terminal, in a Docker log stream, or redirected to a
file. We keep it simple (no external logging framework) but consistent:

    [INFO] AAPL score=0.72 decision=BUY
    [INFO] Risk approved quantity=10
    [INFO] Paper order executed AAPL qty=10 price=213.42
    [INFO] BUY AAPL rejected by risk manager: max_position_size
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("stockbot")
    root.setLevel(level.upper())
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"stockbot.{name}")
