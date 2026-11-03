"""Pluggable document source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List

from ..models import Document


class DocumentSource(ABC):
    name: str = "source"

    @abstractmethod
    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        """Documents for `symbol` with start <= timestamp <= end, oldest first.

        Must be a pure read: no trading side effects, and never return a
        document published after `end` (the caller relies on this for
        backtest no-lookahead, and re-filters defensively anyway).
        """


class StaticSource(DocumentSource):
    """In-memory source; handy for tests and for wrapping pre-parsed data."""

    def __init__(self, documents: Iterable[Document], name: str = "static"):
        self.name = name
        self._docs = list(documents)

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        sym = symbol.upper()
        return sorted(
            (d for d in self._docs if d.symbol.upper() == sym and start <= d.timestamp <= end),
            key=lambda d: d.timestamp,
        )


class CompositeSource(DocumentSource):
    """Merges several sources, de-duplicating by doc_id. A failing source is
    logged and skipped so one flaky feed never blanks the whole signal."""

    name = "composite"

    def __init__(self, sources: Iterable[DocumentSource]):
        self._sources = list(sources)

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        from core.logging import get_logger

        log = get_logger("signals_sentiment.sources")
        seen, out = set(), []
        for src in self._sources:
            try:
                docs = src.fetch(symbol, start, end)
            except Exception as exc:  # noqa: BLE001 - isolate source failures
                log.warning("source %s failed for %s: %s", getattr(src, "name", src), symbol, exc)
                continue
            for d in docs:
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    out.append(d)
        return sorted(out, key=lambda d: d.timestamp)
