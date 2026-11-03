"""Caching for scores and source fetches.

Scores are cached by (scorer name, doc_id): the same document is never
re-scored (matters for paid LLM calls). Optionally persisted to a JSON file.
`None` results ("no opinion") are cached too, so unscorable docs aren't
retried every cycle.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .models import Document
from .scorers import DocumentScorer
from .sources.base import DocumentSource

_MISSING = object()


class ScoreCache:
    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else None
        self._data: Dict[str, Optional[float]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self._path and self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}  # corrupt cache is disposable

    @staticmethod
    def key(scorer: str, doc: Document) -> str:
        return f"{scorer}:{doc.doc_id}"

    def get(self, key: str):
        with self._lock:
            if key in self._data:
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return _MISSING

    def put(self, key: str, value: Optional[float]) -> None:
        with self._lock:
            self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def save(self) -> None:
        if not self._path:
            return
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self._path)  # atomic: no half-written cache on crash


class CachedScorer(DocumentScorer):
    def __init__(self, inner: DocumentScorer, cache: ScoreCache):
        self._inner, self._cache = inner, cache
        self.name = inner.name

    def score(self, doc: Document) -> Optional[float]:
        key = self._cache.key(self._inner.name, doc)
        hit = self._cache.get(key)
        if hit is not _MISSING:
            return hit
        value = self._inner.score(doc)
        self._cache.put(key, value)
        return value


class CachedSource(DocumentSource):
    """TTL cache over a source, keyed by (symbol, start, end). `clock` is
    injectable so tests control time."""

    def __init__(self, inner: DocumentSource, ttl_seconds: float, clock: Callable[[], float]):
        self._inner, self._ttl, self._clock = inner, ttl_seconds, clock
        self.name = getattr(inner, "name", "cached")
        self._store: Dict[Tuple[str, datetime, datetime], Tuple[float, List[Document]]] = {}

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        key = (symbol.upper(), start, end)
        now = self._clock()
        entry = self._store.get(key)
        if entry and now - entry[0] < self._ttl:
            return list(entry[1])
        docs = self._inner.fetch(symbol, start, end)
        self._store[key] = (now, docs)
        return list(docs)
