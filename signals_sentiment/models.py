"""Shared types for the sentiment signal package."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class DocKind(str, Enum):
    NEWS = "news"
    FILING = "filing"


@dataclass(frozen=True)
class Document:
    """One piece of text about one symbol, as emitted by a DocumentSource.

    `timestamp` is when the document became PUBLIC. Everything downstream
    (decay, no-lookahead filtering) keys off it, so sources must never
    backdate or forward-date it.
    """

    symbol: str
    timestamp: datetime
    title: str
    body: str = ""
    source: str = "unknown"
    kind: DocKind = DocKind.NEWS
    url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Document.timestamp must be timezone-aware")

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}".strip() if self.body else self.title

    @property
    def doc_id(self) -> str:
        """Stable content id, used for de-duplication and cache keys."""
        raw = "|".join([self.symbol, self.source, self.kind.value, self.timestamp.isoformat(), self.title, self.url or ""])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: Optional[float]  # [-1, 1], None = scorer had no opinion
    scorer: str
