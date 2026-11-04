"""SentimentSignal: source -> score -> cache -> decay -> one number.

Also provides `SentimentNewsProvider`, a drop-in `NewsDataProvider`, so the
existing pipeline/strategies consume this package without any changes.
Signal only: it never creates orders or touches execution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from core.models import NewsItem
from data.base import NewsDataProvider

from .cache import CachedScorer, ScoreCache
from .decay import aggregate
from .models import Document, ScoredDocument
from .scorers import DocumentScorer, LexiconScorer
from .sources.base import DocumentSource


@dataclass(frozen=True)
class SignalConfig:
    lookback_hours: float = 24.0 * 14
    half_life_hours: float = 24.0
    filing_half_life_hours: float = 24.0 * 7
    filing_boost: float = 1.5
    saturation: float = 1.0
    max_docs: int = 200


class SentimentSignal:
    def __init__(
        self,
        source: DocumentSource,
        scorer: Optional[DocumentScorer] = None,
        cache: Optional[ScoreCache] = None,
        config: Optional[SignalConfig] = None,
    ):
        self.source = source
        self.config = config or SignalConfig()
        base = scorer or LexiconScorer()
        self.cache = cache if cache is not None else ScoreCache()
        self.scorer: DocumentScorer = CachedScorer(base, self.cache)

    def score_documents(self, symbol: str, as_of: datetime) -> List[ScoredDocument]:
        cfg = self.config
        start = as_of - timedelta(hours=cfg.lookback_hours)
        docs = [d for d in self.source.fetch(symbol, start, as_of) if start <= d.timestamp <= as_of]
        docs = sorted(docs, key=lambda d: d.timestamp)[-cfg.max_docs:]
        return [ScoredDocument(d, self.scorer.score(d), self.scorer.name) for d in docs]

    def value(self, symbol: str, as_of: datetime) -> Optional[float]:
        """Sentiment in [-1, 1] using only documents public at `as_of`, or
        None when there is nothing to base an opinion on."""
        cfg = self.config
        return aggregate(
            self.score_documents(symbol, as_of), as_of,
            half_life_hours=cfg.half_life_hours,
            filing_half_life_hours=cfg.filing_half_life_hours,
            filing_boost=cfg.filing_boost,
            saturation=cfg.saturation,
        )


class SentimentNewsProvider(NewsDataProvider):
    """Adapts a SentimentSignal to NewsDataProvider. Each Document becomes a
    NewsItem with `.sentiment` pre-filled, so downstream scorers leave it alone."""

    def __init__(self, signal: SentimentSignal):
        self.signal = signal

    def get_news(self, symbol: str, start: datetime, end: datetime) -> List[NewsItem]:
        items = []
        for sd in self.signal.score_documents(symbol, end):
            d = sd.document
            if d.timestamp < start:
                continue
            items.append(NewsItem(symbol=d.symbol, timestamp=d.timestamp, headline=d.title,
                                  source=d.source, sentiment=sd.score, url=d.url))
        return items
