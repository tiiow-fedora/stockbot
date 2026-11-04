"""Sentiment signal package: pluggable sources, scorers, cache and decay.

Paper-trading research only. Offline by default (fixtures / injected fetchers).
"""

from .cache import CachedScorer, CachedSource, ScoreCache
from .decay import aggregate, decay_weight
from .models import DocKind, Document, ScoredDocument
from .scorers import DocumentScorer, FallbackScorer, HeadlineAdapter, LexiconScorer, LLMScorer, StubLLMScorer
from .signal import SentimentNewsProvider, SentimentSignal, SignalConfig
from .sources import CompositeSource, DocumentSource, FilingsSource, RSSSource, StaticSource

__all__ = [n for n in dir() if not n.startswith("_")]
