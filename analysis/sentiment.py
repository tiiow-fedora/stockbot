"""
News sentiment scoring.

Every scorer implements the same tiny interface, so the strategy never
knows (or cares) whether a NewsItem's `sentiment` came from a keyword
lexicon or an LLM. Per the project's design rules: an LLM is never the
trading decision-maker and is never required - `LexiconSentimentScorer`
is the default, requires no API key, and is fully deterministic.
`LLMSentimentScorer` is an entirely optional plug-in that, if enabled,
produces exactly one more component score, weighted the same as every
other signal by strategy/scoring.py.
"""

from __future__ import annotations

import dataclasses
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from core.logging import get_logger
from core.models import NewsItem

logger = get_logger("analysis.sentiment")


class SentimentScorer(ABC):
    @abstractmethod
    def score(self, headline: str) -> Optional[float]:
        """Return a sentiment score in [-1, 1], or None if no signal could
        be extracted. None means "no opinion" - strategy/scoring.py treats
        that differently from a confident 0.0 (it redistributes the
        sentiment weight across the other components rather than diluting
        the composite score with a fabricated neutral reading)."""

    def score_news_items(self, items: List[NewsItem]) -> List[NewsItem]:
        """Fill in `.sentiment` for any item that doesn't already have one
        (a provider that scores its own news, e.g. a future classroom
        platform, is left untouched)."""
        scored: List[NewsItem] = []
        for item in items:
            if item.sentiment is not None:
                scored.append(item)
                continue
            sentiment = self.score(item.headline)
            scored.append(dataclasses.replace(item, sentiment=sentiment) if sentiment is not None else item)
        return scored


# A deliberately simple, finance-flavored keyword lexicon. Not a research-
# grade sentiment model - just enough signal to be one honest input among
# several, which is the whole point of the weighted scoring design.
_POSITIVE_WORDS = {
    "beat", "beats", "beating", "surge", "surges", "surging", "rally", "rallies",
    "rallying", "upgrade", "upgraded", "upgrades", "growth", "profit", "profits",
    "profitable", "record", "strong", "stronger", "bullish", "outperform",
    "outperforms", "gain", "gains", "gaining", "soar", "soars", "soaring",
    "jump", "jumps", "jumping", "positive", "raises", "raised", "raise",
    "expansion", "expands", "breakthrough", "win", "wins", "winning", "boost",
    "boosts", "boosted", "optimistic", "recovery", "recovers", "exceeds",
    "exceeded", "buyback", "dividend", "partnership", "innovation", "milestone",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "missed", "plunge", "plunges", "plunging", "downgrade",
    "downgraded", "downgrades", "loss", "losses", "losing", "weak", "weaker",
    "bearish", "underperform", "underperforms", "decline", "declines",
    "declining", "drop", "drops", "dropping", "fall", "falls", "falling",
    "cut", "cuts", "cutting", "lawsuit", "investigation", "recall", "resign",
    "resigns", "resignation", "fraud", "negative", "warns", "warning",
    "warned", "layoff", "layoffs", "bankruptcy", "default", "crash",
    "crashes", "crashing", "slump", "slumps", "slumping", "probe", "scandal",
    "delay", "delays", "delayed", "shortfall", "concerns", "risk", "risks",
}
_WORD_RE = re.compile(r"[A-Za-z']+")


class LexiconSentimentScorer(SentimentScorer):
    def __init__(self, positive_words: Optional[set] = None, negative_words: Optional[set] = None):
        self._positive = {w.lower() for w in (positive_words or _POSITIVE_WORDS)}
        self._negative = {w.lower() for w in (negative_words or _NEGATIVE_WORDS)}

    def score(self, headline: str) -> Optional[float]:
        if not headline:
            return None
        words = {w.lower() for w in _WORD_RE.findall(headline)}
        pos = len(words & self._positive)
        neg = len(words & self._negative)
        if pos == 0 and neg == 0:
            return None
        return (pos - neg) / (pos + neg)


class LLMSentimentScorer(SentimentScorer):
    """Optional. Never enabled unless news.sentiment_scorer: llm is set in
    config.yaml AND ANTHROPIC_API_KEY is present - the strategy works
    identically with this disabled, which is the default."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - only hit if extra not installed
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic` "
                "or set news.sentiment_scorer: lexicon in config.yaml."
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or set "
                "news.sentiment_scorer: lexicon in config.yaml to skip the LLM entirely."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def score(self, headline: str) -> Optional[float]:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Score the sentiment of this financial news headline for "
                            "the stock it concerns, from -1.00 (very negative) to 1.00 "
                            "(very positive). Reply with ONLY the number, nothing else."
                            f"\n\nHeadline: {headline}"
                        ),
                    }
                ],
            )
            value = float(response.content[0].text.strip())
            return max(-1.0, min(1.0, value))
        except Exception as exc:
            logger.warning("LLM sentiment scoring failed for headline %r: %s", headline, exc)
            return None


def build_sentiment_scorer(name: str) -> Optional[SentimentScorer]:
    if name == "none":
        return None
    if name == "lexicon":
        return LexiconSentimentScorer()
    if name == "llm":
        return LLMSentimentScorer()
    raise ValueError(f"unknown sentiment scorer: {name!r}")
