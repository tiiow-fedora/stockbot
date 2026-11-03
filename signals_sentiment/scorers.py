"""Document scorers.

- `LexiconScorer`: deterministic, offline, default. Reuses the project's
  finance word lists and adds negation / intensifier handling plus a small
  filing-specific vocabulary.
- `LLMScorer`: optional interface. `StubLLMScorer` is the only
  implementation shipped here; it makes no network calls. A real adapter
  would subclass `LLMScorer` and implement `_complete`.

Scorers return a float in [-1, 1] or None ("no opinion").
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Optional

from analysis.sentiment import SentimentScorer, _NEGATIVE_WORDS, _POSITIVE_WORDS

from .models import DocKind, Document

_WORD_RE = re.compile(r"[A-Za-z']+")

_FILING_NEG = {"restatement", "impairment", "material", "weakness", "going", "concern", "subpoena",
               "delisting", "termination", "covenant", "breach", "writedown", "write-down", "deficiency"}
_FILING_POS = {"authorization", "repurchase", "accretive", "reaffirms", "reaffirmed", "approval", "approved"}
_NEGATORS = {"not", "no", "never", "without", "isn't", "wasn't", "won't", "doesn't", "didn't", "fails", "failed", "n't"}
_INTENSIFIERS = {"sharply": 1.5, "significantly": 1.5, "massively": 1.8, "slightly": 0.5, "marginally": 0.5, "modestly": 0.6}
_NEGATION_WINDOW = 2


class DocumentScorer(ABC):
    name = "scorer"

    @abstractmethod
    def score(self, doc: Document) -> Optional[float]:
        ...


class LexiconScorer(DocumentScorer):
    name = "lexicon"

    def __init__(self, title_weight: float = 2.0, saturation: float = 2.0):
        # Headline words are worth more than body words; `saturation` controls
        # how fast the tanh squash approaches +-1 as evidence accumulates.
        self.title_weight = title_weight
        self.saturation = saturation

    def _tally(self, text: str, weight: float, kind: DocKind) -> float:
        pos = set(_POSITIVE_WORDS) | (_FILING_POS if kind == DocKind.FILING else set())
        neg = set(_NEGATIVE_WORDS) | (_FILING_NEG if kind == DocKind.FILING else set())
        words = [w.lower() for w in _WORD_RE.findall(text)]
        total = 0.0
        for i, w in enumerate(words):
            polarity = 1.0 if w in pos else -1.0 if w in neg else 0.0
            if not polarity:
                continue
            window = words[max(0, i - _NEGATION_WINDOW):i]
            if any(t in _NEGATORS for t in window):
                polarity = -polarity * 0.75  # "not strong" is bad, but milder than "weak"
            for t in window:
                polarity *= _INTENSIFIERS.get(t, 1.0)
            total += polarity * weight
        return total

    def score(self, doc: Document) -> Optional[float]:
        raw = self._tally(doc.title, self.title_weight, doc.kind) + self._tally(doc.body, 1.0, doc.kind)
        if raw == 0.0:
            return None
        return max(-1.0, min(1.0, math.tanh(raw / self.saturation)))


class LLMScorer(DocumentScorer):
    """Optional LLM-backed scorer interface. Subclasses implement `_complete`
    (prompt -> raw text). Parsing/clamping/failure handling is shared so every
    backend behaves identically. An LLM is never required and never decides
    trades; it only supplies one score."""

    name = "llm"

    @abstractmethod
    def _complete(self, prompt: str) -> str:
        ...

    @staticmethod
    def build_prompt(doc: Document) -> str:
        return (
            "Rate the sentiment of this text for the stock "
            f"{doc.symbol} from -1.00 (very negative) to 1.00 (very positive). "
            "Reply with ONLY the number.\n\n"
            f"Type: {doc.kind.value}\nText: {doc.text[:2000]}"
        )

    @staticmethod
    def parse_reply(reply: str) -> Optional[float]:
        m = re.search(r"-?\d+(?:\.\d+)?", reply or "")
        if not m:
            return None
        value = float(m.group())
        return max(-1.0, min(1.0, value)) if math.isfinite(value) else None

    def score(self, doc: Document) -> Optional[float]:
        try:
            return self.parse_reply(self._complete(self.build_prompt(doc)))
        except Exception:  # noqa: BLE001 - a flaky backend must not break the signal
            return None


class StubLLMScorer(LLMScorer):
    """Deterministic, offline stand-in. Replies are scripted (by substring)
    else neutral "0.0"; records every prompt for assertions."""

    name = "llm-stub"

    def __init__(self, scripted: Optional[dict] = None, fail: bool = False):
        self._scripted = scripted or {}
        self._fail = fail
        self.prompts: list = []

    def _complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._fail:
            raise RuntimeError("stub backend failure")
        for needle, reply in self._scripted.items():
            if needle in prompt:
                return reply
        return "0.0"


class FallbackScorer(DocumentScorer):
    """Try scorers in order; first non-None wins. e.g. LLM -> lexicon."""

    name = "fallback"

    def __init__(self, *scorers: DocumentScorer):
        if not scorers:
            raise ValueError("need at least one scorer")
        self._scorers = scorers

    def score(self, doc: Document) -> Optional[float]:
        for s in self._scorers:
            v = s.score(doc)
            if v is not None:
                return v
        return None


class HeadlineAdapter(SentimentScorer):
    """Exposes a DocumentScorer through the existing analysis.sentiment
    SentimentScorer interface (headline-only), so it can be used anywhere the
    current code expects one."""

    def __init__(self, inner: DocumentScorer, symbol: str = "?"):
        from datetime import datetime, timezone

        self._inner, self._symbol = inner, symbol
        self._now = datetime.now(timezone.utc)

    def score(self, headline: str) -> Optional[float]:
        return self._inner.score(Document(symbol=self._symbol, timestamp=self._now, title=headline))
