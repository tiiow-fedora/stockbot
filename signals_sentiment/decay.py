"""Time decay and aggregation of per-document scores into one signal."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Optional

from .models import DocKind, ScoredDocument


def decay_weight(age_seconds: float, half_life_seconds: float) -> float:
    """Exponential decay: weight halves every `half_life_seconds`.
    Negative age (document from the future) -> 0: that would be lookahead."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    if age_seconds < 0:
        return 0.0
    return 0.5 ** (age_seconds / half_life_seconds)


def aggregate(
    scored: Iterable[ScoredDocument],
    as_of: datetime,
    half_life_hours: float = 24.0,
    filing_half_life_hours: float = 24.0 * 7,
    filing_boost: float = 1.5,
    min_weight: float = 1e-3,
    saturation: float = 1.0,
) -> Optional[float]:
    """Weighted decayed mean of scores, in [-1, 1].

    - None scores are ignored; if nothing usable remains -> None (no opinion).
    - Filings decay slower and count `filing_boost`x.
    - The mean is shrunk by total_weight / (total_weight + saturation) so one
      stale headline cannot produce the same signal as a burst of fresh news.
    - Documents published after `as_of` are ignored (no lookahead).
    """
    num = den = 0.0
    for s in scored:
        if s.score is None:
            continue
        age = (as_of - s.document.timestamp).total_seconds()
        is_filing = s.document.kind == DocKind.FILING
        hl = (filing_half_life_hours if is_filing else half_life_hours) * 3600.0
        w = decay_weight(age, hl) * (filing_boost if is_filing else 1.0)
        if w < min_weight:
            continue
        num += w * s.score
        den += w
    if den <= 0:
        return None
    mean = num / den
    confidence = den / (den + saturation) if saturation > 0 else 1.0
    return max(-1.0, min(1.0, mean * confidence))
