"""
Configurable weighted scoring engine.

Component scores (technical, momentum, volume, risk, sentiment) are each
expected in the range [-1, 1], where positive means "bullish" and
negative means "bearish" (risk is the exception - see below). They are
combined into a single composite score using weights from config.yaml,
and the composite is mapped to BUY / HOLD / SELL using configurable
thresholds.

This is intentionally the ONLY place decision thresholds live, so
changing the strategy's aggressiveness never requires touching indicator
math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import StrategyWeights
from core.models import Action, ScoreBreakdown


def clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class ScoringEngine:
    weights: StrategyWeights
    buy_threshold: float = 0.25
    sell_threshold: float = -0.25

    def score(
        self,
        technical: float,
        momentum: float,
        volume: float,
        risk: float,
        sentiment: Optional[float] = None,
    ) -> ScoreBreakdown:
        """`risk` should already be <= 0 (a volatility/drawdown penalty);
        it only ever pulls the composite score toward SELL/HOLD, never
        pushes it toward BUY. If `sentiment` is None (no news configured
        or no news for this symbol/day), its weight is redistributed
        across the remaining components so it never silently zeroes out
        part of the score."""

        active_weights = {
            "technical": self.weights.technical,
            "momentum": self.weights.momentum,
            "volume": self.weights.volume,
            "risk": self.weights.risk,
        }
        if sentiment is not None:
            active_weights["sentiment"] = self.weights.sentiment

        total_weight = sum(active_weights.values()) or 1.0
        normalized = {k: v / total_weight for k, v in active_weights.items()}

        composite = (
            normalized["technical"] * technical
            + normalized["momentum"] * momentum
            + normalized["volume"] * volume
            + normalized["risk"] * risk
        )
        if sentiment is not None:
            composite += normalized["sentiment"] * sentiment

        composite = clip(composite)

        return ScoreBreakdown(
            technical=technical,
            momentum=momentum,
            volume=volume,
            risk=risk,
            sentiment=sentiment,
            composite=composite,
            weights=normalized,
        )

    def decide(self, composite: float) -> Action:
        if composite >= self.buy_threshold:
            return Action.BUY
        if composite <= self.sell_threshold:
            return Action.SELL
        return Action.HOLD
