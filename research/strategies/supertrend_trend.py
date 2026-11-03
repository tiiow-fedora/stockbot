"""
SuperTrend trend follower (ATR trailing-band direction flips).

Family: trend.

Idea
----
SuperTrend places a trailing stop line ``mult`` ATRs from the bar midpoint. While
price stays on one side of the line the trend is considered intact; a close through
the line flips the direction. Buying the flip to +1 and selling the flip to -1
captures sustained moves with a volatility-scaled stop that only ever tightens.

Rules (long-only, daily bars)
-----------------------------
The strategy is stateless, so "a flip" is reconstructed from the history: ``age`` is
the number of consecutive trailing bars whose SuperTrend direction is +1
(``age == 1`` on the flip bar itself).
ENTER  (flat) direction == +1 AND close > line AND age <= ``max_age``
       i.e. we buy the flip and the first ``max_age - 1`` bars after it (this
       tolerates a missed fill or a stop-out re-entry), but we do NOT chase an
       old uptrend that is already more than ``max_age`` bars in progress.
EXIT   (long) direction == -1 (close fell through the line)
Signal strength is 0.5 + 0.5 * clip(distance / (2 * mult ATR), 0, 1) where
distance = close - line, so conviction grows with room above the stop line.

Expected behaviour
------------------
Wins on persistent trends and big one-way moves (low win rate, big winners);
the volatility-scaled stop gives back a chunk of each move. Loses to repeated
whipsaw flips in sideways or mean-reverting markets, and gaps through the line
cost more than the line implies. Smaller ``mult`` trades more and whipsaws more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class SuperTrendParams(StrategyParams):
    period: int = 10
    mult: float = 3.0
    max_age: int = 10  # do not enter an uptrend whose flip is more than this many bars old

    def validate(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if not 0 < self.mult < float("inf"):  # also rejects NaN
            raise ValueError("mult must be finite and > 0")
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")


def _bars_in_uptrend(direction: np.ndarray) -> int:
    """Consecutive trailing bars with direction == +1 (0 if the last bar is not +1)."""
    not_up = np.flatnonzero(direction != 1.0)  # NaN != 1 too, so warm-up counts as "not up"
    return len(direction) - 1 - int(not_up[-1]) if len(not_up) else len(direction)


class SuperTrendTrend(ResearchStrategy):
    """Long-only SuperTrend direction-flip follower. Parameters (``SuperTrendParams``):
    ``period`` ATR length, ``mult`` ATR multiple of the band, ``max_age`` newest-flip
    window in bars within which entries are allowed."""

    name = "supertrend_trend"
    family = "trend"
    description = "Buy when SuperTrend flips up (not chasing stale uptrends), sell when it flips down."
    PARAMS = SuperTrendParams
    PARAM_GRID = {"period": [7, 10, 14, 21], "mult": [2.0, 3.0, 4.0], "max_age": [3, 10]}

    @property
    def min_bars(self) -> int:
        return self.params.period + 2

    @property
    def lookback(self) -> int:
        return max(4 * self.min_bars, 60) + self.params.max_age

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        line, direction = ind.supertrend(bars.high, bars.low, bars.close, p.period, p.mult)
        a = ind.atr(bars.high, bars.low, bars.close, p.period)
        close, line_v, d, atr_v = bars.close[-1], line[-1], direction[-1], a[-1]
        info = dict(close=close, line=line_v, direction=d, atr=atr_v)
        if not ind.last_finite(close, line_v, d, atr_v) or atr_v <= 0:
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if d < 0:
                return Signal.exit(1.0, f"SuperTrend flipped down (line {line_v:.2f})", **info)
            return Signal.hold("uptrend intact", **info)

        age = _bars_in_uptrend(direction)
        info["age"] = float(age)
        if d > 0 and close > line_v:
            if age > p.max_age:
                return Signal.hold(f"uptrend is {age} bars old (> {p.max_age}), not chasing", **info)
            strength = 0.5 + 0.5 * clip01((close - line_v) / (2.0 * p.mult * atr_v))
            return Signal.enter(strength, f"SuperTrend up for {age} bar(s), close {close:.2f} > line {line_v:.2f}", **info)
        return Signal.hold("downtrend", **info)
