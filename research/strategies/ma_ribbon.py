"""
Triple EMA ribbon (fast / mid / slow alignment).

Family: trend.

Idea
----
When a fast, a medium and a slow exponential average are stacked in order and the
slow one is itself rising, the move has persisted across three time scales. That
alignment is a stricter trend test than a single crossover; requiring the fan to
be open (visible gap) filters the tangled-average state typical of chop.

Rules (long-only, daily bars)
-----------------------------
ENTER  EMA(fast) > EMA(mid) > EMA(slow)
       AND linear-regression slope of EMA(slow) over ``slope_period`` bars > 0
       AND (EMA(fast) - EMA(slow)) >= ``min_separation_atr`` * ATR(``atr_period``)
EXIT   EMA(fast) < EMA(mid)
Exit is deliberately looser than entry (only the fast/mid pair must break) so a
trade is not cut by the separation or slope filters that merely gate entries.
``min_separation_atr`` must be finite and >= 0 (NaN would silently disable the filter).
Signal strength is 0.5 + 0.5 * clip(separation / (4 ATR), 0, 1).

Expected behaviour
------------------
Profits in smooth, persistent trends; the slow EMA lag means entries come late and
exits give back part of the move. In sideways markets the separation filter reduces
(but does not eliminate) whipsaw trades; with ``min_separation_atr`` = 0 it
trades noticeably more. No edge in mean-reverting or choppy regimes; sharp V
reversals are caught late on the way back up.
"""

from __future__ import annotations

from dataclasses import dataclass

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class MARibbonParams(StrategyParams):
    fast: int = 10
    mid: int = 30
    slow: int = 60
    slope_period: int = 10
    atr_period: int = 14
    min_separation_atr: float = 0.5

    def validate(self) -> None:
        if self.fast < 1 or self.mid < 1 or self.slow < 1:
            raise ValueError("MA periods must be >= 1")
        if not self.fast < self.mid < self.slow:
            raise ValueError("require fast < mid < slow")
        if self.slope_period < 2 or self.atr_period < 2:
            raise ValueError("slope_period and atr_period must be >= 2")
        if not 0 <= self.min_separation_atr < float("inf"):  # also rejects NaN
            raise ValueError("min_separation_atr must be finite and >= 0")


class MARibbon(ResearchStrategy):
    """Long-only triple-EMA ribbon. Parameters (``MARibbonParams``): ``fast``/``mid``/``slow``
    EMA periods, ``slope_period`` bars for the slow-EMA slope, ``atr_period`` and
    ``min_separation_atr`` (minimum fast-slow gap in ATRs required to enter)."""

    name = "ma_ribbon"
    family = "trend"
    description = "Buy when fast>mid>slow EMAs align with a rising slow EMA and a wide enough fan; sell when fast<mid."
    PARAMS = MARibbonParams
    PARAM_GRID = {
        "fast": [8, 13],
        "mid": [21, 34],
        "slow": [55, 89],
        "min_separation_atr": [0.0, 0.5, 1.0],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.slow + p.slope_period, p.atr_period + 1) + 1

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        fast, mid, slow = (ind.ema(bars.close, n) for n in (p.fast, p.mid, p.slow))
        slope = ind.linreg_slope(slow, p.slope_period)
        a = ind.atr(bars.high, bars.low, bars.close, p.atr_period)
        f, m, s, sl, atr_v = fast[-1], mid[-1], slow[-1], slope[-1], a[-1]
        info = dict(fast=f, mid=m, slow=s, slow_slope=sl, atr=atr_v)
        if not ind.last_finite(f, m, s, sl, atr_v) or atr_v <= 0:
            return Signal.hold("indicators not ready", **info)
        sep = (f - s) / atr_v
        info["separation_atr"] = sep

        if in_position:
            if f < m:
                return Signal.exit(1.0, f"fast EMA {f:.2f} < mid EMA {m:.2f}", **info)
            return Signal.hold("ribbon intact", **info)

        if not (f > m > s):
            return Signal.hold("ribbon not aligned", **info)
        if sl <= 0:
            return Signal.hold("slow EMA not rising", **info)
        if sep < p.min_separation_atr:
            return Signal.hold(f"fan too narrow ({sep:.2f} ATR < {p.min_separation_atr})", **info)
        return Signal.enter(0.5 + 0.5 * clip01(sep / 4.0), f"ribbon aligned, fan {sep:.2f} ATR", **info)
