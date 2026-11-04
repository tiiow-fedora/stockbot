"""
EMA trend filter gated by ADX trend strength.

Family: trend.

Idea
----
A fast EMA above a slow EMA says price is trending up; but moving-average
systems lose money in trendless markets, where the averages braid together
and keep crossing. Wilder's ADX measures how strong a trend is irrespective of
direction (low = range-bound, high = trending), so it is used as a gate: only
take the EMA's long side when the market is actually trending, and leave when
the trend strength collapses even before the averages cross.

Rules (long-only, daily bars)
-----------------------------
ENTER  EMA(fast) > EMA(slow)  AND  ADX(adx_period) > ``adx_enter``
EXIT   EMA(fast) < EMA(slow)  OR   ADX < ``adx_exit``

``adx_exit`` < ``adx_enter`` is deliberate hysteresis: ADX is noisy around any
single level, and one threshold would churn in and out. Entry is state-based
(fast above slow) rather than "the exact bar of the cross" because ADX lags
price - it typically clears its threshold some bars AFTER the cross - and
because a stateless strategy must be able to re-enter after a stop-out.
Signal strength = ADX / 50 clipped to [0, 1]: stronger trends get larger size.

Expected behaviour
------------------
Wins on persistent, directional trends (up trends; it stays flat in down
trends because fast < slow). The ADX gate removes many of the false crosses
of a plain EMA system in sideways markets, but not all: ADX is itself slow,
so the strategy enters a trend late and gives back part of it at the end,
and a sharp V-shaped reversal hits it before ADX falls. ADX can also spike
on a one-way crash, so a downward move followed by a quick rebound is
a typical loss. No edge on a random walk.
"""

from __future__ import annotations

from dataclasses import dataclass

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class EmaAdxTrendParams(StrategyParams):
    fast: int = 20
    slow: int = 50
    adx_period: int = 14
    adx_enter: float = 25.0
    adx_exit: float = 18.0

    def validate(self) -> None:
        if min(self.fast, self.slow, self.adx_period) <= 0:
            raise ValueError("periods must be > 0")
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        if not 0 <= self.adx_exit < self.adx_enter <= 100:
            raise ValueError("need 0 <= adx_exit < adx_enter <= 100")


class EmaAdxTrend(ResearchStrategy):
    """Long-only EMA fast/slow trend filter with an ADX gate and hysteresis. Parameters
    (``EmaAdxTrendParams``): ``fast``/``slow`` EMA periods, ``adx_period``, ``adx_enter`` threshold to
    open, ``adx_exit`` lower threshold below which an open trade is closed."""

    name = "ema_adx_trend"
    family = "trend"
    description = "Hold while fast EMA > slow EMA, entering only when ADX shows a strong trend and leaving if ADX collapses."
    PARAMS = EmaAdxTrendParams
    PARAM_GRID = {
        "fast": [10, 20, 30],
        "slow": [50, 100],
        "adx_enter": [20.0, 25.0, 30.0],
        "adx_exit": [15.0, 18.0],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.slow, 2 * p.adx_period + 1) + 2

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        f = ind.ema(bars.close, p.fast)[-1]
        s = ind.ema(bars.close, p.slow)[-1]
        adx_v = ind.adx(bars.high, bars.low, bars.close, p.adx_period)[0][-1]
        info = dict(ema_fast=f, ema_slow=s, adx=adx_v)
        if not ind.last_finite(f, s, adx_v):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if f < s:
                return Signal.exit(1.0, f"EMA{p.fast} {f:.2f} < EMA{p.slow} {s:.2f}", **info)
            if adx_v < p.adx_exit:
                return Signal.exit(1.0, f"ADX {adx_v:.1f} < exit threshold {p.adx_exit:.1f}", **info)
            return Signal.hold("in trade, trend intact", **info)

        if f > s and adx_v > p.adx_enter:
            return Signal.enter(clip01(adx_v / 50.0), f"EMA{p.fast} > EMA{p.slow} with ADX {adx_v:.1f} > {p.adx_enter:.1f}", **info)
        return Signal.hold("no strong uptrend", **info)
