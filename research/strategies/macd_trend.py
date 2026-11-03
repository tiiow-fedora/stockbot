"""
MACD trend-following with a zero-line regime filter.

Family: trend.

Idea
----
MACD (difference of a fast and a slow EMA) measures trend direction; its
signal line (an EMA of MACD) smooths it, so the histogram (MACD - signal) is
a momentum-of-the-trend gauge. A bullish cross of MACD over its signal says
the trend's acceleration has turned up. Taken in isolation these crosses fire
constantly in sideways markets, so a regime filter is added.

Rules (long-only, daily bars)
-----------------------------
line, sig, hist = MACD(fast, slow, signal);  hist = line - sig
ENTER  a bullish cross (hist <= 0 on the previous bar, hist > 0 on the next)
       happened within the last ``cross_window`` bars AND hist > 0 now
       [and, if ``zero_filter``, line > 0 : only buy crosses that occur inside an
        established uptrend (fast EMA above slow EMA), i.e. pullback resumptions;
        counter-trend crosses from deep negative territory are skipped]
       [and, if ``slope_confirm``, hist is still rising (hist > previous hist)]
EXIT   hist < 0 (bearish cross: MACD below its signal line)

``cross_window`` > 1 lets a missed or stopped-out signal be re-taken for a few
bars (the strategy is stateless and cannot remember a cross it did not act on).
Signal strength = 0.5 + 0.5 * tanh(hist / (0.1 * ATR)): histogram size relative
to the typical daily range, so a decisive cross gets the full position.

Expected behaviour
------------------
Profits in sustained trends of any pace slower than the slow EMA; the zero
filter cuts losing trades in downtrends and bear-market bounces. Loses
repeatedly (many small whipsaws, a low win rate) in sideways or mean-reverting
markets where MACD oscillates around its signal line, and always exits late:
a bearish cross needs a meaningful part of the move to be given back. No edge
on a random walk beyond what its time in market gives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class MacdTrendParams(StrategyParams):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    cross_window: int = 3  # a bullish cross this recent still counts as an entry
    atr_period: int = 14
    zero_filter: bool = True
    slope_confirm: bool = True

    def validate(self) -> None:
        if min(self.fast, self.slow, self.signal, self.cross_window, self.atr_period) <= 0:
            raise ValueError("periods and cross_window must be > 0")
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")


class MacdTrend(ResearchStrategy):
    """Long-only MACD signal-line crossover. Parameters (``MacdTrendParams``): ``fast``/``slow``/``signal``
    EMA periods, ``cross_window`` bars a bullish cross stays actionable, ``atr_period`` for strength,
    ``zero_filter`` (require MACD line > 0 to enter) and ``slope_confirm`` (require a rising histogram)."""

    name = "macd_trend"
    family = "trend"
    description = "Buy a bullish MACD/signal cross (optionally only above the zero line), sell the bearish cross."
    PARAMS = MacdTrendParams
    PARAM_GRID = {
        "fast": [8, 12],
        "slow": [21, 26, 35],
        "signal": [5, 9],
        "zero_filter": [True, False],
        "slope_confirm": [True, False],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.slow + p.signal + p.cross_window + 1, p.atr_period + 2)

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        line, sig, hist = ind.macd(bars.close, p.fast, p.slow, p.signal)
        a = ind.atr(bars.high, bars.low, bars.close, p.atr_period)
        h, ln, atr_v = hist[-1], line[-1], a[-1]
        info = dict(macd=ln, signal=sig[-1], hist=h, atr=atr_v)
        recent = hist[-(p.cross_window + 1):]
        eps = 1e-7 * bars.close[-1]  # float-noise dead band: a perfectly smooth ramp has hist ~ 1e-15, not a cross
        if not ind.last_finite(h, ln, atr_v) or len(recent) < p.cross_window + 1 or not np.all(np.isfinite(recent)):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if h < -eps:
                return Signal.exit(1.0, f"bearish cross: hist {h:.4f} < 0", **info)
            return Signal.hold("in trade, MACD above signal", **info)

        crossed = bool(np.any((recent[1:] > eps) & (recent[:-1] <= eps)))
        if not (crossed and h > eps):
            return Signal.hold("no fresh bullish cross", **info)
        if p.zero_filter and ln <= eps:
            return Signal.hold("bullish cross below zero line (filtered)", **info)
        if p.slope_confirm and not h > hist[-2]:
            return Signal.hold("histogram not rising", **info)
        strength = clip01(0.5 + 0.5 * float(np.tanh(h / (0.1 * atr_v)))) if atr_v > 0 else 0.5
        return Signal.enter(strength, f"bullish cross: hist {h:.4f} > 0, macd {ln:.4f}", **info)
