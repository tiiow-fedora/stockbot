"""
Nearness-to-high momentum (George & Hwang, "The 52-week high and momentum investing").

Family: momentum.

Idea
----
Anchoring: investors treat the trailing high as a reference point and are
reluctant to bid a stock through it, so stocks trading close to their high
tend to be under-reacted to good news and keep drifting up, while stocks far
below their high continue to lag. The signal is therefore the ratio
``close / trailing high``, not past return alone.

Rules (long-only, daily bars; H = highest high of the last ``window`` bars, current included)
---------------------------------------------------------------------------------------------
ENTER  close >= (1 - ``proximity``) * H
       AND H was set at most ``recent`` bars before the decision bar (the high is
       fresh, not a stale level the stock has been sitting under)
       AND close / close[``mom_period`` bars ago] - 1 > 0
EXIT   close < (1 - ``exit_proximity``) * H   (proximity has decayed)
Signal strength is the closeness to the high: 1.0 at the high, 0.5 at the
edge of the ``proximity`` band.

Expected behaviour
------------------
Profits in persistent uptrends and momentum / trend-persistent markets, where
leaders keep making new highs; the wide exit band keeps it in through normal
pullbacks. In sideways markets the stock oscillates around its high and the
strategy will buy near the top of the range and sell 10% lower: slow bleed.
It enters late (only near highs) and exits late (only after a large drawdown
from the high), so it gives back a sizeable chunk at trend ends, and it
needs max(``window``, ``mom_period`` + 1) + 1 bars of history before the first trade. It
does not trade downtrends: the close is never near a high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class HighProximityParams(StrategyParams):
    window: int = 120
    proximity: float = 0.03
    recent: int = 20
    mom_period: int = 60
    exit_proximity: float = 0.10

    def validate(self) -> None:
        if self.window < 2 or self.recent < 1 or self.mom_period < 1:
            raise ValueError("window must be >= 2, recent and mom_period >= 1")
        if self.recent > self.window:
            raise ValueError("recent must be <= window")
        if not 0 < self.proximity < 1 or not 0 < self.exit_proximity < 1:
            raise ValueError("proximity and exit_proximity must be in (0, 1)")
        if self.exit_proximity <= self.proximity:
            raise ValueError("exit_proximity must be > proximity")


class HighProximityMomentum(ResearchStrategy):
    """Long-only 52-week-high style momentum. Parameters (``HighProximityParams``): ``window``
    bars for the trailing high, ``proximity`` the entry band below it, ``recent`` how fresh
    the high must be, ``mom_period`` the return window that must be positive,
    ``exit_proximity`` how far below the high triggers the exit."""

    name = "high_proximity_momentum"
    family = "momentum"
    description = "Buy stocks trading within a few percent of a recent N-bar high with positive momentum; sell when they sag well below it."
    PARAMS = HighProximityParams
    PARAM_GRID = {
        "window": [60, 120, 252],
        "proximity": [0.02, 0.03, 0.05],
        "exit_proximity": [0.08, 0.10, 0.15],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.window, p.mom_period + 1) + 1

    @property
    def lookback(self) -> int:
        return self.min_bars  # no recursive indicators: exactly the needed history

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        close = bars.close[-1]
        if len(bars) < max(p.window, p.mom_period + 1):
            return Signal.hold("indicators not ready", close=close)
        win = bars.high[-p.window :]
        past = bars.close[-1 - p.mom_period]
        high = float(np.max(win))
        info = dict(close=close, high=high)
        if not ind.last_finite(close, past, high) or high <= 0 or past <= 0:
            return Signal.hold("indicators not ready", **info)
        dist = 1.0 - close / high
        mom = close / past - 1.0
        age = len(win) - 1 - int(np.flatnonzero(win == high)[-1])  # bars since the (latest) high
        info.update(distance=dist, momentum=mom, high_age=age)

        if in_position:
            if dist > p.exit_proximity:
                return Signal.exit(1.0, f"close {dist:.1%} below {p.window}-bar high (> {p.exit_proximity:.0%})", **info)
            return Signal.hold("in trade, still near the high", **info)

        if dist > p.proximity:
            return Signal.hold("not near the high", **info)
        if age > p.recent:
            return Signal.hold("high is stale", **info)
        if mom <= 0:
            return Signal.hold("momentum not positive", **info)
        strength = clip01(1.0 - 0.5 * max(dist, 0.0) / p.proximity)
        return Signal.enter(strength, f"close {dist:.1%} below {p.window}-bar high set {age} bars ago", **info)
