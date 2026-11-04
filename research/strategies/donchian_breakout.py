"""
Donchian channel breakout (the "Turtle" entry/exit rule).

Family: breakout.

Idea
----
Price making a new N-day high signals that the previous range has been
resolved upward and that demand is willing to pay up; trend followers
historically get paid for buying such strength and holding until a
shorter-term low shows the trend has paused.

Rules (long-only, daily bars)
-----------------------------
ENTER  close > highest high of the previous ``entry_period`` bars
       (current bar excluded, so a breakout means a genuinely NEW high)
       [and, if ``vol_filter`` > 0, volume >= vol_filter * 20-bar average]
EXIT   close < lowest low of the previous ``exit_period`` bars

Asymmetric periods (long entry, short exit) are deliberate: it is easy to
get in on a breakout, and the shorter exit channel gives back less of the move.
Signal strength grows with how far the close is above the channel, measured
in ATRs, saturating at 2 ATR.

Expected behaviour
------------------
Profits on persistent trends and range->breakout cycles; bleeds small
whipsaw losses in choppy, mean-reverting markets (many failed breakouts).
Low win rate, high payoff ratio. See the README for measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class DonchianParams(StrategyParams):
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 14
    vol_filter: float = 0.0  # 0 disables; e.g. 1.2 demands volume >= 1.2x its 20-bar mean

    def validate(self) -> None:
        if self.entry_period < 2 or self.exit_period < 2 or self.atr_period < 2:
            raise ValueError("periods must be >= 2")
        if self.exit_period > self.entry_period:
            raise ValueError("exit_period must be <= entry_period")
        if self.vol_filter < 0:
            raise ValueError("vol_filter must be >= 0")


class DonchianBreakout(ResearchStrategy):
    """Long-only Donchian breakout. Parameters (``DonchianParams``): ``entry_period`` bars
    for the breakout high, ``exit_period`` bars for the exit low, ``atr_period`` for sizing the
    signal strength, optional ``vol_filter`` volume-confirmation multiple."""

    name = "donchian_breakout"
    family = "breakout"
    description = "Buy a close above the N-day high, sell a close below the M-day low (Turtle rules)."
    PARAMS = DonchianParams
    PARAM_GRID = {"entry_period": [10, 20, 40, 55], "exit_period": [5, 10, 20]}

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.entry_period, p.exit_period, p.atr_period) + 2

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        up, _ = ind.donchian(bars.high, bars.low, p.entry_period)
        _, dn = ind.donchian(bars.high, bars.low, p.exit_period)
        a = ind.atr(bars.high, bars.low, bars.close, p.atr_period)
        close, upper, lower, atr_v = bars.close[-1], up[-1], dn[-1], a[-1]
        info = dict(close=close, upper=upper, lower=lower, atr=atr_v)
        if not ind.last_finite(close, upper, lower, atr_v):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if close < lower:
                return Signal.exit(1.0, f"close {close:.2f} < {p.exit_period}-bar low {lower:.2f}", **info)
            return Signal.hold("in trade, above exit channel", **info)

        if close > upper:
            if p.vol_filter > 0:
                avg_vol = float(np.mean(bars.volume[-21:-1])) if len(bars) > 21 else 0.0
                if avg_vol <= 0 or bars.volume[-1] < p.vol_filter * avg_vol:
                    return Signal.hold("breakout on weak volume", **info)
            strength = clip01(0.5 + 0.25 * (close - upper) / atr_v)  # 0.5 at the line, 1.0 at +2 ATR
            return Signal.enter(strength, f"close {close:.2f} > {p.entry_period}-bar high {upper:.2f}", **info)
        return Signal.hold("inside channel", **info)
