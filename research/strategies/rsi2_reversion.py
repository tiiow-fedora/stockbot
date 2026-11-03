"""
Connors-style short-period RSI pullback ("RSI(2)").

Family: mean_reversion.

Idea
----
A 2-4 bar RSI measures how one-sided the last few days were. Within an
established uptrend, an extreme short-term oversold reading is usually a brief
pullback that is bought back within days (Larry Connors / Cesar Alvarez
research). The long-term SMA filter restricts trades to dips of rising
markets, avoiding the broken-trend case where "oversold" can persist.

Rules (long-only, daily bars)
-----------------------------
ENTER  RSI(rsi_period) < oversold
       [and, if ``trend_period`` > 0, close > SMA(close, trend_period)]
EXIT   RSI(rsi_period) > exit_level  OR  close > SMA(close, exit_sma)
Defaults: RSI(2) < 10 above the 200-bar SMA; exit at RSI > 70 or close above
the 5-bar SMA. Signal strength is 0.5 at the oversold threshold rising to 1.0
as RSI approaches 0 (the more washed-out, the more conviction).
There is no price stop inside the strategy (stateless); the engine's risk
overlay is the only loss limit.

Expected behaviour
------------------
Wins on mean-reverting and cyclical series and on choppy uptrends: frequent
trades, high win rate, small average gains. It LOSES when the dip is the start
of a persistent decline (the trend filter delays but does not prevent this) and
when returns are autocorrelated (momentum): a down day then predicts another.
In a steady downtrend the filter keeps it mostly flat, which is the point.
With the 200-bar filter it needs 200+ bars of history before its first trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class Rsi2ReversionParams(StrategyParams):
    rsi_period: int = 2
    oversold: float = 10.0
    exit_level: float = 70.0
    trend_period: int = 200  # 0 disables the uptrend filter
    exit_sma: int = 5

    def validate(self) -> None:
        if self.rsi_period < 2:
            raise ValueError("rsi_period must be >= 2")
        if not 0 < self.oversold < self.exit_level < 100:
            raise ValueError("need 0 < oversold < exit_level < 100")
        if self.trend_period < 0:
            raise ValueError("trend_period must be >= 0 (0 disables)")
        if self.exit_sma < 1:
            raise ValueError("exit_sma must be >= 1")


class Rsi2Reversion(ResearchStrategy):
    """Long-only Connors RSI pullback. Parameters (``Rsi2ReversionParams``): ``rsi_period`` (2-4
    typical), ``oversold`` entry threshold, ``exit_level`` RSI exit, ``trend_period`` uptrend SMA
    filter (0 = off) and ``exit_sma`` for the alternative close-above-SMA exit."""

    name = "rsi2_reversion"
    family = "mean_reversion"
    description = "Buy a deeply oversold short-period RSI inside an uptrend, sell on RSI rebound or close above a short SMA."
    PARAMS = Rsi2ReversionParams
    PARAM_GRID = {
        "rsi_period": [2, 3, 4],
        "oversold": [5.0, 10.0, 20.0],
        "exit_level": [60.0, 70.0],
        "trend_period": [0, 100, 200],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.rsi_period + 1, p.trend_period, p.exit_sma) + 2

    @property
    def lookback(self) -> int:
        # Wilder smoothing of a 2-4 bar RSI forgets its seed within ~20 bars.
        return self.min_bars + 20

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        rsi_v = ind.rsi(bars.close, p.rsi_period)[-1]
        close = bars.close[-1]
        exit_ma = float(np.mean(bars.close[-p.exit_sma:]))
        info = dict(close=close, rsi=rsi_v, exit_sma=exit_ma)
        if not ind.last_finite(close, rsi_v, exit_ma):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if rsi_v > p.exit_level:
                return Signal.exit(1.0, f"RSI {rsi_v:.1f} > {p.exit_level}", **info)
            if close > exit_ma:
                return Signal.exit(1.0, f"close {close:.2f} > {p.exit_sma}-bar SMA {exit_ma:.2f}", **info)
            return Signal.hold("in trade, awaiting rebound", **info)

        if rsi_v >= p.oversold:
            return Signal.hold("RSI not oversold", **info)
        if p.trend_period > 0:
            trend = float(np.mean(bars.close[-p.trend_period:]))
            info["trend_sma"] = trend
            if close <= trend:
                return Signal.hold("oversold but below long-term trend, skipped", **info)
        strength = clip01(0.5 + 0.5 * (p.oversold - rsi_v) / p.oversold)
        return Signal.enter(strength, f"RSI {rsi_v:.1f} < {p.oversold}", **info)
