"""
Volatility-regime-gated trend following.

Family: volatility.

Idea
----
Trend following earns its keep in orderly markets and gets hurt when volatility
explodes (crashes, gap risk, violent reversals). This strategy only owns a trend
while realized volatility is calm *relative to its own recent history*, and steps
aside (volatility risk-off) when it spikes - even if price is still above trend.

Rules (long-only, daily bars)
-----------------------------
vol      = std of daily log returns over ``vol_period`` bars
vol_pct  = fraction of the last ``vol_window`` vol readings (current included)
           that are <= the current reading, in (0, 1]
ENTER  close > SMA(``trend_period``) AND vol_pct <= ``entry_pct``
       (default 0.5: realized vol at or below its trailing median)
EXIT   close < SMA(``trend_period``) OR vol_pct > ``exit_pct``
       (``exit_pct`` >= ``entry_pct`` gives hysteresis, avoiding flip-flopping at the cap)
Signal strength is 0.5 + 0.5 * (1 - vol_pct / entry_pct), so it falls as the
vol percentile rises (1.0 in the calmest regime, 0.5 at the entry cap).

Expected behaviour
------------------
Aims for trend returns with shallower drawdowns: it should lose less than plain
trend following when volatility clusters (GARCH storms, bear-market selloffs, jumps).
The price is missed upside - rallies that start with high vol (V-shape recoveries) are
skipped, and in a calm grind the filter is mostly inactive, so there it adds nothing
over the SMA rule. In choppy low-vol sideways markets it still whipsaws around the SMA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class VolRegimeTrendParams(StrategyParams):
    trend_period: int = 50
    vol_period: int = 20
    vol_window: int = 120
    entry_pct: float = 0.5
    exit_pct: float = 0.9

    def validate(self) -> None:
        if self.trend_period < 1:
            raise ValueError("trend_period must be >= 1")
        if self.vol_period < 2:
            raise ValueError("vol_period must be >= 2")
        if self.vol_window < 10:
            raise ValueError("vol_window must be >= 10")
        if not 0.0 < self.entry_pct <= self.exit_pct <= 1.0:
            raise ValueError("require 0 < entry_pct <= exit_pct <= 1")


class VolRegimeTrend(ResearchStrategy):
    """Long-only SMA trend filter gated by realized-volatility percentile. Parameters
    (``VolRegimeTrendParams``): ``trend_period`` SMA length, ``vol_period`` realized-vol
    lookback, ``vol_window`` history used for the percentile, ``entry_pct`` max percentile
    to enter, ``exit_pct`` percentile above which an open trade is closed."""

    name = "vol_regime_trend"
    family = "volatility"
    description = "Hold an SMA uptrend only while realized vol is calm versus its own history; exit on a vol spike."
    PARAMS = VolRegimeTrendParams
    PARAM_GRID = {
        "trend_period": [30, 50, 100],
        "entry_pct": [0.4, 0.5, 0.7],
        "exit_pct": [0.8, 0.9, 0.95],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.trend_period, p.vol_period + p.vol_window) + 1

    @property
    def lookback(self) -> int:
        return self.min_bars + 20

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        sma = ind.rolling_mean(bars.close, p.trend_period)
        vol = ind.realized_vol(bars.close, p.vol_period)
        close, sma_v, vol_v = bars.close[-1], sma[-1], vol[-1]
        hist = vol[-p.vol_window:]
        pct = float(np.mean(hist <= vol_v)) if len(hist) == p.vol_window and np.all(np.isfinite(hist)) else float("nan")
        info = dict(close=close, sma=sma_v, vol=vol_v, vol_pct=pct)
        if not ind.last_finite(close, sma_v, vol_v, pct):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if close < sma_v:
                return Signal.exit(1.0, f"close {close:.2f} < SMA{p.trend_period} {sma_v:.2f}", **info)
            if pct > p.exit_pct:
                return Signal.exit(1.0, f"vol percentile {pct:.2f} > {p.exit_pct} (risk-off)", **info)
            return Signal.hold("trend intact, vol calm enough", **info)

        if close <= sma_v:
            return Signal.hold("below trend", **info)
        if pct > p.entry_pct:
            return Signal.hold(f"vol percentile {pct:.2f} > {p.entry_pct}, regime too volatile", **info)
        strength = 0.5 + 0.5 * clip01(1.0 - pct / p.entry_pct)
        return Signal.enter(strength, f"uptrend with calm vol (pct {pct:.2f})", **info)
