"""
Volatility-squeeze breakout (Bollinger Bands inside Keltner Channels).

Family: breakout.

Idea
----
Volatility is cyclical: unusually quiet periods tend to be followed by
unusually large moves. A "squeeze" is flagged when the Bollinger Bands
(standard-deviation based) contract inside the Keltner Channels (ATR based),
i.e. realised dispersion of closes is small relative to the typical bar range.
The trade is not the compression itself but its RELEASE with an upward
bias: buy when the bands expand back out and price is already above the
middle band and rising.

Rules (long-only, daily bars)
-----------------------------
SQUEEZE  bar t is "on" when BB(``bb_period``, ``bb_std``) upper < KC upper
         and BB lower > KC lower, where KC = EMA(``kc_period``) +/-
         ``kc_mult`` * ATR(``kc_atr_period``).
ENTER    the squeeze is OFF on the current bar, was ON on at least one of the
         previous ``squeeze_lookback`` bars (stateless memory of the squeeze),
         AND close > BB middle (SMA) AND the OLS slope of the last
         ``mom_period`` closes is > 0
         [and, if ``vol_filter`` > 0, volume >= vol_filter * 20-bar average]
EXIT     close < BB middle - ``exit_atr`` * ATR, OR the OLS slope of the last
         ``mom_period`` closes turns negative.
Signal strength grows with the momentum slope, measured in ATRs gained over
the momentum window (0.5 at zero, 1.0 at 3 ATR).

Expected behaviour
------------------
Wins in range -> expansion cycles (compression followed by a directional
burst), which is exactly what ``synth.range_then_breakout`` generates. In
persistent trends squeezes are rare, so it trades little and mostly misses the
trend. In choppy / mean-reverting markets a release is as likely to be a fake
move as a real one, and the momentum-flip exit adds whipsaw losses. Because a
release is only a volatility event, roughly half of releases go DOWN; the
long-only filter simply skips those, it does not predict direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class SqueezeParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_period: int = 10
    kc_mult: float = 1.5
    squeeze_lookback: int = 5
    mom_period: int = 12
    exit_atr: float = 1.0
    vol_filter: float = 0.0  # 0 disables; e.g. 1.2 demands volume >= 1.2x its 20-bar mean

    def validate(self) -> None:
        if min(self.bb_period, self.kc_period, self.kc_atr_period, self.mom_period) < 2:
            raise ValueError("periods must be >= 2")
        if self.squeeze_lookback < 1:
            raise ValueError("squeeze_lookback must be >= 1")
        if self.bb_std <= 0 or self.kc_mult <= 0:
            raise ValueError("bb_std and kc_mult must be > 0")
        if self.exit_atr < 0 or self.vol_filter < 0:
            raise ValueError("exit_atr and vol_filter must be >= 0")


class SqueezeBreakout(ResearchStrategy):
    """Long-only Bollinger-in-Keltner squeeze release. Parameters (``SqueezeParams``):
    Bollinger (``bb_period``, ``bb_std``) and Keltner (``kc_period``, ``kc_atr_period``,
    ``kc_mult``) settings define the squeeze, ``squeeze_lookback`` is how many bars back a
    squeeze still counts as "just released", ``mom_period`` the slope window, ``exit_atr``
    the ATR buffer under the middle band for the exit, optional ``vol_filter``."""

    name = "squeeze_breakout"
    family = "breakout"
    description = "Buy the release of a Bollinger-inside-Keltner volatility squeeze when price is above the mid band and rising."
    PARAMS = SqueezeParams
    PARAM_GRID = {"kc_mult": [1.0, 1.5, 2.0], "squeeze_lookback": [3, 5, 10], "mom_period": [8, 12, 20]}

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.bb_period, p.kc_period, p.kc_atr_period, p.mom_period) + p.squeeze_lookback + 2

    @property
    def lookback(self) -> int:
        p = self.params
        return 3 * max(p.kc_period, p.kc_atr_period) + self.min_bars

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        mid, bb_up, bb_lo = ind.bollinger(bars.close, p.bb_period, p.bb_std)
        _, kc_up, kc_lo = ind.keltner(bars.high, bars.low, bars.close, p.kc_period, p.kc_atr_period, p.kc_mult)
        a = ind.atr(bars.high, bars.low, bars.close, p.kc_atr_period)
        slope = ind.linreg_slope(bars.close, p.mom_period)
        close, mid_v, atr_v, slope_v = bars.close[-1], mid[-1], a[-1], slope[-1]
        info = dict(close=close, mid=mid_v, atr=atr_v, slope=slope_v)
        if not ind.last_finite(close, mid_v, atr_v, slope_v) or atr_v <= 0:
            return Signal.hold("indicators not ready", **info)

        if in_position:
            floor = mid_v - p.exit_atr * atr_v
            if close < floor:
                return Signal.exit(1.0, f"close {close:.2f} < mid band - {p.exit_atr} ATR ({floor:.2f})", **info)
            if slope_v < 0:
                return Signal.exit(1.0, f"momentum slope {slope_v:.3f} < 0", **info)
            return Signal.hold("in trade, above mid band with positive momentum", **info)

        with np.errstate(invalid="ignore"):
            squeeze_on = (bb_up < kc_up) & (bb_lo > kc_lo)  # NaN comparisons are False -> warm-up is "off"
        was_squeezed = bool(squeeze_on[-p.squeeze_lookback : -1].any())
        info["squeeze_now"] = float(squeeze_on[-1])
        if squeeze_on[-1] or not was_squeezed:
            return Signal.hold("no fresh squeeze release", **info)
        if close <= mid_v or slope_v <= 0:
            return Signal.hold("squeeze released without upward momentum", **info)
        if p.vol_filter > 0:
            avg_vol = float(np.mean(bars.volume[-21:-1])) if len(bars) > 21 else 0.0
            if avg_vol <= 0 or bars.volume[-1] < p.vol_filter * avg_vol:
                return Signal.hold("release on weak volume", **info)
        strength = clip01(0.5 + 0.5 * slope_v * p.mom_period / (3.0 * atr_v))
        return Signal.enter(strength, f"squeeze released, close {close:.2f} > mid {mid_v:.2f}, slope {slope_v:.3f}", **info)
