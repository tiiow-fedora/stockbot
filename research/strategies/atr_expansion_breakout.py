"""
ATR range-expansion breakout with a Chandelier exit (Crabel / Larry Williams style).

Family: volatility.

Idea
----
A bar that is much larger than recent bars, closes decisively in the
direction of the move and finishes near its extreme is a sign of
initiative buying (a "range expansion" or "thrust" bar). Such days often
start short-lived trends because the move was not faded into the close.
Volatility, not a price level, defines the breakout. The trade is then
managed with a Chandelier stop that trails the highest high by a multiple of
ATR, so the exit adapts to how noisy the stock currently is.

Rules (long-only, daily bars; ATR_prev = ATR(``atr_period``) as of the PREVIOUS bar)
-----------------------------------------------------------------------------------
ENTER  close > previous close + ``k`` * ATR_prev
       AND true range of the bar > ``expansion`` * ATR_prev
       AND (close - low) / (high - low) >= 1 - ``close_pos`` (close in the top
       ``close_pos`` fraction of the bar's range)
       AND close >= the Chandelier level below (otherwise the very next bar
       would sell it: stateless entry/exit must not contradict each other)
EXIT   close < max(high over the last ``chand_period`` bars, current included)
              - ``chand_mult`` * ATR(``atr_period``)
       (recomputed every bar from visible data, so no stored stop is needed;
       once price has made new highs the level only ever moves up).
Signal strength grows with the size of the thrust in ATRs (0.5 when it just
qualifies, 1.0 when it exceeds the threshold by 2 ATR).

Expected behaviour
------------------
Does best in volatility-cluster / impulse markets where one big up bar is
followed by continuation (trend starts, range->breakout cycles). It is
selective, so it trades rarely. In mean-reverting or jumpy markets the thrust
bar is often a one-off spike that retraces (the Chandelier exit then hands
back a few ATRs per false start). It sits out steady downtrends because thrust
up-bars are rare there. It is not a trend-continuation filter: after entry
the only protection is the trailing stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class AtrExpansionParams(StrategyParams):
    atr_period: int = 14
    k: float = 0.75  # required close-to-close gain, in ATRs
    expansion: float = 1.3  # required true range, in multiples of ATR
    close_pos: float = 0.3  # close must sit in the top this fraction of the bar's range
    chand_period: int = 22
    chand_mult: float = 3.0

    def validate(self) -> None:
        if self.atr_period < 2 or self.chand_period < 1:
            raise ValueError("atr_period must be >= 2 and chand_period >= 1")
        if self.k < 0 or self.expansion < 0:
            raise ValueError("k and expansion must be >= 0")
        if not 0 < self.close_pos <= 1:
            raise ValueError("close_pos must be in (0, 1]")
        if self.chand_mult <= 0:
            raise ValueError("chand_mult must be > 0")


class AtrExpansionBreakout(ResearchStrategy):
    """Long-only volatility-expansion breakout. Parameters (``AtrExpansionParams``):
    ``atr_period`` for ATR, ``k`` the minimum close-to-close gain in ATRs, ``expansion``
    the minimum true range in ATRs, ``close_pos`` how close to the bar's high the close
    must be, ``chand_period`` / ``chand_mult`` the Chandelier exit (highest high minus
    ``chand_mult`` ATR)."""

    name = "atr_expansion_breakout"
    family = "volatility"
    description = "Buy a bar whose range and close-to-close gain are large vs ATR and closes near its high; exit on a Chandelier stop."
    PARAMS = AtrExpansionParams
    PARAM_GRID = {
        "k": [0.5, 0.75, 1.0],
        "expansion": [1.0, 1.5],
        "chand_mult": [2.5, 3.0, 4.0],
        "close_pos": [0.25, 0.4],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.atr_period, p.chand_period) + 2

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        a = ind.atr(bars.high, bars.low, bars.close, p.atr_period)
        tr = ind.true_range(bars.high, bars.low, bars.close)
        hh = ind.rolling_max(bars.high, p.chand_period)
        close, prev_close, atr_now, atr_prev = bars.close[-1], bars.close[-2], a[-1], a[-2]
        info = dict(close=close, atr=atr_now, true_range=tr[-1], chandelier_high=hh[-1])
        if not ind.last_finite(close, prev_close, atr_now, atr_prev, tr[-1], hh[-1]) or atr_prev <= 0:
            return Signal.hold("indicators not ready", **info)

        level = hh[-1] - p.chand_mult * atr_now
        info["chandelier"] = level
        if in_position:
            if close < level:
                return Signal.exit(1.0, f"close {close:.2f} < Chandelier {level:.2f}", **info)
            return Signal.hold("in trade, above Chandelier", **info)

        thrust = (close - prev_close) / atr_prev
        rng = bars.high[-1] - bars.low[-1]
        pos = (close - bars.low[-1]) / rng if rng > 0 else 0.0
        info.update(thrust=thrust, range_ratio=tr[-1] / atr_prev, close_position=pos)
        if thrust <= p.k:
            return Signal.hold("no close-to-close thrust", **info)
        if tr[-1] <= p.expansion * atr_prev:
            return Signal.hold("range not expanded", **info)
        if pos < 1.0 - p.close_pos:
            return Signal.hold("close not near the bar's high", **info)
        if close < level:
            return Signal.hold("thrust bar already below the Chandelier, would be sold at once", **info)
        strength = clip01(0.5 + 0.25 * (thrust - p.k))
        return Signal.enter(strength, f"thrust {thrust:.2f} ATR, range {tr[-1] / atr_prev:.2f}x ATR", **info)
