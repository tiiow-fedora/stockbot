"""
Internal Bar Strength (IBS) + z-score mean reversion.

Family: mean_reversion.

Idea
----
IBS = (close - low) / (high - low) says where the bar closed inside its own
range. A close at the very bottom of the range after a drop below the recent
mean means sellers exhausted themselves into the close; in range-bound markets
the next day frequently opens higher. Requiring a negative price z-score too
ensures the weak close is a genuine stretch, not noise inside a flat range.

Rules (long-only, daily bars; needs real OHLC)
----------------------------------------------
Let z = (close - mean) / std over the last ``z_period`` closes.
ENTER  IBS < ibs_entry  AND  z < -z_entry
EXIT   IBS > ibs_exit   OR   z > 0
Zero-range bars have no IBS (NaN) and cannot trigger an entry; while in a trade
they can still exit via the z > 0 leg. Signal strength is 0.5 plus up to 0.25
for how close to the low the bar closed and up to 0.25 for each extra standard
deviation of stretch beyond ``z_entry``. There is no price stop inside the
strategy (stateless); the engine's risk overlay is the only loss limit.

Expected behaviour
------------------
Wins on mean-reverting and cyclical series: trades are very short (1-3 bars)
and frequent. It has no trend filter, so in a persistent downtrend it keeps
buying weak closes that keep getting weaker and loses; it also loses on momentum
series where a weak close predicts another. Small average gain per trade, so it
is sensitive to costs (5 bps slippage per fill in the lab).
"""

from __future__ import annotations

from dataclasses import dataclass

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class IbsReversionParams(StrategyParams):
    z_period: int = 20
    z_entry: float = 1.0
    ibs_entry: float = 0.2
    ibs_exit: float = 0.8

    def validate(self) -> None:
        if self.z_period < 3:
            raise ValueError("z_period must be >= 3")
        if self.z_entry <= 0:
            raise ValueError("z_entry must be > 0")
        if not 0 < self.ibs_entry < self.ibs_exit <= 1:
            raise ValueError("need 0 < ibs_entry < ibs_exit <= 1")


class IbsReversion(ResearchStrategy):
    """Long-only IBS + z-score reversion. Parameters (``IbsReversionParams``): ``z_period`` window
    of the price z-score, ``z_entry`` required stretch below the mean, ``ibs_entry`` max IBS to buy
    and ``ibs_exit`` IBS above which to sell (also sells once z > 0)."""

    name = "ibs_reversion"
    family = "mean_reversion"
    description = "Buy a close near the bar low that is also stretched below its mean; sell when the close strengthens or reverts."
    PARAMS = IbsReversionParams
    PARAM_GRID = {
        "z_period": [10, 20, 40],
        "z_entry": [0.5, 1.0, 1.5],
        "ibs_entry": [0.1, 0.2, 0.3],
        "ibs_exit": [0.7, 0.8],
    }

    @property
    def min_bars(self) -> int:
        return self.params.z_period + 2

    @property
    def lookback(self) -> int:
        return self.min_bars + 5  # only rolling (non-recursive) indicators

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        z_v = ind.zscore(bars.close, p.z_period)[-1]
        ibs_v = ind.ibs(bars.high, bars.low, bars.close)[-1]
        close = bars.close[-1]
        info = dict(close=close, z=z_v, ibs=ibs_v)
        if not ind.last_finite(close, z_v):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if z_v > 0:
                return Signal.exit(1.0, f"z {z_v:.2f} back above the mean", **info)
            if ind.last_finite(ibs_v) and ibs_v > p.ibs_exit:
                return Signal.exit(1.0, f"IBS {ibs_v:.2f} > {p.ibs_exit}", **info)
            return Signal.hold("in trade, awaiting reversion", **info)

        if not ind.last_finite(ibs_v):
            return Signal.hold("zero-range bar, no IBS", **info)
        if ibs_v < p.ibs_entry and z_v < -p.z_entry:
            low_part = 0.25 * (1.0 - ibs_v / p.ibs_entry)
            stretch_part = 0.25 * min(1.0, -z_v - p.z_entry)
            return Signal.enter(clip01(0.5 + low_part + stretch_part), f"IBS {ibs_v:.2f}, z {z_v:.2f}", **info)
        return Signal.hold("no setup", **info)
