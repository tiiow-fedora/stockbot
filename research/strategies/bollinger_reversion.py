"""
Bollinger-band mean reversion ("buy the dip below the lower band").

Family: mean_reversion.

Idea
----
Over short horizons prices stretch away from their recent average and tend to
snap back. A close more than ``num_std`` standard deviations below the
``period``-bar mean is a statistically unusual selloff; when the market is
range-bound or in an uptrend, such stretches are often overreactions that
retrace to the mean. The real danger is a regime break (a crash that keeps
going), so two optional guards are provided: a long-term trend filter and a
"panic" cap that refuses to catch a knife falling unusually far.

Rules (long-only, daily bars)
-----------------------------
Let mid = SMA(close, period), sd = population std of the same window,
lower = mid - num_std * sd, z = (close - mid) / sd.
ENTER  close < lower
       [and, if ``trend_period`` > 0, close > SMA(close, trend_period)]
       [and, if ``panic_std`` > 0, z >= -panic_std; a close MORE than panic_std
        deviations below the mean is treated as a regime break and skipped]
EXIT   ``exit_at`` = 'mid':   close >= mid
       ``exit_at`` = 'upper': close >= mid + num_std * sd
There is no price stop inside the strategy (it is stateless); only the engine's
risk overlay limits a loss, so a position that never recovers is held. In a
persistent downtrend with the lab's loose 50% stop it lost about 63%; with the
shipped 8% stop about 11%. ``trend_period`` is the intended guard (it removed
the loss entirely there), so no separate drawdown guard is provided.
Signal strength is 0.5 at the lower band and grows to 1.0 when the close is one
further standard deviation below it (deeper dip = stronger conviction), which
is why ``panic_std`` should sit above ``num_std`` + 1 to leave that range usable.

Expected behaviour
------------------
Wins on mean-reverting series (many small, high win-rate trades) and on
FAST cycles (period about 1-2x the band window, with a band narrow enough to be
crossed: a pure sine never reaches -2 sd, so num_std 2.0 does not trade it). On
slow cycles (synth.cyclical default, period 40) it LOSES: each leg is a trend
over the 20-bar window (about -31% on scenario_suite cyclical).
It LOSES in persistent downtrends and sharp regime breaks: each new low looks
like a "dip", so it buys repeatedly into a falling market and holds losers
until (if ever) price returns to the mean. The trend filter removes most of
that at the cost of fewer trades; ``panic_std`` only helps against very
violent breaks. Pure momentum series are also unfavourable (dips continue).
Low payoff ratio: small wins, occasional large losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class BollingerReversionParams(StrategyParams):
    period: int = 20
    num_std: float = 2.0
    exit_at: str = "mid"  # 'mid' or 'upper'
    trend_period: int = 0  # 0 disables the "only buy dips above the long SMA" filter
    panic_std: float = 0.0  # 0 disables; else skip closes more than this many std below the mean

    def validate(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.num_std <= 0:
            raise ValueError("num_std must be > 0")
        if self.exit_at not in ("mid", "upper"):
            raise ValueError("exit_at must be 'mid' or 'upper'")
        if self.trend_period < 0:
            raise ValueError("trend_period must be >= 0 (0 disables)")
        if self.panic_std < 0:
            raise ValueError("panic_std must be >= 0 (0 disables)")
        if self.panic_std and self.panic_std <= self.num_std:
            raise ValueError("panic_std must exceed num_std, otherwise no entry is possible")


class BollingerReversion(ResearchStrategy):
    """Long-only Bollinger dip buyer. Parameters (``BollingerReversionParams``): ``period`` and
    ``num_std`` define the bands, ``exit_at`` ('mid' | 'upper') the profit target, ``trend_period``
    an optional long-SMA uptrend filter (0 = off) and ``panic_std`` an optional cap on how deep a
    close may be (in std devs below the mean) before it is treated as a regime break (0 = off)."""

    name = "bollinger_reversion"
    family = "mean_reversion"
    description = "Buy a close below the lower Bollinger band, sell on recovery to the middle (or upper) band."
    PARAMS = BollingerReversionParams
    PARAM_GRID = {
        "period": [10, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
        "exit_at": ["mid", "upper"],
        "trend_period": [0, 100],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.period, p.trend_period) + 2

    @property
    def lookback(self) -> int:
        return self.min_bars + 5  # only rolling (non-recursive) indicators

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        mid, upper, lower = ind.bollinger(bars.close, p.period, p.num_std)
        sd = ind.rolling_std(bars.close, p.period)
        close, mid_v, up_v, lo_v, sd_v = bars.close[-1], mid[-1], upper[-1], lower[-1], sd[-1]
        info = dict(close=close, mid=mid_v, upper=up_v, lower=lo_v)
        if not ind.last_finite(close, mid_v, up_v, lo_v, sd_v):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            target = mid_v if p.exit_at == "mid" else up_v
            if close >= target:
                return Signal.exit(1.0, f"close {close:.2f} recovered to {p.exit_at} band {target:.2f}", **info)
            return Signal.hold("in trade, awaiting recovery", **info)

        if sd_v <= 0 or close >= lo_v:
            return Signal.hold("close not below lower band", **info)
        z = (close - mid_v) / sd_v
        info["z"] = z
        if p.panic_std > 0 and z < -p.panic_std:
            return Signal.hold(f"panic: z {z:.2f} < -{p.panic_std}, regime break not bought", **info)
        if p.trend_period > 0:
            trend = float(np.mean(bars.close[-p.trend_period:]))
            info["trend_sma"] = trend
            if close <= trend:
                return Signal.hold("dip below long-term trend, skipped", **info)
        beyond = -z - p.num_std  # std devs past the band
        strength = clip01(0.5 + 0.5 * beyond)
        return Signal.enter(strength, f"close {close:.2f} < lower band {lo_v:.2f} (z={z:.2f})", **info)
