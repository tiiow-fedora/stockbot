"""
Time-series (absolute) momentum, after Moskowitz, Ooi & Pedersen (2012).

Family: momentum.

Idea
----
An asset's own trailing return predicts the sign of its next return: assets
that went up over the past several months tend to keep going up (investor
under-reaction, herding, slow-moving capital). Unlike cross-sectional momentum
this needs no peer group - each instrument is judged against zero, not against
other instruments - so on a single symbol it acts as a slow "is the trend up?"
switch. The classic 12-1 recipe (12-month return, skipping the latest month to
avoid short-term reversal) is rescaled here to daily bars.

Rules (long-only, daily bars)
-----------------------------
mom      = close[t-skip] / close[t-skip-lookback] - 1      (the last ``skip`` bars are ignored)
vol      = stdev of the last ``vol_window`` daily log returns
ENTER    mom > hurdle                                      (flat only)
EXIT     mom < 0                                           (long only)
HOLD     otherwise - the band 0 <= mom <= hurdle is hysteresis: a position
         opened above the hurdle is not closed until momentum turns negative,
         and a flat book does not buy marginally positive momentum.

Exit depends only on ``mom``: a zero-volatility (perfectly flat) tape cannot trap a
losing position (only ENTER needs ``vol`` > 0, for the strength). ``hurdle`` must be finite.

Signal strength = tanh(mom / (vol * sqrt(lookback))) - the momentum "t-stat":
the return in units of the return a random walk of this volatility would
typically produce over the window, squashed into [0, 1]. A strong, smooth
trend scores near 1; a small gain in a noisy market scores near 0.

Expected behaviour
------------------
Wins on persistent drifts and on autocorrelated (momentum) returns; stays flat
through sustained declines. Because the signal is a ~6-month return it reacts
slowly: it gives back a good part of every trend at the turn, and it is
whipsawed by V-shaped reversals and by range-bound / mean-reverting markets
(there the trailing return flips sign at random with no follow-through). On a
pure random walk it has no edge - expect roughly buy-and-hold-like noise with
fewer days in the market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


@dataclass(frozen=True)
class TsmomParams(StrategyParams):
    lookback: int = 120
    skip: int = 5  # most recent bars excluded from the return; 0 disables the skip
    vol_window: int = 20
    hurdle: float = 0.02  # minimum trailing return required to ENTER (0 = any positive)

    def validate(self) -> None:
        if self.lookback <= 0 or self.vol_window <= 1:
            raise ValueError("lookback must be >= 1 and vol_window >= 2")
        if self.skip < 0:
            raise ValueError("skip must be >= 0")
        if self.skip >= self.lookback:
            raise ValueError("skip must be < lookback")
        if not 0 <= self.hurdle < float("inf"):  # also rejects NaN
            raise ValueError("hurdle must be finite and >= 0")


class Tsmom(ResearchStrategy):
    """Long-only time-series momentum. Parameters (``TsmomParams``): ``lookback`` bars of trailing
    return, ``skip`` most recent bars ignored, ``vol_window`` for the volatility scaling of the
    signal strength, ``hurdle`` minimum return to enter (exit when the return turns negative)."""

    name = "tsmom"
    family = "momentum"
    description = "Hold while the trailing (lookback minus skip) return is positive and above a hurdle (Moskowitz-Ooi-Pedersen)."
    PARAMS = TsmomParams
    PARAM_GRID = {"lookback": [60, 90, 120, 180], "skip": [0, 5, 10], "hurdle": [0.0, 0.02, 0.05]}

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.lookback + p.skip + 1, p.vol_window + 2)

    @property
    def lookback(self) -> int:
        # Nothing here is recursive, so a short cushion beyond the minimum is enough.
        return self.min_bars + 10

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        close = bars.close
        end = len(close) - 1 - p.skip
        mom = close[end] / close[end - p.lookback] - 1.0 if end - p.lookback >= 0 else np.nan
        vol = ind.realized_vol(close, p.vol_window)[-1]
        info = dict(mom=mom, vol=vol, hurdle=p.hurdle)
        if not ind.last_finite(mom):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            # Exit needs only the momentum: a dead-flat tape (vol == 0) must not trap a losing position.
            if mom < 0:
                return Signal.exit(1.0, f"{p.lookback}-bar momentum {mom:+.1%} < 0", **info)
            return Signal.hold(f"momentum {mom:+.1%} still non-negative", **info)

        if not ind.last_finite(vol) or vol <= 0:
            return Signal.hold("indicators not ready", **info)
        if mom > p.hurdle:
            strength = clip01(float(np.tanh(mom / (vol * np.sqrt(p.lookback)))))
            return Signal.enter(strength, f"{p.lookback}-bar momentum {mom:+.1%} > hurdle {p.hurdle:.1%}", **info)
        return Signal.hold("momentum below hurdle", **info)
