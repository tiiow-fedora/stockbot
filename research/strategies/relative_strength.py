"""
Relative strength vs a benchmark / peer (dual momentum).

Family: pairs.

Idea
----
Instruments that have outperformed a benchmark over the last few months tend to keep
outperforming it for a while (cross-sectional momentum). Requiring the absolute return to
be positive as well (dual momentum) avoids owning the "least bad" instrument in a falling
market. The partner (benchmark or peer, a ``List[Candle]`` passed as ``partner=``) is
only used as the yardstick; it is never traded.

Rules (long-only, daily bars)
-----------------------------
r_self = close[t] / close[t - rs_period] - 1, r_peer likewise for the partner,
rs = r_self - r_peer.
ENTER  rs > margin AND r_self > 0
EXIT   rs < -hysteresis (the peer now clearly leads)
       or r_self < -hysteresis (absolute momentum has turned clearly negative)
Between the two thresholds the position is simply held; ``hysteresis`` > 0 keeps the rule from
flip-flopping when rs hovers around zero. Signal strength grows with the margin by which
the instrument leads: 0.5 just above ``margin``, 1.0 at ``margin`` + 10 percentage points.

Partner data handling (no look-ahead): partner closes are looked up ONLY at timestamps in
the visible bars (see ``_partner.py``); bars after the decision bar are never touched. A
missing partner timestamp is forward-filled for at most 3 consecutive bars; if the partner
value at the decision bar or at the ``rs_period``-ago bar is unavailable the strategy HOLDs.

Expected behaviour
------------------
Does well when leadership persists and the market trends (the instrument is the leader in
an up-market). It lags at turning points (it needs rs_period bars to notice a new leader),
whipsaws when leadership flips often (mean-reverting relative performance), and, with the
absolute filter, sits in cash in broad downtrends - which limits drawdowns but forgoes any
rebound. If the instrument is simply the benchmark's twin there is no signal at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from core.models import Candle
from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01
from research.strategies import _partner


@dataclass(frozen=True)
class RelativeStrengthParams(StrategyParams):
    rs_period: int = 60
    margin: float = 0.02
    hysteresis: float = 0.01

    def validate(self) -> None:
        if self.rs_period < 2:
            raise ValueError("rs_period must be >= 2")
        if not (np.isfinite(self.margin) and np.isfinite(self.hysteresis)) or self.margin < 0 or self.hysteresis < 0:
            raise ValueError("margin and hysteresis must be >= 0")


class RelativeStrength(ResearchStrategy):
    """Dual-momentum relative strength. Parameters (``RelativeStrengthParams``): ``rs_period`` bars
    for the trailing returns, ``margin`` the lead (fraction) needed to enter, ``hysteresis`` the
    deficit (fraction) tolerated before exiting. Needs ``partner=<List[Candle]>`` as benchmark."""

    name = "relative_strength"
    family = "pairs"
    description = "Hold the instrument while its trailing return beats the benchmark's and is positive (dual momentum)."
    PARAMS = RelativeStrengthParams
    PARAM_GRID = {"rs_period": [20, 40, 60, 120], "margin": [0.0, 0.02, 0.05], "hysteresis": [0.0, 0.02]}
    requires_partner = True

    def __init__(self, params: Optional[StrategyParams] = None, partner: Optional[List[Candle]] = None, **overrides: Any):
        super().__init__(params, **overrides)
        self._partner = _partner.partner_close_map(partner)  # raises on None / empty

    @property
    def min_bars(self) -> int:
        return self.params.rs_period + 2

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        partner = _partner.align_partner(self._partner, bars.timestamps)
        own = np.asarray(bars.close, float)
        if not (ind.last_finite(partner[-1], partner[-1 - p.rs_period], own[-1], own[-1 - p.rs_period]) and own[-1 - p.rs_period] > 0):
            return Signal.hold("partner or own price unavailable", close=own[-1])
        r_self = own[-1] / own[-1 - p.rs_period] - 1.0
        r_peer = partner[-1] / partner[-1 - p.rs_period] - 1.0
        rs = r_self - r_peer
        info = dict(r_self=r_self, r_peer=r_peer, rs=rs, close=own[-1])

        if in_position:
            if rs < -p.hysteresis:
                return Signal.exit(1.0, f"rs {rs:+.3f} < -{p.hysteresis}: peer leads", **info)
            if r_self < -p.hysteresis:
                return Signal.exit(1.0, f"own return {r_self:+.3f} < -{p.hysteresis}: absolute momentum lost", **info)
            return Signal.hold("still leading", **info)

        if rs > p.margin and r_self > 0:
            strength = clip01(0.5 + 0.5 * (rs - p.margin) / 0.10)
            return Signal.enter(strength, f"rs {rs:+.3f} > {p.margin} and own return {r_self:+.3f} > 0", **info)
        return Signal.hold("no relative-strength lead", **info)
