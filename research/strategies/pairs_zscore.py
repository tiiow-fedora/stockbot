"""
Pairs spread z-score, long leg only.

Family: pairs.

Idea
----
Two instruments that share a common driver (same sector, ETF vs basket, two share
classes) keep a roughly stable log-price relationship: ``log(P) - beta * log(Q)`` is
stationary even though both prices wander. When the instrument becomes unusually cheap
against its partner the gap tends to close. The engine is single-symbol and long-only, so
only the LONG LEG is traded: we buy the instrument when it is cheap vs the partner and
sell when the gap has closed. The short partner leg (which would hedge the common trend)
is NOT traded - this is half a pairs trade and keeps full market exposure.

Rules (long-only, daily bars; the partner is a ``List[Candle]`` passed as ``partner=``)
-----------------------------------------------------------------------------
spread_t = log(P_t) - beta * log(Q_t); beta is ``fixed_beta`` (default 1, i.e. the log price
ratio) or, with ``rolling_beta=True``, the rolling OLS slope of log P on log Q over
``beta_window`` bars. Measured on synth.cointegrated_pair the rolling fit is WORSE: a short
window partly fits the very spread deviations we want to trade and removes most of the
signal. Use it only when the true hedge ratio is unknown and drifting, with a long window.
z = (spread - mean) / std over the last ``z_window`` spread values.
ENTER  -stop_z <= z < -entry_z (this leg is cheap vs the partner, but not so far gone that the
       stop would fire immediately)
       [and, if ``cointegration_check``, the correlation of daily log returns over the
        last ``corr_window`` bars exceeds ``min_corr``]
EXIT   z > -exit_z  (gap closed; exit_z = 0 exits at the mean)
       or z < -stop_z (relationship broke: stop out rather than average down)
Signal strength grows with how deep the entry is, from 0.5 at ``entry_z`` to 1.0 at ``stop_z``.

Partner data handling (no look-ahead): partner closes are looked up ONLY at timestamps
in the visible bars (dict timestamp -> close built once); partner bars after the decision
bar are never touched. A missing partner timestamp is forward-filled for at most
3 consecutive bars; longer gaps leave the bar out of the spread statistics, and if the
decision bar itself has no partner value the strategy HOLDs.

Expected behaviour
------------------
Wins when the spread is genuinely mean-reverting (cointegrated pairs): it buys dips that
are idiosyncratic to this leg. It does NOT hedge: when both legs fall together the long
leg loses with the market, and if the relationship breaks permanently (spread trends) it
keeps buying cheapness that never reverts until the stop triggers. On independent random
walks there is no edge: z-score signals are noise and costs make it lose slowly.
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
class PairsZScoreParams(StrategyParams):
    z_window: int = 60
    beta_window: int = 120
    corr_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    rolling_beta: bool = False
    fixed_beta: float = 1.0
    cointegration_check: bool = False
    min_corr: float = 0.5

    def validate(self) -> None:
        if self.z_window < 5:
            raise ValueError("z_window must be >= 5")
        if self.beta_window < 5 or self.corr_window < 5:
            raise ValueError("beta_window and corr_window must be >= 5")
        if not 0.0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        if not np.isfinite(self.fixed_beta) or self.fixed_beta <= 0:
            raise ValueError("fixed_beta must be > 0")
        if not -1.0 <= self.min_corr <= 1.0:
            raise ValueError("min_corr must be in [-1, 1]")


class PairsZScore(ResearchStrategy):
    """Long-leg-only pairs trade. Parameters (``PairsZScoreParams``): ``z_window`` for the spread
    z-score, ``beta_window`` for the rolling hedge ratio (or ``fixed_beta`` if ``rolling_beta`` is
    False), ``entry_z`` / ``exit_z`` / ``stop_z`` thresholds, and an optional return-correlation
    gate (``cointegration_check``, ``corr_window``, ``min_corr``). Needs ``partner=<List[Candle]>``."""

    name = "pairs_zscore"
    family = "pairs"
    description = "Buy this instrument when its log spread vs a partner is unusually low; sell when it reverts (long leg only)."
    PARAMS = PairsZScoreParams
    PARAM_GRID = {
        "z_window": [30, 60, 100], "entry_z": [1.5, 2.0, 2.5], "exit_z": [0.0, 0.5], "rolling_beta": [False, True],
    }
    requires_partner = True

    def __init__(self, params: Optional[StrategyParams] = None, partner: Optional[List[Candle]] = None, **overrides: Any):
        super().__init__(params, **overrides)
        self._partner = _partner.partner_close_map(partner)  # raises on None / empty

    @property
    def min_bars(self) -> int:
        p = self.params
        beta_need = p.beta_window if p.rolling_beta else 1
        need = beta_need + p.z_window
        if p.cointegration_check:
            need = max(need, p.corr_window + 1)
        return need + 1

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        partner = _partner.align_partner(self._partner, bars.timestamps)
        if not np.isfinite(partner[-1]):
            return Signal.hold("partner has no bar at the decision timestamp")
        mask = _partner.usable_mask(bars.close, partner)
        if not mask[-1]:
            return Signal.hold("decision bar has no usable own/partner price")
        la, lb = np.log(bars.close[mask]), np.log(partner[mask])

        if p.rolling_beta:
            beta = ind.rolling_beta(la, lb, p.beta_window)
        else:
            beta = np.full(len(la), p.fixed_beta)
        z = ind.zscore(la - beta * lb, p.z_window)
        z_now, beta_now = z[-1], beta[-1]
        info = dict(z=z_now, beta=beta_now, close=bars.close[-1])
        if not ind.last_finite(z_now, beta_now):
            return Signal.hold("spread statistics not ready", **info)

        if in_position:
            if z_now < -p.stop_z:
                return Signal.exit(1.0, f"z {z_now:.2f} < -{p.stop_z}: relationship broke, stop", **info)
            if z_now > -p.exit_z:
                return Signal.exit(1.0, f"z {z_now:.2f} > -{p.exit_z}: spread reverted", **info)
            return Signal.hold("in trade, spread still cheap", **info)

        if z_now < -p.entry_z and z_now >= -p.stop_z:
            if p.cointegration_check:
                corr = self._return_corr(la, lb)
                info["corr"] = corr
                if not ind.last_finite(corr) or corr <= p.min_corr:
                    return Signal.hold("return correlation below min_corr", **info)
            strength = clip01(0.5 + 0.5 * (-z_now - p.entry_z) / (p.stop_z - p.entry_z))
            return Signal.enter(strength, f"z {z_now:.2f} < -{p.entry_z}: cheap vs partner", **info)
        return Signal.hold("spread not cheap enough", **info)

    def _return_corr(self, la: np.ndarray, lb: np.ndarray) -> float:
        n = self.params.corr_window
        if len(la) < n + 1:
            return float("nan")
        ra, rb = np.diff(la)[-n:], np.diff(lb)[-n:]
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])
