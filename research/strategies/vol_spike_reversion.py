"""
Volatility-spike (panic) reversion.

Family: volatility.

Idea
----
Sudden jumps in short-term realised volatility together with a sharp drop from the
recent high mark capitulation: forced sellers and fear premium push price below
fair value, and both volatility and price tend to mean-revert afterwards. The strategy
buys the panic and sells when calm returns or price has recovered.

Rules (long-only, daily bars)
-----------------------------
ratio = realised vol over ``short_vol`` bars / realised vol over ``long_vol`` bars,
        where realised vol is the root-mean-square of daily log returns (zero-mean, so a
        one-way crash counts as high vol; a plain std would discount its drift).
ENTER  ratio >= spike_ratio AND close <= (1 - drop_pct) * highest close of the last
       ``high_window`` bars
       [and, if ``trend_filter``, close > SMA(``trend_period``): only buy panics inside a
        long-term uptrend]
EXIT   ratio < normal_ratio (volatility normalised)
       or close >= SMA(``recover_sma``) (price recovered)
Signal strength grows with the spike: 0.5 at ``spike_ratio``, 1.0 at twice that ratio.

Expected behaviour
------------------
Wins in V-shaped or choppy markets with sharp, short-lived selloffs (fear spikes that fade)
and in mean-reverting series. It is a catch-the-falling-knife strategy: in a persistent
downtrend or a regime-change crash the volatility stays high and price keeps falling, so
it buys early and takes large losses (the trend filter reduces but does not remove this).
Trades are rare; in calm or smoothly trending data it does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research import indicators as ind
from research.base import Bars, ResearchStrategy, Signal, StrategyParams, clip01


def _rms_vol(close: np.ndarray, window: int) -> float:
    """Root-mean-square of the last ``window`` daily log returns (NaN if not enough data)."""
    r = ind.log_returns(close)[1:]
    if len(r) < window:
        return float("nan")
    return float(np.sqrt(np.mean(r[-window:] ** 2)))


@dataclass(frozen=True)
class VolSpikeParams(StrategyParams):
    short_vol: int = 5
    long_vol: int = 40
    high_window: int = 20
    recover_sma: int = 20
    trend_period: int = 100
    spike_ratio: float = 1.6
    normal_ratio: float = 1.2
    drop_pct: float = 0.06
    trend_filter: bool = False

    def validate(self) -> None:
        for name in ("short_vol", "long_vol", "high_window", "recover_sma", "trend_period"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.short_vol >= self.long_vol:
            raise ValueError("short_vol must be < long_vol")
        if not 0.0 < self.drop_pct < 1.0:
            raise ValueError("drop_pct must be in (0, 1)")
        if not 0.0 < self.normal_ratio < self.spike_ratio:
            raise ValueError("require 0 < normal_ratio < spike_ratio")


class VolSpikeReversion(ResearchStrategy):
    """Buy a fear spike. Parameters (``VolSpikeParams``): ``short_vol``/``long_vol`` realised-vol
    windows and ``spike_ratio`` trigger, ``drop_pct`` below the ``high_window`` high, exits on
    ``normal_ratio`` or recovery to ``recover_sma``, optional ``trend_filter`` on ``trend_period``."""

    name = "vol_spike_reversion"
    family = "volatility"
    description = "Buy a volatility spike combined with a sharp drop from the recent high; sell on calm or recovery."
    PARAMS = VolSpikeParams
    PARAM_GRID = {
        "spike_ratio": [1.5, 2.0, 2.5],
        "drop_pct": [0.04, 0.06, 0.10],
        "normal_ratio": [1.0, 1.3],
        "trend_filter": [False, True],
    }

    @property
    def min_bars(self) -> int:
        p = self.params
        need = max(p.long_vol + 1, p.high_window, p.recover_sma)
        if p.trend_filter:
            need = max(need, p.trend_period)
        return need + 1

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        p = self.params
        close = bars.close
        short, long_ = _rms_vol(close, p.short_vol), _rms_vol(close, p.long_vol)
        sma = ind.rolling_mean(close, p.recover_sma)[-1]
        high = ind.rolling_max(close, p.high_window)[-1]
        now = close[-1]
        ratio = short / long_ if ind.last_finite(short, long_) and long_ > 0 else np.nan
        info = dict(close=now, vol_ratio=ratio, sma=sma, recent_high=high)
        if not ind.last_finite(ratio, sma, high):
            return Signal.hold("indicators not ready", **info)

        if in_position:
            if ratio < p.normal_ratio:
                return Signal.exit(1.0, f"vol ratio {ratio:.2f} < {p.normal_ratio}: calm again", **info)
            if now >= sma:
                return Signal.exit(1.0, f"close {now:.2f} >= {p.recover_sma}-bar SMA {sma:.2f}: recovered", **info)
            return Signal.hold("in trade, still stressed", **info)

        if ratio >= p.spike_ratio and now <= (1.0 - p.drop_pct) * high:
            if p.trend_filter:
                trend = ind.rolling_mean(close, p.trend_period)[-1]
                info["trend_sma"] = trend
                if not ind.last_finite(trend) or now <= trend:
                    return Signal.hold("panic but below long-term trend", **info)
            strength = clip01(0.5 + 0.5 * (ratio - p.spike_ratio) / p.spike_ratio)
            return Signal.enter(strength, f"vol ratio {ratio:.2f} >= {p.spike_ratio} and {now / high - 1:+.1%} from high", **info)
        return Signal.hold("no panic", **info)
