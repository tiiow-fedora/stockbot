"""
Synthetic market-data generators with KNOWN structure.

Every generator is deterministic given ``seed`` and returns ordinary
``core.models.Candle`` lists (business-day timestamps, valid OHLCV), so the
output drops straight into ``Backtester.run``. Because we know the data
generating process, tests can make *falsifiable* claims such as "a trend
follower must make money on a persistent trend" or "mean reversion must
beat buy-and-hold on an Ornstein-Uhlenbeck series" - something real data
can never give you.

Generators
----------
``gbm``                  geometric Brownian motion (the no-edge null model)
``trending``             constant drift + noise (clean uptrend / downtrend)
``ar1_momentum``         autocorrelated returns (genuine momentum)
``mean_reverting``       Ornstein-Uhlenbeck log price (genuine mean reversion)
``regime_switching``     Markov chain over bull / bear / sideways regimes
``garch``                volatility clustering (GARCH(1,1) returns)
``jump_diffusion``       GBM + Poisson jumps (gap risk)
``range_then_breakout``  repeated consolidation -> breakout cycles
``cyclical``             sine-wave cycle + noise
``v_shape``              crash then recovery
``flat``                 (almost) constant price - degenerate input
``cointegrated_pair``    two series sharing a stochastic trend (pairs trading)
``scenario_suite``       dict of all of the above, for strategy x scenario matrices
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.models import Candle

DEFAULT_START = datetime(2020, 1, 1, tzinfo=timezone.utc)  # a Wednesday


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def business_days(n: int, start: datetime = DEFAULT_START) -> List[datetime]:
    """``n`` consecutive Mon-Fri timestamps starting at (or after) ``start``."""
    out: List[datetime] = []
    ts = start
    while len(out) < n:
        if ts.weekday() < 5:
            out.append(ts)
        ts += timedelta(days=1)
    return out


def candles_from_closes(
    closes: Sequence[float],
    symbol: str = "SYN",
    seed: int = 0,
    start: datetime = DEFAULT_START,
    intraday_vol: Optional[float] = None,
    gap_vol: Optional[float] = None,
    base_volume: float = 1_000_000.0,
    volume_spikes: Optional[Sequence[float]] = None,
) -> List[Candle]:
    """Wrap a close path in realistic OHLCV bars.

    open  = previous close * (1 + small overnight gap)
    high / low = envelope of open/close widened by ``intraday_vol`` noise
    volume = lognormal, boosted on big absolute moves (and by ``volume_spikes``,
    a per-bar multiplier array, e.g. to confirm breakouts).
    """
    closes = np.asarray(closes, dtype=float)
    if np.any(closes <= 0) or not np.all(np.isfinite(closes)):
        raise ValueError("closes must be positive and finite")
    n = len(closes)
    rng = np.random.default_rng([seed, 7919])
    rets = np.concatenate([[0.0], np.diff(np.log(closes))])
    typical = float(np.std(rets)) or 0.005
    iv = intraday_vol if intraday_vol is not None else max(typical * 0.6, 1e-4)
    gv = gap_vol if gap_vol is not None else typical * 0.15

    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1] * np.exp(rng.normal(0.0, gv, n - 1))
    top = np.maximum(opens, closes)
    bot = np.minimum(opens, closes)
    highs = top * np.exp(np.abs(rng.normal(0.0, iv, n)))
    lows = bot * np.exp(-np.abs(rng.normal(0.0, iv, n)))
    move = np.abs(rets) / typical
    vol = base_volume * rng.lognormal(0.0, 0.25, n) * (1.0 + 0.5 * np.minimum(move, 4.0))
    if volume_spikes is not None:
        vol = vol * np.asarray(volume_spikes, dtype=float)

    stamps = business_days(n, start)
    return [
        Candle(
            symbol=symbol,
            timestamp=stamps[i],
            open=float(round(opens[i], 4)),
            high=float(round(max(highs[i], opens[i], closes[i]), 4)),
            low=float(round(min(lows[i], opens[i], closes[i]), 4)),
            close=float(round(closes[i], 4)),
            volume=int(max(1, vol[i])),
        )
        for i in range(n)
    ]


def _path(start_price: float, log_returns: np.ndarray) -> np.ndarray:
    return start_price * np.exp(np.concatenate([[0.0], np.cumsum(log_returns[1:])]))


def _rng(seed: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, salt])


# --------------------------------------------------------------------------
# Generators (all: n bars, seed -> List[Candle])
# --------------------------------------------------------------------------

def gbm(n: int = 500, mu: float = 0.0003, sigma: float = 0.012, seed: int = 0,
        start_price: float = 100.0, symbol: str = "GBM") -> List[Candle]:
    """Daily GBM: log-return ~ N(mu - sigma^2/2, sigma). The no-edge null."""
    r = _rng(seed).normal(mu - 0.5 * sigma**2, sigma, n)
    return candles_from_closes(_path(start_price, r), symbol, seed)


def trending(n: int = 300, daily_drift: float = 0.004, noise: float = 0.006, seed: int = 0,
             start_price: float = 100.0, symbol: str = "TREND") -> List[Candle]:
    """Persistent trend: positive ``daily_drift`` = uptrend, negative = downtrend.
    Drift dominates noise (drift/noise ~ 0.7 per bar), so trend followers must profit."""
    r = daily_drift + _rng(seed).normal(0.0, noise, n)
    return candles_from_closes(_path(start_price, r), symbol, seed)


def ar1_momentum(n: int = 600, phi: float = 0.25, sigma: float = 0.01, mu: float = 0.0002,
                 seed: int = 0, start_price: float = 100.0, symbol: str = "MOM") -> List[Candle]:
    """Returns follow r_t = mu + phi*(r_{t-1}-mu) + eps. phi > 0 -> momentum, phi < 0 -> reversal."""
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must be in (-1, 1)")
    eps = _rng(seed).normal(0.0, sigma, n)
    r = np.empty(n)
    r[0] = mu + eps[0]
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
    return candles_from_closes(_path(start_price, r), symbol, seed)


def mean_reverting(n: int = 600, half_life: float = 8.0, sigma: float = 0.012,
                   seed: int = 0, start_price: float = 100.0, symbol: str = "OU") -> List[Candle]:
    """Ornstein-Uhlenbeck log price around ``start_price``; ``half_life`` in bars.
    Deviations from the long-run mean decay by half every ``half_life`` bars."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    kappa = np.log(2.0) / half_life
    rho = np.exp(-kappa)
    eps = _rng(seed).normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return candles_from_closes(start_price * np.exp(x), symbol, seed)


# regime -> (daily mu, daily sigma)
REGIMES: Dict[str, Tuple[float, float]] = {
    "bull": (0.0012, 0.008),
    "bear": (-0.0015, 0.016),
    "sideways": (0.0, 0.006),
}


def regime_switching(
    n: int = 900, seed: int = 0, stay_prob: float = 0.985, start_price: float = 100.0,
    symbol: str = "REGIME", return_regimes: bool = False,
):
    """Markov chain over bull / bear / sideways regimes (``REGIMES``).
    With ``return_regimes=True`` returns ``(candles, regime_labels)``."""
    rng = _rng(seed)
    names = list(REGIMES)
    state = rng.integers(len(names))
    labels, r = [], np.empty(n)
    for t in range(n):
        if rng.random() > stay_prob:
            state = rng.choice([i for i in range(len(names)) if i != state])
        mu, sig = REGIMES[names[state]]
        r[t] = rng.normal(mu, sig)
        labels.append(names[state])
    candles = candles_from_closes(_path(start_price, r), symbol, seed)
    return (candles, labels) if return_regimes else candles


def garch(n: int = 800, omega: float = 2e-6, alpha: float = 0.10, beta: float = 0.86, mu: float = 0.0003,
          seed: int = 0, start_price: float = 100.0, symbol: str = "GARCH") -> List[Candle]:
    """GARCH(1,1) returns: volatility clusters (calm stretches, then storms)."""
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1 for a stationary GARCH")
    rng = _rng(seed)
    var = omega / (1.0 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        eps = rng.normal() * np.sqrt(var)
        r[t] = mu + eps
        var = omega + alpha * eps**2 + beta * var
    return candles_from_closes(_path(start_price, r), symbol, seed)


def jump_diffusion(n: int = 600, mu: float = 0.0004, sigma: float = 0.009, jump_prob: float = 0.02,
                   jump_mean: float = -0.01, jump_sigma: float = 0.05, seed: int = 0,
                   start_price: float = 100.0, symbol: str = "JUMP") -> List[Candle]:
    """GBM with Poisson-ish jumps (daily probability ``jump_prob``) - tests gap risk."""
    rng = _rng(seed)
    r = rng.normal(mu, sigma, n)
    jumps = rng.random(n) < jump_prob
    r = r + jumps * rng.normal(jump_mean, jump_sigma, n)
    return candles_from_closes(_path(start_price, r), symbol, seed)


def range_then_breakout(
    n: int = 600, cycle: int = 60, range_frac: float = 0.5, range_vol: float = 0.004,
    breakout_drift: float = 0.012, seed: int = 0, start_price: float = 100.0, symbol: str = "BRK",
) -> List[Candle]:
    """Repeating cycle: a tight, drift-less consolidation (``range_frac`` of the cycle),
    then a burst of strong directional movement on elevated volume (direction alternates
    up / up / down). Breakout and squeeze strategies have a clear, repeatable edge here."""
    rng = _rng(seed)
    r = np.empty(n)
    spikes = np.ones(n)
    direction_cycle = [1.0, 1.0, -1.0]
    for t in range(n):
        k, pos = divmod(t, cycle)
        if pos < int(cycle * range_frac):
            # mean-reverting wiggle inside a band: contribute noise, pull-back toward 0
            r[t] = rng.normal(0.0, range_vol)
        else:
            d = direction_cycle[k % len(direction_cycle)]
            r[t] = d * breakout_drift + rng.normal(0.0, range_vol * 1.5)
            spikes[t] = 2.0
    return candles_from_closes(_path(start_price, r), symbol, seed, volume_spikes=spikes)


def cyclical(n: int = 600, period: int = 40, amplitude: float = 0.08, noise: float = 0.004,
             seed: int = 0, start_price: float = 100.0, symbol: str = "CYC") -> List[Candle]:
    """Sine-wave log price with period ``period`` bars plus noise: oscillators/reversion shine."""
    t = np.arange(n)
    x = amplitude * np.sin(2 * np.pi * t / period) + np.cumsum(_rng(seed).normal(0.0, noise, n)) * 0.2
    return candles_from_closes(start_price * np.exp(x), symbol, seed)


def v_shape(n: int = 300, crash_frac: float = 0.4, crash_drift: float = -0.006, recovery_drift: float = 0.006,
            noise: float = 0.007, seed: int = 0, start_price: float = 100.0, symbol: str = "VSHAPE") -> List[Candle]:
    """Sustained decline followed by a sustained recovery - a whipsaw stress test."""
    k = int(n * crash_frac)
    drift = np.concatenate([np.full(k, crash_drift), np.full(n - k, recovery_drift)])
    r = drift + _rng(seed).normal(0.0, noise, n)
    return candles_from_closes(_path(start_price, r), symbol, seed)


def flat(n: int = 200, price: float = 50.0, seed: int = 0, symbol: str = "FLAT") -> List[Candle]:
    """Essentially constant price with tiny noise - degenerate input every strategy must survive."""
    r = _rng(seed).normal(0.0, 1e-5, n)
    return candles_from_closes(price * np.exp(np.cumsum(r)), symbol, seed, intraday_vol=1e-5, gap_vol=1e-6)


def cointegrated_pair(
    n: int = 600, beta: float = 1.0, half_life: float = 10.0, trend_sigma: float = 0.010,
    spread_sigma: float = 0.015, seed: int = 0, price_a: float = 100.0, price_b: float = 80.0,
    symbols: Tuple[str, str] = ("PAIR_A", "PAIR_B"),
) -> Tuple[List[Candle], List[Candle]]:
    """Two series sharing one random-walk trend: log A = beta*trend + s, log B = trend (+ consts),
    where s is mean-reverting (OU, half-life ``half_life``) - so log A - beta*log B = s is
    stationary even though both legs wander. The classic pairs-trading setup (hedge ratio ``beta``)."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    rng = _rng(seed)
    trend = np.cumsum(rng.normal(0.0002, trend_sigma, n))
    rho = np.exp(-np.log(2.0) / half_life)
    s = np.empty(n)
    s[0] = 0.0
    eps = rng.normal(0.0, spread_sigma, n)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + eps[t]
    a = price_a * np.exp(beta * trend + s)
    b = price_b * np.exp(trend)
    return (
        candles_from_closes(a, symbols[0], seed),
        candles_from_closes(b, symbols[1], seed + 1),
    )


def scenario_suite(n: int = 500, seed: int = 0) -> Dict[str, List[Candle]]:
    """One series per market condition - the rows of every strategy x scenario matrix."""
    return {
        "uptrend": trending(n, 0.003, 0.008, seed),
        "downtrend": trending(n, -0.003, 0.008, seed + 1),
        "momentum": ar1_momentum(n, 0.25, 0.01, 0.0003, seed + 2),
        "mean_reverting": mean_reverting(n, 8.0, 0.012, seed + 3),
        "random_walk": gbm(n, 0.0003, 0.012, seed + 4),
        "regime_switching": regime_switching(n, seed + 5),
        "vol_clustering": garch(n, seed=seed + 6),
        "jumpy": jump_diffusion(n, seed=seed + 7),
        "range_breakout": range_then_breakout(n, seed=seed + 8),
        "cyclical": cyclical(n, seed=seed + 9),
        "v_shape": v_shape(n, seed=seed + 10),
    }
