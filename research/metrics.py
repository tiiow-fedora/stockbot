"""
Risk / performance metrics.

Pure numpy functions over an equity curve (or a returns array) plus a
``PerformanceReport`` bundle. Conventions:

* Returns are simple per-bar returns ``equity[t]/equity[t-1] - 1``.
* ``periods_per_year`` defaults to 252 (daily bars).
* ``risk_free`` is an ANNUAL rate (default 0); it is converted to a per-bar rate.
* **Undefined metrics are ``nan``**, never silently 0 or +inf (e.g. Sharpe of a
  flat curve, Sortino with no down bars, profit factor with no losing trades
  is ``inf`` only when there ARE winners). Callers/aggregators use ``np.nanmean`` etc.
* Drawdowns are reported as POSITIVE fractions (0.25 == a 25% peak-to-trough fall).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

NAN = float("nan")
TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def to_returns(equity: Sequence[float]) -> np.ndarray:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return np.array([], dtype=float)
    return eq[1:] / eq[:-1] - 1.0


def total_return(equity: Sequence[float]) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return NAN
    return float(eq[-1] / eq[0] - 1.0)


def cagr(equity: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate; NaN for < 2 points or non-positive endpoints."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return NAN
    years = (len(eq) - 1) / periods_per_year
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def annualized_vol(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return NAN
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe: mean(excess)/std(excess, ddof=1) * sqrt(ppy). NaN if std == 0."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return NAN
    excess = r - risk_free / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd < 1e-15:
        return NAN
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float], risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sortino using the target-semideviation over ALL bars:
    sqrt(mean(min(excess, 0)^2)). NaN if there is no downside at all."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return NAN
    excess = r - risk_free / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    if downside < 1e-15:
        return NAN
    return float(np.mean(excess) / downside * math.sqrt(periods_per_year))


def drawdown_series(equity: Sequence[float]) -> np.ndarray:
    """Per-bar drawdown as a positive fraction of the running peak."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return eq
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
    return dd


def max_drawdown(equity: Sequence[float]) -> float:
    dd = drawdown_series(equity)
    return float(dd.max()) if len(dd) else 0.0


def max_drawdown_duration(equity: Sequence[float]) -> int:
    """Longest run of consecutive bars spent below a previous equity peak."""
    dd = drawdown_series(equity)
    longest = current = 0
    for v in dd:
        current = current + 1 if v > 0 else 0
        longest = max(longest, current)
    return int(longest)


def ulcer_index(equity: Sequence[float]) -> float:
    dd = drawdown_series(equity)
    return float(np.sqrt(np.mean(dd**2))) if len(dd) else NAN


def calmar_ratio(equity: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    c = cagr(equity, periods_per_year)
    if mdd <= 1e-12 or not math.isfinite(c):
        return NAN
    return float(c / mdd)


def value_at_risk(returns: Sequence[float], level: float = 0.95) -> float:
    """Historical VaR as a POSITIVE loss fraction at ``level`` (0.95 -> 5th percentile)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return NAN
    return float(max(0.0, -np.percentile(r, (1.0 - level) * 100.0)))


def conditional_var(returns: Sequence[float], level: float = 0.95) -> float:
    """Expected shortfall: mean loss in the worst (1-level) tail, as a positive fraction."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return NAN
    cutoff = np.percentile(r, (1.0 - level) * 100.0)
    tail = r[r <= cutoff]
    return float(max(0.0, -tail.mean())) if len(tail) else NAN


def exposure(position_value: Sequence[float], equity: Sequence[float]) -> Dict[str, float]:
    """Market exposure from per-bar position value and equity.

    Returns ``time_in_market`` (fraction of bars with any position),
    ``avg_exposure`` (mean of position_value/equity over ALL bars, i.e. average
    capital deployed) and ``max_exposure``. All in [0, ~1] (can exceed 1 only with leverage).
    """
    pv = np.asarray(position_value, dtype=float)
    eq = np.asarray(equity, dtype=float)
    if len(pv) != len(eq):
        raise ValueError("position_value and equity must have the same length")
    if len(eq) == 0:
        return {"time_in_market": NAN, "avg_exposure": NAN, "max_exposure": NAN}
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(eq > 0, pv / eq, 0.0)
    return {
        "time_in_market": float(np.mean(pv > 1e-9)),
        "avg_exposure": float(np.mean(frac)),
        "max_exposure": float(np.max(frac)),
    }


def trade_stats(trade_pnls: Sequence[float]) -> Dict[str, float]:
    """Round-trip statistics from realised P&L per closed trade."""
    p = np.asarray(trade_pnls, dtype=float)
    n = len(p)
    if n == 0:
        return {"num_trades": 0, "win_rate": NAN, "profit_factor": NAN, "avg_trade": NAN,
                "avg_win": NAN, "avg_loss": NAN, "expectancy_ratio": NAN, "best_trade": NAN, "worst_trade": NAN}
    wins, losses = p[p > 0], p[p <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    if gross_loss > 0:
        pf = float(gross_win / gross_loss)
    else:
        pf = float("inf") if gross_win > 0 else NAN
    avg_win = float(wins.mean()) if len(wins) else NAN
    avg_loss = float(losses.mean()) if len(losses) else NAN
    exp_ratio = NAN
    if len(wins) and len(losses) and avg_loss != 0:
        exp_ratio = float(avg_win / abs(avg_loss))
    return {
        "num_trades": int(n), "win_rate": float(len(wins) / n), "profit_factor": pf,
        "avg_trade": float(p.mean()), "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy_ratio": exp_ratio, "best_trade": float(p.max()), "worst_trade": float(p.min()),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    total_return: float = NAN
    cagr: float = NAN
    annual_vol: float = NAN
    sharpe: float = NAN
    sortino: float = NAN
    calmar: float = NAN
    max_drawdown: float = NAN
    max_dd_duration: int = 0
    ulcer_index: float = NAN
    var_95: float = NAN
    cvar_95: float = NAN
    time_in_market: float = NAN
    avg_exposure: float = NAN
    max_exposure: float = NAN
    num_trades: int = 0
    win_rate: float = NAN
    profit_factor: float = NAN
    avg_trade: float = NAN
    expectancy_ratio: float = NAN
    best_trade: float = NAN
    worst_trade: float = NAN
    n_bars: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        f = lambda v, p=2: "inf" if v == float("inf") else ("n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{p}f}")
        return (
            f"ret={f(self.total_return * 100, 1)}% cagr={f(self.cagr * 100, 1)}% sharpe={f(self.sharpe)} "
            f"sortino={f(self.sortino)} maxDD={f(self.max_drawdown * 100, 1)}% exposure={f(self.avg_exposure * 100, 0)}% "
            f"trades={self.num_trades} win={f(self.win_rate * 100, 0)}% pf={f(self.profit_factor)}"
        )


def compute_report(
    equity: Sequence[float],
    position_value: Optional[Sequence[float]] = None,
    trade_pnls: Optional[Sequence[float]] = None,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> PerformanceReport:
    """Build a full ``PerformanceReport`` from an equity curve (+ optional exposure / trade data)."""
    eq = np.asarray(equity, dtype=float)
    rets = to_returns(eq)
    rep = PerformanceReport(
        total_return=total_return(eq),
        cagr=cagr(eq, periods_per_year),
        annual_vol=annualized_vol(rets, periods_per_year),
        sharpe=sharpe_ratio(rets, risk_free, periods_per_year),
        sortino=sortino_ratio(rets, risk_free, periods_per_year),
        calmar=calmar_ratio(eq, periods_per_year),
        max_drawdown=max_drawdown(eq),
        max_dd_duration=max_drawdown_duration(eq),
        ulcer_index=ulcer_index(eq),
        var_95=value_at_risk(rets, 0.95),
        cvar_95=conditional_var(rets, 0.95),
        n_bars=len(eq),
    )
    if position_value is not None:
        e = exposure(position_value, eq)
        rep.time_in_market, rep.avg_exposure, rep.max_exposure = e["time_in_market"], e["avg_exposure"], e["max_exposure"]
    if trade_pnls is not None:
        t = trade_stats(trade_pnls)
        for k, v in t.items():
            setattr(rep, k, v)
    return rep
