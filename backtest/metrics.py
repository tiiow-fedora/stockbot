"""Extra performance statistics derived from an engine `BacktestResult`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple


@dataclass
class TradeStats:
    num_round_trips: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float          # positive number
    net_pnl: float
    profit_factor: Optional[float]  # None when there are no losses
    avg_win: float
    avg_loss: float            # positive number
    best_trade: float
    worst_trade: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_trade_stats(trade_log: Sequence[dict]) -> TradeStats:
    """Stats over realized SELL fills (each SELL = one closed/partial trade)."""
    pnls = [t["realized_pnl"] for t in trade_log if t.get("side") == "SELL" and t.get("realized_pnl") is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    n = len(pnls)
    return TradeStats(
        num_round_trips=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / n if n else 0.0,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=sum(pnls),
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else None,
        avg_win=gross_profit / len(wins) if wins else 0.0,
        avg_loss=gross_loss / len(losses) if losses else 0.0,
        best_trade=max(pnls) if pnls else 0.0,
        worst_trade=min(pnls) if pnls else 0.0,
    )


def max_drawdown_detail(curve: Sequence[Tuple[datetime, float]]) -> dict:
    """Max drawdown as a fraction and in dollars, plus peak/trough timestamps."""
    if not curve:
        return {"max_drawdown_pct": 0.0, "max_drawdown_abs": 0.0, "peak_time": None, "trough_time": None}
    peak_v, peak_t = curve[0][1], curve[0][0]
    best = {"max_drawdown_pct": 0.0, "max_drawdown_abs": 0.0, "peak_time": None, "trough_time": None}
    for t, v in curve:
        if v > peak_v:
            peak_v, peak_t = v, t
        if peak_v > 0:
            dd = (peak_v - v) / peak_v
            if dd > best["max_drawdown_pct"]:
                best = {
                    "max_drawdown_pct": dd,
                    "max_drawdown_abs": peak_v - v,
                    "peak_time": peak_t,
                    "trough_time": t,
                }
    return best


def combine_equity_curves(curves: List[List[Tuple[datetime, float]]]) -> List[Tuple[datetime, float]]:
    """Sum several equal-capital runs into one aggregate curve (by timestamp)."""
    totals = {}
    for curve in curves:
        for t, v in curve:
            totals[t] = totals.get(t, 0.0) + v
    # Only keep timestamps present in every run so partial overlaps don't
    # create artificial cliffs.
    counts = {}
    for curve in curves:
        for t, _ in curve:
            counts[t] = counts.get(t, 0) + 1
    return [(t, totals[t]) for t in sorted(totals) if counts[t] == len(curves)]
