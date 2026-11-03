"""Helpers shared by research tests (import as ``from research.tests.helpers import ...``)."""

from __future__ import annotations

from datetime import timedelta
from typing import List, Optional, Sequence

from core.models import Action, Candle, Decision, MarketSnapshot
from research.base import ResearchStrategy

CASH = 100_000.0


def snapshot_of(candle: Candle) -> MarketSnapshot:
    """The snapshot the Backtester builds from a decision bar."""
    return MarketSnapshot(
        symbol=candle.symbol, timestamp=candle.timestamp, price=candle.close, volume=candle.volume,
        open=candle.open, high=candle.high, low=candle.low, close=candle.close,
    )


def decide(strategy: ResearchStrategy, candles: Sequence[Candle], i: int, in_position: bool = False) -> Decision:
    """Decision a backtester would request at bar ``i`` (bar i = decision bar, history = candles[:i])."""
    c = candles[i]
    return strategy.generate_decision(
        symbol=c.symbol, snapshot=snapshot_of(c), history=list(candles[:i]), news=[],
        current_position_qty=10.0 if in_position else 0.0, cash=CASH, portfolio_value=CASH,
    )


def decisions(strategy: ResearchStrategy, candles: Sequence[Candle], in_position: bool = False,
              start: Optional[int] = None) -> List[Decision]:
    start = strategy.min_bars - 1 if start is None else start
    return [decide(strategy, candles, i, in_position) for i in range(start, len(candles))]


def actions(strategy: ResearchStrategy, candles: Sequence[Candle], in_position: bool = False) -> List[Action]:
    return [d.action for d in decisions(strategy, candles, in_position)]


def simulate_long_flat(strategy: ResearchStrategy, candles: Sequence[Candle]):
    """Cheap signal-only simulation (no engine): follow the strategy's BUY/SELL toggles at the
    decision bar's close. Returns (list of (entry_i, exit_i, return)), final_position_open."""
    pos, entry, trades = False, None, []
    for i in range(strategy.min_bars - 1, len(candles)):
        d = decide(strategy, candles, i, pos)
        if d.action == Action.BUY and not pos:
            pos, entry = True, i
        elif d.action == Action.SELL and pos:
            trades.append((entry, i, candles[i].close / candles[entry].close - 1.0))
            pos = False
    return trades, pos
