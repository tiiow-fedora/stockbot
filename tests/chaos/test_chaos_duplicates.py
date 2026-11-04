"""Duplicate / repeated / concurrent signals."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.config import ExecutionConfig
from execution.manual import ManualExecutor
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config

TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


def cfg(db_url, symbols=("AAPL",), **risk):
    c = make_config(db_url, symbols=symbols, **risk)
    c.execution.mirror_to_alpaca = True
    return c


@pytest.fixture
def seeded(broker, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    return broker


def test_same_buy_signal_every_cycle_respects_position_cap_and_daily_limit(seeded, db_url):
    bot = Bot(cfg(db_url, max_trades_per_symbol_per_day=3))
    for _ in range(40):
        bot.cycle()
        seeded.clock.advance(seconds=60)
    prices = {"AAPL": seeded.prices["AAPL"]}
    total = bot.portfolio.total_value(prices)
    assert bot.portfolio.positions["AAPL"].quantity * prices["AAPL"] <= total * 0.20 + prices["AAPL"]
    assert len(bot.db.get_recent_trades()) <= 3
    assert seeded.positions["AAPL"]["qty"] == bot.portfolio.position_qty("AAPL")


def test_duplicate_symbols_in_config_do_not_double_trade(seeded, db_url):
    once = Bot(cfg(db_url + "x", symbols=("AAPL",)))
    once.cycle()
    solo_qty = once.portfolio.position_qty("AAPL")

    seeded2 = Bot(cfg(db_url, symbols=("AAPL", "AAPL", "AAPL")))
    seeded2.cycle()
    # position cap applies to the symbol, not to each mention of it
    assert seeded2.portfolio.position_qty("AAPL") <= solo_qty


def test_stop_loss_and_strategy_sell_in_one_cycle_never_oversell(seeded, db_url):
    bot = Bot(cfg(db_url))
    bot.cycle()
    last = seeded.bars["AAPL"][-1]
    last.close = last.high = last.low = last.open = last.close * 0.6   # triggers stop-loss AND stays oversold
    bot.cycle()
    for pos in bot.portfolio.positions.values():
        assert pos.quantity > 0
    assert bot.portfolio.cash >= 0
    assert seeded.positions.get("AAPL", {}).get("qty", 0) >= 0


@pytest.mark.xfail(strict=True, reason="BUG-13: a stop-loss exit is followed by same-cycle re-entry on the strategy's "
                                       "BUY (no cooldown) - realises the loss then buys the knife again (README)")
def test_no_same_day_reentry_right_after_a_stop_loss(seeded, db_url):
    bot = Bot(cfg(db_url))
    bot.cycle()
    last = seeded.bars["AAPL"][-1]
    last.close = last.high = last.low = last.open = last.close * 0.6
    bot.cycle()
    trades = list(reversed(bot.db.get_recent_trades()))
    sides = [t["side"] for t in trades]
    assert "SELL" in sides
    assert sides[sides.index("SELL"):].count("BUY") == 0


@pytest.mark.xfail(strict=True, reason="BUG-14: two bot processes (deploy overlap / accidental second container) each "
                                       "keep a private Portfolio, so together they exceed max_position_size_pct (README)")
def test_two_concurrent_processes_cannot_exceed_the_position_cap(seeded, db_url):
    a, b = Bot(cfg(db_url)), Bot(cfg(db_url))
    a.cycle()
    b.cycle()
    prices = {"AAPL": seeded.prices["AAPL"]}
    broker_value = seeded.positions["AAPL"]["qty"] * prices["AAPL"]
    assert broker_value <= 0.20 * 100_000 * 1.01


def test_manual_executor_never_mutates_ledger_or_db(seeded, db_url):
    c = cfg(db_url)
    c.execution = ExecutionConfig(mode="manual")
    bot = Bot(c)
    for _ in range(5):
        bot.cycle()
    assert isinstance(bot.executor, ManualExecutor)
    assert not bot.portfolio.positions and bot.portfolio.cash == 100_000
    assert bot.db.get_recent_trades() == [] and not seeded.orders
