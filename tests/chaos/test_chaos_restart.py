"""Process restarts - including crashes in the middle of an order."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config
from tests.chaos.fake_alpaca import FakeAlpacaBroker
from tests.chaos.loop_driver import run_loop

TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


def cfg(db_url, **risk):
    c = make_config(db_url, **risk)
    c.execution.mirror_to_alpaca = True
    return c


class ProcessDied(BaseException):
    """Like SIGKILL: not an Exception, so no `except Exception` can absorb it."""


@pytest.fixture
def seeded(broker, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    return broker


@pytest.mark.xfail(strict=True, reason="BUG-9: live loop starts with a fresh Portfolio(starting_balance) and never "
                                       "restores cash/positions from the DB or broker - see tests/chaos/README.md")
def test_restart_restores_open_positions(seeded, db_url, monkeypatch):
    first = Bot(cfg(db_url))
    first.cycle()
    held, cash = dict(first.portfolio.positions), first.portfolio.cash
    assert held

    seeded.market_open = False       # restart overnight: heartbeat only, no trading to mask the loss
    out = run_loop(monkeypatch, seeded, cfg(db_url), iterations=1)
    assert out["portfolio"].positions.keys() == held.keys()
    assert out["portfolio"].cash == pytest.approx(cash)


@pytest.mark.xfail(strict=True, reason="BUG-9: after a restart the bot has forgotten its position, so the signal "
                                       "that already bought re-buys on top of it (broker ends up with 2x)")
def test_restart_does_not_double_the_position(seeded, db_url):
    first = Bot(cfg(db_url))
    first.cycle()
    after_first = seeded.positions["AAPL"]["qty"]

    second = Bot(cfg(db_url))          # same DB, same broker, empty memory
    second.cycle()
    assert seeded.positions["AAPL"]["qty"] == after_first


def test_daily_trade_limit_survives_restart(seeded, db_url):
    """The one piece of state that IS durable (counted from fills in the DB)."""
    Bot(cfg(db_url, max_trades_per_symbol_per_day=1)).cycle()
    qty = seeded.positions["AAPL"]["qty"]
    for _ in range(3):
        Bot(cfg(db_url, max_trades_per_symbol_per_day=1)).cycle()
    assert seeded.positions["AAPL"]["qty"] == qty


@pytest.mark.parametrize("crash_point", ["after_broker_submit", "after_db_order_row"])
def test_crash_mid_order_never_leaves_a_half_written_trade(seeded, db_url, crash_point):
    """Kill the process between the broker submit and the DB write (or between
    the order row and its fill row). Whatever happened, the DB must not claim a
    fill that does not exist, and a restarted bot must run cleanly."""
    bot = Bot(cfg(db_url))
    if crash_point == "after_db_order_row":
        real = bot.db.save_fill
        bot.db.save_fill = lambda *a, **k: (_ for _ in ()).throw(ProcessDied())
    else:
        real = bot.db.save_order
        bot.db.save_order = lambda *a, **k: (_ for _ in ()).throw(ProcessDied())
    with pytest.raises(ProcessDied):
        bot.cycle()

    assert seeded.positions["AAPL"]["qty"] > 0                       # broker did get the order
    trades = Bot(cfg(db_url)).db.get_recent_trades()
    assert trades == []                                              # no fill row without a completed write
    Bot(cfg(db_url)).cycle()                                         # restart is clean (see BUG-9 re: position)


def test_orphaned_order_row_is_not_counted_as_a_trade(seeded, db_url):
    """If we die after save_order but before save_fill, the order row stays
    'submitted' forever; it must not count against the daily trade limit
    (limit counts fills) nor show up as a trade."""
    bot = Bot(cfg(db_url))
    bot.db.save_fill = lambda *a, **k: (_ for _ in ()).throw(ProcessDied())
    with pytest.raises(ProcessDied):
        bot.cycle()
    fresh = Bot(cfg(db_url))
    assert fresh.db.count_trades_today("AAPL", seeded.clock.true_now) == 0


def test_known_drift_db_failure_after_fill_leaves_ledger_ahead_of_db(seeded, db_url):
    """Documented (BUG-10): route_order mutates the in-memory ledger BEFORE the
    DB writes. A transient DB error afterwards leaves memory (and the broker)
    ahead of the DB: the position exists, the audit trail doesn't."""
    bot = Bot(cfg(db_url))
    bot.db.save_order = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database is locked"))
    bot.cycle()                                   # error is contained per-symbol
    assert bot.portfolio.position_qty("AAPL") > 0
    assert bot.db.get_recent_trades() == []
