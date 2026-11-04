"""Seeded random chaos soak: data jumps, gaps, outages, executor faults,
partial fills, rejections, clock steps - checking ledger invariants after
every cycle. Deterministic per seed, so a failure is replayable with
`pytest tests/chaos/test_chaos_fuzz.py -k seed7`."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.models import OrderSide
from execution.base import OrderExecutor
from portfolio.portfolio import Portfolio
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config
from tests.chaos.fake_alpaca import ConnectionDrop, HTTPError, PartialFills, FullFills, Timeout

SYMBOLS = ("AAPL", "MSFT", "NVDA", "JPM")
CYCLES = 45
TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


class ChaosExecutor(OrderExecutor):
    def __init__(self, inner, rng):
        self.inner, self.rng = inner, rng

    def execute(self, order, price, ts):
        r = self.rng.random()
        if r < 0.10:
            raise TimeoutError("executor timeout")
        if r < 0.18:
            return None
        fill = self.inner.execute(order, price, ts)
        if r < 0.30 and order.quantity > 1:
            fill.quantity = float(max(1, int(order.quantity * self.rng.uniform(0.2, 0.9))))
        return fill


def _mutate_market(broker, rng, day_roll):
    for sym in SYMBOLS:
        bars = broker.bars[sym]
        last = bars[-1]
        move = rng.gauss(0, 0.01)
        if rng.random() < 0.08:
            move += rng.choice([-0.20, -0.12, 0.15, 0.25])      # gap / halt-resume style jump
        close = max(1.0, last.close * (1 + move))
        vol = 0 if rng.random() < 0.05 else int(last.volume * rng.uniform(0.6, 1.6))
        if day_roll:
            ts = last.timestamp + timedelta(days=rng.choice([1, 1, 1, 3, 5]))    # incl. long gaps
            bars.append(SimpleNamespace(symbol=sym, timestamp=ts, open=last.close, high=max(last.close, close),
                                        low=min(last.close, close), close=close, volume=max(vol, 1)))
        else:
            last.close = close
            last.high, last.low = max(last.high, close), min(last.low, close)
            last.volume = vol
        broker.prices[sym] = close


def _inject_faults(broker, rng):
    for method in ("get_stock_bars", "get_clock"):
        if rng.random() < 0.12:
            broker.fail_next(method, rng.choice([Timeout(), ConnectionDrop(), HTTPError(503), HTTPError(429)]))
    if rng.random() < 0.15:
        broker.fail_next("submit_order", rng.choice([Timeout(), Timeout(applied=True), HTTPError(504), ConnectionDrop()]))
    if rng.random() < 0.10:
        broker.reject_next()
    broker.fill_mode = PartialFills(rng.uniform(0.2, 0.8)) if rng.random() < 0.25 else FullFills()
    if rng.random() < 0.05:
        broker.clock.local_skew = timedelta(seconds=rng.randint(-600, 600))


def _assert_invariants(bot, cfg):
    pf = bot.portfolio
    assert pf.cash >= -1e-6, f"negative cash {pf.cash}"
    assert len(pf.positions) <= cfg.risk.max_open_positions
    for sym, pos in pf.positions.items():
        assert pos.quantity > 0 and pos.avg_entry_price > 0, (sym, pos)

    # ledger <-> DB fills (order-independent replay)
    net, cash = {}, cfg.starting_balance
    for t in bot.db.get_recent_trades(limit=10_000):
        sign = 1 if t["side"] == "BUY" else -1
        net[t["symbol"]] = net.get(t["symbol"], 0.0) + sign * t["quantity"]
        cash -= sign * t["quantity"] * t["price"] + t["commission"]
    for sym in SYMBOLS:
        assert net.get(sym, 0.0) == pytest.approx(pf.position_qty(sym), abs=1e-6), f"{sym}: DB net != ledger"
        assert net.get(sym, 0.0) >= -1e-6
    assert cash == pytest.approx(pf.cash, abs=1e-3)


@pytest.mark.parametrize("seed", range(15), ids=lambda s: f"seed{s}")
def test_random_chaos_soak_keeps_ledger_sane(broker, db_url, monkeypatch, seed):
    rng = random.Random(seed)
    patch_clock(monkeypatch, broker.clock)
    # Candle persistence is O(bars) queries per call (see README: perf note) and
    # irrelevant to the ledger invariants checked here; stub it to keep the soak fast.
    monkeypatch.setattr("database.database.DatabaseManager.save_candles", lambda self, candles: None)
    for sym in SYMBOLS:
        broker.set_bars_from_candles(sym, dip_candles(sym, end=TODAY_BAR, crash=rng.choice([0.0, 0.05, 0.12])))
    cfg = make_config(db_url, symbols=SYMBOLS)
    cfg.execution.mirror_to_alpaca = True
    bot = Bot(cfg, executor_wrapper=lambda inner: ChaosExecutor(inner, rng))

    for i in range(CYCLES):
        _mutate_market(broker, rng, day_roll=(i % 9 == 8))
        _inject_faults(broker, rng)
        if i % 9 == 8:
            broker.clock.advance(days=1)
        else:
            broker.clock.advance(seconds=60)
        bot.cycle()                       # must never raise
        _assert_invariants(bot, cfg)
