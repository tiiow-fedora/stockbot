"""Order-path chaos: timeouts, 5xx, rejections, partial fills, ambiguous
outcomes - against the Alpaca mirror executor AND the generic route_order
pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from core.config import ExecutionConfig, RiskConfig
from core.models import Fill, Order, OrderSide
from execution.alpaca_mirror import AlpacaMirrorExecutor
from execution.base import OrderExecutor
from execution.order_builder import route_order
from execution.paper import PaperExecutor
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config
from tests.chaos.fake_alpaca import ConnectionDrop, HTTPError, PartialFills, Timeout

NOW = datetime(2024, 3, 12, 15, 0, tzinfo=timezone.utc)
TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


def buy(qty=10, symbol="AAPL"):
    return Order(symbol, OrderSide.BUY, qty, timestamp=NOW, reason="t")


def sell(qty=10, symbol="AAPL"):
    return Order(symbol, OrderSide.SELL, qty, timestamp=NOW, reason="t")


@pytest.fixture
def mirror(broker):
    broker.prices["AAPL"] = 100.0
    return AlpacaMirrorExecutor(PaperExecutor(ExecutionConfig(slippage_bps=0)))


# -- mirror: local fill is authoritative whatever the broker does --------------------

@pytest.mark.parametrize(
    "fault", [Timeout(), ConnectionDrop(), HTTPError(500), HTTPError(504), HTTPError(429)],
)
def test_transport_fault_never_loses_or_raises_the_local_fill(mirror, broker, fault):
    broker.fail_next("submit_order", fault)
    fill = mirror.execute(buy(), 100.0, NOW)
    assert fill is not None and fill.quantity == 10
    assert not broker.positions          # broker never saw it -> known drift


def test_broker_rejection_never_loses_the_local_fill(mirror, broker):
    broker.reject_next("insufficient buying power")
    assert mirror.execute(buy(), 100.0, NOW).quantity == 10


def test_ambiguous_timeout_leaves_broker_and_local_in_agreement(mirror, broker):
    """Response lost AFTER Alpaca accepted the order: the mirror logs a failure
    but the order exists on both sides, so the books still agree."""
    broker.fail_next("submit_order", Timeout(applied=True))
    fill = mirror.execute(buy(), 100.0, NOW)
    assert fill.quantity == 10 and broker.positions["AAPL"]["qty"] == 10


def test_mirrored_order_carries_an_idempotency_key(mirror, broker):
    """BUG-3 (fixed): no client_order_id meant alpaca-py's built-in 504/429
    retry of POST /orders could place the order twice."""
    mirror.execute(buy(), 100.0, NOW)
    (_, _, _, client_id), = broker.calls_to("submit_order")
    assert client_id


def test_remirroring_the_same_order_is_deduplicated_by_the_broker(mirror, broker):
    o = buy()
    mirror.execute(o, 100.0, NOW)
    mirror._mirror(o)   # what a retry layer (ours or alpaca-py's) would do
    assert broker.positions["AAPL"]["qty"] == 10   # not 20


def test_distinct_orders_get_distinct_keys(mirror, broker):
    mirror.execute(buy(), 100.0, NOW)
    mirror.execute(buy(), 100.0, NOW)
    assert broker.positions["AAPL"]["qty"] == 20


def test_known_drift_failed_buy_mirror_then_sell_mirror_is_rejected(mirror, broker):
    """Documented limitation (see module docstring of alpaca_mirror.py): the
    mirror is best-effort, so books drift. Pinned here so nobody 'fixes' it by
    accident without also adding reconciliation."""
    broker.fail_next("submit_order", Timeout())
    mirror.execute(buy(), 100.0, NOW)
    mirror.execute(sell(), 100.0, NOW)       # local: flat. broker: SELL w/o position -> rejected
    assert not broker.positions
    assert broker.orders["ord-1"].status == "rejected"


def test_known_drift_broker_partial_fill_is_invisible_locally(mirror, broker):
    broker.fill_mode = PartialFills(0.5)
    fill = mirror.execute(buy(10), 100.0, NOW)
    assert fill.quantity == 10                       # local believes it got everything
    assert broker.positions["AAPL"]["qty"] == 5      # broker only gave half


def test_mirror_only_ever_builds_paper_clients(broker, monkeypatch):
    AlpacaMirrorExecutor(PaperExecutor(ExecutionConfig()))
    from alpaca.trading.client import TradingClient

    with pytest.raises(AssertionError):
        TradingClient("k", "s", paper=False)
    broker.live_attempts = 0


def test_real_alpaca_clients_get_a_request_timeout(monkeypatch):
    """BUG-4 (fixed): alpaca-py never passes timeout= to requests, so a silent
    socket would hang the live loop forever."""
    import requests

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    captured = {}

    def fake_request(self, method, url, **kw):
        captured.update(kw)
        raise requests.exceptions.ReadTimeout("stop here")

    monkeypatch.setattr(requests.Session, "request", fake_request)
    mirror = AlpacaMirrorExecutor(PaperExecutor(ExecutionConfig()))
    with pytest.raises(requests.exceptions.ReadTimeout):
        mirror._client.get_clock()
    assert captured.get("timeout"), "no timeout on Alpaca trading requests"

    captured.clear()
    from data.market_calendar import AlpacaMarketCalendar
    from data.providers.alpaca import AlpacaMarketDataProvider

    with pytest.raises(requests.exceptions.ReadTimeout):
        AlpacaMarketCalendar().is_open(NOW)
    assert captured.get("timeout"), "no timeout on calendar requests"
    captured.clear()
    with pytest.raises(Exception):
        AlpacaMarketDataProvider().get_snapshot("AAPL")
    assert captured.get("timeout"), "no timeout on market-data requests"


# -- route_order with misbehaving executors ----------------------------------------------

class ScriptedExecutor(OrderExecutor):
    """Wraps PaperExecutor with a scripted misbehaviour."""

    def __init__(self, inner: PaperExecutor, mode: str = "ok", fraction: float = 0.5):
        self.inner, self.mode, self.fraction = inner, mode, fraction
        self.calls = 0

    def execute(self, order, market_price, timestamp) -> Optional[Fill]:
        self.calls += 1
        if self.mode == "raise":
            raise TimeoutError("executor timed out")
        if self.mode == "none":
            return None
        fill = self.inner.execute(order, market_price, timestamp)
        if self.mode == "partial":
            fill.quantity = float(int(order.quantity * self.fraction))
        return fill


def make_pipeline(db, mode="ok", **risk):
    return (
        RiskManager(RiskConfig(**risk), db=db),
        Portfolio(100_000.0),
        ScriptedExecutor(PaperExecutor(ExecutionConfig(slippage_bps=5)), mode),
    )


def test_partial_fill_updates_portfolio_with_filled_quantity_only(db):
    rm, pf, ex = make_pipeline(db, "partial")
    fill = route_order(buy(100), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert fill.quantity == 50
    assert pf.position_qty("AAPL") == 50
    assert pf.cash == pytest.approx(100_000 - fill.fill_price * 50)
    assert db.get_recent_trades()[0]["quantity"] == 50


def test_partial_sell_leaves_remainder_that_a_second_sell_clears(db):
    rm, pf, ex = make_pipeline(db)
    route_order(buy(100), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    ex.mode = "partial"
    route_order(sell(100), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert pf.position_qty("AAPL") == 50
    ex.mode = "ok"
    route_order(sell(50), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert pf.position_qty("AAPL") == 0 and "AAPL" not in pf.positions


def test_unfilled_order_changes_nothing_and_leaves_no_trade_row(db):
    rm, pf, ex = make_pipeline(db, "none")
    assert route_order(buy(10), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db) is None
    assert pf.cash == 100_000 and not pf.positions
    assert db.get_recent_trades() == []


def test_executor_exception_propagates_without_touching_the_ledger(db):
    rm, pf, ex = make_pipeline(db, "raise")
    with pytest.raises(TimeoutError):
        route_order(buy(10), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert pf.cash == 100_000 and not pf.positions and db.get_recent_trades() == []


def test_failed_attempts_do_not_burn_the_daily_trade_allowance(db):
    """With a DB-backed RiskManager the limit counts FILLS, so repeated
    executor failures must not lock the symbol out for the day."""
    rm, pf, ex = make_pipeline(db, "raise", max_trades_per_symbol_per_day=1)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            route_order(buy(10), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    ex.mode = "ok"
    assert route_order(buy(10), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db) is not None


@pytest.mark.xfail(strict=True, reason="BUG-7: fill applied to broker before ledger affordability is checked "
                                       "(only reachable with min_cash_reserve_pct=0) - see tests/chaos/README.md")
def test_slippage_cannot_push_a_fill_beyond_available_cash(db):
    rm, pf, ex = make_pipeline(db, max_position_size_pct=1.0, max_portfolio_exposure_pct=1.0,
                               min_cash_reserve_pct=0.0)
    order = buy(1000)  # 1000 * $100 == all the cash; 5 bps slippage makes the fill unaffordable
    route_order(order, rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert ex.calls == 0 or pf.cash >= 0


def test_default_reserve_absorbs_adverse_slippage(db):
    ex = ScriptedExecutor(PaperExecutor(ExecutionConfig(slippage_bps=500, commission_per_trade=5)))
    rm, pf = RiskManager(RiskConfig(), db=db), Portfolio(100_000.0)
    qty = rm.size_position("AAPL", 100.0, pf, {"AAPL": 100.0}, 1.0)
    route_order(buy(qty), rm, pf, ex, {"AAPL": 100.0}, 100.0, NOW, db)
    assert pf.cash >= 0


# -- full cycle: executor failures --------------------------------------------------------

class FlakyExecutor(OrderExecutor):
    def __init__(self, inner, fail_sells=False, fail_all=False):
        self.inner, self.fail_sells, self.fail_all = inner, fail_sells, fail_all

    def execute(self, order, price, ts):
        if self.fail_all or (self.fail_sells and order.side == OrderSide.SELL):
            raise ConnectionError("executor down")
        return self.inner.execute(order, price, ts)


def test_failing_stop_loss_exit_does_not_wedge_the_rest_of_the_cycle(broker, db_url, monkeypatch):
    """BUG-8 (fixed): the stop-loss loop in _run_cycle was unguarded, so one
    failing exit aborted the cycle - no other exits, no decisions, no portfolio
    snapshot - and repeated every cycle for as long as the failure lasted."""
    patch_clock(monkeypatch, broker.clock)
    for sym in ("AAPL", "MSFT"):
        broker.set_bars_from_candles(sym, dip_candles(sym, end=TODAY_BAR))
    cfg = make_config(db_url, symbols=("AAPL", "MSFT"))
    bot = Bot(cfg)
    bot.cycle()
    assert set(bot.portfolio.positions) == {"AAPL", "MSFT"}

    # AAPL craters (-30%) and exits start failing; MSFT is unchanged.
    broker.bars["AAPL"] = [b for b in broker.bars["AAPL"]]
    last = broker.bars["AAPL"][-1]
    last.close = last.low = last.high = last.open = last.close * 0.7
    bot.executor = FlakyExecutor(bot.executor, fail_sells=True)
    snaps_before = len(bot.db.get_portfolio_history())
    bot.cycle()                                    # must not raise
    assert len(bot.db.get_portfolio_history()) == snaps_before + 1   # cycle completed


def test_executor_outage_on_buys_is_contained_per_symbol(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    for sym in ("AAPL", "MSFT"):
        broker.set_bars_from_candles(sym, dip_candles(sym, end=TODAY_BAR))
    bot = Bot(make_config(db_url, symbols=("AAPL", "MSFT")))
    bot.executor = FlakyExecutor(bot.executor, fail_all=True)
    bot.cycle()
    assert not bot.portfolio.positions and bot.portfolio.cash == 100_000
    assert len(bot.db.get_recent_decisions()) == 2     # both symbols still evaluated
    bot.executor = bot.executor.inner                  # outage over
    bot.cycle()
    assert set(bot.portfolio.positions) == {"AAPL", "MSFT"}
