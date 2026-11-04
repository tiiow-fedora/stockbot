"""Stale / gappy / corrupt / failing market data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.models import Action, Candle, MarketSnapshot
from data.base import MarketDataError
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config
from tests.chaos.fake_alpaca import ConnectionDrop, HTTPError, Timeout

TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)  # broker clock is Tue 15:00Z


def provider():
    from data.providers.alpaca import AlpacaMarketDataProvider

    return AlpacaMarketDataProvider()


def bar(ts, close=100.0, volume=1_000_000, **kw):
    return SimpleNamespace(symbol="AAPL", timestamp=ts, open=kw.get("open", close), high=kw.get("high", close),
                           low=kw.get("low", close), close=close, volume=volume)


# -- transport failures --------------------------------------------------------

@pytest.mark.parametrize("fault", [Timeout(), ConnectionDrop(), HTTPError(500), HTTPError(429), HTTPError(504)])
def test_transport_faults_become_market_data_error(broker, monkeypatch, fault):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    broker.fail_next("get_stock_bars", fault)
    with pytest.raises(MarketDataError):
        provider().get_snapshot("AAPL")


def test_empty_response_raises_market_data_error_not_keyerror(broker, monkeypatch):
    """BUG-2 (fixed): alpaca-py's BarSet raises KeyError for a symbol with no
    bars (delisted, halted all week, bad ticker). It used to leak out of
    get_historical_candles as a bare KeyError."""
    patch_clock(monkeypatch, broker.clock)
    assert provider().get_historical_candles(
        "AAPL", broker.clock.true_now - timedelta(days=30), broker.clock.true_now) == []
    with pytest.raises(MarketDataError):
        provider().get_snapshot("AAPL")


def test_one_corrupt_bar_does_not_discard_the_whole_history(broker):
    """A single malformed bar (high<low, zero price, NaN) used to raise out of
    get_historical_candles and kill the symbol's entire history."""
    good = dip_candles(end=TODAY_BAR)
    broker.set_bars_from_candles("AAPL", good)
    broker.bars["AAPL"].insert(40, bar(good[40].timestamp + timedelta(hours=1), close=0.0))
    broker.bars["AAPL"].insert(41, bar(good[41].timestamp + timedelta(hours=1), close=float("nan")))
    out = provider().get_historical_candles("AAPL", broker.clock.true_now - timedelta(days=200), broker.clock.true_now)
    assert len(out) == len(good)


# -- NaN / inf must never become a tradable price ----------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_prices_are_rejected_by_core_models(bad):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        MarketSnapshot("AAPL", ts, bad, 100)
    with pytest.raises(ValueError):
        Candle("AAPL", ts, bad, bad, bad, bad, 100)


# -- stale data --------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="BUG-5: no staleness check on snapshot bar age (see tests/chaos/README.md)")
def test_snapshot_built_from_a_very_stale_bar_is_refused(broker, monkeypatch):
    """BUG-5 (documented, unfixed): get_snapshot() stamps `now` on whatever the
    latest bar is. If the feed stalls (or a symbol is halted) for the whole
    5-day lookback, the 5-day-old close is returned as if current, and the bot
    will trade on it at an open market."""
    patch_clock(monkeypatch, broker.clock)
    stale_end = broker.clock.true_now - timedelta(days=5)
    broker.set_bars_from_candles("AAPL", dip_candles(end=stale_end))
    with pytest.raises(MarketDataError):
        provider().get_snapshot("AAPL")


def test_weekend_gap_is_not_treated_as_stale(broker, monkeypatch):
    patch_clock(monkeypatch, broker.clock)  # Tuesday
    broker.set_bars_from_candles("AAPL", dip_candles(end=datetime(2024, 3, 8, 5, 0, tzinfo=timezone.utc)))  # Fri bar
    assert provider().get_snapshot("AAPL").price > 0


# -- gappy data -------------------------------------------------------------------

def _decide(candles, price=None):
    from core.config import StrategyConfig
    from strategy.registry import build_strategy_by_name

    s = build_strategy_by_name("mean_reversion", StrategyConfig(name="mean_reversion"))
    price = price or candles[-1].close
    snap = MarketSnapshot("AAPL", candles[-1].timestamp + timedelta(days=1), price, 1_000_000)
    return s.generate_decision("AAPL", snap, candles, [], 0, 100_000, 100_000)


def test_short_history_holds_instead_of_trading():
    assert _decide(dip_candles(n=80)[-10:]).action == Action.HOLD


def test_history_with_multi_week_hole_does_not_crash():
    c = dip_candles(n=80)
    holey = c[:30] + c[50:]  # 20 missing trading days
    d = _decide(holey)
    assert d.action in (Action.BUY, Action.HOLD, Action.SELL)


def test_duplicate_and_unsorted_bars_do_not_crash():
    c = dip_candles(n=80)
    d = _decide(c[:40] + c[39:40] * 3 + c[40:])
    assert d.action in Action
    shuffled = list(reversed(c))
    assert _decide(shuffled).action in Action


def test_zero_volume_bars_do_not_divide_by_zero():
    c = dip_candles(n=80)
    z = [Candle(x.symbol, x.timestamp, x.open, x.high, x.low, x.close, 0) for x in c]
    assert _decide(z).action in Action


# -- live-cycle behaviour with the real provider ------------------------------------

def test_history_does_not_contain_the_bar_the_snapshot_is_built_from(broker, db_url, monkeypatch):
    """BUG-1 (fixed): the live loop's history included today's still-forming
    daily bar, and the snapshot is built from that very same bar, so the
    strategy saw today's price/volume twice (an extra zero-return day in
    RSI/momentum, and today's partial volume in its own baseline). The
    backtester feeds bars[:i] as history and bar i as the snapshot, so
    live and backtest disagreed."""
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url))
    seen = {}
    orig = bot.strategy.generate_decision

    def spy(**kw):
        seen.update(history=kw["history"], snapshot=kw["snapshot"])
        return orig(**kw)

    bot.strategy.generate_decision = spy
    bot.cycle()
    assert seen["history"][-1].timestamp.date() < seen["snapshot"].timestamp.date()


def test_provider_outage_for_one_symbol_does_not_block_others(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    for sym in ("AAPL", "MSFT"):
        broker.set_bars_from_candles(sym, dip_candles(sym, end=TODAY_BAR))
    cfg = make_config(db_url, symbols=("AAPL", "MSFT"))
    cfg.execution.mirror_to_alpaca = True
    bot = Bot(cfg)
    # first get_stock_bars (AAPL snapshot) fails, MSFT is healthy
    broker.fail_next("get_stock_bars", Timeout())
    bot.cycle()
    assert "MSFT" in bot.portfolio.positions and "AAPL" not in bot.portfolio.positions


def test_history_fetch_failure_after_snapshot_is_contained(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url))
    # snapshot call ok, history call fails
    orig = bot.market_data.get_history
    bot.market_data.get_history = lambda *a, **k: (_ for _ in ()).throw(MarketDataError("boom"))
    bot.cycle()  # must not raise
    assert not bot.portfolio.positions
    assert len(bot.db.get_portfolio_history()) == 1  # cycle still wrote its snapshot


@pytest.mark.xfail(strict=True, reason="BUG-6: no outlier/sanity filter before stop-loss exits (see tests/chaos/README.md)")
def test_outlier_tick_must_not_liquidate_a_position(broker, db_url, monkeypatch):
    """BUG-6 (documented, unfixed): one bad print (-60% for a single poll)
    triggers the 8% stop-loss and SELLS the whole position at that price. No
    sanity band / confirmation on price jumps."""
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url))
    bot.cycle()
    assert "AAPL" in bot.portfolio.positions
    last = broker.bars["AAPL"][-1]
    broker.bars["AAPL"][-1] = bar(last.timestamp, close=last.close * 0.4, volume=last.volume)
    bot.cycle()
    sells = [t for t in bot.db.get_recent_trades() if t["side"] == "SELL"]
    assert not sells, f"liquidated on a single outlier print: {sells}"
