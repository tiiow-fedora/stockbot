"""Other poison-input and bookkeeping chaos: bad news items, valuation during
data outages, portfolio op fuzz, the other strategy."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from analysis.sentiment import LexiconSentimentScorer
from core.models import Fill, Order, OrderSide
from portfolio.portfolio import InsufficientFundsError, InsufficientPositionError, Portfolio
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config

TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)
NOW = datetime(2024, 3, 12, 15, tzinfo=timezone.utc)


# -- news ----------------------------------------------------------------------------------

class _FakeNewsClient:
    def __init__(self, articles=None, exc=None):
        self.articles, self.exc = articles or [], exc

    def get_news(self, request):
        if self.exc:
            raise self.exc
        return SimpleNamespace(data={"news": self.articles})


def article(created_at, headline="Apple beats earnings", source="x"):
    return SimpleNamespace(created_at=created_at, headline=headline, source=source, url=None)


def news_provider(monkeypatch, client):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr("alpaca.data.historical.NewsClient", lambda *a, **k: client)
    from data.providers.alpaca import AlpacaNewsProvider

    return AlpacaNewsProvider()


def test_news_outage_returns_no_news_instead_of_raising(monkeypatch):
    p = news_provider(monkeypatch, _FakeNewsClient(exc=TimeoutError("news down")))
    assert p.get_news("AAPL", NOW - timedelta(days=1), NOW) == []


def test_one_malformed_article_does_not_discard_the_whole_news_batch(monkeypatch):
    """BUG-15 (fixed): a single article with a naive timestamp raised ValueError
    out of get_news; _run_cycle then dropped the ENTIRE symbol for the cycle
    (no decision, no trade) for as long as that article stayed in the 1-day
    window."""
    good = article(NOW - timedelta(hours=2))
    bad = article(datetime(2024, 3, 12, 10, 0))                   # naive -> NewsItem raises
    p = news_provider(monkeypatch, _FakeNewsClient([bad, good]))
    items = p.get_news("AAPL", NOW - timedelta(days=1), NOW)
    assert len(items) == 1


def test_symbol_with_a_poisoned_news_feed_still_trades(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    cfg = make_config(db_url)
    cfg.execution.mirror_to_alpaca = True
    bot = Bot(cfg)
    bot.news = news_provider(monkeypatch, _FakeNewsClient([article(datetime(2024, 3, 12, 9, 0))]))
    bot.cycle()
    assert bot.portfolio.position_qty("AAPL") > 0


@pytest.mark.parametrize("headline", ["", " ", "x" * 200_000, "💥📉 CRASH — ñandú 破产", "\x00\x01", "NOT not no no no", None])
def test_sentiment_scorer_survives_garbage_headlines(headline):
    from core.models import NewsItem

    item = NewsItem("AAPL", NOW, headline if headline is not None else "", "src")
    scored = LexiconSentimentScorer().score_news_items([item])
    assert scored[0].sentiment is None or -1.0 <= scored[0].sentiment <= 1.0


# -- valuation during outages ---------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="BUG-16: a held symbol whose quote fails this cycle is valued at its ENTRY "
                                       "price in the saved portfolio snapshot, so equity jumps during data outages")
def test_portfolio_snapshot_does_not_jump_when_a_held_symbol_loses_its_quote(broker, db_url, monkeypatch):
    from tests.chaos.fake_alpaca import Timeout

    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    cfg = make_config(db_url, stop_loss_pct=0.5)
    cfg.execution.mirror_to_alpaca = True
    bot = Bot(cfg)
    bot.cycle()
    last = broker.bars["AAPL"][-1]
    last.close = last.high = last.low = last.open = last.close * 0.80     # unrealised -20%
    bot.cycle()
    before = bot.db.get_latest_portfolio_snapshot()["total_value"]
    broker.fail_next("get_stock_bars", Timeout())                           # quote outage for the only symbol
    bot.cycle()                                                             # "no market data" -> skip (no snapshot)
    # a second symbol being healthy is what writes a snapshot with the missing one at cost:
    broker.set_bars_from_candles("MSFT", dip_candles("MSFT", end=TODAY_BAR))
    bot.config.symbols = ["AAPL", "MSFT"]
    broker.fail_next("get_stock_bars", Timeout())                           # AAPL fails, MSFT ok
    bot.cycle()
    after = bot.db.get_latest_portfolio_snapshot()["total_value"]
    assert abs(after - before) / before < 0.02


# -- portfolio ledger fuzz ------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_portfolio_random_fill_sequences_never_corrupt_the_ledger(seed):
    rng = random.Random(seed)
    pf = Portfolio(50_000.0)
    shares = {s: 0 for s in ("A", "B", "C")}
    cash = 50_000.0
    for _ in range(400):
        sym = rng.choice(list(shares))
        qty = float(rng.randint(1, 80))
        price = round(rng.uniform(1, 400), 4)
        comm = round(rng.uniform(0, 3), 4)
        side = rng.choice([OrderSide.BUY, OrderSide.SELL])
        fill = Fill(Order(sym, side, qty), price, qty, NOW, commission=comm)
        try:
            pf.apply_fill(fill)
        except (InsufficientFundsError, InsufficientPositionError):
            assert pf.cash >= -1e-6                 # a rejected fill changes nothing
            continue
        if side == OrderSide.BUY:
            shares[sym] += qty
            cash -= price * qty + comm
        else:
            shares[sym] -= qty
            cash += price * qty - comm
        assert pf.cash == pytest.approx(cash, abs=1e-6) and pf.cash >= -1e-6
        assert pf.position_qty(sym) == pytest.approx(shares[sym], abs=1e-9)
        assert shares[sym] >= 0


# -- the other strategy ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_technical_strategy_under_chaos_keeps_ledger_sane(broker, db_url, monkeypatch, seed):
    from tests.chaos.test_chaos_fuzz import SYMBOLS, ChaosExecutor, _assert_invariants, _inject_faults, _mutate_market

    rng = random.Random(100 + seed)
    patch_clock(monkeypatch, broker.clock)
    monkeypatch.setattr("database.database.DatabaseManager.save_candles", lambda self, candles: None)
    for sym in SYMBOLS:
        broker.set_bars_from_candles(sym, dip_candles(sym, end=TODAY_BAR, crash=0.0))
    cfg = make_config(db_url, symbols=SYMBOLS)
    cfg.strategy.name = "technical"
    cfg.execution.mirror_to_alpaca = True
    bot = Bot(cfg, executor_wrapper=lambda inner: ChaosExecutor(inner, rng))
    for i in range(40):
        _mutate_market(broker, rng, day_roll=(i % 9 == 8))
        _inject_faults(broker, rng)
        broker.clock.advance(days=1) if i % 9 == 8 else broker.clock.advance(seconds=60)
        bot.cycle()
        _assert_invariants(bot, cfg)
