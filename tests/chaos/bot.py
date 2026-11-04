"""A thin wrapper that assembles the REAL StockBot live pipeline (Alpaca data
provider, Alpaca calendar, PaperExecutor + AlpacaMirrorExecutor, RiskManager,
DB, strategy) on top of the fake broker, so chaos tests drive production code
rather than re-implementations of it."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

import main
from core.config import AppConfig
from data.market_data import MarketDataService
from data.news_data import build_news_provider
from database.database import DatabaseManager
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager
from analysis.sentiment import build_sentiment_scorer
from tests.chaos.fake_alpaca import FakeAlpacaBroker, FakeClock


def patch_clock(monkeypatch, clock: FakeClock) -> None:
    """Make every `datetime.now()` in the live path read the BOT's (possibly
    skewed) machine clock instead of the real one."""

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            t = clock.local_now()
            return t.astimezone(tz) if tz else t.replace(tzinfo=None)

    for mod in ("main", "data.market_data", "data.providers.alpaca", "portfolio.risk"):
        monkeypatch.setattr(f"{mod}.datetime", _DT, raising=True)


class Bot:
    """One 'process' of the bot. Create a second Bot over the same db_url to
    simulate a restart (all in-memory state is lost, the DB survives)."""

    def __init__(self, config: AppConfig, executor_wrapper: Optional[Callable] = None):
        from execution.paper import PaperExecutor

        self.config = config
        self.db = DatabaseManager(config.database.url)
        self.db.init_db()
        self.provider = main.build_market_data_provider(config)
        self.news = build_news_provider(config.news)
        self.sentiment = build_sentiment_scorer(config.news.sentiment_scorer)
        self.market_data = MarketDataService(self.provider, self.db)
        self.strategy = main.build_strategy(config)
        self.risk = RiskManager(config.risk, db=self.db)
        self.executor = main.build_executor(config)
        if executor_wrapper:
            self.executor = executor_wrapper(self.executor)
        self.portfolio = Portfolio(config.starting_balance)
        self.calendar = main.build_market_calendar(config)

    def cycle(self) -> None:
        main._run_cycle(
            self.config, self.db, self.provider, self.market_data, self.news, self.sentiment,
            self.strategy, self.risk, self.executor, self.portfolio,
        )

    def is_open(self) -> bool:
        return self.calendar.is_open(datetime.now(timezone.utc))
