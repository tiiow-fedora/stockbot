"""
Alpaca market data (and optional news) provider.

This is ONE provider implementation among several - the strategy never
imports this module directly. It is used only for development against a
real (free, paper) market data feed before the classroom platform is
known.

Credentials are read from environment variables ONLY:

    ALPACA_API_KEY
    ALPACA_SECRET_KEY

Never hard-code keys here. This module never places trades - it is a
market-data (and optionally news) READ ONLY client. Order execution, even
in Alpaca's own paper trading account, is intentionally out of scope; see
execution/paper.py for the local paper simulator used by this project.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from core.logging import get_logger
from core.models import Candle, MarketSnapshot, NewsItem
from data.base import MarketDataError, MarketDataProvider, NewsDataProvider

logger = get_logger("data.alpaca")


def _get_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise MarketDataError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. "
            "Copy .env.example to .env and fill them in, or choose a "
            "different data.provider in config.yaml (e.g. 'synthetic' or 'csv')."
        )
    return api_key, secret_key


class AlpacaMarketDataProvider(MarketDataProvider):
    def __init__(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:  # pragma: no cover - only hit if extra not installed
            raise MarketDataError(
                "alpaca-py is not installed. Run `pip install alpaca-py` "
                "or select a different data provider."
            ) from exc

        api_key, secret_key = _get_credentials()
        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_historical_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Minute.unit),
            "15Min": TimeFrame(15, TimeFrame.Minute.unit),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        if timeframe not in tf_map:
            raise MarketDataError(f"unsupported timeframe: {timeframe!r}")

        try:
            # Free/paper accounts only have IEX feed access, not the full-
            # market SIP feed (which 403s on recent data without a paid
            # subscription). IEX is a real, live exchange feed - just a
            # single venue's view rather than a consolidated tape - which
            # is a fine tradeoff for a paper-trading research bot.
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map[timeframe],
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = self._client.get_stock_bars(request)
        except Exception as exc:  # network/auth/rate-limit failures
            raise MarketDataError(f"Alpaca historical bars request failed for {symbol}: {exc}") from exc

        candles: List[Candle] = []
        for bar in bars[symbol]:
            candles.append(
                Candle(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume),
                    timeframe=timeframe,
                )
            )
        return candles

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Uses today's still-forming DAILY bar, not the latest single
        trade. This matters for `volume`: a single trade's size (e.g.
        100 shares) is not remotely comparable to the daily volumes in
        `history` (millions of shares) that `analysis.indicators.
        volume_change` compares it against - using trade size there made
        every cycle read as a ~100% volume collapse regardless of actual
        conditions, silently suppressing the volume component of every
        decision. Today's cumulative daily-bar volume-so-far is directly
        comparable to prior full days' volumes instead. This also
        correctly fills in open/high/low, which the latest-trade version
        always left as None."""
        now = datetime.now(timezone.utc)
        # A few days back guarantees today's still-forming bar is the
        # last entry even right after a weekend/holiday reopen, before
        # today's first trade has printed.
        candles = self.get_historical_candles(symbol, now - timedelta(days=5), now, "1Day")
        if not candles:
            raise MarketDataError(f"no recent daily bar available for {symbol}")

        latest = candles[-1]
        return MarketSnapshot(
            symbol=symbol,
            timestamp=now,
            price=latest.close,
            volume=latest.volume,
            open=latest.open,
            high=latest.high,
            low=latest.low,
            close=latest.close,
        )


class AlpacaNewsProvider(NewsDataProvider):
    def __init__(self):
        try:
            from alpaca.data.historical import NewsClient
        except ImportError as exc:  # pragma: no cover
            raise MarketDataError(
                "alpaca-py is not installed. Run `pip install alpaca-py` "
                "or select news.provider: none in config.yaml."
            ) from exc

        api_key, secret_key = _get_credentials()
        self._client = NewsClient(api_key, secret_key)

    def get_news(self, symbol: str, start: datetime, end: datetime) -> List[NewsItem]:
        from alpaca.data.requests import NewsRequest

        try:
            request = NewsRequest(symbols=symbol, start=start, end=end)
            response = self._client.get_news(request)
        except Exception as exc:
            logger.warning("Alpaca news request failed for %s: %s", symbol, exc)
            return []

        items: List[NewsItem] = []
        for article in response.data.get("news", []):
            items.append(
                NewsItem(
                    symbol=symbol,
                    timestamp=article.created_at,
                    headline=article.headline,
                    source=article.source or "alpaca",
                    sentiment=None,  # Alpaca does not score sentiment; left for an optional module.
                    url=getattr(article, "url", None),
                )
            )
        return items
