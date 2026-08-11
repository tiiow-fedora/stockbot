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
from datetime import datetime
from typing import List, Optional

from core.logging import get_logger
from core.models import Candle, MarketSnapshot, NewsItem
from data.base import MarketDataError, MarketDataProvider, NewsDataProvider

logger = get_logger("data.alpaca")

_TIMEFRAME_MAP = {
    "1Min": "Min",
    "5Min": "5Min",
    "15Min": "15Min",
    "1Hour": "Hour",
    "1Day": "Day",
}


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
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf_map[timeframe], start=start, end=end
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
        from alpaca.data.requests import StockLatestTradeRequest

        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self._client.get_stock_latest_trade(request)[symbol]
        except Exception as exc:
            raise MarketDataError(f"Alpaca latest trade request failed for {symbol}: {exc}") from exc

        return MarketSnapshot(
            symbol=symbol,
            timestamp=trade.timestamp,
            price=float(trade.price),
            volume=int(getattr(trade, "size", 0) or 0),
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
