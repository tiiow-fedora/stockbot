"""
Factory for NewsDataProvider implementations.

Kept separate from data/base.py so that importing "no news configured"
(the default, and the only path Milestone 1-3 exercise) never requires
importing the optional Alpaca SDK.
"""

from __future__ import annotations

from core.config import NewsConfig
from data.base import NewsDataProvider, NullNewsProvider


def build_news_provider(config: NewsConfig) -> NewsDataProvider:
    if config.provider == "none":
        return NullNewsProvider()
    if config.provider == "alpaca":
        from data.providers.alpaca import AlpacaNewsProvider

        return AlpacaNewsProvider()
    raise ValueError(f"unknown news provider: {config.provider!r}")
