"""Build `{label: candles}` series from stored CSVs or synthetic scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from backtest.scenarios import ALL_SCENARIOS, generate_scenario
from core.models import Candle
from data.providers.csv import CSVMarketDataProvider

_FAR_PAST = datetime(1970, 1, 1, tzinfo=timezone.utc)
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)


def load_csv_series(
    csv_dir: str,
    symbols: List[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[str, List[Candle]]:
    provider = CSVMarketDataProvider(csv_dir)
    return {
        s: provider.get_historical_candles(s, start or _FAR_PAST, end or _FAR_FUTURE) for s in symbols
    }


def synthetic_series(
    scenarios: Optional[List[str]] = None, n: int = 250, seed: int = 7
) -> Dict[str, List[Candle]]:
    names = scenarios or ALL_SCENARIOS
    return {name: generate_scenario(name, symbol=name.upper(), n=n, seed=seed) for name in names}
