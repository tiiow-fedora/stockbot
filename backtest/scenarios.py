"""Deterministic synthetic bar generators for repeatable backtests.

Each scenario is a seeded geometric random walk with a regime-specific
drift/volatility, so the same (scenario, seed, n) always yields the same
bars. Weekends are skipped to look like daily equity data.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from core.models import Candle

# name -> (daily drift, daily vol)
SCENARIOS: Dict[str, tuple] = {
    "bull": (0.0020, 0.008),
    "bear": (-0.0020, 0.010),
    "sideways": (0.0000, 0.006),
    "volatile": (0.0002, 0.025),
}
# "crash" is path-dependent, handled separately below.
ALL_SCENARIOS = sorted(list(SCENARIOS) + ["crash"])


def generate_scenario(
    name: str,
    symbol: str = "SYN",
    n: int = 250,
    seed: int = 7,
    start_price: float = 100.0,
    start: Optional[datetime] = None,
) -> List[Candle]:
    if name not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; available: {ALL_SCENARIOS}")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(f"{name}:{symbol}:{seed}")
    ts = start or datetime(2023, 1, 2, tzinfo=timezone.utc)
    price = start_price
    candles: List[Candle] = []
    for i in range(n):
        while ts.weekday() >= 5:
            ts += timedelta(days=1)
        if name == "crash":
            # steady grind up, then a sharp ~35% drop around 60% of the way in
            crash_at = int(n * 0.6)
            drift, vol = (0.0010, 0.007)
            if crash_at <= i < crash_at + 5:
                drift, vol = (-0.08, 0.01)
            elif i >= crash_at + 5:
                drift, vol = (0.0003, 0.012)
        else:
            drift, vol = SCENARIOS[name]
        o = price
        c = max(0.5, o * (1 + drift + rng.gauss(0, vol)))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 3)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, vol / 3)))
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=ts,
                open=round(o, 2),
                high=round(max(h, o, c), 2),
                low=round(min(lo, o, c), 2),
                close=round(c, 2),
                volume=int(rng.uniform(5e5, 3e6)),
            )
        )
        price = c
        ts += timedelta(days=1)
    return candles
