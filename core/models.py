"""
Shared, provider-agnostic data types.

Every layer of the system (data, analysis, strategy, risk, execution,
portfolio, database) communicates using ONLY the types defined here.
No component upstream of these types should ever be imported by a
strategy or risk module - that is what keeps the strategy ignorant of
where data came from and the executor ignorant of where orders end up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


def _require_tz_aware(ts: datetime, label: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware (got naive datetime {ts!r})")


@dataclass(frozen=True)
class MarketSnapshot:
    """A single point-in-time quote. This is what a strategy actually sees."""

    symbol: str
    timestamp: datetime
    price: float
    volume: int = 0
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "MarketSnapshot.timestamp")
        # `not (x > 0)` (rather than `x <= 0`) so NaN is rejected too: every
        # comparison with NaN is False, and a NaN price would otherwise sail
        # through into indicators and sizing and silently suppress signals.
        if self.price is None or not math.isfinite(self.price) or not (self.price > 0):
            raise ValueError(f"invalid price for {self.symbol}: {self.price!r}")
        if self.volume is not None and self.volume < 0:
            raise ValueError(f"invalid volume for {self.symbol}: {self.volume!r}")


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar for a given timeframe (e.g. '1Day', '1Min')."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1Day"

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "Candle.timestamp")
        for name, val in (("open", self.open), ("high", self.high), ("low", self.low), ("close", self.close)):
            if val is None or not math.isfinite(val) or not (val > 0):
                raise ValueError(f"invalid {name} for {self.symbol} at {self.timestamp}: {val!r}")
        if self.high < self.low:
            raise ValueError(f"high < low for {self.symbol} at {self.timestamp}")
        if self.volume is not None and self.volume < 0:
            raise ValueError(f"invalid volume for {self.symbol} at {self.timestamp}: {self.volume!r}")


@dataclass(frozen=True)
class NewsItem:
    """A single news item. sentiment is None until/unless a sentiment provider scores it."""

    symbol: str
    timestamp: datetime
    headline: str
    source: str
    sentiment: Optional[float] = None  # -1.0 (very negative) .. +1.0 (very positive)
    url: Optional[str] = None

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "NewsItem.timestamp")
        if self.sentiment is not None and not (-1.0 <= self.sentiment <= 1.0):
            raise ValueError(f"sentiment out of range [-1,1]: {self.sentiment!r}")


@dataclass
class ScoreBreakdown:
    """Component scores that were combined into the composite decision score."""

    technical: float
    momentum: float
    volume: float
    risk: float
    sentiment: Optional[float]
    composite: float
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    """A fully reproducible record of what the strategy decided and why."""

    timestamp: datetime
    symbol: str
    price: float
    action: Action
    confidence: float
    scores: ScoreBreakdown
    indicators: Dict[str, Optional[float]]
    current_position_qty: float
    cash: float
    portfolio_value: float
    reason: str


@dataclass
class Order:
    """An abstract instruction to trade. The strategy only ever produces these."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    reason: str = ""
    confidence: Optional[float] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity!r}")


@dataclass
class Fill:
    """What actually happened when an order was executed (paper or otherwise)."""

    order: Order
    fill_price: float
    quantity: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_entry_price) * self.quantity

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    positions: Dict[str, dict]
