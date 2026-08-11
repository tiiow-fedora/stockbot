"""
Persistence layer.

Works against SQLite (default, zero-config, used in dev/tests) or
PostgreSQL (recommended for the long-running server deployment) via the
same SQLAlchemy models - only DATABASE_URL / config.database.url changes.

Everything a human would need to answer "why did the bot make this
trade" is written here: raw observations, candles, computed indicators,
the full scoring breakdown behind every decision, orders, fills,
portfolio snapshots, and news.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from core.models import (
    Action,
    Candle,
    Decision,
    Fill,
    MarketSnapshot,
    NewsItem,
    Order,
    PortfolioSnapshot,
)


class Base(DeclarativeBase):
    pass


def _as_utc(ts: datetime) -> datetime:
    """SQLite has no native timezone-aware datetime type - SQLAlchemy
    stores the value fine but reads it back naive. Postgres preserves
    tzinfo natively, so this is a no-op there. Every stored timestamp is
    written in UTC, so re-attaching UTC on read is always correct."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


class MarketObservationORM(Base):
    __tablename__ = "market_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer, default=0)


class CandleORM(Base):
    __tablename__ = "candles"
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_symbol_tf_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


class DecisionORM(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    scores: Mapped[dict] = mapped_column(JSON)
    indicators: Mapped[dict] = mapped_column(JSON)
    current_position_qty: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    portfolio_value: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(1024))


class OrderORM(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(8))
    limit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    reason: Mapped[str] = mapped_column(String(1024), default="")
    status: Mapped[str] = mapped_column(String(16), default="submitted")

    fills: Mapped[List["FillORM"]] = relationship(back_populates="order")


class FillORM(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    fill_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)

    order: Mapped["OrderORM"] = relationship(back_populates="fills")


class PortfolioSnapshotORM(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)
    positions: Mapped[dict] = mapped_column(JSON)


class NewsItemORM(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    headline: Mapped[str] = mapped_column(String(1024))
    source: Mapped[str] = mapped_column(String(128))
    sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)


class RiskRejectionORM(Base):
    __tablename__ = "risk_rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    reason: Mapped[str] = mapped_column(String(1024))


class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///./stockbot.db"):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    # -- writers ----------------------------------------------------------

    def save_snapshot(self, snapshot: MarketSnapshot) -> None:
        with self.Session() as session:
            session.add(
                MarketObservationORM(
                    symbol=snapshot.symbol,
                    timestamp=snapshot.timestamp,
                    price=snapshot.price,
                    volume=snapshot.volume,
                )
            )
            session.commit()

    def save_candles(self, candles: List[Candle]) -> None:
        if not candles:
            return
        with self.Session() as session:
            for c in candles:
                exists = (
                    session.query(CandleORM)
                    .filter_by(symbol=c.symbol, timeframe=c.timeframe, timestamp=c.timestamp)
                    .first()
                )
                if exists:
                    continue
                session.add(
                    CandleORM(
                        symbol=c.symbol,
                        timeframe=c.timeframe,
                        timestamp=c.timestamp,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                    )
                )
            session.commit()

    def save_decision(self, decision: Decision) -> None:
        with self.Session() as session:
            session.add(
                DecisionORM(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    price=decision.price,
                    action=decision.action.value,
                    confidence=decision.confidence,
                    scores={
                        "technical": decision.scores.technical,
                        "momentum": decision.scores.momentum,
                        "volume": decision.scores.volume,
                        "risk": decision.scores.risk,
                        "sentiment": decision.scores.sentiment,
                        "composite": decision.scores.composite,
                        "weights": decision.scores.weights,
                    },
                    indicators=decision.indicators,
                    current_position_qty=decision.current_position_qty,
                    cash=decision.cash,
                    portfolio_value=decision.portfolio_value,
                    reason=decision.reason,
                )
            )
            session.commit()

    def save_order(self, order: Order) -> int:
        with self.Session() as session:
            row = OrderORM(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                limit_price=order.limit_price,
                timestamp=order.timestamp,
                reason=order.reason,
                status="submitted",
            )
            session.add(row)
            session.commit()
            return row.id

    def save_fill(self, fill: Fill, order_id: int) -> None:
        with self.Session() as session:
            session.add(
                FillORM(
                    order_id=order_id,
                    fill_price=fill.fill_price,
                    quantity=fill.quantity,
                    timestamp=fill.timestamp,
                    commission=fill.commission,
                    slippage=fill.slippage,
                )
            )
            order_row = session.get(OrderORM, order_id)
            if order_row:
                order_row.status = "filled"
            session.commit()

    def save_risk_rejection(self, symbol: str, side: str, quantity: float, timestamp: datetime, reason: str) -> None:
        with self.Session() as session:
            session.add(
                RiskRejectionORM(
                    symbol=symbol, side=side, quantity=quantity, timestamp=timestamp, reason=reason
                )
            )
            session.commit()

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        with self.Session() as session:
            session.add(
                PortfolioSnapshotORM(
                    timestamp=snapshot.timestamp,
                    cash=snapshot.cash,
                    positions_value=snapshot.positions_value,
                    total_value=snapshot.total_value,
                    positions=snapshot.positions,
                )
            )
            session.commit()

    def save_news(self, items: List[NewsItem]) -> None:
        if not items:
            return
        with self.Session() as session:
            for n in items:
                session.add(
                    NewsItemORM(
                        symbol=n.symbol,
                        timestamp=n.timestamp,
                        headline=n.headline,
                        source=n.source,
                        sentiment=n.sentiment,
                        url=n.url,
                    )
                )
            session.commit()

    # -- readers ------------------------------------------------------------

    def get_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        with self.Session() as session:
            rows = (
                session.query(CandleORM)
                .filter(
                    CandleORM.symbol == symbol,
                    CandleORM.timeframe == timeframe,
                    CandleORM.timestamp >= start,
                    CandleORM.timestamp <= end,
                )
                .order_by(CandleORM.timestamp.asc())
                .all()
            )
            return [
                Candle(
                    symbol=r.symbol,
                    timestamp=_as_utc(r.timestamp),
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    timeframe=r.timeframe,
                )
                for r in rows
            ]

    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        with self.Session() as session:
            rows = session.query(DecisionORM).order_by(DecisionORM.timestamp.desc()).limit(limit).all()
            return [
                {
                    "symbol": r.symbol,
                    "timestamp": _as_utc(r.timestamp).isoformat(),
                    "price": r.price,
                    "action": r.action,
                    "confidence": r.confidence,
                    "scores": r.scores,
                    "reason": r.reason,
                }
                for r in rows
            ]

    def get_latest_decisions(self, symbols: List[str]) -> Dict[str, Dict]:
        """Most recent decision for each symbol - used for "current signals"."""
        result: Dict[str, Dict] = {}
        with self.Session() as session:
            for symbol in symbols:
                row = (
                    session.query(DecisionORM)
                    .filter(DecisionORM.symbol == symbol)
                    .order_by(DecisionORM.timestamp.desc())
                    .first()
                )
                if row is None:
                    continue
                result[symbol] = {
                    "symbol": row.symbol,
                    "timestamp": _as_utc(row.timestamp).isoformat(),
                    "price": row.price,
                    "action": row.action,
                    "confidence": row.confidence,
                    "scores": row.scores,
                    "indicators": row.indicators,
                    "reason": row.reason,
                }
        return result

    def get_latest_portfolio_snapshot(self) -> Optional[Dict]:
        with self.Session() as session:
            row = session.query(PortfolioSnapshotORM).order_by(PortfolioSnapshotORM.timestamp.desc()).first()
            if row is None:
                return None
            return {
                "timestamp": _as_utc(row.timestamp).isoformat(),
                "cash": row.cash,
                "positions_value": row.positions_value,
                "total_value": row.total_value,
                "positions": row.positions,
            }

    def get_portfolio_history(self, limit: int = 500) -> List[Dict]:
        """Ascending by time (oldest first) - ready to feed straight into a chart."""
        with self.Session() as session:
            rows = (
                session.query(PortfolioSnapshotORM)
                .order_by(PortfolioSnapshotORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            rows = list(reversed(rows))
            return [
                {
                    "timestamp": _as_utc(r.timestamp).isoformat(),
                    "cash": r.cash,
                    "positions_value": r.positions_value,
                    "total_value": r.total_value,
                }
                for r in rows
            ]

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Every filled order (BUY/SELL), most recent first."""
        with self.Session() as session:
            rows = (
                session.query(FillORM, OrderORM)
                .join(OrderORM, FillORM.order_id == OrderORM.id)
                .order_by(FillORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": _as_utc(fill.timestamp).isoformat(),
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": fill.quantity,
                    "price": fill.fill_price,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                    "reason": order.reason,
                }
                for fill, order in rows
            ]

    def get_recent_risk_rejections(self, limit: int = 50) -> List[Dict]:
        with self.Session() as session:
            rows = (
                session.query(RiskRejectionORM)
                .order_by(RiskRejectionORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": _as_utc(r.timestamp).isoformat(),
                    "symbol": r.symbol,
                    "side": r.side,
                    "quantity": r.quantity,
                    "reason": r.reason,
                }
                for r in rows
            ]
