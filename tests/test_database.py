from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.exc import ProgrammingError

from core.models import Fill, Order, OrderSide, OrderType
from database.database import Base, DatabaseManager

NOW = datetime(2024, 6, 10, 15, 0, tzinfo=timezone.utc)


def _make_fill(symbol: str, ts: datetime) -> tuple[Order, Fill]:
    order = Order(symbol=symbol, side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=ts)
    fill = Fill(order=order, fill_price=100.0, quantity=1, timestamp=ts)
    return order, fill


def test_init_db_creates_tables(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    with db.Session() as session:
        # A trivial query against a real table proves the schema exists.
        session.execute(Base.metadata.tables["candles"].select())


def test_init_db_survives_a_concurrent_creation_race(tmp_path):
    """Simulates two processes (the live loop and the dashboard) both
    calling init_db() against a fresh database at once - the loser of
    that race must not crash the process."""
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")

    fake_error = ProgrammingError("CREATE TABLE ...", {}, Exception("relation already exists"))
    with patch("database.database.Base.metadata.create_all", side_effect=fake_error):
        db.init_db()  # must not raise


def test_count_trades_today_counts_only_todays_fills_for_the_symbol(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()

    for ts in [NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2)]:
        order, fill = _make_fill("INTC", ts)
        order_id = db.save_order(order)
        db.save_fill(fill, order_id)

    # A different symbol and a different day must not be counted.
    other_symbol_order, other_symbol_fill = _make_fill("AAPL", NOW)
    db.save_fill(other_symbol_fill, db.save_order(other_symbol_order))
    yesterday_order, yesterday_fill = _make_fill("INTC", NOW - timedelta(days=1))
    db.save_fill(yesterday_fill, db.save_order(yesterday_order))

    assert db.count_trades_today("INTC", NOW) == 3
    assert db.count_trades_today("AAPL", NOW) == 1
    assert db.count_trades_today("NVDA", NOW) == 0


def test_count_trades_today_survives_process_restart_simulation(tmp_path):
    """The whole point of this method: a fresh DatabaseManager (standing
    in for a restarted process) must still see prior fills - unlike an
    in-memory counter, which would reset to zero."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    db1 = DatabaseManager(db_url)
    db1.init_db()
    for _ in range(3):
        order, fill = _make_fill("INTC", NOW)
        db1.save_fill(fill, db1.save_order(order))

    db2 = DatabaseManager(db_url)  # simulates a brand-new process after a restart
    assert db2.count_trades_today("INTC", NOW) == 3
