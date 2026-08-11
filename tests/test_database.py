from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy.exc import ProgrammingError

from database.database import Base, DatabaseManager


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
