"""
Read-only dashboard.

Deliberately simple, per the project spec: it never writes anything back
to the trading engine - it only queries `DatabaseManager`'s read methods
and renders one static HTML page. The trading engine (main.py run /
backtest) is a completely separate process; the dashboard just tails
whatever it wrote to the database.

Run standalone with `python main.py dashboard`, or `uvicorn dashboard.app:app`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from core.config import AppConfig, load_config
from database.database import DatabaseManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    config = config or load_config()
    db = DatabaseManager(config.database.url)
    db.init_db()

    app = FastAPI(title="StockBot Dashboard")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz():
        from sqlalchemy import text

        try:
            with db.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            return JSONResponse({"status": "db_down"}, status_code=503)
        return {"status": "ok"}

    @app.get("/api/status")
    def status():
        snapshot = db.get_latest_portfolio_snapshot()
        bot_status = "not_started"
        last_update = None
        if snapshot is not None:
            last_update = snapshot["timestamp"]
            last_ts = datetime.fromisoformat(last_update)
            age_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
            # Must tolerate BOTH cadences: normal polling while the market
            # is open, and the slower closed-market heartbeat (up to
            # max_closed_market_heartbeat_seconds between updates) - using
            # only the polling interval here would falsely show "stale"
            # every time the market closes, even with the bot working fine.
            staleness_threshold = max(
                config.polling_interval_seconds * 3,
                config.max_closed_market_heartbeat_seconds * 1.5,
            )
            bot_status = "running" if age_seconds <= staleness_threshold else "stale"
        return {
            "status": bot_status,
            "last_update": last_update,
            "symbols": config.symbols,
            "data_provider": config.data.provider,
            "execution_mode": config.execution.mode,
            "polling_interval_seconds": config.polling_interval_seconds,
            "strategy": config.strategy.name,
        }

    @app.get("/api/portfolio")
    def portfolio():
        snapshot = db.get_latest_portfolio_snapshot()
        if snapshot is None:
            return {
                "cash": config.starting_balance,
                "positions_value": 0.0,
                "total_value": config.starting_balance,
                "starting_balance": config.starting_balance,
                "return_pct": 0.0,
                "positions": {},
                "timestamp": None,
            }
        return_pct = (snapshot["total_value"] - config.starting_balance) / config.starting_balance
        return {**snapshot, "starting_balance": config.starting_balance, "return_pct": return_pct}

    @app.get("/api/portfolio/history")
    def portfolio_history(limit: int = 500):
        return db.get_portfolio_history(limit=limit)

    @app.get("/api/positions")
    def positions():
        snapshot = db.get_latest_portfolio_snapshot()
        return snapshot["positions"] if snapshot else {}

    @app.get("/api/trades")
    def trades(limit: int = 50):
        return db.get_recent_trades(limit=limit)

    @app.get("/api/decisions")
    def decisions(limit: int = 50):
        return db.get_recent_decisions(limit=limit)

    @app.get("/api/signals")
    def signals():
        return db.get_latest_decisions(config.symbols)

    @app.get("/api/rejections")
    def rejections(limit: int = 50):
        return db.get_recent_risk_rejections(limit=limit)

    if STATIC_DIR.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
