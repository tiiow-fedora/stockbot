"""
Typed configuration loader.

All tunable behavior (symbols, strategy weights, risk limits, execution
costs, database URL, polling interval, ...) lives in config/config.yaml
and/or environment variables. Nothing here should ever contain a secret -
credentials are read directly from the environment by provider classes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class StrategyWeights(BaseModel):
    technical: float = 0.30
    momentum: float = 0.25
    volume: float = 0.20
    risk: float = 0.15
    sentiment: float = 0.10


class StrategyConfig(BaseModel):
    name: str = "technical"
    weights: StrategyWeights = Field(default_factory=StrategyWeights)
    buy_threshold: float = 0.25
    sell_threshold: float = -0.25
    sma_short_period: int = 10
    sma_long_period: int = 30
    ema_period: int = 12
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_period: int = 10
    volume_period: int = 10
    volatility_period: int = 20
    min_history_bars: int = 35


class RiskConfig(BaseModel):
    max_position_size_pct: float = 0.20       # max fraction of portfolio value in one symbol
    max_portfolio_exposure_pct: float = 0.80   # max fraction of portfolio value in all positions combined
    max_open_positions: int = 8
    max_trades_per_symbol_per_day: int = 3
    stop_loss_pct: float = 0.08                # exit if position down 8% from entry
    take_profit_pct: float = 0.20              # exit if position up 20% from entry
    min_cash_reserve_pct: float = 0.05         # always keep at least 5% of portfolio in cash


    @field_validator(
        "max_position_size_pct", "max_portfolio_exposure_pct", "stop_loss_pct",
        "take_profit_pct", "min_cash_reserve_pct",
    )
    @classmethod
    def _is_fraction(cls, v: float, info) -> float:
        # These are FRACTIONS (0.08 = 8%), despite the "_pct" suffix. A
        # value like 8 would silently disable the limit (stop at -800%).
        lo_ok = v >= 0.0 if info.field_name == "min_cash_reserve_pct" else v > 0.0
        if not (lo_ok and v <= 1.0):
            raise ValueError(f"{info.field_name} must be a fraction in (0, 1] (0.08 means 8%), got {v!r}")
        return v


class ExecutionConfig(BaseModel):
    mode: str = "paper"                # "paper" or "manual"
    slippage_bps: float = 5.0          # basis points of adverse price movement on fill
    commission_per_trade: float = 0.0  # flat fee per fill
    commission_pct: float = 0.0        # percentage fee per fill (in addition to flat fee)
    # Optional, off by default: best-effort mirror every local paper fill to
    # your own Alpaca PAPER account too, purely so trades also show up on
    # Alpaca's dashboard. Only takes effect when mode: paper. Requires
    # ALPACA_API_KEY/ALPACA_SECRET_KEY. Never touches a live account - see
    # execution/alpaca_mirror.py. Ignored during backtesting (live loop only).
    mirror_to_alpaca: bool = False
    # mode: class_simulator - CSV ticket outbox (whole shares) for the classroom platform.
    class_simulator_outbox: str = "class_orders/outbox.csv"

    @field_validator("slippage_bps")
    @classmethod
    def _bps_range(cls, v: float) -> float:
        # basis points: 5 = 0.05%. 10_000 bps = 100% - anything near that is a units slip.
        if not (0.0 <= v < 1000.0):
            raise ValueError(f"slippage_bps must be in [0, 1000) basis points, got {v!r}")
        return v

    @field_validator("commission_pct")
    @classmethod
    def _commission_fraction(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"commission_pct is a fraction in [0, 1) (0.001 = 0.1%), got {v!r}")
        return v

    @field_validator("commission_per_trade")
    @classmethod
    def _flat_fee(cls, v: float) -> float:
        if v < 0:
            raise ValueError("commission_per_trade must be >= 0 dollars")
        return v


class DataConfig(BaseModel):
    provider: str = "synthetic"        # "synthetic", "csv", "alpaca", or "class_simulator"
    csv_dir: str = "data_files"
    timeframe: str = "1Day"
    history_lookback_days: int = 180
    # "auto" (alpaca calendar if provider: alpaca, else always_open),
    # "always_open" (synthetic/CSV or a 24/7 classroom platform),
    # "us_equity" (deterministic Mon-Fri 9:30-16:00 ET, no holidays, no
    # network), or "alpaca" (exact hours + real holidays, needs credentials).
    market_calendar: str = "auto"
    # provider: class_simulator - see data/providers/class_simulator.py for the file contract.
    class_simulator_quotes_path: str = "data_files/class_quotes.json"


class NewsConfig(BaseModel):
    provider: str = "none"             # "none" or "alpaca"
    sentiment_scorer: str = "lexicon"  # "none", "lexicon" (default, no API key needed), or "llm" (optional)


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./stockbot.db"


class BacktestConfig(BaseModel):
    starting_balance: float = 100000.0
    warmup_bars: int = 40


class AppConfig(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    starting_balance: float = 100000.0
    polling_interval_seconds: int = 60
    # While the market is closed, the live loop re-checks (and writes a
    # heartbeat portfolio snapshot) at most this often rather than sleeping
    # straight through to the next open. Shared with the dashboard's
    # "running" vs "stale" status check, which must tolerate this slower
    # cadence during closed-market hours instead of assuming the normal
    # polling_interval_seconds pace around the clock.
    max_closed_market_heartbeat_seconds: int = 1800
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/stockbot.log"

    data: DataConfig = Field(default_factory=DataConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config.yaml (if present) and let environment variables override
    the database URL, since that's the one setting that commonly differs
    between local dev and the Docker deployment."""
    data: Dict = {}
    if path is None:
        default = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        path = str(default) if default.exists() else None

    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    config = AppConfig(**data)

    env_db_url = os.environ.get("DATABASE_URL")
    if env_db_url:
        config.database.url = env_db_url

    env_log_level = os.environ.get("LOG_LEVEL")
    if env_log_level:
        config.log_level = env_log_level

    return config
