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
from pydantic import BaseModel, Field


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


class DataConfig(BaseModel):
    provider: str = "synthetic"        # "synthetic", "csv", or "alpaca"
    csv_dir: str = "data_files"
    timeframe: str = "1Day"
    history_lookback_days: int = 180
    # "auto" (alpaca calendar if provider: alpaca, else always_open),
    # "always_open" (synthetic/CSV or a 24/7 classroom platform),
    # "us_equity" (deterministic Mon-Fri 9:30-16:00 ET, no holidays, no
    # network), or "alpaca" (exact hours + real holidays, needs credentials).
    market_calendar: str = "auto"


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
