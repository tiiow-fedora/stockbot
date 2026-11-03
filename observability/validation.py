"""Config validation beyond what pydantic types give us.

pydantic accepts any float for a percentage and any string for a provider;
a typo like ``max_position_size_pct: 20`` (meant 0.20) or ``mode: live``
would only surface mid-run, or never. ``validate_config`` returns every
problem at once so they can be fixed in one pass; ``ensure_valid`` raises.

Paper trading only: anything that smells like a live-trading setup is an
error, not a warning.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Mapping, Optional

from core.config import AppConfig

ERROR = "error"
WARNING = "warning"

DATA_PROVIDERS = {"synthetic", "csv", "alpaca"}
NEWS_PROVIDERS = {"none", "alpaca"}
SCORERS = {"none", "lexicon", "llm"}
EXEC_MODES = {"paper", "manual"}
CALENDARS = {"auto", "always_open", "us_equity", "alpaca"}
LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass(frozen=True)
class Issue:
    level: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.path}: {self.message}"


class ConfigError(ValueError):
    def __init__(self, issues: List[Issue]):
        self.issues = issues
        super().__init__("invalid configuration:\n  " + "\n  ".join(str(i) for i in issues))


def validate_config(config: AppConfig, env: Optional[Mapping[str, str]] = None) -> List[Issue]:
    env = os.environ if env is None else env
    out: List[Issue] = []

    def err(path, msg):
        out.append(Issue(ERROR, path, msg))

    def warn(path, msg):
        out.append(Issue(WARNING, path, msg))

    def one_of(path, value, allowed):
        if value not in allowed:
            err(path, f"{value!r} is not one of {sorted(allowed)}")

    # --- symbols
    if not config.symbols:
        err("symbols", "watchlist is empty")
    for s in config.symbols:
        if not _SYMBOL_RE.match(s):
            err("symbols", f"{s!r} is not a valid ticker (expected upper-case, e.g. AAPL or BRK.B)")
    dupes = sorted({s for s in config.symbols if config.symbols.count(s) > 1})
    if dupes:
        warn("symbols", f"duplicate symbols: {dupes}")

    # --- top level
    if config.starting_balance <= 0:
        err("starting_balance", "must be > 0")
    if config.polling_interval_seconds < 1:
        err("polling_interval_seconds", "must be >= 1")
    elif config.polling_interval_seconds < 10 and config.data.provider == "alpaca":
        warn("polling_interval_seconds", "very short interval risks Alpaca rate limits")
    if config.max_closed_market_heartbeat_seconds < config.polling_interval_seconds:
        err("max_closed_market_heartbeat_seconds", "must be >= polling_interval_seconds")
    if config.log_level.upper() not in LOG_LEVELS:
        err("log_level", f"{config.log_level!r} is not one of {sorted(LOG_LEVELS)}")

    # --- providers / modes
    one_of("data.provider", config.data.provider, DATA_PROVIDERS)
    one_of("news.provider", config.news.provider, NEWS_PROVIDERS)
    one_of("news.sentiment_scorer", config.news.sentiment_scorer, SCORERS)
    one_of("execution.mode", config.execution.mode, EXEC_MODES)
    one_of("data.market_calendar", config.data.market_calendar, CALENDARS)
    if config.data.history_lookback_days < 1:
        err("data.history_lookback_days", "must be >= 1")

    # --- strategy
    from strategy.registry import STRATEGY_BUILDERS

    one_of("strategy.name", config.strategy.name, set(STRATEGY_BUILDERS))
    s = config.strategy
    if s.sell_threshold >= s.buy_threshold:
        err("strategy.sell_threshold", "must be below buy_threshold")
    if not (-1 <= s.sell_threshold <= 1 and -1 <= s.buy_threshold <= 1):
        err("strategy.buy_threshold", "thresholds are on a -1..1 score scale")
    if s.sma_short_period >= s.sma_long_period:
        err("strategy.sma_short_period", "must be less than sma_long_period")
    if s.macd_fast >= s.macd_slow:
        err("strategy.macd_fast", "must be less than macd_slow")
    if s.rsi_oversold >= s.rsi_overbought:
        err("strategy.rsi_oversold", "must be below rsi_overbought")
    for name in ("sma_short_period", "sma_long_period", "ema_period", "rsi_period", "macd_fast",
                 "macd_slow", "macd_signal", "momentum_period", "volume_period", "volatility_period"):
        if getattr(s, name) < 1:
            err(f"strategy.{name}", "must be >= 1")
    weights = s.weights.model_dump()
    if any(w < 0 for w in weights.values()):
        err("strategy.weights", "weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        err("strategy.weights", "weights sum to zero")
    elif abs(total - 1.0) > 0.01:
        warn("strategy.weights", f"weights sum to {total:.2f}, not 1.0")
    needed = max(s.sma_long_period, s.macd_slow + s.macd_signal)
    if s.min_history_bars < needed:
        warn("strategy.min_history_bars", f"{s.min_history_bars} is below the {needed} bars the indicators need")
    if config.data.history_lookback_days < s.min_history_bars:
        warn("data.history_lookback_days", "fewer calendar days than min_history_bars; strategy may never trade")

    # --- risk
    r = config.risk
    for name in ("max_position_size_pct", "max_portfolio_exposure_pct", "stop_loss_pct", "take_profit_pct"):
        v = getattr(r, name)
        if not 0 < v <= 1:
            err(f"risk.{name}", f"{v} must be a fraction in (0, 1] (did you write a percent?)")
    if not 0 <= r.min_cash_reserve_pct < 1:
        err("risk.min_cash_reserve_pct", f"{r.min_cash_reserve_pct} must be in [0, 1)")
    if r.max_position_size_pct > r.max_portfolio_exposure_pct:
        warn("risk.max_position_size_pct", "larger than max_portfolio_exposure_pct; exposure cap will bind first")
    if r.max_portfolio_exposure_pct + r.min_cash_reserve_pct > 1.0 + 1e-9:
        warn("risk", "max_portfolio_exposure_pct + min_cash_reserve_pct exceeds 100%")
    if r.max_open_positions < 1:
        err("risk.max_open_positions", "must be >= 1")
    if r.max_trades_per_symbol_per_day < 1:
        err("risk.max_trades_per_symbol_per_day", "must be >= 1")
    if r.stop_loss_pct >= r.take_profit_pct:
        warn("risk.stop_loss_pct", "stop loss is not tighter than take profit")
    if config.symbols and r.max_open_positions * r.max_position_size_pct < 0.01:
        warn("risk", "limits make meaningful positions impossible")

    # --- execution
    e = config.execution
    if e.slippage_bps < 0:
        err("execution.slippage_bps", "must be >= 0")
    if e.commission_per_trade < 0 or e.commission_pct < 0:
        err("execution.commission", "must be >= 0")
    if e.mirror_to_alpaca and e.mode != "paper":
        err("execution.mirror_to_alpaca", "only valid with mode: paper")

    # --- credentials (presence only; values are never logged)
    needs_alpaca = (
        config.data.provider == "alpaca" or config.news.provider == "alpaca"
        or e.mirror_to_alpaca or config.data.market_calendar == "alpaca"
    )
    if needs_alpaca and not (env.get("ALPACA_API_KEY") and env.get("ALPACA_SECRET_KEY")):
        err("env", "ALPACA_API_KEY / ALPACA_SECRET_KEY are required by the configured providers")
    if config.news.sentiment_scorer == "llm" and not env.get("ANTHROPIC_API_KEY"):
        err("env", "ANTHROPIC_API_KEY is required for news.sentiment_scorer: llm")

    # --- paper-only guard
    for var in ("APCA_API_BASE_URL", "ALPACA_BASE_URL", "ALPACA_API_BASE_URL"):
        url = env.get(var, "")
        if url and "paper" not in url.lower():
            err(f"env.{var}", "points at a non-paper Alpaca endpoint; StockBot is paper-trading only")
    if str(env.get("ALPACA_PAPER", "true")).lower() in ("0", "false", "no"):
        err("env.ALPACA_PAPER", "must not disable paper mode; StockBot is paper-trading only")

    # --- database
    if not config.database.url:
        err("database.url", "is empty")
    elif "://" not in config.database.url:
        err("database.url", "is not a SQLAlchemy URL")
    return out


def ensure_valid(config: AppConfig, env: Optional[Mapping[str, str]] = None) -> List[Issue]:
    """Raise ConfigError on any error; return the (warning) issues otherwise."""
    issues = validate_config(config, env)
    errors = [i for i in issues if i.level == ERROR]
    if errors:
        raise ConfigError(errors)
    return issues
