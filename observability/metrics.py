"""Prometheus metrics for StockBot.

A dedicated registry (not the global default) keeps tests isolated and
avoids exporting unrelated process collectors unless asked. All helper
functions are cheap no-throw wrappers, so instrumentation can never break
the trading loop.
"""

from __future__ import annotations

from typing import Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

REGISTRY = CollectorRegistry(auto_describe=True)

CYCLES = Counter("stockbot_cycles_total", "Live-loop cycles by outcome.", ["outcome"], registry=REGISTRY)
CYCLE_SECONDS = Histogram(
    "stockbot_cycle_duration_seconds", "Wall time of one trading cycle.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120), registry=REGISTRY,
)
LAST_CYCLE_TS = Gauge("stockbot_last_successful_cycle_timestamp_seconds", "Unix time of last good cycle/heartbeat.", registry=REGISTRY)
CONSECUTIVE_FAILURES = Gauge("stockbot_consecutive_failures", "Failed loop iterations in a row.", registry=REGISTRY)
MARKET_OPEN = Gauge("stockbot_market_open", "1 if the market calendar says open.", registry=REGISTRY)
DECISIONS = Counter("stockbot_decisions_total", "Strategy decisions.", ["symbol", "action"], registry=REGISTRY)
ORDERS = Counter("stockbot_orders_total", "Orders by side and result (filled/rejected).", ["side", "result"], registry=REGISTRY)
RISK_REJECTIONS = Counter("stockbot_risk_rejections_total", "Risk-manager rejections.", ["reason"], registry=REGISTRY)
DATA_ERRORS = Counter("stockbot_data_errors_total", "Market data failures.", ["symbol"], registry=REGISTRY)
PORTFOLIO_VALUE = Gauge("stockbot_portfolio_value_dollars", "Total paper portfolio value.", registry=REGISTRY)
PORTFOLIO_CASH = Gauge("stockbot_portfolio_cash_dollars", "Paper cash balance.", registry=REGISTRY)
POSITIONS_VALUE = Gauge("stockbot_positions_value_dollars", "Market value of open positions.", registry=REGISTRY)
OPEN_POSITIONS = Gauge("stockbot_open_positions", "Number of open positions.", registry=REGISTRY)
BUILD_INFO = Gauge("stockbot_info", "Static info (always 1).", ["execution_mode", "data_provider", "strategy"], registry=REGISTRY)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def _safe(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:  # instrumentation must never break trading
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_safe
def record_cycle(outcome: str, seconds: Optional[float] = None, now: Optional[float] = None) -> None:
    import time as _t
    CYCLES.labels(outcome).inc()
    if seconds is not None:
        CYCLE_SECONDS.observe(seconds)
    if outcome in ("ok", "closed"):
        LAST_CYCLE_TS.set(now if now is not None else _t.time())
        CONSECUTIVE_FAILURES.set(0)


@_safe
def record_failure(consecutive: int) -> None:
    CYCLES.labels("error").inc()
    CONSECUTIVE_FAILURES.set(consecutive)


@_safe
def record_decision(symbol: str, action: str) -> None:
    DECISIONS.labels(symbol, action).inc()


@_safe
def record_order(side: str, result: str, reason: Optional[str] = None) -> None:
    ORDERS.labels(side, result).inc()
    if result == "rejected" and reason:
        # Reasons can embed numbers; keep label cardinality bounded.
        RISK_REJECTIONS.labels(_normalize_reason(reason)).inc()


def _normalize_reason(reason: str) -> str:
    head = reason.split(":")[0].split("(")[0].strip().lower()
    return (head[:40] or "unknown").replace(" ", "_")


@_safe
def record_data_error(symbol: str) -> None:
    DATA_ERRORS.labels(symbol).inc()


@_safe
def record_portfolio(total: float, cash: float, positions_value: float, open_positions: int) -> None:
    PORTFOLIO_VALUE.set(total)
    PORTFOLIO_CASH.set(cash)
    POSITIONS_VALUE.set(positions_value)
    OPEN_POSITIONS.set(open_positions)


@_safe
def set_market_open(is_open: bool) -> None:
    MARKET_OPEN.set(1 if is_open else 0)


@_safe
def set_info(execution_mode: str, data_provider: str, strategy: str) -> None:
    BUILD_INFO.clear()
    BUILD_INFO.labels(execution_mode, data_provider, strategy).set(1)
