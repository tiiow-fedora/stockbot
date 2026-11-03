"""
StockBot entry point.

    python main.py run                              # continuous paper-trading loop
    python main.py backtest                         # backtest all configured symbols
    python main.py backtest --symbol AAPL --days 365
    python main.py backtest --compare               # run every registered strategy side by side
    python main.py optimize --symbol AAPL           # grid-search strategy parameters (research tool)
    python main.py dashboard                        # read-only web dashboard
    python main.py init-db                          # create database tables and exit

Every object below is built by a small factory function so that swapping
a component (e.g. Alpaca -> a future classroom-platform adapter) means
changing one factory function, never the pipeline itself. This is the
dependency-injection seam described in the project spec.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from analysis.sentiment import SentimentScorer, build_sentiment_scorer
from core.config import AppConfig, load_config
from core.logging import get_logger
from core.models import Action
from data.base import MarketDataProvider
from data.market_data import MarketDataService
from data.news_data import build_news_provider
from database.database import DatabaseManager
from execution.base import OrderExecutor
from execution.manual import ManualExecutor
from execution.order_builder import decision_to_order, route_order
from execution.paper import PaperExecutor
from observability import metrics
from observability.health import HealthState, start_server
from observability.jsonlog import configure_logging
from observability.shutdown import GracefulShutdown
from observability.validation import ConfigError, ensure_valid, validate_config
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager
from strategy.base import Strategy
from strategy.registry import STRATEGY_BUILDERS, build_strategy_by_name

logger = get_logger("main")


# --------------------------------------------------------------------------
# Factories (the dependency-injection seam)
# --------------------------------------------------------------------------

def build_market_data_provider(config: AppConfig) -> MarketDataProvider:
    provider = config.data.provider
    if provider == "synthetic":
        from data.providers.synthetic import SyntheticMarketDataProvider

        return SyntheticMarketDataProvider(symbols=config.symbols)
    if provider == "csv":
        from data.providers.csv import CSVMarketDataProvider

        return CSVMarketDataProvider(csv_dir=config.data.csv_dir)
    if provider == "alpaca":
        from data.providers.alpaca import AlpacaMarketDataProvider

        return AlpacaMarketDataProvider()
    raise ValueError(f"unknown data provider: {provider!r}")


def build_strategy(config: AppConfig) -> Strategy:
    return build_strategy_by_name(config.strategy.name, config.strategy)


def build_executor(config: AppConfig) -> OrderExecutor:
    if config.execution.mode == "paper":
        paper = PaperExecutor(config.execution)
        if config.execution.mirror_to_alpaca:
            from execution.alpaca_mirror import AlpacaMirrorExecutor

            return AlpacaMirrorExecutor(paper)
        return paper
    if config.execution.mode == "manual":
        return ManualExecutor()
    raise ValueError(f"unknown execution mode: {config.execution.mode!r}")


def build_market_calendar(config: AppConfig):
    from data.market_calendar import build_market_calendar as _build_calendar

    name = config.data.market_calendar
    if name == "auto":
        name = "alpaca" if config.data.provider == "alpaca" else "always_open"
    return _build_calendar(name)


# --------------------------------------------------------------------------
# Live (paper) trading loop
# --------------------------------------------------------------------------

# How long the live loop waits after a failed iteration before retrying, and
# the ceiling that backoff doubles up to. The cap sits below the closed-market
# heartbeat interval so a long outage still re-checks on a sane cadence.
RETRY_BACKOFF_BASE_SECONDS = 30.0
RETRY_BACKOFF_MAX_SECONDS = 900.0


def run_live_loop(config: AppConfig) -> None:
    db = DatabaseManager(config.database.url)
    db.init_db()

    provider = build_market_data_provider(config)
    news_provider = build_news_provider(config.news)
    sentiment_scorer = build_sentiment_scorer(config.news.sentiment_scorer)
    market_data = MarketDataService(provider, db)
    strategy = build_strategy(config)
    risk_manager = RiskManager(config.risk, db=db)
    executor = build_executor(config)
    portfolio = Portfolio(config.starting_balance)
    calendar = build_market_calendar(config)

    shutdown = GracefulShutdown().install()
    health, health_server = _start_observability(config, db)
    metrics.set_info(config.execution.mode, config.data.provider, config.strategy.name)

    logger.info(
        "StockBot starting: symbols=%s provider=%s execution_mode=%s starting_balance=%.2f market_calendar=%s",
        config.symbols,
        config.data.provider,
        config.execution.mode,
        config.starting_balance,
        type(calendar).__name__,
    )

    consecutive_failures = 0
    try:
        while not shutdown.requested:
            try:
                now = datetime.now(timezone.utc)
                is_open = calendar.is_open(now)
                metrics.set_market_open(is_open)
                if not is_open:
                    sleep_seconds = _heartbeat_while_closed(config, db, market_data, portfolio, calendar, now)
                    consecutive_failures = 0
                    metrics.record_cycle("closed")
                    health.beat()
                    with shutdown.interruptible():
                        time.sleep(sleep_seconds)
                    continue

                cycle_started = time.monotonic()
                _run_cycle(
                    config, db, provider, market_data, news_provider, sentiment_scorer,
                    strategy, risk_manager, executor, portfolio,
                )
                consecutive_failures = 0
                metrics.record_cycle("ok", time.monotonic() - cycle_started)
                health.beat()
                with shutdown.interruptible():
                    time.sleep(config.polling_interval_seconds)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # A transient outage - a DNS blip, an Alpaca 5xx, a dropped
                # connection - must not kill the process. calendar.is_open()
                # and calendar.next_open() are unguarded network calls, so
                # before this a single failed lookup propagated straight out
                # of main() and the container died; restart:unless-stopped
                # then papered over it (336 restarts in production). Back off
                # and retry instead, and only log the traceback on the first
                # failure of a streak so a long outage does not flood the log.
                consecutive_failures += 1
                metrics.record_failure(consecutive_failures)
                backoff = min(
                    RETRY_BACKOFF_BASE_SECONDS * (2 ** (consecutive_failures - 1)),
                    RETRY_BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    "Live loop iteration failed (%d consecutive): %s. Retrying in %.0fs.",
                    consecutive_failures, exc, backoff,
                    exc_info=(consecutive_failures == 1),
                )
                with shutdown.interruptible():
                    time.sleep(backoff)
    except KeyboardInterrupt:
        pass
    finally:
        health.shutting_down = True
        logger.info("StockBot shutting down (%s).", "signal" if shutdown.requested else "interrupt")
        shutdown.uninstall()
        if health_server is not None:
            health_server.shutdown()
            health_server.server_close()


def _db_ping(db) -> bool:
    from sqlalchemy import text

    with db.engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def _start_observability(config: AppConfig, db):
    """Health/metrics HTTP server on $METRICS_PORT (default 9108, 0 disables).
    Failure to bind is logged, never fatal - observability must not stop trading."""
    stale_after = max(config.polling_interval_seconds * 5, config.max_closed_market_heartbeat_seconds * 2, 120)
    health = HealthState(stale_after=stale_after, db_check=lambda: _db_ping(db))
    port = int(os.environ.get("METRICS_PORT", "9108"))
    if port == 0:
        return health, None
    try:
        server = start_server(health, port)
        logger.info("Health/metrics server listening on :%d (/healthz /readyz /metrics)", port)
        return health, server
    except OSError as exc:
        logger.warning("Could not start health/metrics server on :%d: %s", port, exc)
        return health, None


def _heartbeat_while_closed(config, db, market_data, portfolio, calendar, now: datetime) -> float:
    """The market is closed: mark the portfolio to last-known prices (so
    the dashboard shows "running," not "stale," across a long
    overnight/weekend gap) and skip the strategy/risk/order pipeline
    entirely. The local paper simulator has no concept of trading hours
    on its own - if we ran the full pipeline here, it would happily fill
    orders instantly at a stale price, which a real trading session never
    would. Returns how many seconds to sleep before checking again."""
    prices: Dict[str, float] = {}
    for symbol in config.symbols:
        try:
            snapshot = market_data.get_snapshot(symbol)
            prices[symbol] = snapshot.price
        except Exception as exc:
            logger.warning("Market data unavailable for %s while closed: %s", symbol, exc)

    if prices:
        closed_snapshot = portfolio.snapshot(now, prices)
        db.save_portfolio_snapshot(closed_snapshot)
        metrics.record_portfolio(
            closed_snapshot.total_value, closed_snapshot.cash, closed_snapshot.positions_value,
            len(portfolio.positions),
        )

    next_open = calendar.next_open(now)
    seconds_until_open = max(1.0, (next_open - now).total_seconds())
    sleep_seconds = min(seconds_until_open, config.max_closed_market_heartbeat_seconds)
    logger.info(
        "Market closed - next open %s (in %.0f min). No trading; rechecking in %.0f min.",
        next_open.isoformat(),
        seconds_until_open / 60,
        sleep_seconds / 60,
    )
    return sleep_seconds


def _run_cycle(
    config, db, provider, market_data, news_provider, sentiment_scorer, strategy, risk_manager, executor, portfolio
) -> None:
    prices: Dict[str, float] = {}
    snapshots = {}

    for symbol in config.symbols:
        try:
            snapshot = market_data.get_snapshot(symbol)
        except Exception as exc:
            logger.error("Market data outage for %s: %s", symbol, exc)
            metrics.record_data_error(symbol)
            continue
        snapshots[symbol] = snapshot
        prices[symbol] = snapshot.price

    if not prices:
        logger.warning("No market data available this cycle; skipping.")
        return

    # Protective exits first, across every symbol currently held.
    for order in risk_manager.check_stop_loss_take_profit(portfolio, prices):
        route_order(order, risk_manager, portfolio, executor, prices, prices[order.symbol], order.timestamp, db)

    for symbol, snapshot in snapshots.items():
        try:
            history = market_data.get_history(symbol, config.data.history_lookback_days, config.data.timeframe)
            history = [c for c in history if c.timestamp < snapshot.timestamp]

            news_start = snapshot.timestamp - timedelta(days=1)
            news = news_provider.get_news(symbol, news_start, snapshot.timestamp)
            if news and sentiment_scorer is not None:
                news = sentiment_scorer.score_news_items(news)
            if news:
                db.save_news(news)

            decision = strategy.generate_decision(
                symbol=symbol,
                snapshot=snapshot,
                history=history,
                news=news,
                current_position_qty=portfolio.position_qty(symbol),
                cash=portfolio.cash,
                portfolio_value=portfolio.total_value(prices),
            )
            db.save_decision(decision)
            metrics.record_decision(symbol, decision.action.value)
            logger.info(
                "%s score=%.3f decision=%s confidence=%.2f",
                symbol,
                decision.scores.composite,
                decision.action.value,
                decision.confidence,
            )

            order = decision_to_order(decision, risk_manager, portfolio, prices)
            if order is not None:
                route_order(order, risk_manager, portfolio, executor, prices, snapshot.price, snapshot.timestamp, db)

        except Exception:
            logger.exception("Unhandled error processing %s this cycle", symbol)

    snapshot_ts = next(iter(snapshots.values())).timestamp
    portfolio_snapshot = portfolio.snapshot(snapshot_ts, prices)
    db.save_portfolio_snapshot(portfolio_snapshot)
    metrics.record_portfolio(
        portfolio_snapshot.total_value, portfolio_snapshot.cash, portfolio_snapshot.positions_value,
        len(portfolio.positions),
    )
    logger.info(
        "Portfolio: cash=%.2f positions_value=%.2f total_value=%.2f",
        portfolio_snapshot.cash,
        portfolio_snapshot.positions_value,
        portfolio_snapshot.total_value,
    )


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

def run_backtest(
    config: AppConfig, symbol_filter: Optional[str], days: int, persist: bool, compare: bool
) -> None:
    from backtesting.engine import Backtester

    provider = build_market_data_provider(config)
    symbols = [symbol_filter] if symbol_filter else config.symbols

    db = DatabaseManager(config.database.url) if persist else None
    if db:
        db.init_db()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    if compare:
        _run_compare(config, provider, symbols, start, end, db)
        return

    results = []
    for symbol in symbols:
        candles = provider.get_historical_candles(symbol, start, end, config.data.timeframe)
        strategy = build_strategy(config)  # fresh strategy instance per symbol; strategies are stateless here
        backtester = Backtester(
            strategy=strategy,
            risk_config=config.risk,
            execution_config=config.execution,
            starting_balance=config.backtest.starting_balance,
            warmup_bars=config.backtest.warmup_bars,
            db=db,
        )
        try:
            result = backtester.run(symbol, candles)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            continue
        results.append(result)
        print(result.summary())

    if not results:
        print("No backtest results produced (not enough historical data).")
        return

    avg_return = sum(r.total_return_pct for r in results) / len(results)
    avg_benchmark = sum(r.benchmark_return_pct for r in results) / len(results)
    print(f"\nAverage across {len(results)} symbols: strategy {avg_return:+.2%} vs buy-and-hold {avg_benchmark:+.2%}")


def _run_compare(config: AppConfig, provider, symbols, start, end, db) -> None:
    from backtesting.compare import compare_strategies, format_comparison_table

    for symbol in symbols:
        candles = provider.get_historical_candles(symbol, start, end, config.data.timeframe)
        print(f"\n=== {symbol} ===")
        try:
            results = compare_strategies(
                symbol=symbol,
                candles=candles,
                strategy_config=config.strategy,
                risk_config=config.risk,
                execution_config=config.execution,
                starting_balance=config.backtest.starting_balance,
                warmup_bars=config.backtest.warmup_bars,
                db=db,
            )
        except ValueError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            continue
        print(format_comparison_table(results))


def run_optimize(config: AppConfig, symbol: str, days: int, top: int, objective: str) -> None:
    from backtesting.optimize import format_grid_search_table, grid_search

    provider = build_market_data_provider(config)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    candles = provider.get_historical_candles(symbol, start, end, config.data.timeframe)

    results = grid_search(
        symbol=symbol,
        candles=candles,
        base_strategy_config=config.strategy,
        risk_config=config.risk,
        execution_config=config.execution,
        strategy_name=config.strategy.name,
        starting_balance=config.backtest.starting_balance,
        warmup_bars=config.backtest.warmup_bars,
        objective=objective,
    )
    print(format_grid_search_table(results, top=top, objective=objective))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="StockBot - modular paper trading research system")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: config/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="run the continuous paper-trading loop")

    backtest_parser = sub.add_parser("backtest", help="backtest the configured strategy")
    backtest_parser.add_argument("--symbol", default=None, help="backtest a single symbol instead of all configured symbols")
    backtest_parser.add_argument("--days", type=int, default=365, help="lookback window in days")
    backtest_parser.add_argument("--persist", action="store_true", help="write decisions/orders/fills to the database")
    backtest_parser.add_argument(
        "--compare", action="store_true",
        help="run every registered strategy (see strategy/registry.py) and print a comparison table",
    )

    sub.add_parser("init-db", help="create database tables and exit")
    sub.add_parser("validate-config", help="validate config + environment and exit (non-zero on errors)")

    dashboard_parser = sub.add_parser("dashboard", help="run the read-only web dashboard")
    dashboard_parser.add_argument("--host", default="0.0.0.0")
    dashboard_parser.add_argument("--port", type=int, default=8000)

    optimize_parser = sub.add_parser(
        "optimize", help="grid-search strategy parameters against historical data (research tool, not a promise of future returns)"
    )
    optimize_parser.add_argument("--symbol", required=True, help="symbol to optimize against")
    optimize_parser.add_argument("--days", type=int, default=365, help="lookback window in days")
    optimize_parser.add_argument("--top", type=int, default=5, help="how many top parameter combos to print")
    optimize_parser.add_argument(
        "--objective", default="total_return_pct", choices=["total_return_pct", "sharpe_ratio"],
        help="metric to rank parameter combinations by",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config.log_level, config.log_file)

    if args.command == "validate-config":
        issues = validate_config(config)
        for issue in issues:
            print(issue)
        errors = [i for i in issues if i.level == "error"]
        print(f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
        sys.exit(1 if errors else 0)
    if args.command in ("run", "dashboard"):
        try:
            for issue in ensure_valid(config):
                logger.warning("config: %s", issue)
        except ConfigError as exc:
            logger.error("%s", exc)
            sys.exit(2)

    if args.command == "run":
        run_live_loop(config)
    elif args.command == "backtest":
        run_backtest(config, args.symbol, args.days, args.persist, args.compare)
    elif args.command == "optimize":
        run_optimize(config, args.symbol, args.days, args.top, args.objective)
    elif args.command == "init-db":
        db = DatabaseManager(config.database.url)
        db.init_db()
        print(f"Database initialized at {config.database.url}")
    elif args.command == "dashboard":
        import uvicorn

        from dashboard.app import create_app

        uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
