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
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from analysis.sentiment import SentimentScorer, build_sentiment_scorer
from core.config import AppConfig, load_config
from core.logging import get_logger, setup_logging
from core.models import Action
from data.base import MarketDataProvider
from data.market_data import MarketDataService
from data.news_data import build_news_provider
from database.database import DatabaseManager
from execution.base import OrderExecutor
from execution.manual import ManualExecutor
from execution.order_builder import decision_to_order, route_order
from execution.paper import PaperExecutor
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


# --------------------------------------------------------------------------
# Live (paper) trading loop
# --------------------------------------------------------------------------

def run_live_loop(config: AppConfig) -> None:
    db = DatabaseManager(config.database.url)
    db.init_db()

    provider = build_market_data_provider(config)
    news_provider = build_news_provider(config.news)
    sentiment_scorer = build_sentiment_scorer(config.news.sentiment_scorer)
    market_data = MarketDataService(provider, db)
    strategy = build_strategy(config)
    risk_manager = RiskManager(config.risk)
    executor = build_executor(config)
    portfolio = Portfolio(config.starting_balance)

    logger.info(
        "StockBot starting: symbols=%s provider=%s execution_mode=%s starting_balance=%.2f",
        config.symbols,
        config.data.provider,
        config.execution.mode,
        config.starting_balance,
    )

    try:
        while True:
            _run_cycle(
                config, db, provider, market_data, news_provider, sentiment_scorer,
                strategy, risk_manager, executor, portfolio,
            )
            time.sleep(config.polling_interval_seconds)
    except KeyboardInterrupt:
        logger.info("StockBot stopped by user.")


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
    setup_logging(config.log_level, config.log_file)

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
