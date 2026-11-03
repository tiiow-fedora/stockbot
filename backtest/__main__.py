"""CLI:  python -m backtest --synthetic --strategy mean_reversion --out backtest_reports"""

from __future__ import annotations

import argparse
import sys

from backtest.loaders import load_csv_series, synthetic_series
from backtest.report import save_report
from backtest.runner import BacktestHarness
from core.config import StrategyConfig
from strategy.registry import STRATEGY_BUILDERS, build_strategy_by_name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m backtest", description="Paper-only backtest harness")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use built-in synthetic scenarios")
    src.add_argument("--csv-dir", help="directory of {SYMBOL}.csv files")
    p.add_argument("--symbols", nargs="+", help="symbols (with --csv-dir)")
    p.add_argument("--scenarios", nargs="+", help="scenario names (with --synthetic)")
    p.add_argument("--bars", type=int, default=250)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strategy", default="technical", choices=sorted(STRATEGY_BUILDERS))
    p.add_argument("--balance", type=float, default=100_000.0)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="backtest_reports")
    args = p.parse_args(argv)

    if args.synthetic:
        series = synthetic_series(args.scenarios, n=args.bars, seed=args.seed)
    else:
        if not args.symbols:
            p.error("--symbols is required with --csv-dir")
        series = load_csv_series(args.csv_dir, args.symbols)

    strategy = build_strategy_by_name(args.strategy, StrategyConfig())
    harness = BacktestHarness(strategy, args.strategy, starting_balance=args.balance)
    name = args.name or f"{args.strategy}-{'synthetic' if args.synthetic else 'csv'}"
    result = harness.run(series, name=name)
    path = save_report(result.report, args.out)
    for row in result.report.per_symbol:
        print(f"  {row['label']:<10} pnl=${row['pnl']:+,.2f} ret={row['return_pct']:+.2%} "
              f"dd={row['max_drawdown_pct']:.2%} win={row['win_rate']:.0%} trades={row['num_trades']}")
    print(result.report.summary())
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
