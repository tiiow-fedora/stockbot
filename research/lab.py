"""
Strategy comparison runner + CLI.

    python -m research.lab compare            # every strategy x every synthetic scenario
    python -m research.lab compare --bars 750 --seed 3 --markdown out.md --json out.json
    python -m research.lab list               # registered strategies by family

``compare`` runs each strategy through the REAL engine (``research.harness``) on the
synthetic scenario suite plus a cointegrated-pair scenario, and prints Sharpe,
total return, max drawdown, exposure and trade counts - the raw material for the
README's comparison tables. Paper trading / simulation only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.models import Candle
from research import metrics, synth
from research.base import ResearchStrategy
from research.harness import RunResult, run_backtest
from research.registry import STRATEGIES, by_family


@dataclass
class Scenario:
    name: str
    candles: List[Candle]
    partner: List[Candle]   # second instrument for partner strategies (benchmark / pair leg)


def build_scenarios(bars: int = 500, seed: int = 0) -> List[Scenario]:
    """The scenario suite + a ``cointegrated_pair`` scenario. Non-pair scenarios use a fixed random-walk
    benchmark as the partner (same timestamps), so relative-strength has something to be relative to."""
    benchmark = synth.gbm(bars, mu=0.0003, sigma=0.012, seed=seed + 1000, symbol="BENCH")
    out = [Scenario(n, c, benchmark) for n, c in synth.scenario_suite(bars, seed).items()]
    a, b = synth.cointegrated_pair(bars, seed=seed + 500)
    out.append(Scenario("cointegrated_pair", a, b))
    return out


def make_strategy(name: str, scenario: Scenario) -> ResearchStrategy:
    cls = STRATEGIES[name]
    return cls(partner=scenario.partner) if cls.requires_partner else cls()


def run_matrix(
    names: Optional[Sequence[str]] = None, bars: int = 500, seed: int = 0
) -> Dict[str, Dict[str, RunResult]]:
    names = list(names or STRATEGIES)
    scenarios = build_scenarios(bars, seed)
    return {n: {s.name: run_backtest(make_strategy(n, s), s.candles) for s in scenarios} for n in names}


def _fmt(v: float, pct: bool = False, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def markdown_table(matrix: Dict[str, Dict[str, RunResult]], metric: str = "sharpe") -> str:
    """Strategy x scenario table of one metric (``sharpe``, ``total_return``, ``max_drawdown``,
    ``avg_exposure``, ``num_trades``) with a buy-and-hold row for reference."""
    pct = metric in {"total_return", "max_drawdown", "avg_exposure", "time_in_market", "cagr"}
    scen = list(next(iter(matrix.values())))
    rows = ["| strategy | " + " | ".join(scen) + " | mean |", "|---|" + "---|" * (len(scen) + 1)]
    for name, by_scen in matrix.items():
        vals = [getattr(by_scen[s].report, metric) for s in scen]
        mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rows.append(f"| `{name}` | " + " | ".join(_fmt(v, pct, 2 if metric != "num_trades" else 0) for v in vals)
                    + f" | {_fmt(mean, pct)} |")
    first = next(iter(matrix.values()))
    bench = [getattr(first[s].benchmark_report, metric, float('nan')) for s in scen]
    rows.append("| *buy & hold* | " + " | ".join(_fmt(v, pct) for v in bench) + f" | {_fmt(float(np.nanmean(bench)), pct)} |")
    return "\n".join(rows)


def to_json(matrix: Dict[str, Dict[str, RunResult]]) -> dict:
    def clean(d):
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}

    return {n: {s: clean(r.report.as_dict()) for s, r in by.items()} for n, by in matrix.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.lab", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered strategies")
    c = sub.add_parser("compare", help="run strategy x scenario matrix")
    c.add_argument("--bars", type=int, default=500)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--strategy", action="append", help="restrict to these strategies (repeatable)")
    c.add_argument("--markdown", help="write markdown tables here")
    c.add_argument("--json", help="write JSON metrics here")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for fam, names in by_family().items():
            print(f"{fam}:")
            for n in names:
                print(f"  {n:28s} {STRATEGIES[n].description}")
        return 0

    matrix = run_matrix(args.strategy, args.bars, args.seed)
    out = []
    for metric, title in [("sharpe", "Sharpe"), ("total_return", "Total return"), ("max_drawdown", "Max drawdown"),
                          ("avg_exposure", "Average exposure"), ("num_trades", "Closed trades")]:
        out.append(f"### {title}\n\n{markdown_table(matrix, metric)}\n")
    text = "\n".join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(matrix), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
