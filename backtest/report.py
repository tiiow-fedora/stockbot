"""JSON-serialisable run reports (what the dashboard reads)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunReport:
    name: str
    strategy: str
    created_at: str
    starting_balance: float
    ending_balance: float
    total_pnl: float
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    max_drawdown_abs: float
    win_rate: float
    num_trades: int
    profit_factor: Optional[float]
    sharpe_ratio: Optional[float]
    per_symbol: List[dict] = field(default_factory=list)
    equity_curve: List[List] = field(default_factory=list)   # [[iso, value], ...]
    trades: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        pf = "n/a" if self.profit_factor is None else f"{self.profit_factor:.2f}"
        return (
            f"[{self.name}] strategy={self.strategy} pnl=${self.total_pnl:+,.2f} "
            f"({self.total_return_pct:+.2%}) vs buy&hold {self.benchmark_return_pct:+.2%} | "
            f"max_dd={self.max_drawdown_pct:.2%} win_rate={self.win_rate:.1%} "
            f"trades={self.num_trades} profit_factor={pf}"
        )


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_report(report: RunReport, directory: str) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_slug(report.name)}.json"
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    return path


def load_report(path: str) -> RunReport:
    return RunReport(**json.loads(Path(path).read_text(encoding="utf-8")))


def list_report_files(directory: str) -> List[Path]:
    d = Path(directory)
    return sorted(d.glob("*.json")) if d.is_dir() else []
