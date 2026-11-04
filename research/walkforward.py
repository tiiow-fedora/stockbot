"""
Walk-forward parameter optimisation.

Why: picking the best parameters on the whole history and reporting that
backtest is circular - the parameters have seen the data they are scored on.
Walk-forward analysis (WFA) repeats the honest procedure a live trader would
follow: fit on the past, trade the unseen next block, roll forward, and judge
the strategy ONLY on the stitched out-of-sample (OOS) blocks.

Procedure per window (see :func:`generate_windows`)::

    |<---------- train ---------->|<-- test -->|
                                  ^ test_start == train_end

1. Grid-search ``strategy_cls.param_combinations(grid)`` on the TRAIN slice
   through the real engine (:func:`research.harness.run_backtest`).
2. Pick the best candidate by ``cfg.objective`` (ties -> earliest in grid order,
   so results are deterministic).
3. Run ONLY that candidate on the TEST slice.

Timing / no look-ahead
----------------------
A slice ``[start, end)`` is run as ``Backtester(candles[start-extra:end],
warmup_bars=extra + 1)`` with ``extra = min(warmup_bars, start)``. The engine's
first decision bar is ``candles[warmup_bars - 1]`` of the slice, i.e. GLOBAL
index ``start`` - exactly the window start - and that decision's order fills at
the open of bar ``start + 1``. The last decision is on bar ``end - 2`` (its
order fills at ``end - 1``), so a slice never touches bar ``end`` or later and
a train slice never touches a test bar. A window therefore yields ``end -
start - 1`` equity points. The ``extra`` bars before ``start`` are history only
(they warm indicators up); they are never traded.

Stitching
---------
Every OOS block starts flat with the full ``starting_balance`` (positions are not
carried across blocks - each block is an independent paper run, which is how the
parameters changed anyway). The stitched curve chains the blocks
multiplicatively: ``equity = running_equity * block_equity / starting_balance``,
beginning with one point at ``starting_balance``. Windows where no candidate was
eligible contribute a flat (cash) block. With ``step_bars > test_bars`` the bars
between test blocks are simply absent from the curve.

Cost control
------------
Cost is ``windows x grid`` full engine backtests. :class:`WalkForwardOptimizer`
refuses grids bigger than ``MAX_GRID`` (ValueError) unless ``max_combos`` is set,
in which case the grid is subsampled ONCE (seeded, same subset in every window,
original grid order preserved for tie-breaking) - see :func:`random_search`.
``max_workers > 1`` evaluates windows in a ``ProcessPoolExecutor``; results are
collected in window order and each window is computed by the same pure function,
so the output is identical to the serial default (the objective callable must
then be picklable, i.e. a module-level function, not a lambda).

Everything is paper trading; nothing here places orders.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from core.config import ExecutionConfig, RiskConfig
from core.models import Candle
from research import metrics
from research.base import ResearchStrategy
from research.harness import run_backtest, trade_pnls

Objective = Union[str, Callable[[metrics.PerformanceReport], float]]

#: Hard cap on the number of candidates per window when ``max_combos`` is not given.
MAX_GRID = 2000

_NAMED_OBJECTIVES = ("sharpe", "sortino", "calmar", "total_return")
_NEG_INF = float("-inf")
#: Errors from a single candidate backtest that make the candidate ineligible instead of aborting.
_CANDIDATE_ERRORS = (ValueError, TypeError, ArithmeticError)


# --------------------------------------------------------------------------
# Config and windows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward settings (all sizes in bars).

    * ``train_bars`` / ``test_bars``: size of the in-sample / out-of-sample block.
    * ``step_bars``: how far the window advances; default ``test_bars`` (OOS blocks
      tile the series). Must be ``>= test_bars`` so OOS blocks never overlap.
    * ``anchored``: True -> expanding train window (always starts at bar 0, ends at
      ``test_start``); False -> rolling window of exactly ``train_bars``.
    * ``objective``: ``'sharpe' | 'sortino' | 'calmar' | 'total_return'`` or a
      callable ``PerformanceReport -> float``. Non-finite values (NaN, +-inf) are
      treated as ``-inf`` (worst possible), so an undefined metric never wins.
    * ``min_trades``: a candidate with fewer than this many CLOSED trades on the
      train window is ineligible. Default 0: note that a position still open at
      the end of the window is NOT a closed trade, so in a clean trend a
      buy-and-hold-like candidate has 0 closed trades; raising the bar excludes it.
      A candidate whose objective is non-finite (NaN/inf, e.g. Sharpe of a flat
      curve) can never be chosen. If no candidate is eligible AND finite the window
      is skipped: no params are chosen, the OOS block is held in cash and it is
      flagged ``skipped``.
    * ``warmup_bars``: extra bars before each window given to the strategy as
      history so indicators are warm when the first decision is made.
    """

    train_bars: int = 250
    test_bars: int = 60
    step_bars: Optional[int] = None
    anchored: bool = False
    objective: Objective = "sharpe"
    min_trades: int = 0
    warmup_bars: int = 60
    starting_balance: float = 100_000.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.step_bars is None:
            object.__setattr__(self, "step_bars", self.test_bars)
        if self.train_bars < 10:
            raise ValueError("train_bars must be >= 10")
        if self.test_bars < 5:
            raise ValueError("test_bars must be >= 5")
        if self.step_bars < self.test_bars:
            raise ValueError("step_bars must be >= test_bars (OOS blocks may not overlap)")
        if self.min_trades < 0:
            raise ValueError("min_trades must be >= 0")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if not callable(self.objective) and self.objective not in _NAMED_OBJECTIVES:
            raise ValueError(f"objective must be one of {_NAMED_OBJECTIVES} or a callable, got {self.objective!r}")


@dataclass(frozen=True)
class Window:
    """Half-open index ranges ``[train_start, train_end)`` and ``[test_start, test_end)``;
    ``test_start == train_end``."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_windows(n_bars: int, cfg: WalkForwardConfig) -> List[Window]:
    """All walk-forward windows that fit in ``n_bars`` bars (pure function).

    Window ``k`` has ``train_end = train_bars + k*step`` and ``test_end = train_end +
    test_bars``; the train block starts at ``k*step`` (rolling) or 0 (anchored). Count =
    ``(n_bars - train_bars - test_bars) // step + 1`` when that is >= 0, else none.
    """
    if n_bars < 0:
        raise ValueError("n_bars must be >= 0")
    out: List[Window] = []
    k = 0
    while True:
        train_end = cfg.train_bars + k * cfg.step_bars
        test_end = train_end + cfg.test_bars
        if test_end > n_bars:
            return out
        train_start = 0 if cfg.anchored else k * cfg.step_bars
        out.append(Window(train_start, train_end, train_end, test_end))
        k += 1


def random_search(
    strategy_cls: Type[ResearchStrategy],
    grid: Optional[Mapping[str, Sequence[Any]]] = None,
    n: int = 50,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Seeded random subset (without replacement) of the VALID grid combinations, returned in
    original grid order. Returns the full list when it is not larger than ``n``."""
    if n <= 0:
        raise ValueError("n must be > 0")
    combos = strategy_cls.param_combinations(grid)
    if len(combos) <= n:
        return combos
    keep = sorted(random.Random(seed).sample(range(len(combos)), n))
    return [combos[i] for i in keep]


# --------------------------------------------------------------------------
# Single-slice backtest (the only place that touches slicing / warm-up)
# --------------------------------------------------------------------------

def backtest_slice(
    strategy: ResearchStrategy,
    candles: Sequence[Candle],
    start: int,
    end: int,
    warmup_bars: int,
    *,
    risk: Optional[RiskConfig],
    execution: Optional[ExecutionConfig],
    starting_balance: float,
    periods_per_year: int,
):
    """Run ``strategy`` so that its first decision is on bar ``start`` and it never sees bar
    ``end`` or later (see module docstring)."""
    extra = min(warmup_bars, start)
    return run_backtest(
        strategy,
        candles[start - extra:end],
        risk=risk,
        execution=execution,
        starting_balance=starting_balance,
        warmup_bars=extra + 1,
        periods_per_year=periods_per_year,
    )


def objective_value(objective: Objective, report: metrics.PerformanceReport) -> float:
    """Objective score of a report; any non-finite or failing value becomes ``-inf``."""
    try:
        v = float(getattr(report, objective) if isinstance(objective, str) else objective(report))
    except _CANDIDATE_ERRORS:
        return _NEG_INF
    return v if math.isfinite(v) else _NEG_INF


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class WindowResult:
    index: int
    window: Window
    params: Optional[Dict[str, Any]]            # chosen overrides; None when skipped
    is_score: float                              # train objective of the chosen params (nan if skipped)
    oos_score: float                             # test objective of the chosen params (nan if skipped)
    is_report: Optional[metrics.PerformanceReport]
    oos_report: Optional[metrics.PerformanceReport]
    n_candidates: int
    n_eligible: int
    skipped: bool
    oos_equity: np.ndarray                       # raw block equity, starts from ~starting_balance
    oos_position_value: np.ndarray
    oos_trade_pnls: List[float]


def _clean(obj: Any) -> Any:
    """Recursively convert to plain JSON types: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, np.ndarray):
        return [_clean(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _clean({k: getattr(obj, k) for k in obj.__dataclass_fields__})
    return obj


@dataclass
class WalkForwardResult:
    """Outcome of a walk-forward run.

    * ``windows``: per-window :class:`WindowResult` records.
    * ``equity``: stitched OOS equity curve (starts at ``starting_balance``).
    * ``oos_report``: ``metrics.compute_report`` of the stitched curve (+ exposure and
      all OOS closed trades).
    * ``efficiency`` (walk-forward efficiency): mean OOS objective / mean IS objective over
      non-skipped windows with finite scores. NaN when no finite pair exists or the IS
      mean is not positive (a ratio of an arbitrary-sign IS mean is meaningless).
      ~1 = the in-sample edge persisted; ~0 or negative = it did not.
    * ``stability``: ``modal_fraction`` (share of non-skipped windows that chose the most
      common param set; 1.0 = same params every time) and ``dispersion`` per parameter
      (numeric: std of the chosen values divided by the grid's value range, 0 = constant,
      ~0.3 = chosen values spread over the grid; non-numeric: 1 - share of the modal value).

    ``overfit_flag`` (heuristic, a warning not a proof): True when there are >= 3 active
    windows, the mean IS objective is positive, and EITHER efficiency < 0.5 (less than half
    of the in-sample edge survived out of sample) OR the mean OOS objective is <= 0.
    """

    config: WalkForwardConfig
    strategy_name: str
    windows: List[WindowResult]
    equity: np.ndarray
    oos_report: metrics.PerformanceReport
    param_ranges: Dict[str, float] = field(default_factory=dict)  # grid value range per numeric param

    # -- derived -------------------------------------------------------

    @property
    def active(self) -> List[WindowResult]:
        return [w for w in self.windows if not w.skipped]

    @property
    def n_skipped(self) -> int:
        return len(self.windows) - len(self.active)

    def _mean(self, attr: str) -> float:
        vals = [getattr(w, attr) for w in self.active]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_is_objective(self) -> float:
        return self._mean("is_score")

    @property
    def mean_oos_objective(self) -> float:
        return self._mean("oos_score")

    @property
    def efficiency(self) -> float:
        pairs = [(w.is_score, w.oos_score) for w in self.active
                 if math.isfinite(w.is_score) and math.isfinite(w.oos_score)]
        if not pairs:
            return float("nan")
        is_mean = float(np.mean([p[0] for p in pairs]))
        oos_mean = float(np.mean([p[1] for p in pairs]))
        return oos_mean / is_mean if is_mean > 1e-12 else float("nan")

    @property
    def stability(self) -> Dict[str, Any]:
        chosen = [w.params for w in self.active if w.params is not None]
        if not chosen:
            return {"modal_params": None, "modal_fraction": float("nan"), "dispersion": {}}
        key = lambda p: tuple(sorted(p.items(), key=lambda kv: kv[0]))
        counts = Counter(key(p) for p in chosen)
        modal, n_modal = max(counts.items(), key=lambda kv: kv[1])  # ties -> earliest window's choice
        disp: Dict[str, float] = {}
        for name in chosen[0]:
            vals = [p[name] for p in chosen]
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
            if numeric:
                rng = self.param_ranges.get(name, 0.0)
                disp[name] = float(np.std(vals) / rng) if rng > 0 else 0.0
            else:
                disp[name] = 1.0 - Counter(map(repr, vals)).most_common(1)[0][1] / len(vals)
        return {"modal_params": dict(modal), "modal_fraction": n_modal / len(chosen), "dispersion": disp}

    @property
    def overfit_flag(self) -> bool:
        if len(self.active) < 3:
            return False
        is_mean, oos_mean, wfe = self.mean_is_objective, self.mean_oos_objective, self.efficiency
        if not (math.isfinite(is_mean) and is_mean > 0):
            return False
        return (math.isfinite(wfe) and wfe < 0.5) or (math.isfinite(oos_mean) and oos_mean <= 0)

    # -- output --------------------------------------------------------

    def summary(self) -> str:
        c = self.config
        obj = c.objective if isinstance(c.objective, str) else getattr(c.objective, "__name__", "custom")
        f = lambda v, p=2: "n/a" if not (v is not None and math.isfinite(v)) else f"{v:.{p}f}"
        st = self.stability
        lines = [
            f"Walk-forward {self.strategy_name}: {len(self.windows)} windows "
            f"({'anchored' if c.anchored else 'rolling'}, train={c.train_bars}, test={c.test_bars}, "
            f"step={c.step_bars}, objective={obj}, skipped={self.n_skipped})",
            f"  OOS (stitched): {self.oos_report.summary()}",
            f"  mean objective  IS={f(self.mean_is_objective)}  OOS={f(self.mean_oos_objective)}  "
            f"efficiency={f(self.efficiency)}",
            f"  stability: modal params {st['modal_params']} in {f(st['modal_fraction'] * 100, 0) if st['modal_fraction'] == st['modal_fraction'] else 'n/a'}% of windows",
            f"  overfit_flag={self.overfit_flag}",
        ]
        for w in self.windows:
            if w.skipped:
                lines.append(f"  [{w.index}] test {w.window.test_start}-{w.window.test_end}: SKIPPED (no eligible candidate)")
            else:
                lines.append(
                    f"  [{w.index}] test {w.window.test_start}-{w.window.test_end}: {w.params} "
                    f"IS={f(w.is_score)} OOS={f(w.oos_score)} oos_ret={f(w.oos_report.total_return * 100, 1)}%"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable (plain python types, NaN/inf -> None)."""
        c = self.config
        cfg = {k: getattr(c, k) for k in c.__dataclass_fields__}
        if callable(c.objective):
            cfg["objective"] = getattr(c.objective, "__name__", "custom")
        return _clean({
            "strategy": self.strategy_name,
            "config": cfg,
            "windows": [
                {
                    "index": w.index,
                    "train": [w.window.train_start, w.window.train_end],
                    "test": [w.window.test_start, w.window.test_end],
                    "params": w.params,
                    "skipped": w.skipped,
                    "is_score": w.is_score,
                    "oos_score": w.oos_score,
                    "n_candidates": w.n_candidates,
                    "n_eligible": w.n_eligible,
                    "is_report": w.is_report.as_dict() if w.is_report else None,
                    "oos_report": w.oos_report.as_dict() if w.oos_report else None,
                }
                for w in self.windows
            ],
            "equity": self.equity,
            "oos_report": self.oos_report.as_dict(),
            "mean_is_objective": self.mean_is_objective,
            "mean_oos_objective": self.mean_oos_objective,
            "efficiency": self.efficiency,
            "stability": self.stability,
            "overfit_flag": self.overfit_flag,
            "n_skipped": self.n_skipped,
        })


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Job:
    """Everything one window needs (picklable for the process pool)."""

    index: int
    window: Window
    candles: Sequence[Candle]
    strategy_cls: Type[ResearchStrategy]
    combos: List[Dict[str, Any]]
    cfg: WalkForwardConfig
    risk: Optional[RiskConfig]
    execution: Optional[ExecutionConfig]
    partner: Optional[Sequence[Candle]]


def _build(job: _Job, overrides: Dict[str, Any]) -> ResearchStrategy:
    if job.partner is not None:
        return job.strategy_cls(partner=job.partner, **overrides)
    return job.strategy_cls(**overrides)


def _slice_run(job: _Job, overrides: Dict[str, Any], start: int, end: int):
    return backtest_slice(
        _build(job, overrides), job.candles, start, end, job.cfg.warmup_bars,
        risk=job.risk, execution=job.execution,
        starting_balance=job.cfg.starting_balance, periods_per_year=job.cfg.periods_per_year,
    )


def _run_window(job: _Job) -> WindowResult:
    """Pure function: train-search then one OOS run. Only ``candles[:test_start]`` is read
    until the choice is made."""
    w, cfg = job.window, job.cfg
    best: Optional[Tuple[float, int]] = None   # (score, combo index) - strictly-greater wins => first in grid order
    best_report: Optional[metrics.PerformanceReport] = None
    n_eligible = 0
    for i, combo in enumerate(job.combos):
        try:
            rep = _slice_run(job, combo, w.train_start, w.train_end).report
        except _CANDIDATE_ERRORS:
            continue
        if rep.num_trades < cfg.min_trades:
            continue
        n_eligible += 1
        score = objective_value(cfg.objective, rep)
        if score == _NEG_INF:
            continue
        if best is None or score > best[0]:
            best, best_report = (score, i), rep

    n_oos = w.test_end - w.test_start - 1
    flat = np.full(n_oos, cfg.starting_balance, dtype=float)
    if best is None:
        return WindowResult(job.index, w, None, float("nan"), float("nan"), None, None,
                            len(job.combos), 0, True, flat, np.zeros(n_oos), [])
    params = job.combos[best[1]]
    try:
        oos = _slice_run(job, params, w.test_start, w.test_end)
    except _CANDIDATE_ERRORS:
        return WindowResult(job.index, w, params, best[0], float("nan"), best_report, None,
                            len(job.combos), n_eligible, True, flat, np.zeros(n_oos), [])
    return WindowResult(
        job.index, w, params, best[0], objective_value(cfg.objective, oos.report),
        best_report, oos.report, len(job.combos), n_eligible, False,
        oos.equity, oos.position_value, trade_pnls(oos.result),
    )


class WalkForwardOptimizer:
    """Walk-forward grid search for one ``ResearchStrategy`` class.

    ``param_grid`` defaults to ``strategy_cls.PARAM_GRID`` (invalid combinations are dropped
    by ``param_combinations``). ``partner`` (candles of a second instrument) is passed to
    the strategy constructor as ``partner=`` for ``requires_partner`` strategies; it is
    handed over whole, which is safe because partner strategies may only read partner bars
    up to the decision timestamp. ``max_combos``/``seed`` subsample big grids (see
    :func:`random_search`); without ``max_combos`` a grid above ``MAX_GRID`` raises
    ValueError. ``max_workers`` > 1 parallelises across windows (same results as serial).
    """

    def __init__(
        self,
        strategy_cls: Type[ResearchStrategy],
        param_grid: Optional[Mapping[str, Sequence[Any]]] = None,
        cfg: Optional[WalkForwardConfig] = None,
        risk: Optional[RiskConfig] = None,
        execution: Optional[ExecutionConfig] = None,
        partner: Optional[Sequence[Candle]] = None,
        *,
        max_combos: Optional[int] = None,
        seed: int = 0,
        max_workers: Optional[int] = None,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.cfg = cfg or WalkForwardConfig()
        self.risk = risk
        self.execution = execution
        self.partner = partner
        self.max_workers = max_workers
        if max_combos is not None and max_combos <= 0:
            raise ValueError("max_combos must be > 0")
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        if strategy_cls.requires_partner and partner is None:
            raise ValueError(f"{strategy_cls.name} requires a partner series")
        grid = strategy_cls.PARAM_GRID if param_grid is None else param_grid
        self.grid_size = len(strategy_cls.param_combinations(grid))
        if self.grid_size == 0:
            raise ValueError("param grid has no valid combinations")
        if max_combos is None and self.grid_size > MAX_GRID:
            raise ValueError(
                f"grid has {self.grid_size} valid combinations (> MAX_GRID={MAX_GRID}); "
                "shrink it or pass max_combos=N for seeded subsampling"
            )
        self.combos = (random_search(strategy_cls, grid, max_combos, seed)
                       if max_combos is not None else strategy_cls.param_combinations(grid))
        self._grid = grid

    def _param_ranges(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in (self.combos[0] if self.combos else {}):
            vals = [c[name] for c in self.combos]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                out[name] = float(max(vals) - min(vals))
        return out

    def run(self, candles: Sequence[Candle]) -> WalkForwardResult:
        cfg = self.cfg
        windows = generate_windows(len(candles), cfg)
        if not windows:
            raise ValueError(
                f"{len(candles)} bars is too short for train={cfg.train_bars} + test={cfg.test_bars}"
            )
        candles = list(candles)
        jobs = [_Job(i, w, candles, self.strategy_cls, self.combos, cfg, self.risk, self.execution, self.partner)
                for i, w in enumerate(windows)]
        if self.max_workers and self.max_workers > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                results = list(ex.map(_run_window, jobs))
        else:
            results = [_run_window(j) for j in jobs]
        equity, report = self._stitch(results)
        return WalkForwardResult(cfg, self.strategy_cls.name or self.strategy_cls.__name__,
                                 results, equity, report, self._param_ranges())

    def _stitch(self, results: List[WindowResult]) -> Tuple[np.ndarray, metrics.PerformanceReport]:
        sb = self.cfg.starting_balance
        eq, pv, pnls = [sb], [0.0], []
        running = sb
        for r in results:
            scale = running / sb
            eq.extend((r.oos_equity * scale).tolist())
            pv.extend((r.oos_position_value * scale).tolist())
            pnls.extend(p * scale for p in r.oos_trade_pnls)
            running = eq[-1]
        equity = np.array(eq, dtype=float)
        report = metrics.compute_report(equity, np.array(pv), pnls, self.cfg.periods_per_year)
        return equity, report
