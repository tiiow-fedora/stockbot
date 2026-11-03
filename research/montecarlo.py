"""
Monte Carlo robustness toolkit.

A single backtest is ONE draw from a noisy process. These tools ask how much
of the result is skill and how much is luck / path / parameter fit:

``bootstrap_trades``         resample the realised trade P&Ls (with replacement, or
                             reshuffled order) -> distribution of final equity, max
                             drawdown, probability of loss / ruin.
``block_bootstrap_returns``  circular / stationary (Politis-Romano) block bootstrap of
                             per-bar returns -> distribution of total return, Sharpe,
                             max drawdown that keeps short-range autocorrelation.
``permutation_test``         "does the timing beat random?": circularly shift (or
                             shuffle) the position series against the real market
                             returns and report the p-value of the observed metric.
``price_path_robustness``    re-run a strategy through the real engine on synthetic
                             price paths built from the series' own log returns.
``parameter_perturbation``   jitter the parameters and see whether the result sits on
                             a plateau (robust) or a spike (over-fit).
``monte_carlo_report``       runs the sensible subset for one strategy + candles.

Determinism: every function takes an explicit ``seed``; all randomness comes from
``np.random.default_rng``. Multi-simulation runs draw one independent child stream
per simulation via ``SeedSequence.spawn`` so simulation ``k`` does not depend on
``n_sims`` (the first k sims of a longer run equal a shorter run).

Caveats worth remembering: bootstrapping trades assumes trades are i.i.d. (no
regime dependence); block bootstraps destroy structure longer than ``block_len``;
synthetic paths test robustness to *re-ordered* history, not to regimes that
never occurred. Treat results as stress tests, not forecasts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from core.models import Candle
from research import metrics
from research.base import ResearchStrategy

DEFAULT_PERCENTILES: Tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
_MAX_CELLS = 60_000_000  # n_sims * n_steps guard for the vectorised matrices


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

def _clean(v: Any) -> Any:
    """Recursively make a value JSON-safe (nan/inf -> None, numpy -> python)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


@dataclass
class MonteCarloSummary:
    """Distribution of one headline ``metric`` over ``n_sims`` simulations.

    ``percentiles`` maps e.g. ``"p5"`` -> value. ``prob_loss`` is P(total return < 0)
    (or P(metric below its loss line) where the method says so); ``prob_ruin`` is
    P(equity ever falls below ``ruin_level`` x start) and ``None`` when not applicable.
    ``extras`` holds the same description for secondary metrics (e.g. max drawdown),
    ``probabilities`` other named probabilities, ``meta`` the settings used.
    """

    method: str
    metric: str
    n_sims: int
    mean: float
    std: float
    minimum: float
    maximum: float
    percentiles: Dict[str, float]
    prob_loss: float
    prob_ruin: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict (non-finite numbers become ``None``)."""
        return _clean(asdict(self))

    def summary(self) -> str:
        p = self.percentiles
        s = (f"{self.method}/{self.metric}: mean={_fmt(self.mean)} p5={_fmt(p.get('p5'))} "
             f"p50={_fmt(p.get('p50'))} p95={_fmt(p.get('p95'))} P(loss)={_fmt(self.prob_loss, 0, pct=True)}")
        if self.prob_ruin is not None:
            s += f" P(ruin)={_fmt(self.prob_ruin, 0, pct=True)}"
        for k, v in self.probabilities.items():
            s += f" {k}={_fmt(v, 0, pct=True)}"
        return s


def _fmt(v: Any, digits: int = 3, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v * 100:.{digits}f}%" if pct else f"{v:.{digits}f}"


def describe(values: Sequence[float], percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> Dict[str, Any]:
    """Describe a sample ignoring NaNs: mean/std/min/max, percentile dict and ``n_valid``."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "n_valid": 0,
                "percentiles": {_pkey(q): nan for q in percentiles}}
    qs = np.percentile(a, list(percentiles))
    return {
        "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()), "n_valid": int(len(a)),
        "percentiles": {_pkey(q): float(v) for q, v in zip(percentiles, qs)},
    }


def _pkey(q: float) -> str:
    return f"p{q:g}"


def _summary(method: str, metric: str, values: np.ndarray, prob_loss: float, *,
             percentiles: Sequence[float], prob_ruin: Optional[float] = None,
             probabilities: Optional[Dict[str, float]] = None,
             extras: Optional[Dict[str, Dict[str, Any]]] = None,
             meta: Optional[Dict[str, Any]] = None) -> MonteCarloSummary:
    d = describe(values, percentiles)
    return MonteCarloSummary(
        method=method, metric=metric, n_sims=int(len(values)), mean=d["mean"], std=d["std"],
        minimum=d["min"], maximum=d["max"], percentiles=d["percentiles"], prob_loss=float(prob_loss),
        prob_ruin=prob_ruin, probabilities=probabilities or {}, extras=extras or {}, meta=meta or {},
    )


# --------------------------------------------------------------------------
# Validation / RNG helpers
# --------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> List[np.random.Generator]:
    """``n`` statistically independent generators derived from ``seed``."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def _check_n_sims(n_sims: int) -> None:
    if not isinstance(n_sims, (int, np.integer)) or isinstance(n_sims, bool) or n_sims < 1:
        raise ValueError(f"n_sims must be a positive integer, got {n_sims!r}")


def _as_1d(x: Sequence[float], name: str, min_len: int) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if len(a) < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {len(a)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def _check_cells(n_sims: int, n_steps: int) -> None:
    if n_sims * n_steps > _MAX_CELLS:
        raise ValueError(f"n_sims * n_steps = {n_sims * n_steps} exceeds the memory guard of {_MAX_CELLS}")


def _max_dd_matrix(equity: np.ndarray) -> np.ndarray:
    """Row-wise ``research.metrics.max_drawdown`` (vectorised; verified equal in the tests)."""
    peak = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return dd.max(axis=1)


def _sharpe_matrix(rets: np.ndarray, periods_per_year: int) -> np.ndarray:
    """Row-wise ``research.metrics.sharpe_ratio`` (NaN for zero variance)."""
    if rets.shape[1] < 2:
        return np.full(rets.shape[0], np.nan)
    sd = rets.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = rets.mean(axis=1) / sd * math.sqrt(periods_per_year)
    return np.where(sd < 1e-15, np.nan, out)


# --------------------------------------------------------------------------
# 1. Trade bootstrap
# --------------------------------------------------------------------------

def bootstrap_trades(
    trade_pnls: Sequence[float],
    n_sims: int = 1000,
    starting_equity: float = 100_000.0,
    seed: int = 0,
    replace: bool = True,
    ruin_level: float = 0.5,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> MonteCarloSummary:
    """Resample realised trade P&Ls (dollars, added to equity in sequence).

    ``replace=True`` draws ``len(trades)`` trades with replacement (final equity varies:
    luck in WHICH trades occurred). ``replace=False`` reshuffles the same trades
    (final equity is identical in every sim, only the PATH / drawdown / ruin risk
    varies: luck in ORDER). The headline metric is final equity; max drawdown is in
    ``extras["max_drawdown"]``. Ruin = equity at any point < ``ruin_level`` x start.
    """
    pnl = _as_1d(trade_pnls, "trade_pnls", 2)
    _check_n_sims(n_sims)
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if not 0.0 < ruin_level < 1.0:
        raise ValueError("ruin_level must be in (0, 1)")
    n = len(pnl)
    _check_cells(n_sims, n)
    rng = _rng(seed)
    if replace:
        draws = pnl[rng.integers(0, n, size=(n_sims, n))]
    else:
        draws = rng.permuted(np.tile(pnl, (n_sims, 1)), axis=1)
    equity = np.concatenate([np.full((n_sims, 1), starting_equity), starting_equity + np.cumsum(draws, axis=1)], axis=1)
    final = equity[:, -1]
    mdd = _max_dd_matrix(equity)
    ruined = equity[:,-1] < ruin_level * starting_equity
    return _summary(
        "bootstrap_trades", "final_equity", final, float(np.mean(final < starting_equity)),
        percentiles=percentiles, prob_ruin=float(np.mean(ruined)),
        extras={"max_drawdown": describe(mdd, percentiles),
                "total_return": describe(final / starting_equity - 1.0, percentiles)},
        meta={"replace": bool(replace), "n_trades": n, "starting_equity": starting_equity,
              "ruin_level": ruin_level, "seed": seed,
              "observed_final_equity": float(starting_equity + pnl.sum())},
    )


# --------------------------------------------------------------------------
# 2. Block bootstrap of returns
# --------------------------------------------------------------------------

def block_bootstrap_indices(
    n: int, n_sims: int, block_len: int, rng: np.random.Generator,
    stationary: bool = True, circular: bool = True,
) -> np.ndarray:
    """Index matrix ``(n_sims, n)`` of a block bootstrap of ``n`` observations.

    * ``stationary=True`` (Politis-Romano): a new block starts at each step with
      probability ``1/block_len`` (geometric block lengths, mean ``block_len``);
      otherwise the previous index advances by one (wrapping circularly).
      ``block_len=1`` is the iid bootstrap.
    * ``stationary=False``: fixed-length blocks. ``circular=True`` lets blocks wrap
      around the end; ``circular=False`` is the moving-block bootstrap (block starts
      in ``[0, n - block_len]``), so ``block_len=n`` reproduces the original series.
    Index ``t`` of a path depends only on random draws at positions ``<= t``.
    """
    if not isinstance(block_len, (int, np.integer)) or isinstance(block_len, bool) or not 1 <= block_len <= n:
        raise ValueError(f"block_len must be an integer in [1, {n}], got {block_len!r}")
    pos = np.arange(n)
    if stationary:
        new = rng.random((n_sims, n)) < 1.0 / block_len
        new[:, 0] = True
        starts = rng.integers(0, n, size=(n_sims, n))
        anchor = np.maximum.accumulate(np.where(new, pos[None, :], 0), axis=1)
        base = np.take_along_axis(starts, anchor, axis=1)
        return (base + pos[None, :] - anchor) % n
    n_blocks = -(-n // block_len)
    hi = n if circular else n - block_len + 1
    starts = rng.integers(0, hi, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]).reshape(n_sims, -1)[:, :n]
    return idx % n if circular else idx


def block_bootstrap_returns(
    returns: Sequence[float],
    n_sims: int = 1000,
    block_len: int = 10,
    seed: int = 0,
    stationary: bool = True,
    circular: bool = True,
    ruin_level: float = 0.5,
    periods_per_year: int = 252,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> MonteCarloSummary:
    """Block-bootstrap per-bar simple returns into ``n_sims`` equity paths (start = 1).

    Headline metric: total return; ``extras`` has ``sharpe`` and ``max_drawdown``.
    See ``block_bootstrap_indices`` for the resampling schemes.
    """
    r = _as_1d(returns, "returns", 2)
    _check_n_sims(n_sims)
    if np.any(r <= -1.0):
        raise ValueError("returns must be > -1 (a return of -100% or worse wipes out the path)")
    if not 0.0 < ruin_level < 1.0:
        raise ValueError("ruin_level must be in (0, 1)")
    n = len(r)
    _check_cells(n_sims, n)
    idx = block_bootstrap_indices(n, n_sims, block_len, _rng(seed), stationary, circular)
    paths = r[idx]
    equity = np.concatenate([np.ones((n_sims, 1)), np.cumprod(1.0 + paths, axis=1)], axis=1)
    total = equity[:, -1] - 1.0
    ruined = equity.min(axis=1) < ruin_level
    return _summary(
        "block_bootstrap" if stationary else "moving_block_bootstrap", "total_return", total,
        float(np.mean(total < 0)), percentiles=percentiles, prob_ruin=float(np.mean(ruined)),
        extras={"sharpe": describe(_sharpe_matrix(paths, periods_per_year), percentiles),
                "max_drawdown": describe(_max_dd_matrix(equity), percentiles)},
        meta={"block_len": int(block_len), "stationary": bool(stationary), "circular": bool(circular),
              "n_bars": n, "seed": seed, "observed_total_return": float(np.prod(1.0 + r) - 1.0),
              "observed_sharpe": metrics.sharpe_ratio(r, periods_per_year=periods_per_year)},
    )


# --------------------------------------------------------------------------
# 3. Permutation test of timing
# --------------------------------------------------------------------------

@dataclass
class PermutationResult:
    """Outcome of ``permutation_test``. ``p_value`` = P(null statistic >= observed),
    with the +1 correction so it is never exactly 0."""

    statistic: str
    mode: str
    observed: float
    null_mean: float
    null_std: float
    null_percentiles: Dict[str, float]
    p_value: float
    n_sims: int
    n_valid: int

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))

    def summary(self) -> str:
        return (f"permutation[{self.mode}] {self.statistic}: observed={_fmt(self.observed, 4)} "
                f"null={_fmt(self.null_mean, 4)}+/-{_fmt(self.null_std, 4)} p={_fmt(self.p_value, 3)}")


def _stat_rows(rets: np.ndarray, statistic: str, periods_per_year: int) -> np.ndarray:
    return rets.mean(axis=1) if statistic == "mean" else _sharpe_matrix(rets, periods_per_year)


def permutation_test(
    strategy_returns: Optional[Sequence[float]],
    position: Sequence[float],
    market_returns: Sequence[float],
    n_sims: int = 1000,
    seed: int = 0,
    statistic: str = "mean",
    mode: str = "shift",
    min_shift: int = 1,
    periods_per_year: int = 252,
) -> PermutationResult:
    """Does the position timing beat random timing with the SAME exposure profile?

    ``position[t]`` is the exposure held DURING bar ``t`` (already lagged: it must not
    use bar ``t``'s return). The null re-pairs the position series with the real
    ``market_returns`` after (``mode="shift"``) a random circular shift of at least
    ``min_shift`` bars, which preserves its autocorrelation and average exposure, or
    (``mode="shuffle"``) a full random permutation, which also destroys the holding
    pattern (weaker null: any trend-like holding looks "significant" vs shuffle in a
    trending market, so prefer ``shift``).

    ``strategy_returns`` is the observed return series; pass ``None`` to use
    ``position * market_returns`` (gross, apples-to-apples with the null). If you pass
    net returns the test is conservative (costs are not in the null).
    ``statistic`` is ``"mean"`` or ``"sharpe"``. The p-value is one-sided.
    """
    pos = _as_1d(position, "position", 3)
    mkt = _as_1d(market_returns, "market_returns", 3)
    if len(pos) != len(mkt):
        raise ValueError(f"position and market_returns must have equal length, got {len(pos)} vs {len(mkt)}")
    if statistic not in ("mean", "sharpe"):
        raise ValueError("statistic must be 'mean' or 'sharpe'")
    if mode not in ("shift", "shuffle"):
        raise ValueError("mode must be 'shift' or 'shuffle'")
    _check_n_sims(n_sims)
    n = len(pos)
    if not 1 <= min_shift <= n // 2:
        raise ValueError(f"min_shift must be in [1, {n // 2}] for {n} bars, got {min_shift}")
    if strategy_returns is None:
        obs_rets = pos * mkt
    else:
        obs_rets = _as_1d(strategy_returns, "strategy_returns", 3)
        if len(obs_rets) != n:
            raise ValueError(f"strategy_returns must have length {n}, got {len(obs_rets)}")
    _check_cells(n_sims, n)
    rng = _rng(seed)
    if mode == "shift":
        shifts = rng.integers(min_shift, n - min_shift + 1, size=n_sims)
        idx = (np.arange(n)[None, :] + shifts[:, None]) % n
    else:
        idx = rng.permuted(np.tile(np.arange(n), (n_sims, 1)), axis=1)
    null = _stat_rows(pos[idx] * mkt[None, :], statistic, periods_per_year)
    observed = float(_stat_rows(obs_rets[None, :], statistic, periods_per_year)[0])
    valid = null[np.isfinite(null)]
    if not math.isfinite(observed) or len(valid) == 0:
        p = float("nan")
    else:
        p = float((1 + np.sum(valid >= observed - 1e-15)) / (len(valid) + 1))
    d = describe(valid)
    return PermutationResult(statistic, mode, observed, d["mean"], d["std"], d["percentiles"], p, n_sims, int(len(valid)))


# --------------------------------------------------------------------------
# 4. Synthetic price paths through the real engine
# --------------------------------------------------------------------------

PATH_METHODS = ("block", "gbm_matched", "shuffle_returns")


def build_path_candles(
    base: Sequence[Candle], log_returns: np.ndarray, shape_idx: np.ndarray,
) -> List[Candle]:
    """Rebuild OHLCV from a log-return path.

    Bar ``k >= 1`` gets close ``close[k-1] * exp(log_returns[k-1])`` and borrows the
    intrabar SHAPE (open gap vs previous close, high / low excursion beyond the
    open-close body) and the volume of source bar ``base[shape_idx[k-1] + 1]``. Bar 0
    is the original first bar. Causal: bars ``<= k`` depend only on the first ``k``
    returns / shape indices, and timestamps are the originals.
    """
    n = len(base)
    if len(log_returns) != n - 1 or len(shape_idx) != n - 1:
        raise ValueError("log_returns and shape_idx must have len(base) - 1 entries")
    first = base[0]
    out = [first]
    prev_close = first.close
    for k in range(1, n):
        src = base[int(shape_idx[k - 1]) + 1]
        src_prev = base[int(shape_idx[k - 1])]
        close = prev_close * math.exp(float(log_returns[k - 1]))
        open_ = prev_close * (src.open / src_prev.close)
        hi_r = src.high / max(src.open, src.close)
        lo_r = src.low / min(src.open, src.close)
        high = max(open_, close) * hi_r
        low = min(open_, close) * lo_r
        out.append(Candle(first.symbol, base[k].timestamp, open_, high, low, close, src.volume, first.timeframe))
        prev_close = close
    return out


def _closes_log_returns(candles: Sequence[Candle]) -> np.ndarray:
    closes = np.array([c.close for c in candles], dtype=float)
    return np.diff(np.log(closes))


def generate_price_path(
    candles: Sequence[Candle], method: str, rng: np.random.Generator, block_len: int = 10,
) -> List[Candle]:
    """One synthetic candle path from the real series' log returns.

    * ``block``: stationary block bootstrap (keeps drift, vol and short-range autocorrelation).
    * ``gbm_matched``: iid normal log returns with the sample mean and std (keeps drift and vol only).
    * ``shuffle_returns``: random permutation (same terminal price and return distribution, no ordering).
    """
    if method not in PATH_METHODS:
        raise ValueError(f"method must be one of {PATH_METHODS}, got {method!r}")
    lr = _closes_log_returns(candles)
    m = len(lr)
    if method == "block":
        idx = block_bootstrap_indices(m, 1, min(block_len, m), rng, stationary=True)[0]
        path = lr[idx]
    elif method == "shuffle_returns":
        idx = rng.permutation(m)
        path = lr[idx]
    else:
        path = rng.normal(lr.mean(), lr.std(ddof=1), size=m)
        idx = rng.integers(0, m, size=m)
    return build_path_candles(candles, path, idx)


def price_path_robustness(
    strategy_factory: Callable[[], ResearchStrategy],
    candles: Sequence[Candle],
    n_sims: int = 100,
    seed: int = 0,
    method: str = "block",
    block_len: int = 10,
    max_sims: int = 500,
    warmup_bars: int = 40,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    **backtest_kwargs: Any,
) -> MonteCarloSummary:
    """Run a fresh strategy (``strategy_factory()`` per sim) over synthetic paths built from
    ``candles`` and aggregate the engine's metrics.

    Headline metric: total return. ``extras``: sharpe, max_drawdown, avg_exposure,
    num_trades, benchmark_return. ``probabilities``: ``profitable`` (return > 0) and
    ``beats_buy_hold`` (return > that path's buy & hold). ``meta["real"]`` holds the
    same metrics on the real series for comparison. Partner strategies: close over them
    in the factory (``lambda: Strategy(partner=Other())``). ``n_sims > max_sims`` raises.
    """
    from research.harness import run_backtest  # local: harness pulls in the engine

    _check_n_sims(n_sims)
    if n_sims > max_sims:
        raise ValueError(f"n_sims={n_sims} exceeds max_sims={max_sims}; raise max_sims explicitly if intended")
    if method not in PATH_METHODS:
        raise ValueError(f"method must be one of {PATH_METHODS}, got {method!r}")
    if len(candles) < warmup_bars + 22:
        raise ValueError(f"need at least warmup_bars+22={warmup_bars + 22} candles, got {len(candles)}")
    if any(c.close <= 0 for c in candles):
        raise ValueError("candles must have positive closes")

    def one(cs: Sequence[Candle]) -> Dict[str, float]:
        rr = run_backtest(strategy_factory(), cs, warmup_bars=warmup_bars, **backtest_kwargs)
        return {"total_return": rr.report.total_return, "sharpe": rr.report.sharpe,
                "max_drawdown": rr.report.max_drawdown, "avg_exposure": rr.report.avg_exposure,
                "num_trades": float(rr.report.num_trades),
                "benchmark_return": rr.benchmark_report.total_return}

    rows = [one(generate_price_path(candles, method, g, block_len)) for g in spawn_rngs(seed, n_sims)]
    cols = {k: np.array([row[k] for row in rows], dtype=float) for k in rows[0]}
    ret, bench = cols["total_return"], cols["benchmark_return"]
    ok = np.isfinite(ret) & np.isfinite(bench)
    return _summary(
        f"price_paths[{method}]", "total_return", ret, float(np.mean(ret[np.isfinite(ret)] < 0)) if ok.any() else float("nan"),
        percentiles=percentiles,
        probabilities={"profitable": float(np.mean(ret[ok] > 0)) if ok.any() else float("nan"),
                       "beats_buy_hold": float(np.mean(ret[ok] > bench[ok])) if ok.any() else float("nan"),
                       "no_closed_trades": float(np.mean(cols["num_trades"] == 0))},
        extras={k: describe(v, percentiles) for k, v in cols.items() if k != "total_return"},
        meta={"method": method, "block_len": block_len, "seed": seed, "n_bars": len(candles), "real": one(candles)},
    )


# --------------------------------------------------------------------------
# 5. Parameter perturbation
# --------------------------------------------------------------------------

Objective = Union[str, Callable[[Any], float]]


def _objective_value(rr: Any, objective: Objective) -> float:
    if callable(objective):
        return float(objective(rr))
    return float(getattr(rr.report, objective))


def perturb_params(base: Dict[str, Any], rel_noise: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Jitter numeric params multiplicatively by U(1-rel, 1+rel). Ints are rounded and
    kept >= 1; bools, strings and zero-valued numbers are left unchanged."""
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or v == 0:
            out[k] = v
            continue
        factor = 1.0 + rng.uniform(-rel_noise, rel_noise)
        if isinstance(v, (int, np.integer)):
            out[k] = max(1, int(round(int(v) * factor)))
        else:
            out[k] = float(v) * factor
    return out


def parameter_perturbation(
    strategy_cls: Type[ResearchStrategy],
    base_params: Optional[Dict[str, Any]],
    candles: Sequence[Candle],
    n_sims: int = 50,
    rel_noise: float = 0.2,
    seed: int = 0,
    keep: float = 0.8,
    objective: Objective = "sharpe",
    warmup_bars: int = 40,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    **backtest_kwargs: Any,
) -> MonteCarloSummary:
    """Plateau-vs-spike test. Jitter ``base_params`` (on top of the class defaults) by
    +/- ``rel_noise`` and backtest each valid variant.

    ``objective`` is a ``PerformanceReport`` field name (default ``"sharpe"``) or a
    callable on the ``RunResult``. The robustness score
    (``probabilities["plateau_score"]``) is the fraction of valid perturbed runs whose
    objective is >= ``base - (1 - keep) * |base|`` (i.e. >= ``keep`` x base when base > 0).
    Combos failing ``validate()`` are skipped and counted in ``meta["n_invalid"]``;
    runs whose objective is NaN count as failures. A spiky optimum scores low.
    """
    from research.harness import run_backtest

    _check_n_sims(n_sims)
    if rel_noise < 0:
        raise ValueError("rel_noise must be >= 0")
    if not 0.0 <= keep <= 1.0:
        raise ValueError("keep must be in [0, 1]")
    if len(candles) < warmup_bars + 22:
        raise ValueError(f"need at least warmup_bars+22={warmup_bars + 22} candles, got {len(candles)}")

    def evaluate(params: Dict[str, Any]) -> float:
        rr = run_backtest(strategy_cls(**params), candles, warmup_bars=warmup_bars, **backtest_kwargs)
        return _objective_value(rr, objective)

    full_base = strategy_cls(**(base_params or {})).params.to_dict()
    base_obj = evaluate(full_base)
    if not math.isfinite(base_obj):
        raise ValueError("base objective is NaN (e.g. no trades / zero variance); cannot assess robustness")
    threshold = base_obj - (1.0 - keep) * abs(base_obj)

    cache: Dict[Tuple, Optional[float]] = {}  # None marks an invalid parameter set
    values: List[float] = []
    n_invalid = 0
    for g in spawn_rngs(seed, n_sims):
        params = perturb_params(full_base, rel_noise, g)
        key = tuple(sorted(params.items()))
        if key not in cache:
            try:
                cache[key] = evaluate(params)
            except ValueError:
                cache[key] = None
        value = cache[key]
        if value is None:
            n_invalid += 1
        else:
            values.append(value)
    arr = np.array(values, dtype=float)
    n_valid = len(arr)
    nanv = float("nan")
    passed = np.isfinite(arr) & (arr >= threshold - 1e-12)
    return _summary(
        "parameter_perturbation", str(objective) if isinstance(objective, str) else "objective", arr,
        float(np.mean(arr[np.isfinite(arr)] < 0)) if np.isfinite(arr).any() else nanv, percentiles=percentiles,
        probabilities={"plateau_score": float(passed.mean()) if n_valid else nanv,
                       "positive": float(np.mean(arr > 0)) if n_valid else nanv},
        meta={"base_objective": base_obj, "threshold": threshold, "keep": keep, "rel_noise": rel_noise,
              "n_invalid": n_invalid, "n_valid": n_valid, "n_unique_param_sets": len(cache),
              "base_params": full_base, "seed": seed},
    )


# --------------------------------------------------------------------------
# 6. One-call report
# --------------------------------------------------------------------------

def monte_carlo_report(
    strategy_factory: Callable[[], ResearchStrategy],
    candles: Sequence[Candle],
    strategy_cls: Optional[Type[ResearchStrategy]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    n_sims: int = 200,
    path_sims: int = 40,
    perturb_sims: int = 30,
    seed: int = 0,
    block_len: int = 10,
    warmup_bars: int = 40,
) -> Dict[str, Any]:
    """Run the sensible subset for one strategy on one candle series.

    Sections (each a ``to_dict()`` / dict, ``None`` if not computable): ``backtest``
    (the real run), ``trade_bootstrap`` (needs >= 5 trades), ``return_bootstrap``
    (block bootstrap of the realised equity returns), ``timing_permutation``
    (position = invested fraction of equity during each bar vs market returns; an
    approximation because fills happen at the bar open), ``path_robustness`` for the
    block and GBM-matched methods, and ``parameter_perturbation`` when ``strategy_cls``
    is given. ``text`` is a plain-text digest for the README generator.
    """
    from research import harness

    rr = harness.run_backtest(strategy_factory(), candles, warmup_bars=warmup_bars)
    out: Dict[str, Any] = {"backtest": rr.report.as_dict(), "benchmark_return": rr.benchmark_report.total_return}
    lines = [f"real run: {rr.report.summary()}"]
    seeds = np.random.SeedSequence(seed).spawn(5)
    s = [int(x.generate_state(1)[0]) for x in seeds]

    pnls = harness.trade_pnls(rr.result)
    tb = bootstrap_trades(pnls, n_sims, rr.equity[0], s[0]) if len(pnls) >= 5 else None
    out["trade_bootstrap"] = tb.to_dict() if tb else None
    lines.append(tb.summary() if tb else "trade bootstrap: n/a (<5 trades)")

    eq_rets = metrics.to_returns(rr.equity)
    rb = None
    if len(eq_rets) >= 20:
        rb = block_bootstrap_returns(eq_rets, n_sims, min(block_len, len(eq_rets)), s[1])
    out["return_bootstrap"] = rb.to_dict() if rb else None
    lines.append(rb.summary() if rb else "return bootstrap: n/a")

    pt = None
    if len(eq_rets) >= 20 and np.any(rr.position_value > 0):
        closes = np.array([c.close for c in candles], dtype=float)
        mkt = (closes[1:] / closes[:-1] - 1.0)[-len(eq_rets):]
        weight = (rr.position_value[1:] / rr.equity[1:])
        pt = permutation_test(None, weight, mkt, n_sims, s[2], statistic="mean")
    out["timing_permutation"] = pt.to_dict() if pt else None
    lines.append(pt.summary() if pt else "timing permutation: n/a (never invested)")

    paths: Dict[str, Any] = {}
    for j, m in enumerate(("block", "gbm_matched")):
        res = price_path_robustness(strategy_factory, candles, path_sims, s[3] + j, m, block_len, warmup_bars=warmup_bars)
        paths[m] = res.to_dict()
        lines.append(res.summary())
    out["path_robustness"] = paths

    pp = None
    if strategy_cls is not None:
        try:
            pp = parameter_perturbation(strategy_cls, base_params, candles, perturb_sims, seed=s[4], warmup_bars=warmup_bars)
        except ValueError as e:  # base objective undefined (e.g. no trades)
            lines.append(f"parameter perturbation: n/a ({e})")
        else:
            lines.append(pp.summary() + f" plateau_score_threshold={_fmt(pp.meta['threshold'])}")
    out["parameter_perturbation"] = pp.to_dict() if pp else None
    out["text"] = "\n".join(lines)
    return _clean(out)
