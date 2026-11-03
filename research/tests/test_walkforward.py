"""Tests for research.walkforward: window maths, timing/no-look-ahead, the mandatory
leakage test, the overfitting-is-visible test, result bookkeeping and the optional pool.
Only donchian_breakout (plus tiny local strategies) is used so the file never depends on
strategies that other contributors may not have added yet."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from core.models import Candle
from research import metrics, synth
from research.base import Bars, ResearchStrategy, Signal, StrategyParams
from research.strategies.donchian_breakout import DonchianBreakout
from research.walkforward import (
    MAX_GRID,
    Window,
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
    WindowResult,
    backtest_slice,
    generate_windows,
    objective_value,
    random_search,
)

GRID = {"entry_period": [10, 20, 40], "exit_period": [5, 10]}
CFG = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60)


@dataclass(frozen=True)
class LagParams(StrategyParams):
    lag: int = 10
    exit_lag: int = 5

    def validate(self) -> None:
        if self.lag < 1 or self.exit_lag < 1:
            raise ValueError("lags must be >= 1")


class LagMomentum(ResearchStrategy):
    """Cheap many-parameter rule (enter if close > close[-lag], exit if close < close[-exit_lag]).
    Its 36-cell grid on noise is the classic overfitting setup; ~4 ms per backtest."""

    name = "wf_lag_momentum"
    family = "test"
    PARAMS = LagParams
    PARAM_GRID = {"lag": [3, 5, 8, 12, 16, 20, 30, 40, 60], "exit_lag": [2, 4, 6, 10]}

    @property
    def min_bars(self) -> int:
        return max(self.params.lag, self.params.exit_lag) + 1

    @property
    def lookback(self) -> int:
        return self.min_bars

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        c, p = bars.close, self.params
        if in_position:
            return Signal.exit() if c[-1] < c[-1 - p.exit_lag] else Signal.hold()
        return Signal.enter() if c[-1] > c[-1 - p.lag] else Signal.hold()


def opt(cfg=CFG, grid=GRID, cls=DonchianBreakout, **kw) -> WalkForwardOptimizer:
    return WalkForwardOptimizer(cls, grid, cfg, **kw)


# --------------------------------------------------------------------------
# Config + windows
# --------------------------------------------------------------------------

class TestConfig:
    def test_step_defaults_to_test_bars(self):
        assert WalkForwardConfig(train_bars=100, test_bars=30).step_bars == 30

    @pytest.mark.parametrize("kw", [
        dict(train_bars=5), dict(test_bars=2), dict(test_bars=30, step_bars=29), dict(min_trades=-1),
        dict(warmup_bars=-1), dict(objective="alpha"), dict(starting_balance=0), dict(periods_per_year=0),
    ])
    def test_invalid_config_rejected(self, kw):
        with pytest.raises(ValueError):
            WalkForwardConfig(**kw)

    def test_callable_objective_accepted(self):
        assert WalkForwardConfig(objective=lambda r: r.total_return).objective is not None


class TestWindows:
    def test_exact_rolling_and_anchored(self):
        r = generate_windows(200, WalkForwardConfig(train_bars=100, test_bars=40, step_bars=40))
        assert r == [Window(0, 100, 100, 140), Window(40, 140, 140, 180)]
        a = generate_windows(200, WalkForwardConfig(train_bars=100, test_bars=40, anchored=True))
        assert a == [Window(0, 100, 100, 140), Window(0, 140, 140, 180)]

    def test_too_short_gives_no_windows(self):
        assert generate_windows(139, WalkForwardConfig(train_bars=100, test_bars=40)) == []
        assert len(generate_windows(140, WalkForwardConfig(train_bars=100, test_bars=40))) == 1

    @pytest.mark.parametrize("n", [0, 99, 150, 333, 1000])
    @pytest.mark.parametrize("train,test,step", [(100, 20, 20), (50, 10, 25), (200, 60, 60), (30, 7, 11)])
    @pytest.mark.parametrize("anchored", [False, True])
    def test_window_invariants(self, n, train, test, step, anchored):
        cfg = WalkForwardConfig(train_bars=train, test_bars=test, step_bars=step, anchored=anchored)
        ws = generate_windows(n, cfg)
        expected = (n - train - test) // step + 1 if n >= train + test else 0
        assert len(ws) == expected
        for k, w in enumerate(ws):
            assert 0 <= w.train_start < w.train_end == w.test_start < w.test_end <= n
            assert w.test_end - w.test_start == test
            assert w.train_start == (0 if anchored else k * step)
            assert w.train_end - w.train_start >= train
        for a, b in zip(ws, ws[1:]):
            assert b.test_start >= a.test_end                 # OOS blocks never overlap
            assert b.test_start - a.test_start == step

    def test_rolling_train_length_constant_anchored_grows(self):
        cfg = WalkForwardConfig(train_bars=100, test_bars=20)
        assert {w.train_end - w.train_start for w in generate_windows(400, cfg)} == {100}
        lens = [w.train_end - w.train_start for w in generate_windows(400, WalkForwardConfig(train_bars=100, test_bars=20, anchored=True))]
        assert lens == sorted(lens) and lens[-1] > lens[0]


# --------------------------------------------------------------------------
# Timing: first decision exactly at the window start, nothing beyond end-1
# --------------------------------------------------------------------------

class Recorder(ResearchStrategy):
    """Does nothing but record what it can see at every decision."""

    name = "wf_recorder"
    family = "test"
    PARAMS = StrategyParams
    seen: List = []

    @property
    def min_bars(self) -> int:
        return 1

    def signal(self, bars: Bars, in_position: bool) -> Signal:
        Recorder.seen.append((bars.timestamps[-1], bars.timestamps[0], len(bars)))
        return Signal.hold("")


class TestTiming:
    @pytest.mark.parametrize("start,end,warmup", [(100, 160, 30), (100, 160, 0), (10, 70, 50), (0, 60, 20)])
    def test_first_decision_at_start_and_last_before_end(self, start, end, warmup):
        candles = synth.gbm(300, seed=3)
        Recorder.seen = []
        backtest_slice(Recorder(), candles, start, end, warmup, risk=None, execution=None,
                       starting_balance=100_000.0, periods_per_year=252)
        decided = [s[0] for s in Recorder.seen]
        assert decided[0] == candles[start].timestamp                 # not start-1, not start+1
        assert decided[-1] == candles[end - 2].timestamp              # bar end-1 only executes, bar `end` unseen
        assert len(decided) == end - start - 1
        extra = min(warmup, start)
        assert Recorder.seen[0][1] == candles[start - extra].timestamp  # warm-up history is handed over
        assert Recorder.seen[0][2] == extra + 1                       # ... and nothing else
        assert max(decided) < candles[end - 1].timestamp

    def test_equity_length_matches_formula(self):
        res = backtest_slice(DonchianBreakout(), synth.gbm(300, seed=3), 120, 180, 60, risk=None, execution=None,
                             starting_balance=100_000.0, periods_per_year=252)
        assert len(res.equity) == 180 - 120 - 1


# --------------------------------------------------------------------------
# Mandatory: no leakage of the test slice into parameter selection
# --------------------------------------------------------------------------

def corrupt_from(candles: List[Candle], idx: int, seed: int = 99) -> List[Candle]:
    """Replace every bar from ``idx`` on with a wildly different (but valid) price path."""
    rng = np.random.default_rng(seed)
    closes = np.array([c.close for c in candles], dtype=float)
    tail = closes[idx - 1] * np.exp(np.cumsum(rng.normal(0.01, 0.06, len(candles) - idx)))
    closes[idx:] = tail
    return candles[:idx] + synth.candles_from_closes(closes, candles[0].symbol, seed)[idx:]


class TestNoLeakage:
    @pytest.mark.parametrize("anchored,cls,grid", [(False, DonchianBreakout, GRID), (True, LagMomentum, None)])
    def test_test_slice_never_influences_selection(self, anchored, cls, grid):
        candles = synth.regime_switching(420, seed=5)
        cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, anchored=anchored)
        base = opt(cfg, grid, cls).run(candles)
        assert len(base.windows) >= 4
        for k in (1, len(base.windows) - 1):
            w = base.windows[k].window
            res = opt(cfg, grid, cls).run(corrupt_from(candles, w.test_start))
            for j in range(k + 1):                                    # windows 0..k only train on bars < w.test_start
                assert res.windows[j].params == base.windows[j].params
                assert res.windows[j].is_score == base.windows[j].is_score or (
                    math.isnan(res.windows[j].is_score) and math.isnan(base.windows[j].is_score))
            # ... but the corruption genuinely changed the test result (the test has teeth)
            assert not np.array_equal(res.windows[k].oos_equity, base.windows[k].oos_equity)

    def test_oos_result_depends_only_on_chosen_params(self):
        """Re-running the chosen params on the test slice reproduces the recorded OOS exactly."""
        candles = synth.gbm(420, seed=8)
        res = opt().run(candles)
        for w in res.active[:3]:
            again = backtest_slice(DonchianBreakout(**w.params), candles, w.window.test_start, w.window.test_end,
                                   CFG.warmup_bars, risk=None, execution=None, starting_balance=CFG.starting_balance,
                                   periods_per_year=252)
            assert np.allclose(again.equity, w.oos_equity)


@pytest.fixture(scope="module")
def random_walk_runs() -> List[WalkForwardResult]:
    cfg = WalkForwardConfig(train_bars=120, test_bars=60, warmup_bars=60)
    return [opt(cfg, grid=None, cls=LagMomentum).run(synth.gbm(360, 0.0, 0.012, seed)) for seed in range(8)]


class TestOverfittingVisible:
    def test_is_beats_oos_on_random_walks(self, random_walk_runs):
        """On no-edge data the train-selected params look good in-sample and do not repeat out of sample."""
        is_mean = np.nanmean([r.mean_is_objective for r in random_walk_runs])
        oos_mean = np.nanmean([r.mean_oos_objective for r in random_walk_runs])
        assert is_mean > oos_mean + 0.3
        assert is_mean > 0.0

    def test_no_stitched_oos_edge_on_random_walk(self, random_walk_runs):
        assert np.nanmean([r.oos_report.sharpe for r in random_walk_runs]) < 0.5   # IS-selected params promise far more


# --------------------------------------------------------------------------
# Selection logic
# --------------------------------------------------------------------------

class TestSelection:
    def test_uptrend_oos_is_profitable_and_picks_something(self):
        res = opt().run(synth.trending(400, 0.004, 0.006, seed=1))
        assert res.oos_report.total_return > 0 and res.n_skipped == 0
        assert all(w.params in DonchianBreakout.param_combinations(GRID) for w in res.windows)

    def test_best_candidate_is_argmax_of_objective(self):
        candles = synth.regime_switching(300, seed=2)
        res = opt(WalkForwardConfig(train_bars=200, test_bars=50, warmup_bars=60)).run(candles)
        w = res.windows[0]
        scores = []
        for combo in DonchianBreakout.param_combinations(GRID):
            rep = backtest_slice(DonchianBreakout(**combo), candles, w.window.train_start, w.window.train_end, 60,
                                 risk=None, execution=None, starting_balance=100_000.0, periods_per_year=252).report
            if rep.num_trades >= 1:
                scores.append(objective_value("sharpe", rep))
        assert w.is_score == max(scores)

    def test_deterministic(self):
        c = synth.gbm(350, seed=4)
        a, b = opt().run(c), opt().run(c)
        assert [w.params for w in a.windows] == [w.params for w in b.windows]
        assert np.array_equal(a.equity, b.equity)

    def test_custom_objective_callable_and_name(self):
        c = synth.trending(350, 0.004, 0.006, seed=1)
        cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, objective=lambda r: -r.max_drawdown)
        r = opt(cfg).run(c)
        assert all(w.is_score <= 0 for w in r.active)
        for name in ("sortino", "calmar", "total_return"):
            opt(WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, objective=name)).run(c)

    def test_nan_objective_is_minus_inf_and_skips_window(self):
        rep = metrics.PerformanceReport()
        assert objective_value("sharpe", rep) == float("-inf")
        assert objective_value(lambda r: float("inf"), rep) == float("-inf")
        assert objective_value(lambda r: 1 / 0, rep) == float("-inf")
        cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, objective=lambda r: float("nan"))
        r = opt(cfg).run(synth.gbm(350, seed=4))
        assert r.n_skipped == len(r.windows)                         # nothing finite to choose: stay in cash

    def test_min_trades_makes_windows_skipped_with_flat_oos(self):
        cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, min_trades=10_000)
        r = opt(cfg).run(synth.gbm(350, seed=4))
        assert r.n_skipped == len(r.windows) and all(w.params is None for w in r.windows)
        assert np.all(r.equity == cfg.starting_balance)
        assert math.isnan(r.efficiency) and not r.overfit_flag

    def test_invalid_combos_dropped_and_grid_empty_rejected(self):
        o = opt(grid={"entry_period": [5, 20], "exit_period": [10, 5]})   # (5,10) invalid: exit > entry
        assert {(c["entry_period"], c["exit_period"]) for c in o.combos} == {(5, 5), (20, 10), (20, 5)}
        with pytest.raises(ValueError, match="no valid"):
            opt(grid={"entry_period": [5], "exit_period": [10]})

    def test_too_short_series(self):
        with pytest.raises(ValueError, match="too short"):
            opt().run(synth.gbm(150, seed=0))

    def test_partner_is_passed_to_constructor(self):
        seen = []

        class Pairy(Recorder):
            name = "wf_pairy"
            requires_partner = True

            def __init__(self, params=None, partner=None, **kw):
                super().__init__(params, **kw)
                seen.append(partner)

        a, b = synth.gbm(300, seed=1), synth.gbm(300, seed=2)
        WalkForwardOptimizer(Pairy, {}, WalkForwardConfig(train_bars=100, test_bars=50), partner=b).run(a)
        assert seen and all(p is not None and len(p) == len(b) for p in seen)
        with pytest.raises(ValueError, match="partner"):
            WalkForwardOptimizer(Pairy, {}, CFG)


# --------------------------------------------------------------------------
# Grid cost control
# --------------------------------------------------------------------------

class TestGridControl:
    BIG = {"entry_period": list(range(10, 90)), "exit_period": list(range(2, 12))}

    def test_oversized_grid_raises_clear_error(self):
        class Huge(DonchianBreakout):
            pass
        grid = {"entry_period": list(range(10, 70)), "exit_period": list(range(2, 10)), "atr_period": list(range(2, 12))}
        assert len(Huge.param_combinations(grid)) > MAX_GRID
        with pytest.raises(ValueError, match="max_combos"):
            WalkForwardOptimizer(Huge, grid, CFG)

    def test_random_search_seeded_ordered_subset(self):
        full = DonchianBreakout.param_combinations(self.BIG)
        a, b = random_search(DonchianBreakout, self.BIG, 20, seed=1), random_search(DonchianBreakout, self.BIG, 20, seed=1)
        c = random_search(DonchianBreakout, self.BIG, 20, seed=2)
        assert a == b and a != c and len(a) == 20 and len({tuple(x.items()) for x in a}) == 20
        assert [full.index(x) for x in a] == sorted(full.index(x) for x in a)
        assert random_search(DonchianBreakout, GRID, 1000) == DonchianBreakout.param_combinations(GRID)
        with pytest.raises(ValueError):
            random_search(DonchianBreakout, GRID, 0)

    def test_max_combos_subsamples_same_subset_every_window(self):
        o = WalkForwardOptimizer(DonchianBreakout, self.BIG, CFG, max_combos=4, seed=3)
        assert len(o.combos) == 4 and o.grid_size > 4
        res = o.run(synth.gbm(350, seed=4))
        assert all(w.n_candidates == 4 for w in res.windows)
        assert all(w.params in o.combos for w in res.active)


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------

class TestResult:
    @pytest.fixture(scope="class")
    @classmethod
    def res(cls) -> WalkForwardResult:
        return opt().run(synth.regime_switching(400, seed=6))

    def test_stitched_equity_chains_multiplicatively(self, res):
        sb = CFG.starting_balance
        assert res.equity[0] == sb
        expect_len = 1 + sum(len(w.oos_equity) for w in res.windows)
        assert len(res.equity) == expect_len == res.oos_report.n_bars
        run, pos = sb, 1
        for w in res.windows:
            seg = res.equity[pos:pos + len(w.oos_equity)]
            assert np.allclose(seg, run * w.oos_equity / sb)
            run, pos = seg[-1], pos + len(seg)
        total = np.prod([w.oos_equity[-1] / sb for w in res.windows])
        assert res.equity[-1] == pytest.approx(sb * total)
        assert res.oos_report.total_return == pytest.approx(total - 1)

    def test_aggregate_report_counts_all_oos_trades(self, res):
        assert res.oos_report.num_trades == sum(len(w.oos_trade_pnls) for w in res.windows)

    def test_efficiency_math(self):
        def wr(i, is_s, oos_s, skipped=False, params=None):
            return WindowResult(i, Window(0, 1, 1, 2), params, is_s, oos_s, None, None, 1, 1, skipped,
                                np.array([1.0]), np.array([0.0]), [])
        cfg = WalkForwardConfig()
        rep = metrics.compute_report([1.0, 1.0])

        def mk(ws, ranges=None):
            return WalkForwardResult(cfg, "x", ws, np.array([1.0]), rep, ranges or {})

        r = mk([wr(0, 2.0, 1.0, params={"a": 1}), wr(1, 1.0, float("nan"), params={"a": 1}), wr(2, 3.0, 2.0, params={"a": 3})])
        assert r.efficiency == pytest.approx(1.5 / 2.5)                 # NaN pair ignored
        assert mk([wr(0, -1.0, 1.0, params={"a": 1})]).efficiency != mk([wr(0, -1.0, 1.0, params={"a": 1})]).efficiency  # NaN
        assert math.isnan(mk([]).efficiency)

    def test_stability_and_overfit_flag(self):
        def wr(i, is_s, oos_s, params):
            return WindowResult(i, Window(0, 1, 1, 2), params, is_s, oos_s, None, None, 1, 1, False,
                                np.array([1.0]), np.array([0.0]), [])
        rep = metrics.compute_report([1.0, 1.0])
        ws = [wr(0, 2.0, 0.1, {"p": 10, "m": "x"}), wr(1, 2.0, -0.2, {"p": 10, "m": "x"}),
              wr(2, 2.0, 0.0, {"p": 50, "m": "y"}), wr(3, 2.0, 0.1, {"p": 10, "m": "x"})]
        r = WalkForwardResult(WalkForwardConfig(), "x", ws, np.array([1.0]), rep, {"p": 40.0})
        st = r.stability
        assert st["modal_params"] == {"p": 10, "m": "x"} and st["modal_fraction"] == 0.75
        assert st["dispersion"]["p"] == pytest.approx(np.std([10, 10, 50, 10]) / 40.0)
        assert st["dispersion"]["m"] == pytest.approx(0.25)
        assert r.overfit_flag                                            # efficiency ~0.0 < 0.5 with IS > 0
        good = WalkForwardResult(WalkForwardConfig(), "x", [wr(i, 1.0, 0.9, {"p": 1}) for i in range(4)],
                                 np.array([1.0]), rep, {})
        assert not good.overfit_flag and good.stability["modal_fraction"] == 1.0
        few = WalkForwardResult(WalkForwardConfig(), "x", ws[:2], np.array([1.0]), rep, {})
        assert not few.overfit_flag                                      # < 3 windows: not enough evidence

    def test_to_dict_is_strict_json(self, res):
        d = res.to_dict()
        text = json.dumps(d, allow_nan=False)                            # raises on NaN/inf or numpy types
        assert json.loads(text)["strategy"] == "donchian_breakout"
        assert len(d["equity"]) == len(res.equity)

    def test_to_dict_with_skipped_windows_and_callable_objective(self):
        cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60, min_trades=10_000, objective=lambda r: r.sharpe)
        d = opt(cfg).run(synth.gbm(350, seed=4)).to_dict()
        json.dumps(d, allow_nan=False)
        assert d["efficiency"] is None and d["windows"][0]["params"] is None

    def test_summary_mentions_key_facts(self, res):
        s = res.summary()
        assert "donchian_breakout" in s and "efficiency" in s and "overfit_flag" in s
        assert s.count("test ") >= len(res.windows)


# --------------------------------------------------------------------------
# Optional process pool: identical to serial
# --------------------------------------------------------------------------

def test_parallel_matches_serial():
    c = synth.regime_switching(350, seed=7)
    cfg = WalkForwardConfig(train_bars=150, test_bars=50, warmup_bars=60)
    serial, par = opt(cfg).run(c), opt(cfg, max_workers=2).run(c)
    assert [w.params for w in serial.windows] == [w.params for w in par.windows]
    assert np.array_equal(serial.equity, par.equity)
    with pytest.raises(ValueError):
        opt(max_workers=0)
