"""Behavioural tests for SuperTrendTrend: rules on hand-built bars, parameters, synthetic edge."""

from __future__ import annotations

import numpy as np
import pytest

from research import indicators as ind
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.supertrend_trend import SuperTrendParams, SuperTrendTrend
from research.tests.helpers import simulate_long_flat

FLAT_THEN_DROP = [100.0] * 30 + [90.0] * 10  # SuperTrend is -1 at the end


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


def direction_of(closes, period=10, mult=3.0):
    b = bars_from_closes(closes)
    return ind.supertrend(b.high, b.low, b.close, period, mult)[1]


class TestRules:
    def test_fixture_directions(self):
        assert direction_of(FLAT_THEN_DROP)[-1] == -1.0
        assert direction_of(FLAT_THEN_DROP + [101.0])[-1] == 1.0

    def test_enters_on_flip_to_uptrend(self):
        s = SuperTrendTrend()
        sig = s.signal(bars_from_closes(FLAT_THEN_DROP + [101.0]), False)
        assert sig.kind.value == "ENTER" and sig.indicators["age"] == 1.0

    def test_no_entry_while_downtrend(self):
        assert SuperTrendTrend().signal(bars_from_closes(FLAT_THEN_DROP), False).kind.value == "HOLD"

    def test_enters_shortly_after_flip_but_not_when_stale(self):
        s = SuperTrendTrend(max_age=5)
        assert s.signal(bars_from_closes(FLAT_THEN_DROP + [101.0] * 5), False).kind.value == "ENTER"   # age 5
        stale = s.signal(bars_from_closes(FLAT_THEN_DROP + [101.0] * 6), False)                       # age 6
        assert stale.kind.value == "HOLD" and "not chasing" in stale.reason

    def test_does_not_chase_old_uptrend_from_warmup(self):
        """Direction is +1 from the first valid bar; with no flip in view the strategy stays flat."""
        closes = np.linspace(100, 130, 60)
        assert SuperTrendTrend(max_age=10).signal(bars_from_closes(closes), False).kind.value == "HOLD"

    def test_exits_on_flip_down_only_when_long(self):
        s = SuperTrendTrend()
        closes = [100.0] * 30 + [90.0]
        assert s.signal(bars_from_closes(closes), True).kind.value == "EXIT"
        assert s.signal(bars_from_closes(closes), False).kind.value == "HOLD"

    def test_holds_while_uptrend_intact_even_when_stale(self):
        s = SuperTrendTrend(max_age=1)
        closes = list(np.linspace(100, 130, 60))
        assert s.signal(bars_from_closes(closes), True).kind.value == "HOLD"

    def test_small_dip_does_not_exit(self):
        s = SuperTrendTrend()
        closes = list(np.linspace(100, 120, 40)) + [119.5]
        assert s.signal(bars_from_closes(closes), True).kind.value == "HOLD"

    def test_strength_grows_with_distance_above_line(self):
        s = SuperTrendTrend(max_age=50)
        stalled = s.signal(bars_from_closes(FLAT_THEN_DROP + [101.0] * 5), False)  # line catches up
        rallying = s.signal(bars_from_closes(FLAT_THEN_DROP + [101.0, 102.0, 103.0, 104.0, 105.0]), False)
        assert 0.5 <= stalled.strength < rallying.strength <= 1.0

    def test_warmup_nan_is_hold(self):
        sig = SuperTrendTrend().signal(bars_from_closes([100.0] * 5), False)
        assert sig.kind.value == "HOLD" and "not ready" in sig.reason


class TestParams:
    @pytest.mark.parametrize("kw", [{"period": 0}, {"period": 1}, {"mult": 0.0}, {"mult": -1.0}, {"max_age": 0}])
    def test_invalid_params(self, kw):
        with pytest.raises(ValueError):
            SuperTrendParams(**kw)

    def test_grid_size_and_validity(self):
        combos = SuperTrendTrend.param_combinations()
        assert 6 <= len(combos) <= 60
        for c in combos:
            SuperTrendParams(**c)

    def test_lookback_covers_max_age(self):
        assert SuperTrendTrend(max_age=40).lookback >= SuperTrendTrend(max_age=40).min_bars + 40

    def test_wider_band_means_fewer_trades(self):
        c = synth.regime_switching(600, seed=2)
        tight, _ = simulate_long_flat(SuperTrendTrend(period=7, mult=1.5), c)
        wide, _ = simulate_long_flat(SuperTrendTrend(period=7, mult=4.0), c)
        assert len(wide) < len(tight)


class TestSyntheticEdge:
    def test_profits_on_range_breakout_cycles(self):
        r = run_backtest(SuperTrendTrend(), synth.range_then_breakout(600, seed=8))
        assert r.report.total_return > 0.3

    def test_profits_on_v_shape_recovery(self):
        r = run_backtest(SuperTrendTrend(), synth.v_shape(300, seed=10))
        assert r.report.total_return > 0.3 and r.report.max_drawdown < 0.15

    def test_never_enters_a_trend_with_no_flip_in_view(self):
        """Documented limitation: a clean uptrend that was already running when data starts
        never produces a flip, so the strategy stays flat (and earns nothing)."""
        r = run_backtest(SuperTrendTrend(), synth.trending(300, 0.004, 0.006, seed=1))
        assert r.report.avg_exposure == 0.0 and r.report.total_return == 0.0

    def test_stays_out_of_persistent_downtrend(self):
        r = run_backtest(SuperTrendTrend(), synth.trending(300, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.15 and r.report.max_drawdown < 0.10

    def test_loses_in_mean_reverting_market(self):
        """Honest negative result: band flips are noise in an OU series, so no profit."""
        for seed in (3, 9):
            r = run_backtest(SuperTrendTrend(), synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed))
            assert r.report.total_return < 0.05

    def test_stale_entry_guard_delays_first_entry_in_trend(self):
        """With max_age=1 on a smooth uptrend the strategy waits for a real flip, so it is
        invested less than with a generous window that accepts late/re-entries."""
        c = synth.trending(300, 0.004, 0.006, seed=1)
        strict = run_backtest(SuperTrendTrend(max_age=1), c).report.avg_exposure
        loose = run_backtest(SuperTrendTrend(max_age=200), c).report.avg_exposure
        assert strict < loose
