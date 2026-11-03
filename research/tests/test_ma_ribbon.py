"""Behavioural tests for MARibbon: rules on hand-built bars, parameters, synthetic edge."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.ma_ribbon import MARibbon, MARibbonParams
from research.tests.helpers import simulate_long_flat

RAMP = np.linspace(100, 160, 150)
# decline then sharp V-rally: fast > mid > slow, yet the slow EMA is still falling on a 40-bar view
V_REBOUND = np.concatenate([np.linspace(200, 100, 150), np.linspace(100, 144, 22)])


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


class TestRules:
    def test_enters_when_aligned_rising_and_fanned(self):
        sig = MARibbon().signal(bars_from_closes(RAMP), False)
        i = sig.indicators
        assert sig.kind.value == "ENTER"
        assert i["fast"] > i["mid"] > i["slow"] and i["slow_slope"] > 0

    def test_no_entry_in_downtrend(self):
        assert MARibbon().signal(bars_from_closes(RAMP[::-1]), False).kind.value == "HOLD"

    def test_no_entry_when_ribbon_not_aligned(self):
        """Uptrend that just rolled over: fast fell below mid even though mid > slow."""
        closes = np.concatenate([RAMP, np.linspace(160, 140, 15)])
        sig = MARibbon().signal(bars_from_closes(closes), False)
        assert sig.kind.value == "HOLD" and sig.indicators["fast"] < sig.indicators["mid"]

    def test_slope_gate_blocks_aligned_but_falling_slow_ma(self):
        b = bars_from_closes(V_REBOUND)
        blocked = MARibbon(slope_period=40, min_separation_atr=0.0).signal(b, False)
        assert blocked.indicators["fast"] > blocked.indicators["mid"] > blocked.indicators["slow"]
        assert blocked.kind.value == "HOLD" and "not rising" in blocked.reason
        assert MARibbon(slope_period=5, min_separation_atr=0.0).signal(b, False).kind.value == "ENTER"

    def test_separation_filter_blocks_narrow_fan(self):
        gentle = bars_from_closes(np.linspace(100, 100.6, 150))
        assert MARibbon(min_separation_atr=0.0).signal(gentle, False).kind.value == "ENTER"
        narrow = MARibbon(min_separation_atr=1.0).signal(gentle, False)
        assert narrow.kind.value == "HOLD" and "too narrow" in narrow.reason

    def test_separation_boundary_is_inclusive(self):
        b = bars_from_closes(RAMP)
        sep = MARibbon().signal(b, False).indicators["separation_atr"]
        assert MARibbon(min_separation_atr=sep).signal(b, False).kind.value == "ENTER"
        assert MARibbon(min_separation_atr=sep * 1.001).signal(b, False).kind.value == "HOLD"

    def test_exit_when_fast_crosses_below_mid_only_if_long(self):
        closes = np.concatenate([RAMP, np.linspace(160, 140, 15)])
        b = bars_from_closes(closes)
        assert MARibbon().signal(b, True).kind.value == "EXIT"
        assert MARibbon().signal(b, False).kind.value == "HOLD"

    def test_holds_through_a_narrow_fan_while_fast_above_mid(self):
        """The separation and slope filters gate entries only; they must not trigger exits."""
        b = bars_from_closes(np.linspace(100, 100.6, 150))
        assert MARibbon(min_separation_atr=5.0).signal(b, True).kind.value == "HOLD"

    def test_strength_grows_with_fan_width(self):
        wide = MARibbon(min_separation_atr=0.0).signal(bars_from_closes(RAMP), False).strength
        slim = MARibbon(min_separation_atr=0.0).signal(bars_from_closes(np.linspace(100, 103, 150)), False).strength
        assert 0.5 <= slim < wide <= 1.0

    def test_warmup_is_hold(self):
        s = MARibbon()
        sig = s.signal(bars_from_closes(RAMP[: s.params.slow - 1]), False)
        assert sig.kind.value == "HOLD" and "not ready" in sig.reason


class TestParams:
    @pytest.mark.parametrize(
        "kw",
        [{"fast": 0}, {"mid": -1}, {"slow": 0}, {"fast": 30, "mid": 30}, {"mid": 70}, {"slope_period": 1},
         {"atr_period": 0}, {"min_separation_atr": -0.1}],
    )
    def test_invalid_params(self, kw):
        with pytest.raises(ValueError):
            MARibbonParams(**kw)

    def test_grid_size_and_validity(self):
        combos = MARibbon.param_combinations()
        assert 6 <= len(combos) <= 60
        for c in combos:
            MARibbonParams(**c)

    def test_grid_drops_misordered_cells(self):
        combos = MARibbon.param_combinations({"fast": [10, 40], "mid": [30], "slow": [60]})
        assert combos == [{"fast": 10, "mid": 30, "slow": 60}]

    def test_separation_filter_reduces_trades_in_chop(self):
        c = synth.mean_reverting(600, half_life=8.0, sigma=0.012, seed=3)
        loose, _ = simulate_long_flat(MARibbon(min_separation_atr=0.0), c)
        tight, _ = simulate_long_flat(MARibbon(min_separation_atr=1.5), c)
        assert len(tight) < len(loose)


class TestSyntheticEdge:
    def test_profits_on_clean_uptrend(self):
        r = run_backtest(MARibbon(), synth.trending(300, 0.004, 0.006, seed=1))
        assert r.report.total_return > 0.5 and r.report.avg_exposure > 0.6

    def test_stays_out_of_persistent_downtrend(self):
        r = run_backtest(MARibbon(), synth.trending(300, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.1 and r.report.max_drawdown < 0.05

    def test_profits_on_range_breakout_cycles(self):
        r = run_backtest(MARibbon(), synth.range_then_breakout(600, seed=8))
        assert r.report.total_return > 0.3

    def test_loses_in_mean_reverting_market(self):
        """Honest negative result: EMA alignment lags an OU series and buys near the tops."""
        for seed in (3, 9):
            r = run_backtest(MARibbon(), synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed))
            assert r.report.total_return < 0.05

    def test_loses_on_fast_cycles_shorter_than_its_lag(self):
        """A 40-bar sine cycle is shorter than the 60-EMA lag: the ribbon buys late and sells late."""
        r = run_backtest(MARibbon(), synth.cyclical(500, seed=9))
        assert r.report.total_return < 0.0
