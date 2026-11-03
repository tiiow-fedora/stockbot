"""Behavioural tests for HighProximityMomentum: rules on hand-built bars, parameters, synthetic claims."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.high_proximity_momentum import HighProximityMomentum, HighProximityParams
from research.tests.helpers import simulate_long_flat


def bars_from_closes(closes):
    """high == close so the trailing high is exactly the highest close (clean percentages)."""
    c = np.asarray(closes, float)
    return Bars("T", list(range(len(c))), c, c, c * 0.999, c, np.full(len(c), 1e6))


def small(**kw):
    base = dict(window=20, proximity=0.03, recent=5, mom_period=10, exit_proximity=0.10)
    base.update(kw)
    return HighProximityMomentum(**base)


def rising(n=30, start=100.0, stop=120.0):
    return list(np.linspace(start, stop, n))


def kind(s, closes, in_position=False):
    return s.signal(bars_from_closes(closes), in_position).kind.value


class TestRules:
    def test_enters_at_new_high(self):
        sig = small().signal(bars_from_closes(rising()), False)
        assert sig.kind.value == "ENTER" and sig.indicators["distance"] == pytest.approx(0.0)
        assert sig.indicators["high_age"] == 0

    def test_enters_within_proximity_band(self):
        assert kind(small(), rising() + [119.0]) == "ENTER"  # ~0.8% under the high

    def test_no_entry_outside_proximity_band(self):
        closes = rising() + [120.0 * 0.95]  # 5% below the high
        assert kind(small(), closes) == "HOLD"
        assert kind(small(proximity=0.06), closes) == "ENTER"

    def test_proximity_boundary(self):
        high = 120.0
        assert kind(small(), rising() + [high * 0.971]) == "ENTER"
        assert kind(small(), rising() + [high * 0.969]) == "HOLD"

    def test_stale_high_blocks_entry(self):
        """Price sits ~1% under a high set 12 bars ago: near, but not a fresh high."""
        closes = rising(18, 100, 120) + [118.8] * 10 + [119.0]
        assert kind(small(recent=5), closes) == "HOLD"
        assert kind(small(recent=15), closes) == "ENTER"

    def test_non_positive_momentum_blocks_entry(self):
        """Near and recent high, but the close is below where it was ``mom_period`` bars ago."""
        closes = [100.0] * 20 + [108.0] * 10 + [107.5] * 4
        sig = small(recent=10).signal(bars_from_closes(closes), False)
        assert sig.indicators["momentum"] < 0 and sig.indicators["distance"] < 0.03
        assert sig.kind.value == "HOLD"

    def test_current_bar_counts_as_part_of_the_window(self):
        assert small().signal(bars_from_closes([100.0] * 25 + [110.0]), False).indicators["high"] == pytest.approx(110.0)

    def test_old_high_rolls_out_of_window(self):
        """A peak older than ``window`` bars no longer defines the high."""
        closes = [200.0] + [100.0] * 5 + rising(30, 100, 110)
        assert kind(small(window=20), closes) == "ENTER"
        assert kind(small(window=60), closes) == "HOLD"  # 200 still in view -> far below high

    def test_strength_is_closeness_to_the_high(self):
        s = small()
        at_high = s.signal(bars_from_closes(rising()), False).strength
        edge = s.signal(bars_from_closes(rising() + [120.0 * 0.971]), False).strength
        mid = s.signal(bars_from_closes(rising() + [120.0 * 0.985]), False).strength
        assert at_high == pytest.approx(1.0) and 0.5 <= edge < mid < at_high

    def test_exit_when_proximity_decays(self):
        closes = rising() + [120.0 * 0.89]
        assert kind(small(), closes, True) == "EXIT"
        assert kind(small(), closes, False) == "HOLD"

    def test_holds_through_moderate_pullback(self):
        assert kind(small(), rising() + [120.0 * 0.91], True) == "HOLD"

    def test_exit_boundary_follows_exit_proximity(self):
        closes = rising() + [120.0 * 0.87]
        assert kind(small(exit_proximity=0.10), closes, True) == "EXIT"
        assert kind(small(exit_proximity=0.15), closes, True) == "HOLD"

    def test_short_history_is_hold(self):
        assert kind(small(), [100.0, 101.0, 102.0]) == "HOLD"
        assert kind(HighProximityMomentum(), rising(100)) == "HOLD"  # shorter than window=120


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(window=0), dict(window=1), dict(recent=0), dict(mom_period=0), dict(mom_period=-5), dict(recent=500),
        dict(proximity=0.0), dict(proximity=1.0), dict(exit_proximity=1.2), dict(proximity=0.1, exit_proximity=0.05),
        dict(proximity=0.05, exit_proximity=0.05),
    ])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            HighProximityParams(**bad)

    def test_grid_drops_invalid_combinations(self):
        combos = HighProximityMomentum.param_combinations(
            {"proximity": [0.05, 0.2], "exit_proximity": [0.1]})
        assert combos == [{"proximity": 0.05, "exit_proximity": 0.1}]
        assert 6 <= len(HighProximityMomentum.param_combinations()) <= 60

    def test_tighter_proximity_means_less_time_invested(self):
        c = synth.gbm(500, seed=4)
        loose = run_backtest(HighProximityMomentum(window=60, mom_period=30, proximity=0.08), c)
        tight = run_backtest(HighProximityMomentum(window=60, mom_period=30, proximity=0.01), c)
        assert tight.report.avg_exposure < loose.report.avg_exposure


class TestSyntheticEdge:
    def test_profits_in_persistent_uptrend(self):
        c = synth.trending(400, 0.003, 0.008, seed=0)
        r = run_backtest(HighProximityMomentum(), c)
        assert r.report.total_return > 0.5 and r.report.time_in_market > 0.5

    def test_never_buys_a_persistent_downtrend(self):
        c = synth.trending(400, -0.003, 0.008, seed=1)
        r = run_backtest(HighProximityMomentum(), c)
        assert r.report.avg_exposure < 0.05 and r.report.max_drawdown < 0.05

    def test_enters_only_near_a_fresh_high(self):
        """Every simulated entry is within ``proximity`` of the high, with a recent high and positive momentum."""
        from research.tests.helpers import decide
        c = synth.regime_switching(500, seed=5)
        s = HighProximityMomentum(window=60, mom_period=30, recent=15)
        pos, seen = False, 0
        for i in range(s.min_bars - 1, len(c)):
            d = decide(s, c, i, pos)
            if d.action.value == "BUY":
                pos, seen = True, seen + 1
                assert d.indicators["distance"] <= 0.03 + 1e-9 and d.indicators["high_age"] <= 15
                assert d.indicators["momentum"] > 0
            elif d.action.value == "SELL":
                pos = False
        assert seen >= 1

    def test_loses_in_oscillating_market(self):
        """Honest negative: in a range-bound cycle it buys near the top of the swing and sells after a 10% slide."""
        c = synth.cyclical(500, period=40, amplitude=0.10, seed=9)
        r = run_backtest(HighProximityMomentum(window=60, mom_period=20), c)
        assert r.report.total_return < 0.0

    def test_exit_band_limits_loss_after_a_peak(self):
        """Uptrend then crash: the 10% exit band gets it out near the peak instead of riding the whole slide."""
        up = 100 * np.exp(np.cumsum(np.full(150, 0.004)))
        down = up[-1] * np.exp(np.cumsum(np.full(100, -0.006)))
        c = synth.candles_from_closes(np.concatenate([up, down]), seed=1)
        trades, still_open = simulate_long_flat(HighProximityMomentum(window=60, mom_period=30), c)
        assert not still_open and len(trades) == 1
        assert trades[0][2] > 0.2  # riding the whole slide would have ended below the entry price
