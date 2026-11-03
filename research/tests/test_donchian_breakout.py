"""Behavioural tests for DonchianBreakout - the TEMPLATE every strategy test file follows:
(1) rules on hand-built bars, (2) edge/parameter behaviour, (3) falsifiable claims on synthetic
data with a known generating process, (4) engine round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from core.models import Action
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.donchian_breakout import DonchianBreakout, DonchianParams
from research.tests.helpers import decide, simulate_long_flat


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


class TestRules:
    def test_enters_on_new_high_only(self):
        s = DonchianBreakout(entry_period=10, exit_period=5)
        flat = [100.0] * 30
        assert s.signal(bars_from_closes(flat + [100.0]), False).kind.value == "HOLD"       # inside channel
        assert s.signal(bars_from_closes(flat + [105.0]), False).kind.value == "ENTER"      # above 10-bar high

    def test_current_bar_excluded_from_channel(self):
        """A bar that makes the high must be compared to the PREVIOUS highs, not itself."""
        s = DonchianBreakout(entry_period=10, exit_period=5)
        sig = s.signal(bars_from_closes([100.0] * 30 + [101.0]), False)
        assert sig.kind.value == "ENTER" and sig.indicators["upper"] < 101.0

    def test_exit_below_exit_channel_only_when_in_position(self):
        s = DonchianBreakout(entry_period=10, exit_period=5)
        closes = [100.0] * 30 + [90.0]
        assert s.signal(bars_from_closes(closes), True).kind.value == "EXIT"
        assert s.signal(bars_from_closes(closes), False).kind.value == "HOLD"

    def test_stays_in_trade_while_above_exit_channel(self):
        s = DonchianBreakout(entry_period=10, exit_period=5)
        closes = list(np.linspace(100, 120, 30)) + [119.0]
        assert s.signal(bars_from_closes(closes), True).kind.value == "HOLD"

    def test_strength_increases_with_breakout_size(self):
        s = DonchianBreakout()
        small = s.signal(bars_from_closes([100.0] * 40 + [100.5]), False).strength
        big = s.signal(bars_from_closes([100.0] * 40 + [110.0]), False).strength
        assert 0.5 <= small < big <= 1.0

    def test_volume_filter_blocks_weak_breakouts(self):
        closes = np.array([100.0] * 40 + [110.0])
        vol = np.full(41, 1e6)
        vol[-1] = 0.5e6
        b = Bars("T", list(range(41)), closes, closes * 1.001, closes * 0.999, closes, vol)
        assert DonchianBreakout(vol_filter=1.5).signal(b, False).kind.value == "HOLD"
        assert DonchianBreakout(vol_filter=0.0).signal(b, False).kind.value == "ENTER"


class TestParams:
    def test_exit_period_cannot_exceed_entry(self):
        with pytest.raises(ValueError):
            DonchianParams(entry_period=10, exit_period=20)

    def test_grid_drops_invalid_combinations(self):
        combos = DonchianBreakout.param_combinations({"entry_period": [5, 20], "exit_period": [10]})
        assert combos == [{"entry_period": 20, "exit_period": 10}]

    def test_longer_entry_means_fewer_trades(self):
        c = synth.regime_switching(600, seed=2)
        fast, _ = simulate_long_flat(DonchianBreakout(entry_period=10, exit_period=5), c)
        slow, _ = simulate_long_flat(DonchianBreakout(entry_period=55, exit_period=20), c)
        assert len(slow) <= len(fast)


class TestSyntheticEdge:
    def test_profits_on_clean_uptrend(self):
        c = synth.trending(300, 0.004, 0.006, seed=1)
        r = run_backtest(DonchianBreakout(), c)
        assert r.report.total_return > 0.3 and r.report.time_in_market > 0.5

    def test_stays_out_of_persistent_downtrend(self):
        c = synth.trending(300, -0.004, 0.006, seed=2)
        r = run_backtest(DonchianBreakout(), c)
        assert r.report.avg_exposure < 0.15
        assert r.report.max_drawdown < 0.10

    def test_wins_on_range_then_breakout_cycles(self):
        c = synth.range_then_breakout(600, seed=4)
        r = run_backtest(DonchianBreakout(), c)
        assert r.report.total_return > 0.0

    def test_whipsaws_in_mean_reverting_market(self):
        """Honest negative result: breakouts fail in an OU series; the rule must NOT beat buy&hold there."""
        c = synth.mean_reverting(500, half_life=5.0, sigma=0.015, seed=9)
        r = run_backtest(DonchianBreakout(entry_period=10, exit_period=5), c)
        wins = r.report.win_rate
        assert not (wins > 0.6 and r.report.total_return > 0.2)


class TestBoundaries:
    def test_volume_baseline_excludes_the_breakout_bar(self):
        """Prior 20-bar average is 1.0e6; the breakout bar's own 1.25e6 must not lift the baseline."""
        s24, s26 = DonchianBreakout(vol_filter=1.24), DonchianBreakout(vol_filter=1.26)
        b = bars_from_closes([100.0] * 30 + [105.0])
        vol = np.full(31, 1e6)
        vol[-1] = 1.25e6
        bars = Bars("T", list(range(31)), b.open, b.high, b.low, b.close, vol)
        assert s24.signal(bars, False).kind.value == "ENTER"
        assert s26.signal(bars, False).kind.value == "HOLD"
