"""Behavioural tests for AtrExpansionBreakout: rules on hand-built OHLC bars, parameters, synthetic claims."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.atr_expansion_breakout import AtrExpansionBreakout, AtrExpansionParams
from research.tests.helpers import simulate_long_flat


def ohlc(rows):
    """Bars from (open, high, low, close) rows."""
    a = np.asarray(rows, float)
    return Bars("T", list(range(len(a))), a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.full(len(a), 1e6))


# 40 quiet bars: close 100, range 99..101 -> true range 2, ATR 2.
QUIET = [(100.0, 101.0, 99.0, 100.0)] * 40


def with_last(*bar):
    return ohlc(QUIET + [bar])


def kind(s, bars, in_position=False):
    return s.signal(bars, in_position).kind.value


class TestEntryRules:
    def test_enters_on_thrust_bar(self):
        sig = AtrExpansionBreakout().signal(with_last(100.0, 103.2, 100.5, 103.0), False)
        assert sig.kind.value == "ENTER"
        assert sig.indicators["thrust"] == pytest.approx(1.5, abs=0.05)
        assert sig.indicators["range_ratio"] > 1.3

    def test_small_gain_is_not_a_thrust(self):
        assert kind(AtrExpansionBreakout(), with_last(100.0, 101.8, 99.9, 101.0)) == "HOLD"

    def test_gain_threshold_scales_with_k(self):
        bar = with_last(100.0, 102.7, 100.2, 102.0)  # thrust = 1.0 ATR, TR 2.7
        assert kind(AtrExpansionBreakout(k=0.75), bar) == "ENTER"
        assert kind(AtrExpansionBreakout(k=1.25), bar) == "HOLD"

    def test_range_must_expand(self):
        """Gap up and close high but the bar's own range is small: only passes with a lax ``expansion``."""
        bar = with_last(102.9, 103.2, 102.5, 103.0)  # H-L = 0.7, but true range = 3.2 incl. the gap
        assert kind(AtrExpansionBreakout(expansion=1.3), bar) == "ENTER"  # gap counts via true range
        assert kind(AtrExpansionBreakout(expansion=2.0), bar) == "HOLD"

    def test_close_must_be_in_top_of_range(self):
        weak_close = with_last(100.0, 106.0, 100.5, 103.0)  # close at 45% of range
        strong_close = with_last(100.0, 103.2, 100.5, 103.0)
        s = AtrExpansionBreakout(close_pos=0.3)
        assert kind(s, weak_close) == "HOLD" and kind(s, strong_close) == "ENTER"
        assert kind(AtrExpansionBreakout(close_pos=0.6), weak_close) == "ENTER"

    def test_threshold_uses_previous_bar_atr(self):
        """The thrust bar itself must not inflate the ATR it is measured against (its own TR would
        lift ATR(14) by ~10% and the same bar would then fail a threshold set from ATR(prev))."""
        s = AtrExpansionBreakout(k=1.45, expansion=1.0)
        sig = s.signal(with_last(100.0, 103.2, 100.5, 103.0), False)  # thrust 1.5 vs ATR_prev = 2.0
        assert sig.kind.value == "ENTER" and sig.indicators["atr"] > 2.0

    def test_strength_grows_with_thrust(self):
        s = AtrExpansionBreakout()
        small = s.signal(with_last(100.0, 102.7, 100.0, 102.2), False)
        big = s.signal(with_last(100.0, 105.2, 100.5, 105.0), False)
        assert small.kind.value == big.kind.value == "ENTER"
        assert 0.5 <= small.strength < big.strength <= 1.0

    def test_ignored_signal_flat_vs_long(self):
        bar = with_last(100.0, 103.2, 100.5, 103.0)
        assert kind(AtrExpansionBreakout(), bar, in_position=True) == "HOLD"  # already long: no pyramiding

    def test_nan_and_flat_data_are_hold(self):
        s = AtrExpansionBreakout()
        assert kind(s, ohlc([(100.0, 101.0, 99.0, 100.0)] * 5)) == "HOLD"
        flat = ohlc([(100.0, 100.0, 100.0, 100.0)] * 40 + [(100.0, 105.0, 100.0, 105.0)])
        assert kind(s, flat) == "HOLD"  # ATR_prev == 0: no meaningful expansion


class TestExitRules:
    def run_up(self):
        """Steady climb to ~120 (high 121) then a final bar closing at a chosen level."""
        closes = np.linspace(100, 120, 40)
        return [(c - 0.2, c + 1.0, c - 1.0, c) for c in closes]

    def final(self, close):
        return ohlc(self.run_up() + [(120.0, 121.0, close - 0.5, close)])

    def test_holds_above_chandelier(self):
        sig = AtrExpansionBreakout().signal(self.final(119.0), True)
        assert sig.kind.value == "HOLD" and sig.indicators["chandelier"] < 119.0

    def test_exits_below_chandelier(self):
        sig = AtrExpansionBreakout().signal(self.final(110.0), True)
        assert sig.kind.value == "EXIT" and sig.indicators["close"] < sig.indicators["chandelier"]

    def test_chandelier_is_highest_high_minus_mult_atr(self):
        s = AtrExpansionBreakout(chand_mult=3.0)
        sig = s.signal(self.final(119.0), True)
        assert sig.indicators["chandelier"] == pytest.approx(sig.indicators["chandelier_high"] - 3.0 * sig.indicators["atr"])

    def test_wider_multiple_tolerates_deeper_pullback(self):
        b = self.final(114.0)
        assert kind(AtrExpansionBreakout(chand_mult=2.0), b, True) == "EXIT"
        assert kind(AtrExpansionBreakout(chand_mult=6.0), b, True) == "HOLD"

    def test_exit_ignored_when_flat(self):
        assert kind(AtrExpansionBreakout(), self.final(110.0), False) == "HOLD"

    def test_old_highs_leave_the_chandelier_window(self):
        """Highest high is taken over ``chand_period`` bars only: a stale peak must not pin the stop."""
        peak = [(100.0, 150.0, 99.0, 100.0)]
        rest = [(100.0, 101.0, 99.0, 100.0)] * 40
        b = ohlc(peak + rest)
        assert kind(AtrExpansionBreakout(chand_period=22), b, True) == "HOLD"
        assert b.high.max() == 150.0  # sanity: the spike really is in the data


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(atr_period=0), dict(atr_period=1), dict(chand_period=0), dict(chand_period=-3), dict(k=-0.1),
        dict(expansion=-1.0), dict(close_pos=0.0), dict(close_pos=1.5), dict(chand_mult=0.0),
    ])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            AtrExpansionParams(**bad)

    def test_grid_has_valid_cells(self):
        assert 6 <= len(AtrExpansionBreakout.param_combinations()) <= 60

    def test_stricter_expansion_means_fewer_trades(self):
        c = synth.gbm(600, seed=4)
        loose, _ = simulate_long_flat(AtrExpansionBreakout(k=0.25, expansion=0.5), c)
        strict, _ = simulate_long_flat(AtrExpansionBreakout(k=1.25, expansion=2.0), c)
        assert len(strict) < len(loose)

    def test_tighter_chandelier_means_more_trades(self):
        c = synth.regime_switching(600, seed=2)
        tight, _ = simulate_long_flat(AtrExpansionBreakout(chand_mult=1.5), c)
        wide, _ = simulate_long_flat(AtrExpansionBreakout(chand_mult=5.0), c)
        assert len(tight) > len(wide)


class TestSyntheticEdge:
    def test_profits_on_range_then_breakout_cycles(self):
        c = synth.range_then_breakout(500, seed=8)
        r = run_backtest(AtrExpansionBreakout(), c)
        assert r.report.total_return > 0.5 and r.report.sharpe > 1.5

    def test_profits_on_strong_uptrend(self):
        c = synth.trending(400, 0.004, 0.006, seed=1)
        assert run_backtest(AtrExpansionBreakout(), c).report.total_return > 0.3

    def test_limited_exposure_in_downtrend(self):
        c = synth.trending(300, -0.004, 0.006, seed=2)
        r = run_backtest(AtrExpansionBreakout(), c)
        assert r.report.avg_exposure < 0.25 and r.report.total_return < 0.05

    def test_low_hit_rate_in_mean_reverting_and_jumpy_markets(self):
        """Honest negative: thrust bars there are spikes that retrace, so it loses more often than it wins."""
        for c in (synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=3), synth.jump_diffusion(500, seed=7)):
            r = run_backtest(AtrExpansionBreakout(), c)
            assert r.report.win_rate < 0.5
