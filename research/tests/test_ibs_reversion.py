"""Behavioural tests for IbsReversion: rules on hand-built OHLC bars, parameter validation,
and falsifiable claims on synthetic data (wins on mean reversion / fast cycles, loses in trends)."""

from __future__ import annotations

import numpy as np
import pytest

from research import indicators as ind
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.ibs_reversion import IbsReversion, IbsReversionParams
from research.tests.helpers import simulate_long_flat


def ohlc(closes, last_low, last_high):
    """Bars whose history has IBS = 0.5 (H = C+0.5, L = C-0.5) and whose final bar has the given low/high."""
    c = np.asarray(closes, float)
    high, low = c + 0.5, c - 0.5
    high[-1], low[-1] = last_high, last_low
    return Bars("T", list(range(len(c))), c, high, low, c, np.full(len(c), 1e6))


def wiggle(n=40, base=100.0, amp=0.5):
    return [base + amp * (-1) ** i for i in range(n)]


def sig(last_close, last_low, last_high, in_position=False, **params):
    closes = wiggle() + [last_close]
    return IbsReversion(**params).signal(ohlc(closes, last_low, last_high), in_position)


class TestRules:
    def test_enters_on_low_close_below_mean(self):
        s = sig(98.0, 97.9, 100.0)
        assert s.indicators["ibs"] < 0.1 and s.indicators["z"] < -1.0
        assert s.kind.value == "ENTER"

    def test_needs_weak_close_not_just_low_price(self):
        """Same price drop but closing mid-range (IBS 0.5): no entry."""
        assert sig(98.0, 97.0, 99.0).kind.value == "HOLD"

    def test_needs_stretch_not_just_weak_close(self):
        """Weak close (IBS ~0.05) at the mean: z ~ 0, no entry."""
        assert sig(100.0, 99.9, 102.0).kind.value == "HOLD"

    def test_ibs_entry_boundary(self):
        # low 97, close 98: IBS = 1/(H-97); H chosen to straddle 0.2
        assert sig(98.0, 97.0, 97.0 + 1 / 0.19).kind.value == "ENTER"      # IBS 0.19
        assert sig(98.0, 97.0, 97.0 + 1 / 0.21).kind.value == "HOLD"       # IBS 0.21

    def test_z_entry_boundary(self):
        closes = wiggle() + [99.0]
        z = ind.zscore(np.array(closes), 20)[-1]
        assert -2.0 < z < -1.0
        bars = ohlc(closes, 98.95, 101.0)
        assert IbsReversion(z_entry=1.0).signal(bars, False).kind.value == "ENTER"
        assert IbsReversion(z_entry=-z + 0.05).signal(bars, False).kind.value == "HOLD"

    def test_exit_when_price_back_above_mean_even_with_weak_close(self):
        s = sig(100.8, 100.7, 102.0, in_position=True)       # IBS ~0.08 but z > 0
        assert s.kind.value == "EXIT" and "mean" in s.reason

    def test_exit_on_strong_close_while_still_below_mean(self):
        s = sig(99.0, 97.0, 99.1, in_position=True)           # IBS ~0.98, z < 0
        assert s.indicators["z"] < 0
        assert s.kind.value == "EXIT" and "IBS" in s.reason

    def test_holds_while_weak_and_below_mean(self):
        assert sig(98.0, 97.9, 100.0, in_position=True).kind.value == "HOLD"

    def test_exit_never_fires_when_flat(self):
        assert sig(100.8, 100.7, 102.0, in_position=False).kind.value == "HOLD"

    def test_zero_range_bar_cannot_enter_but_can_exit_via_z(self):
        assert sig(98.0, 98.0, 98.0).kind.value == "HOLD"                          # IBS undefined
        assert sig(98.0, 98.0, 98.0, in_position=True).kind.value == "HOLD"       # z < 0 still
        assert sig(101.0, 101.0, 101.0, in_position=True).kind.value == "EXIT"    # z > 0

    def test_strength_grows_with_weaker_close_and_deeper_stretch(self):
        base = sig(98.6, 98.5, 100.0)                  # IBS ~0.07 near the entry limit
        lower_ibs = sig(98.6, 98.599, 100.0)           # closes at the very low
        deeper = sig(97.0, 96.9, 100.0)                # much more stretched
        assert all(x.kind.value == "ENTER" for x in (base, lower_ibs, deeper))
        assert 0.5 <= base.strength < lower_ibs.strength <= 1.0
        assert base.strength < deeper.strength <= 1.0

    def test_constant_price_is_hold(self):
        c = np.full(40, 100.0)
        assert IbsReversion().signal(Bars("T", list(range(40)), c, c, c, c, c), False).kind.value == "HOLD"


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(z_period=0), dict(z_period=2), dict(z_period=-4), dict(z_entry=0.0), dict(z_entry=-1.0),
        dict(ibs_entry=0.0), dict(ibs_entry=0.8, ibs_exit=0.8), dict(ibs_entry=0.9, ibs_exit=0.5),
        dict(ibs_exit=1.2),
    ])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            IbsReversionParams(**bad)

    def test_grid_is_valid_and_sized(self):
        combos = IbsReversion.param_combinations()
        assert 6 <= len(combos) <= 60

    def test_min_bars_scales_with_z_period(self):
        assert IbsReversion(z_period=40).min_bars > IbsReversion(z_period=10).min_bars

    def test_stricter_ibs_gate_means_fewer_trades(self):
        c = synth.mean_reverting(500, 8.0, 0.012, seed=2)
        strict, _ = simulate_long_flat(IbsReversion(ibs_entry=0.1), c)
        loose, _ = simulate_long_flat(IbsReversion(ibs_entry=0.4), c)
        assert len(strict) < len(loose)


class TestSyntheticEdge:
    def test_profits_on_mean_reverting_series(self):
        for seed in (1, 2, 3, 4):
            c = synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed)
            r = run_backtest(IbsReversion(), c)
            assert r.report.total_return > 0.04, seed
            assert r.report.win_rate >= 0.5, seed
            assert r.report.total_return > r.benchmark_report.total_return, seed

    def test_trades_are_short_swing_trades(self):
        trades, _ = simulate_long_flat(IbsReversion(), synth.mean_reverting(500, 8.0, 0.012, seed=2))
        assert len(trades) >= 10
        assert np.median([exit_i - entry_i for entry_i, exit_i, _ in trades]) <= 5

    def test_profits_on_fast_cycles(self):
        for seed in (0, 1, 2):
            c = synth.cyclical(500, period=10, amplitude=0.04, seed=seed)
            assert run_backtest(IbsReversion(), c).report.total_return > 0.3, seed

    def test_loses_on_slow_cycles(self):
        """Honest negative: on an 80-bar sine, 'low close below the 20-bar mean' recurs all the way down."""
        c = synth.cyclical(500, period=80, amplitude=0.1, seed=1)
        r = run_backtest(IbsReversion(), c)
        assert r.report.total_return < -0.2 and r.report.win_rate < 0.3

    def test_loses_badly_in_persistent_downtrend(self):
        """No trend filter: it buys every weak close of a falling market."""
        for seed in (1, 2):
            c = synth.trending(300, -0.004, 0.006, seed=seed)
            r = run_backtest(IbsReversion(), c)
            assert r.report.total_return < -0.3 and r.report.win_rate < 0.4, seed

    def test_loses_on_momentum_series(self):
        for seed in (1, 2, 3):
            c = synth.ar1_momentum(500, phi=0.5, sigma=0.01, mu=0.0003, seed=seed)
            assert run_backtest(IbsReversion(), c).report.total_return < -0.05, seed
