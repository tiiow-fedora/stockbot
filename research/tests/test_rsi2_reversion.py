"""Behavioural tests for Rsi2Reversion: rules on hand-built bars, parameter validation,
and falsifiable claims on synthetic data (wins on mean reversion / uptrend dips, loses on momentum)."""

from __future__ import annotations

import numpy as np
import pytest

from research import indicators as ind
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.rsi2_reversion import Rsi2Reversion, Rsi2ReversionParams


def bars_from_closes(closes):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), 1e6))


UP = list(np.linspace(80, 120, 230))      # steady uptrend: SMA200 ~ 103, last close 120
DOWN = list(np.linspace(160, 120, 230))   # steady downtrend: SMA200 ~ 137, last close 120


def sig(closes, in_position=False, **params):
    return Rsi2Reversion(**params).signal(bars_from_closes(closes), in_position)


class TestRules:
    def test_enters_on_oversold_dip_in_uptrend(self):
        closes = UP + [117.0, 113.0]
        assert ind.rsi(np.array(closes), 2)[-1] < 10 and closes[-1] > np.mean(closes[-200:])
        assert sig(closes).kind.value == "ENTER"

    def test_no_entry_when_not_oversold(self):
        assert sig(UP).kind.value == "HOLD"
        assert sig(UP + [119.0, 118.0], oversold=5.0).kind.value == "HOLD"   # RSI ~5.5 is not < 5

    def test_oversold_boundary(self):
        closes = UP + [119.5, 119.0]
        rsi = ind.rsi(np.array(closes), 2)[-1]
        assert 10.0 < rsi < 11.0
        assert sig(closes, oversold=10.0).kind.value == "HOLD"
        assert sig(closes, oversold=11.0).kind.value == "ENTER"

    def test_trend_filter_blocks_dips_in_downtrend(self):
        closes = DOWN + [117.0, 113.0]
        assert ind.rsi(np.array(closes), 2)[-1] < 10
        blocked = sig(closes, trend_period=200)
        assert blocked.kind.value == "HOLD" and "trend" in blocked.reason
        assert sig(closes, trend_period=0).kind.value == "ENTER"     # filter off: the knife is caught

    def test_exit_on_rsi_rebound_only(self):
        closes = DOWN + [121.0, 125.0, 128.0]
        # exit_sma=100: close (128) is still far below its 100-bar mean, so only the RSI leg can fire
        s = sig(closes, True, exit_sma=100)
        assert s.kind.value == "EXIT" and "RSI" in s.reason
        assert sig(closes, False, exit_sma=100).kind.value == "HOLD"     # never exits when flat

    def test_exit_on_close_above_short_sma_only(self):
        closes = DOWN + [121.0, 125.0, 128.0]
        s = sig(closes, True, exit_level=99.9)       # RSI ~99.2 does not exceed 99.9
        assert s.kind.value == "EXIT" and "SMA" in s.reason

    def test_holds_while_still_depressed(self):
        closes = UP + [119.0, 118.0]                 # RSI ~5.5, close below its 5-bar SMA
        assert sig(closes, True).kind.value == "HOLD"

    def test_strength_grows_with_oversoldness(self):
        mild = sig(UP + [119.0, 118.0], oversold=10.0)       # RSI ~5.5
        deep = sig(UP + [117.0, 113.0], oversold=10.0)       # RSI ~1.6
        assert mild.kind.value == deep.kind.value == "ENTER"
        assert 0.5 <= mild.strength < deep.strength <= 1.0

    def test_nan_rsi_holds(self):
        assert sig([100.0, 99.0]).reason == "indicators not ready"


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(rsi_period=0), dict(rsi_period=1), dict(rsi_period=-2), dict(oversold=0.0),
        dict(oversold=80.0, exit_level=70.0), dict(oversold=10.0, exit_level=100.0),
        dict(trend_period=-1), dict(exit_sma=0), dict(exit_sma=-5),
    ])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            Rsi2ReversionParams(**bad)

    def test_trend_period_zero_allowed(self):
        assert Rsi2ReversionParams(trend_period=0).trend_period == 0

    def test_grid_is_valid_and_sized(self):
        combos = Rsi2Reversion.param_combinations()
        assert 6 <= len(combos) <= 60
        assert {c["rsi_period"] for c in combos} == {2, 3, 4}

    def test_min_bars_covers_trend_filter(self):
        assert Rsi2Reversion().min_bars >= 200
        assert Rsi2Reversion(trend_period=0).min_bars < 20

    def test_stricter_oversold_means_fewer_trades(self):
        c = synth.mean_reverting(500, 8.0, 0.012, seed=2)
        from research.tests.helpers import simulate_long_flat
        loose, _ = simulate_long_flat(Rsi2Reversion(trend_period=0, oversold=25.0), c)
        strict, _ = simulate_long_flat(Rsi2Reversion(trend_period=0, oversold=3.0), c)
        assert len(strict) < len(loose)


class TestSyntheticEdge:
    def test_profits_on_mean_reverting_series(self):
        for seed in (1, 2, 3, 4):
            c = synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed)
            r = run_backtest(Rsi2Reversion(trend_period=0), c)
            assert r.report.total_return > 0.05, seed
            assert r.report.win_rate >= 0.55, seed
            assert r.report.total_return > r.benchmark_report.total_return, seed

    def test_dip_buying_in_uptrend_has_high_win_rate_but_trails_buy_and_hold(self):
        """It wins often but sits mostly in cash, so a strong uptrend is far better held."""
        for seed in (0, 1, 2):
            c = synth.trending(500, 0.003, 0.008, seed=seed)
            r = run_backtest(Rsi2Reversion(), c)
            assert r.report.total_return > 0.0 and r.report.win_rate >= 0.7, seed
            assert r.report.avg_exposure < 0.15
            assert r.report.total_return < 0.5 * r.benchmark_report.total_return, seed

    def test_trend_filter_keeps_it_out_of_a_downtrend(self):
        c = synth.trending(500, -0.003, 0.008, seed=1)
        filtered = run_backtest(Rsi2Reversion(), c).report
        assert filtered.avg_exposure < 0.02 and filtered.max_drawdown < 0.02

    def test_without_filter_downtrend_is_a_disaster(self):
        c = synth.trending(300, -0.004, 0.006, seed=1)
        r = run_backtest(Rsi2Reversion(trend_period=0), c)
        assert r.report.total_return < -0.3

    def test_loses_on_momentum_series(self):
        for seed in (1, 2, 3):
            c = synth.ar1_momentum(500, phi=0.5, sigma=0.01, mu=0.0003, seed=seed)
            assert run_backtest(Rsi2Reversion(trend_period=0), c).report.total_return < -0.1, seed

    def test_loses_on_slow_cycles(self):
        """Honest negative: a 2-bar oscillator cannot see an 80-bar swing; each leg is a trend."""
        c = synth.cyclical(500, period=80, amplitude=0.1, seed=1)
        assert run_backtest(Rsi2Reversion(trend_period=0), c).report.total_return < -0.2


class TestStrengthValues:
    def test_strength_formula_and_cap(self):
        s = sig(UP + [117.0, 113.0])
        assert s.kind.value == "ENTER"
        assert s.strength == pytest.approx(0.5 + 0.5 * (10.0 - s.indicators["rsi"]) / 10.0)
        washed_out = sig(UP + [115.0, 110.0, 105.0, 100.0], trend_period=0)
        assert washed_out.indicators["rsi"] < 1.0 and washed_out.strength > 0.95
