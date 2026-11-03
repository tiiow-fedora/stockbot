"""Behavioural tests for BollingerReversion: rules on hand-built bars, parameter validation,
and falsifiable claims on synthetic data (wins on mean reversion, loses in trends)."""

from __future__ import annotations

import numpy as np
import pytest

from research import indicators as ind
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.bollinger_reversion import BollingerReversion, BollingerReversionParams


def bars_from_closes(closes):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), 1e6))


def wiggle(n=40, base=100.0, amp=0.5):
    """Flat series alternating +-amp: mean = base, std = amp, so band widths are known."""
    return [base + amp * (-1) ** i for i in range(n)]


def kind(strategy, closes, in_position=False):
    return strategy.signal(bars_from_closes(closes), in_position).kind.value


class TestRules:
    def test_enters_only_below_lower_band(self):
        s = BollingerReversion(period=20, num_std=2.0)
        assert kind(s, wiggle() + [100.0]) == "HOLD"      # at the mean
        assert kind(s, wiggle() + [98.0]) == "ENTER"      # far below the band
        assert kind(s, wiggle() + [100.8]) == "HOLD"      # above the mean never buys

    def test_boundary_is_strictly_below_band(self):
        """A close exactly on the lower band is not a signal; epsilon below it is."""
        s = BollingerReversion(period=20, num_std=2.0)
        base = wiggle()
        # solve for the close that lands on the band: close = mean - 2*std of window incl. close
        lo, hi = 90.0, 100.0
        for _ in range(60):
            c = (lo + hi) / 2
            _, _, lower = ind.bollinger(np.array(base + [c]), 20, 2.0)
            lo, hi = (c, hi) if c < lower[-1] else (lo, c)
        assert kind(s, base + [lo - 1e-3]) == "ENTER"
        assert kind(s, base + [hi + 1e-3]) == "HOLD"

    def test_exit_at_mid(self):
        s = BollingerReversion(period=20, num_std=2.0, exit_at="mid")
        assert kind(s, wiggle() + [98.0], True) == "HOLD"     # still depressed
        assert kind(s, wiggle() + [100.5], True) == "EXIT"    # back above the middle band
        assert kind(s, wiggle() + [100.5], False) == "HOLD"   # exit never fires when flat

    def test_exit_at_upper_waits_longer(self):
        closes = wiggle() + [100.3]
        assert kind(BollingerReversion(period=20, exit_at="mid"), closes, True) == "EXIT"
        assert kind(BollingerReversion(period=20, exit_at="upper"), closes, True) == "HOLD"
        assert kind(BollingerReversion(period=20, exit_at="upper"), wiggle() + [103.0], True) == "EXIT"

    def test_strength_grows_with_depth(self):
        s = BollingerReversion(period=20, num_std=2.0)
        shallow = s.signal(bars_from_closes(wiggle() + [98.6]), False)
        deep = s.signal(bars_from_closes(wiggle() + [97.0]), False)
        assert shallow.kind.value == deep.kind.value == "ENTER"
        assert 0.5 <= shallow.strength < deep.strength <= 1.0

    def test_trend_filter_blocks_dips_below_long_sma(self):
        # long decline, then a fresh dip: below the 60-bar SMA -> skipped with the filter, bought without
        closes = list(np.linspace(130, 100, 80)) + wiggle(20, 100.0) + [97.0]
        assert kind(BollingerReversion(period=20, trend_period=0), closes) == "ENTER"
        assert kind(BollingerReversion(period=20, trend_period=60), closes) == "HOLD"

    def test_trend_filter_allows_dips_in_uptrend(self):
        closes = list(np.linspace(70, 100, 80)) + wiggle(20, 100.0) + [97.0]
        assert kind(BollingerReversion(period=20, trend_period=60), closes) == "ENTER"

    def test_panic_cap_skips_regime_breaks(self):
        s_off = BollingerReversion(period=20, num_std=2.0, panic_std=0.0)
        s_on = BollingerReversion(period=20, num_std=2.0, panic_std=3.0)
        mild = wiggle() + [98.8]
        crash = wiggle() + [90.0]
        assert kind(s_on, mild) == kind(s_off, mild) == "ENTER"
        assert kind(s_off, crash) == "ENTER"
        sig = s_on.signal(bars_from_closes(crash), False)
        assert sig.kind.value == "HOLD" and "panic" in sig.reason

    def test_zero_variance_and_warmup_are_hold(self):
        s = BollingerReversion(period=20)
        assert kind(s, [100.0] * 40) == "HOLD"
        assert s.signal(bars_from_closes([100.0] * 5 + [90.0]), False).reason == "indicators not ready"
        assert kind(s, [100.0] * 5 + [90.0], True) == "HOLD"


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(period=0), dict(period=1), dict(period=-5), dict(num_std=0.0), dict(num_std=-1.0),
        dict(exit_at="lower"), dict(trend_period=-1), dict(panic_std=-1.0),
        dict(num_std=2.0, panic_std=2.0), dict(num_std=2.0, panic_std=1.5),
    ])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            BollingerReversionParams(**bad)

    def test_zero_disables_optional_filters(self):
        p = BollingerReversionParams(trend_period=0, panic_std=0.0)
        assert p.trend_period == 0 and p.panic_std == 0.0

    def test_grid_is_nonempty_and_valid(self):
        combos = BollingerReversion.param_combinations()
        assert 6 <= len(combos) <= 60
        assert {c["exit_at"] for c in combos} == {"mid", "upper"}

    def test_min_bars_covers_trend_filter(self):
        assert BollingerReversion(period=20, trend_period=0).min_bars < BollingerReversion(period=20, trend_period=100).min_bars
        assert BollingerReversion(period=20, trend_period=100).min_bars >= 100

    def test_wider_band_means_fewer_trades(self):
        c = synth.mean_reverting(500, 8.0, 0.012, seed=3)
        from research.tests.helpers import simulate_long_flat
        narrow, _ = simulate_long_flat(BollingerReversion(period=20, num_std=1.0), c)
        wide, _ = simulate_long_flat(BollingerReversion(period=20, num_std=2.5), c)
        assert len(wide) < len(narrow)


class TestSyntheticEdge:
    def test_profits_and_beats_buy_hold_on_mean_reverting_series(self):
        for seed in (1, 2, 3, 4):
            c = synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed)
            r = run_backtest(BollingerReversion(), c)
            assert r.report.total_return > 0.04, seed
            assert r.report.win_rate >= 0.6, seed
            assert r.report.total_return > r.benchmark_report.total_return, seed

    def test_profits_on_fast_cycles_when_band_matches_cycle(self):
        """Edge exists only when the lookback is short relative to the cycle (period 16 here)."""
        for seed in (0, 1, 2):
            c = synth.cyclical(500, period=16, amplitude=0.05, seed=seed)
            assert run_backtest(BollingerReversion(period=10, num_std=1.5), c).report.total_return > 0.1

    def test_loses_on_slow_cycles(self):
        """Honest negative: on an 80-bar sine each leg is a persistent trend for the 20-bar band,
        so every 'dip' keeps dipping. Reversion must lose here."""
        for seed in (1, 2):
            c = synth.cyclical(500, period=80, amplitude=0.1, seed=seed)
            r = run_backtest(BollingerReversion(), c)
            assert r.report.total_return < -0.2 and r.report.win_rate < 0.3

    def test_loses_badly_in_persistent_downtrend_without_filter(self):
        c = synth.trending(300, -0.004, 0.006, seed=2)
        r = run_backtest(BollingerReversion(), c)
        assert r.report.total_return < -0.3 and r.report.win_rate == 0.0

    def test_trend_filter_avoids_the_downtrend_losses(self):
        c = synth.trending(300, -0.004, 0.006, seed=2)
        plain = run_backtest(BollingerReversion(), c).report
        filtered = run_backtest(BollingerReversion(trend_period=100), c).report
        assert filtered.total_return > plain.total_return + 0.3
        assert filtered.avg_exposure < 0.1

    def test_loses_on_momentum_series(self):
        for seed in (1, 2, 3):
            c = synth.ar1_momentum(500, phi=0.5, sigma=0.01, mu=0.0003, seed=seed)
            assert run_backtest(BollingerReversion(), c).report.total_return < 0.0
