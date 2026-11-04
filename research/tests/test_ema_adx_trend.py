"""Behavioural tests for EmaAdxTrend: rules on hand-built bars, parameter handling, falsifiable claims on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from core.models import Action
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.ema_adx_trend import EmaAdxTrend, EmaAdxTrendParams
from research.tests.helpers import actions, simulate_long_flat


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


RAMP = np.linspace(100, 200, 120)  # clean uptrend: fast > slow, ADX very high
NOISY_UP = RAMP * np.exp(np.random.default_rng(0).normal(0, 0.02, 120))  # fast > slow but ADX ~21
# strong trend, then a dead-flat range: EMAs stay ordered (fast > slow) while ADX bleeds away
STALL = np.concatenate([np.linspace(100, 200, 100), 200 + 0.3 * np.where(np.arange(80) % 2 == 0, 1.0, -1.0)])


def kind(strategy, closes, in_position):
    return strategy.signal(bars_from_closes(closes), in_position).kind.value


class TestRules:
    def test_enters_on_strong_uptrend(self):
        sig = EmaAdxTrend().signal(bars_from_closes(RAMP), False)
        assert sig.kind.value == "ENTER"
        assert sig.indicators["ema_fast"] > sig.indicators["ema_slow"] and sig.indicators["adx"] > 25

    def test_adx_gate_blocks_weak_trend_even_with_fast_above_slow(self):
        s = EmaAdxTrend()
        sig = s.signal(bars_from_closes(NOISY_UP), False)
        assert sig.indicators["ema_fast"] > sig.indicators["ema_slow"]
        assert sig.indicators["adx"] < 25 and sig.kind.value == "HOLD"

    def test_adx_entry_threshold_boundary(self):
        adx_v = EmaAdxTrend().signal(bars_from_closes(NOISY_UP), False).indicators["adx"]
        assert kind(EmaAdxTrend(adx_enter=adx_v - 0.5, adx_exit=10.0), NOISY_UP, False) == "ENTER"
        assert kind(EmaAdxTrend(adx_enter=adx_v + 0.5, adx_exit=10.0), NOISY_UP, False) == "HOLD"

    def test_no_entry_in_downtrend(self):
        assert kind(EmaAdxTrend(), RAMP[::-1], False) == "HOLD"

    def test_exits_on_ema_cross_down_even_with_high_adx(self):
        sig = EmaAdxTrend().signal(bars_from_closes(RAMP[::-1]), True)
        assert sig.kind.value == "EXIT" and sig.indicators["adx"] > 25 and "EMA" in sig.reason

    def test_exits_when_adx_collapses_before_the_ema_cross(self):
        s = EmaAdxTrend()
        sig = s.signal(bars_from_closes(STALL), True)
        assert sig.kind.value == "EXIT" and "ADX" in sig.reason
        assert sig.indicators["ema_fast"] > sig.indicators["ema_slow"]  # EMAs have NOT crossed yet

    def test_adx_exit_threshold_boundary(self):
        adx_v = EmaAdxTrend().signal(bars_from_closes(STALL), True).indicators["adx"]
        assert kind(EmaAdxTrend(adx_enter=60.0, adx_exit=adx_v + 0.5), STALL, True) == "EXIT"
        assert kind(EmaAdxTrend(adx_enter=60.0, adx_exit=adx_v - 0.5), STALL, True) == "HOLD"

    def test_hysteresis_band(self):
        """ADX between exit and enter thresholds: an open trade is kept, a flat book stays flat."""
        adx_v = EmaAdxTrend().signal(bars_from_closes(NOISY_UP), False).indicators["adx"]
        s = EmaAdxTrend(adx_enter=adx_v + 5, adx_exit=adx_v - 5)
        assert kind(s, NOISY_UP, False) == "HOLD"
        assert kind(s, NOISY_UP, True) == "HOLD"

    def test_exit_ignored_when_flat(self):
        assert kind(EmaAdxTrend(), STALL, False) == "HOLD"

    def test_strength_grows_with_adx(self):
        s = EmaAdxTrend(adx_enter=15.0, adx_exit=10.0)
        weak, strong = s.signal(bars_from_closes(NOISY_UP), False), s.signal(bars_from_closes(RAMP), False)
        assert weak.kind.value == strong.kind.value == "ENTER"
        assert 0.0 < weak.strength < strong.strength <= 1.0
        assert strong.indicators["adx"] > weak.indicators["adx"]

    def test_flat_prices_never_enter_and_close_an_open_trade(self):
        """No trend at all (ADX = 0): don't buy; if somehow long, the ADX-collapse rule gets out."""
        assert kind(EmaAdxTrend(), [100.0] * 120, False) == "HOLD"
        assert kind(EmaAdxTrend(), [100.0] * 120, True) == "EXIT"

    def test_not_ready_before_enough_history(self):
        s = EmaAdxTrend()
        assert kind(s, RAMP[: s.min_bars - 3], False) == "HOLD"
        assert kind(s, RAMP[: s.min_bars - 3], True) == "HOLD"


class TestParams:
    @pytest.mark.parametrize("bad", [dict(fast=0), dict(slow=-1), dict(adx_period=0), dict(fast=50, slow=50),
                                     dict(adx_enter=20.0, adx_exit=20.0), dict(adx_enter=20.0, adx_exit=25.0),
                                     dict(adx_exit=-1.0), dict(adx_enter=101.0)])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            EmaAdxTrendParams(**bad)

    def test_grid_size_and_validity(self):
        combos = EmaAdxTrend.param_combinations()
        assert 6 <= len(combos) <= 60
        assert all(c["adx_exit"] < c["adx_enter"] and c["fast"] < c["slow"] for c in combos)

    def test_grid_drops_invalid_combinations(self):
        combos = EmaAdxTrend.param_combinations({"adx_enter": [20.0], "adx_exit": [15.0, 25.0]})
        assert combos == [{"adx_enter": 20.0, "adx_exit": 15.0}]

    def test_min_bars_covers_slow_ema_and_adx(self):
        assert EmaAdxTrend(fast=10, slow=100).min_bars >= 100
        assert EmaAdxTrend(fast=10, slow=20, adx_period=30).min_bars >= 61


class TestSyntheticEdge:
    def test_profits_on_clean_uptrend(self):
        r = run_backtest(EmaAdxTrend(), synth.trending(300, 0.004, 0.006, seed=1))
        assert r.report.total_return > 0.3 and r.report.avg_exposure > 0.5

    def test_stays_out_of_persistent_downtrend(self):
        r = run_backtest(EmaAdxTrend(), synth.trending(300, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.05 and r.report.max_drawdown < 0.05

    def test_adx_gate_reduces_entries_in_trendless_markets(self):
        """A gated entry is a strict subset of the plain EMA-state entry; on an OU series it removes many."""
        c = synth.mean_reverting(600, 8.0, 0.012, seed=3)
        gated = sum(a == Action.BUY for a in actions(EmaAdxTrend(), c))
        plain = sum(a == Action.BUY for a in actions(EmaAdxTrend(adx_enter=1.0, adx_exit=0.0), c))
        assert gated < 0.7 * plain

    def test_ungated_ema_trades_more_than_gated(self):
        c = synth.regime_switching(700, seed=5)
        gated, _ = simulate_long_flat(EmaAdxTrend(), c)
        plain, _ = simulate_long_flat(EmaAdxTrend(adx_enter=1.0, adx_exit=0.0), c)
        assert len(gated) <= len(plain)

    def test_wins_on_range_then_breakout_cycles(self):
        r = run_backtest(EmaAdxTrend(fast=10, slow=50), synth.range_then_breakout(600, seed=4))
        assert r.report.total_return > 0.0

    @pytest.mark.parametrize("seed", [9, 19, 29])
    def test_loses_on_cyclical_market(self, seed):
        """Honest negative: with a 40-bar cycle the 50-bar EMA lags about a quarter cycle, so the
        system buys near tops and ADX confirms only after the move - it must not profit."""
        r = run_backtest(EmaAdxTrend(), synth.cyclical(600, period=40, seed=seed))
        assert r.report.total_return < 0.05

    def test_no_edge_in_mean_reverting_market(self):
        rets = [run_backtest(EmaAdxTrend(), synth.mean_reverting(500, 8.0, 0.012, seed=s)).report.total_return
                for s in (3, 13, 23)]
        assert np.mean(rets) < 0.05


class TestReviewFixes:
    def test_entry_requires_adx_strictly_above_threshold(self):
        adx_v = EmaAdxTrend().signal(bars_from_closes(NOISY_UP), False).indicators["adx"]
        assert kind(EmaAdxTrend(adx_enter=adx_v, adx_exit=10.0), NOISY_UP, False) == "HOLD"

    def test_exit_requires_adx_strictly_below_threshold(self):
        adx_v = EmaAdxTrend().signal(bars_from_closes(STALL), True).indicators["adx"]
        assert kind(EmaAdxTrend(adx_enter=60.0, adx_exit=adx_v), STALL, True) == "HOLD"

    def test_equal_emas_do_not_exit(self):
        """fast == slow (flat tape) is not a cross-down; with the ADX exit disabled the trade is held."""
        assert kind(EmaAdxTrend(adx_exit=0.0), [100.0] * 120, True) == "HOLD"

    def test_strength_is_adx_over_fifty(self):
        sig = EmaAdxTrend(adx_enter=15.0, adx_exit=10.0).signal(bars_from_closes(NOISY_UP), False)
        assert 0 < sig.indicators["adx"] < 50
        assert sig.strength == pytest.approx(sig.indicators["adx"] / 50.0)
