"""Behavioural tests for MacdTrend: rules on hand-built bars, parameter handling, falsifiable claims on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from research import indicators as ind
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.macd_trend import MacdTrend, MacdTrendParams
from research.tests.helpers import simulate_long_flat


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


def signals(strategy, closes, in_position=False):
    """The strategy's signal at every bar from min_bars on (prefix i+1 of ``closes``)."""
    return [(i, strategy.signal(bars_from_closes(closes[: i + 1]), in_position)) for i in range(strategy.min_bars - 1, len(closes))]


def entries(strategy, closes):
    return [i for i, g in signals(strategy, closes) if g.kind.value == "ENTER"]


# up-trend, shallow pullback, resumption: the cross-up happens ABOVE the zero line
PULLBACK = np.concatenate([np.linspace(100, 140, 60), np.linspace(140, 128, 12), np.linspace(128, 150, 30)])
# long decline then a bounce: the cross-up happens BELOW the zero line
BOUNCE = np.concatenate([np.linspace(150, 100, 70), np.linspace(100, 112, 12)])


class TestRules:
    def test_enters_on_bullish_cross_above_zero_line(self):
        s = MacdTrend(slope_confirm=False)
        got = entries(s, PULLBACK)
        assert got and got[0] == 80  # cross-up in the resumption, not during the initial smooth ramp
        sig = s.signal(bars_from_closes(PULLBACK[:81]), False)
        assert sig.indicators["macd"] > 0 and sig.indicators["hist"] > 0

    def test_smooth_ramp_has_no_cross(self):
        """Float noise in the histogram of a perfectly linear ramp must not be read as crosses."""
        assert entries(MacdTrend(slope_confirm=False), np.linspace(100, 140, 120)) == []

    def test_zero_filter_blocks_cross_below_zero_line(self):
        assert entries(MacdTrend(zero_filter=True, slope_confirm=False), BOUNCE) == []
        got = entries(MacdTrend(zero_filter=False, slope_confirm=False), BOUNCE)
        assert got
        last = MacdTrend().signal(bars_from_closes(BOUNCE[: got[0] + 1]), False)
        assert last.indicators["macd"] < 0 < last.indicators["hist"]

    def test_cross_window_keeps_signal_alive_then_expires(self):
        """Cross at bar 80; with window=3 bars 80..82 qualify, bar 83 does not although hist is still > 0."""
        s = MacdTrend(cross_window=3, slope_confirm=False)
        assert entries(s, PULLBACK) == [80, 81, 82]
        assert signals(s, PULLBACK)[83 - (s.min_bars - 1)][1].indicators["hist"] > 0
        assert entries(MacdTrend(cross_window=1, slope_confirm=False), PULLBACK)[:2] == [80]
        assert 81 not in entries(MacdTrend(cross_window=1, slope_confirm=False), PULLBACK)

    def test_exits_only_on_bearish_cross_and_only_when_long(self):
        s = MacdTrend()
        drop = np.concatenate([np.linspace(100, 140, 70), np.linspace(140, 110, 20)])
        sigs = signals(s, drop, in_position=True)
        _, hist = zip(*[(i, g.indicators["hist"]) for i, g in sigs])
        for (i, g), h in zip(sigs, hist):
            assert (g.kind.value == "EXIT") == (h < -1e-6)
        assert sigs[0][1].kind.value == "HOLD" and sigs[-1][1].kind.value == "EXIT"
        assert all(g.kind.value != "EXIT" for _, g in signals(s, drop, in_position=False))

    def test_slope_confirm_is_a_pure_restriction(self):
        c = synth.regime_switching(600, seed=2)
        closes = np.array([k.close for k in c])
        plain = set(entries(MacdTrend(slope_confirm=False, cross_window=4), closes))
        confirmed = set(entries(MacdTrend(slope_confirm=True, cross_window=4), closes))
        assert confirmed < plain
        for i in confirmed:
            h = ind.macd(closes[: i + 1])[2]
            assert h[-1] > h[-2]

    def test_zero_filter_is_a_pure_restriction(self):
        closes = np.array([k.close for k in synth.regime_switching(600, seed=2)])
        on = set(entries(MacdTrend(zero_filter=True), closes))
        off = set(entries(MacdTrend(zero_filter=False), closes))
        assert on < off

    def test_strength_in_range_and_increases_with_histogram(self):
        s = MacdTrend(slope_confirm=False)
        by_hist = sorted((g.indicators["hist"], g.strength) for i, g in signals(s, PULLBACK) if g.kind.value == "ENTER")
        assert len(by_hist) >= 3 and all(0.5 <= st <= 1.0 for _, st in by_hist)
        assert by_hist[0][1] < by_hist[-1][1]
        assert [st for _, st in by_hist] == sorted(st for _, st in by_hist)

    def test_flat_series_never_signals(self):
        s = MacdTrend()
        assert entries(s, [100.0] * 80) == []
        assert all(g.kind.value == "HOLD" for _, g in signals(s, [100.0] * 80, in_position=True))

    def test_not_ready_before_enough_history(self):
        s = MacdTrend()
        sig = s.signal(bars_from_closes(PULLBACK[: s.min_bars - 2]), False)
        assert sig.kind.value == "HOLD"


class TestParams:
    @pytest.mark.parametrize("bad", [dict(fast=0), dict(slow=-1), dict(signal=0), dict(cross_window=0), dict(atr_period=0),
                                     dict(fast=26, slow=26), dict(fast=30, slow=26)])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            MacdTrendParams(**bad)

    def test_grid_size_and_validity(self):
        combos = MacdTrend.param_combinations()
        assert 6 <= len(combos) <= 60
        assert all(c["fast"] < c["slow"] for c in combos)

    def test_grid_drops_fast_ge_slow(self):
        combos = MacdTrend.param_combinations({"fast": [12, 30], "slow": [26]})
        assert combos == [{"fast": 12, "slow": 26}]

    def test_min_bars_covers_macd_warmup(self):
        assert MacdTrend().min_bars >= 26 + 9


class TestSyntheticEdge:
    def test_profits_on_clean_uptrend(self):
        r = run_backtest(MacdTrend(), synth.trending(300, 0.004, 0.006, seed=1))
        assert r.report.total_return > 0.2

    def test_stays_out_of_persistent_downtrend(self):
        """Zero-line filter: a downtrend never lets MACD cross zero, so no entries."""
        r = run_backtest(MacdTrend(), synth.trending(300, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.05 and r.report.max_drawdown < 0.05

    def test_wins_on_range_then_breakout_cycles(self):
        r = run_backtest(MacdTrend(), synth.range_then_breakout(600, seed=4))
        assert r.report.total_return > 0.0

    def test_whipsaws_in_mean_reverting_market(self):
        """Honest negative: it must not be a reliable winner on an OU series (low win rate, no real profit)."""
        rets, wins = [], []
        for seed in (3, 9, 15):
            rep = run_backtest(MacdTrend(), synth.mean_reverting(500, 8.0, 0.012, seed=seed)).report
            rets.append(rep.total_return)
            wins.append(rep.win_rate)
        assert np.mean(rets) < 0.05
        assert np.nanmean(wins) < 0.6

    def test_exit_lags_the_peak(self):
        up, down = 150, 150
        drift = np.concatenate([np.full(up, 0.004), np.full(down, -0.004)])
        closes = 100 * np.exp(np.cumsum(drift + np.random.default_rng(1).normal(0, 0.006, up + down)))
        trades, still_open = simulate_long_flat(MacdTrend(), synth.candles_from_closes(closes, seed=1))
        assert trades and not still_open
        assert trades[-1][1] > up  # the position carried through the top is only sold after it


class TestReviewFixes:
    def test_strength_formula(self):
        s = MacdTrend(slope_confirm=False)
        checked = 0
        for _, g in signals(s, PULLBACK):
            if g.kind.value == "ENTER":
                x = g.indicators["hist"] / (0.1 * g.indicators["atr"])
                assert g.strength == pytest.approx(0.5 + 0.5 * np.tanh(x))
                checked += 1
        assert checked >= 3
        assert any(0.55 < g.strength < 0.95 for _, g in signals(s, PULLBACK) if g.kind.value == "ENTER")
