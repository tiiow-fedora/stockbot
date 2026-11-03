"""Behavioural tests for Tsmom: rules on hand-built bars, parameter handling, falsifiable claims on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.tsmom import Tsmom, TsmomParams
from research.tests.helpers import simulate_long_flat

SMALL = dict(lookback=20, skip=5, vol_window=10, hurdle=0.02)


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


def path_with_momentum(mom, n=60, lookback=20, skip=5, tail=None):
    """Closes whose (lookback, skip) momentum is exactly ``mom``; a little alternating noise keeps vol > 0."""
    c = 100.0 + 0.1 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    end = n - 1 - skip
    c[end - lookback] = 100.0
    c[end] = 100.0 * (1.0 + mom)
    if tail is not None:
        c[end + 1:] = tail
    return c


class TestRules:
    def test_enters_above_hurdle_and_not_below(self):
        s = Tsmom(**SMALL)
        assert s.signal(bars_from_closes(path_with_momentum(0.025)), False).kind.value == "ENTER"
        assert s.signal(bars_from_closes(path_with_momentum(0.015)), False).kind.value == "HOLD"

    def test_hurdle_is_strict_boundary(self):
        c = path_with_momentum(0.05)
        assert Tsmom(**{**SMALL, "hurdle": 0.049}).signal(bars_from_closes(c), False).kind.value == "ENTER"
        assert Tsmom(**{**SMALL, "hurdle": 0.051}).signal(bars_from_closes(c), False).kind.value == "HOLD"

    def test_exits_when_momentum_turns_negative_only_if_long(self):
        s = Tsmom(**SMALL)
        c = bars_from_closes(path_with_momentum(-0.02))
        assert s.signal(c, True).kind.value == "EXIT"
        assert s.signal(c, False).kind.value == "HOLD"

    def test_hysteresis_band_holds_both_ways(self):
        """0 <= mom <= hurdle: a flat book does not buy it, an open position is not sold."""
        s = Tsmom(**SMALL)
        c = bars_from_closes(path_with_momentum(0.01))
        assert s.signal(c, False).kind.value == "HOLD"
        assert s.signal(c, True).kind.value == "HOLD"

    def test_zero_hurdle_enters_any_positive_momentum(self):
        s = Tsmom(**{**SMALL, "hurdle": 0.0})
        assert s.signal(bars_from_closes(path_with_momentum(0.002)), False).kind.value == "ENTER"
        assert s.signal(bars_from_closes(path_with_momentum(-0.002)), False).kind.value == "HOLD"

    def test_skip_ignores_the_most_recent_bars(self):
        """A last-week crash is invisible with skip=5 (12-1 logic) but decisive with skip=0."""
        c = bars_from_closes(path_with_momentum(0.05, tail=[60.0] * 5))
        assert Tsmom(**SMALL).signal(c, True).kind.value == "HOLD"
        assert Tsmom(**{**SMALL, "skip": 0}).signal(c, True).kind.value == "EXIT"

    def test_reads_the_exact_lookback_window(self):
        """Changing the bar just before the window must not matter; changing its first bar must."""
        base = path_with_momentum(0.05)
        end = 60 - 1 - 5
        outside = base.copy()
        outside[end - 20 - 1] = 1.0
        inside = base.copy()
        inside[end - 20] = 200.0
        s = Tsmom(**SMALL)
        assert s.signal(bars_from_closes(outside), False).indicators["mom"] == pytest.approx(0.05)
        assert s.signal(bars_from_closes(inside), False).kind.value == "HOLD"

    def test_strength_rewards_smooth_trends_over_noisy_ones(self):
        """Same trailing return, more volatility -> lower (vol-scaled) conviction."""
        n = 60

        def series(noise):
            c = np.linspace(100, 110, n) + noise * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
            return bars_from_closes(c)

        s = Tsmom(**SMALL)
        calm, wild = s.signal(series(0.05), False), s.signal(series(1.5), False)
        assert calm.kind.value == wild.kind.value == "ENTER"
        assert 0.0 < wild.strength < calm.strength <= 1.0

    def test_strength_grows_with_return(self):
        s = Tsmom(**SMALL)
        weak = s.signal(bars_from_closes(path_with_momentum(0.03)), False).strength
        strong = s.signal(bars_from_closes(path_with_momentum(0.20)), False).strength
        assert 0.0 < weak < strong <= 1.0

    def test_constant_prices_hold_instead_of_dividing_by_zero(self):
        sig = Tsmom(**SMALL).signal(bars_from_closes([100.0] * 60), False)
        assert sig.kind.value == "HOLD" and "not ready" in sig.reason

    def test_too_few_bars_is_not_ready(self):
        s = Tsmom(**SMALL)
        sig = s.signal(bars_from_closes(np.linspace(100, 110, s.min_bars - 1)), False)
        assert sig.kind.value == "HOLD"
        assert s.signal(bars_from_closes(np.linspace(100, 110, s.min_bars)), False).kind.value == "ENTER"


class TestParams:
    def test_defaults_are_twelve_minus_one_scaled_to_days(self):
        p = TsmomParams()
        assert (p.lookback, p.skip) == (120, 5)

    @pytest.mark.parametrize("bad", [dict(lookback=0), dict(lookback=-5), dict(vol_window=1), dict(vol_window=0),
                                     dict(skip=-1), dict(hurdle=-0.01), dict(lookback=10, skip=10)])
    def test_invalid_params_rejected(self, bad):
        with pytest.raises(ValueError):
            TsmomParams(**bad)

    def test_grid_size_and_validity(self):
        combos = Tsmom.param_combinations()
        assert 6 <= len(combos) <= 60
        assert all(c["skip"] < c["lookback"] for c in combos)

    def test_grid_drops_invalid_combinations(self):
        combos = Tsmom.param_combinations({"lookback": [10], "skip": [5, 10, 20]})
        assert combos == [{"lookback": 10, "skip": 5}]

    def test_min_bars_covers_lookback_plus_skip(self):
        assert Tsmom(lookback=120, skip=5).min_bars == 126
        assert Tsmom(lookback=60, skip=10).min_bars == 71

    def test_longer_lookback_means_fewer_trades(self):
        c = synth.regime_switching(700, seed=2)
        fast, _ = simulate_long_flat(Tsmom(lookback=40, skip=0, hurdle=0.0), c)
        slow, _ = simulate_long_flat(Tsmom(lookback=180, skip=5), c)
        assert len(slow) < len(fast)


class TestSyntheticEdge:
    def test_profits_on_clean_uptrend(self):
        r = run_backtest(Tsmom(), synth.trending(400, 0.004, 0.006, seed=1))
        assert r.report.total_return > 0.5 and r.report.avg_exposure > 0.6

    def test_stays_out_of_persistent_downtrend(self):
        r = run_backtest(Tsmom(), synth.trending(400, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.05 and r.report.max_drawdown < 0.05

    def test_exits_late_after_a_trend_reversal(self):
        """Honest lag: a 6-month return only turns negative well after the peak."""
        up, down = 200, 200
        r = np.concatenate([np.full(up, 0.004), np.full(down, -0.004)])
        closes = 100 * np.exp(np.cumsum(r))
        trades, still_open = simulate_long_flat(Tsmom(), synth.candles_from_closes(closes, seed=1))
        assert trades and not still_open
        exit_bar = trades[0][1]
        assert exit_bar > up + 20  # at least a month after the peak
        assert exit_bar < up + down  # but it does get out

    def test_beats_buy_and_hold_drawdown_in_bear_then_bull_v_shape(self):
        """In a V (crash then recovery) it sits out the crash and joins the recovery."""
        c = synth.v_shape(500, seed=10)
        r = run_backtest(Tsmom(lookback=60, skip=5), c)
        assert r.report.total_return > 0.3
        assert r.report.max_drawdown < r.benchmark_report.max_drawdown

    @pytest.mark.parametrize("seed", [3, 13, 23])
    def test_no_edge_in_mean_reverting_market(self, seed):
        """Honest negative: trailing return flips sign with no follow-through in an OU series."""
        r = run_backtest(Tsmom(), synth.mean_reverting(500, 8.0, 0.012, seed=seed))
        assert r.report.total_return < 0.05


class TestReviewFixes:
    """Defects found in review."""

    def test_flat_tape_does_not_trap_a_losing_position(self):
        """Crash followed by vol_window dead-flat bars: vol == 0, but the 20-bar return is -20%."""
        flat = bars_from_closes([100.0] * 40 + [80.0] * 20)
        s = Tsmom(**SMALL)
        sig = s.signal(flat, True)
        assert sig.indicators["vol"] == 0.0 and sig.indicators["mom"] < 0
        assert sig.kind.value == "EXIT"

    def test_flat_tape_still_never_enters(self):
        flat = bars_from_closes([80.0] * 40 + [100.0] * 20)  # positive momentum, zero vol -> no strength basis
        assert Tsmom(**SMALL).signal(flat, False).kind.value == "HOLD"

    def test_zero_momentum_is_not_an_exit(self):
        c = bars_from_closes(path_with_momentum(0.0))
        sig = Tsmom(**SMALL).signal(c, True)
        assert sig.indicators["mom"] == 0.0 and sig.kind.value == "HOLD"

    def test_entry_requires_strictly_exceeding_hurdle(self):
        c = bars_from_closes(path_with_momentum(0.05))
        mom = Tsmom(**SMALL).signal(c, False).indicators["mom"]
        assert Tsmom(**{**SMALL, "hurdle": mom}).signal(c, False).kind.value == "HOLD"

    def test_strength_is_vol_scaled_tstat(self):
        sig = Tsmom(**SMALL).signal(bars_from_closes(path_with_momentum(0.05)), False)
        i = sig.indicators
        t = i["mom"] / (i["vol"] * np.sqrt(SMALL["lookback"]))
        assert 0.05 < t < 2  # unsaturated, so the sqrt(lookback) scale is really exercised
        assert sig.strength == pytest.approx(np.tanh(t))

    @pytest.mark.parametrize("hurdle", [float("nan"), float("inf")])
    def test_rejects_nan_and_inf_hurdle(self, hurdle):
        with pytest.raises(ValueError):
            TsmomParams(hurdle=hurdle)
