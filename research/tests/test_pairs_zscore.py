"""Behavioural tests for PairsZScore (long leg only): rules, partner handling, no look-ahead, synthetic edge."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.models import Action
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.pairs_zscore import PairsZScore, PairsZScoreParams
from research.tests.helpers import decide, decisions, simulate_long_flat


def make_pair(a_closes, b_closes):
    return synth.candles_from_closes(a_closes, "A", 1), synth.candles_from_closes(b_closes, "B", 2)


def spread_case(last_factor, n=120, seed=0):
    """B random-walks, A = B * stationary noise; the last A close is scaled by ``last_factor``."""
    rng = np.random.default_rng(seed)
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    a = 50 * (b / 100) * np.exp(rng.normal(0, 0.01, n))
    a[-1] = 50 * (b[-1] / 100) * last_factor  # noise-free last bar: z is roughly ln(factor) / 0.01
    return make_pair(a, b)


def verdict(last_factor, in_position=False, **kw):
    a, b = spread_case(last_factor)
    s = PairsZScore(partner=b, **kw)
    bars = Bars.from_candles("A", a)
    return s.signal(bars, in_position)


class TestRules:
    def test_enters_when_cheap_vs_partner(self):
        sig = verdict(0.975)
        assert sig.kind.value == "ENTER" and sig.indicators["z"] < -2.0

    def test_does_not_enter_when_rich_or_neutral(self):
        assert verdict(1.0).kind.value == "HOLD"
        assert verdict(1.05).kind.value == "HOLD"

    def test_entry_threshold_boundary(self):
        assert verdict(0.975, entry_z=1.5).kind.value == "ENTER"
        assert verdict(0.975, entry_z=3.0, stop_z=6.0).kind.value == "HOLD"

    def test_exit_when_spread_reverted(self):
        sig = verdict(1.0, in_position=True)
        assert sig.kind.value == "EXIT" and sig.indicators["z"] > -0.5

    def test_holds_while_still_cheap(self):
        sig = verdict(0.975, in_position=True)
        assert sig.kind.value == "HOLD"

    def test_stop_when_relationship_breaks(self):
        sig = verdict(0.90, in_position=True)  # z far below -stop_z
        assert sig.kind.value == "EXIT" and "broke" in sig.reason

    def test_does_not_enter_beyond_stop_level(self):
        """Deeper than stop_z: entering would be stopped out immediately -> stay out."""
        assert verdict(0.90).kind.value == "HOLD"

    def test_exit_z_zero_exits_only_above_mean(self):
        assert verdict(0.99, in_position=True, exit_z=0.0).kind.value == "HOLD"
        assert verdict(1.02, in_position=True, exit_z=0.0).kind.value == "EXIT"

    def test_strength_grows_with_depth(self):
        mild, deep = verdict(0.975, stop_z=8.0), verdict(0.94, stop_z=8.0)
        assert mild.kind.value == deep.kind.value == "ENTER"
        assert 0.5 <= mild.strength < deep.strength <= 1.0

    def test_rolling_beta_recovers_hedge_ratio(self):
        a, b = synth.cointegrated_pair(300, beta=1.5, seed=3)
        sig = PairsZScore(partner=b, rolling_beta=True, beta_window=100).signal(Bars.from_candles("A", a), False)
        assert sig.indicators["beta"] == pytest.approx(1.5, abs=0.35)

    def test_correlation_gate_blocks_uncorrelated_partner(self):
        a, _ = spread_case(0.975)
        rng = np.random.default_rng(11)
        indep = synth.candles_from_closes(80 * np.exp(np.cumsum(rng.normal(0, 0.01, len(a)))), "B", 2)
        gated = PairsZScore(partner=indep, cointegration_check=True, min_corr=0.5, entry_z=1.0, stop_z=9.0)
        open_ = PairsZScore(partner=indep, cointegration_check=False, entry_z=1.0, stop_z=9.0)
        bars = Bars.from_candles("A", a)
        assert open_.signal(bars, False).kind.value == "ENTER"
        sig = gated.signal(bars, False)
        assert sig.kind.value == "HOLD" and "correlation" in sig.reason

    def test_correlation_gate_passes_correlated_partner(self):
        a, b = spread_case(0.975)
        sig = PairsZScore(partner=b, cointegration_check=True, min_corr=0.3).signal(Bars.from_candles("A", a), False)
        assert sig.kind.value == "ENTER" and sig.indicators["corr"] > 0.3


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(z_window=0), dict(beta_window=-1), dict(corr_window=2), dict(entry_z=0.3, exit_z=0.5),
        dict(stop_z=1.5, entry_z=2.0), dict(exit_z=-0.1), dict(fixed_beta=0.0), dict(min_corr=1.5),
    ])
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValueError):
            PairsZScoreParams(**bad)

    def test_grid_valid_and_sized(self):
        combos = PairsZScore.param_combinations()
        assert 6 <= len(combos) <= 60

    def test_partner_required_and_nonempty(self):
        with pytest.raises(ValueError):
            PairsZScore(partner=[])
        with pytest.raises(ValueError):
            PairsZScore()

    def test_override_kwargs_and_partner_together(self):
        _, b = synth.cointegrated_pair(80, seed=1)
        s = PairsZScore(partner=b, entry_z=2.5)
        assert s.params.entry_z == 2.5


class TestPartnerHandling:
    def test_no_lookahead_on_partner_mutation(self):
        """Corrupting partner bars AFTER the decision bar never changes the decision at/before it."""
        a, b = synth.cointegrated_pair(300, seed=21)
        cut = 200
        garbage = [replace(c, close=c.close * 4.0) for c in b[cut + 1:]]
        b2 = b[: cut + 1] + garbage
        s1, s2 = PairsZScore(partner=b), PairsZScore(partner=b2)
        for i in range(s1.min_bars - 1, cut + 1, 3):
            for pos in (False, True):
                assert decide(s1, a, i, pos).action == decide(s2, a, i, pos).action, i

    def test_mutation_after_cut_would_change_later_decisions(self):
        """Sanity that the previous test is not vacuous: later decisions DO react to partner data."""
        a, b = synth.cointegrated_pair(300, seed=21)
        b2 = b[:200] + [replace(c, close=c.close * 4.0) for c in b[200:]]
        s1, s2 = PairsZScore(partner=b), PairsZScore(partner=b2)
        diffs = [decide(s1, a, i).indicators["z"] != decide(s2, a, i).indicators["z"] for i in range(200, 205)]
        assert any(diffs)

    def test_missing_partner_timestamps_degrade_gracefully(self):
        a, b = synth.cointegrated_pair(300, seed=21)
        sparse = [c for i, c in enumerate(b) if i % 5 != 3]  # drop every 5th bar (isolated gaps)
        s = PairsZScore(partner=sparse)
        ds = decisions(s, a)
        assert all(d.action in (Action.BUY, Action.HOLD) for d in ds)
        assert any(d.indicators.get("z") is not None for d in ds), "isolated gaps are forward-filled"

    def test_long_partner_gap_holds(self):
        a, b = synth.cointegrated_pair(300, seed=21)
        gappy = b[:150] + b[160:]  # 10 consecutive missing bars at 150..159
        s = PairsZScore(partner=gappy)
        d = decide(s, a, 158)
        assert d.action == Action.HOLD and "partner" in d.reason

    def test_partner_with_disjoint_timestamps_never_trades(self):
        a, b = synth.cointegrated_pair(200, seed=21)
        shifted = synth.candles_from_closes([c.close for c in b], "B", 1, start=a[0].timestamp.replace(year=a[0].timestamp.year + 5))
        s = PairsZScore(partner=shifted)
        assert all(d.action == Action.HOLD for d in decisions(s, a))


def mean_over(fn, seeds):
    return float(np.mean([fn(sd) for sd in seeds]))


def relative_after_entries(strategy, a, b, hold=15):
    """Average (A return - B return) over ``hold`` bars after each ENTER decision."""
    out = []
    for i in range(strategy.min_bars - 1, len(a) - hold):
        if decide(strategy, a, i).action == Action.BUY:
            out.append((a[i + hold].close / a[i].close) - (b[i + hold].close / b[i].close))
    return out


class TestSyntheticEdge:
    def test_buys_reverting_cheapness_on_cointegrated_pair(self):
        """After an entry the long leg outperforms its partner (the gap closes)."""
        rel = []
        for seed in range(6):
            a, b = synth.cointegrated_pair(600, seed=seed)
            rel += relative_after_entries(PairsZScore(partner=b, exit_z=0.0), a, b)
        assert len(rel) >= 100 and np.mean(rel) > 0.02

    def test_no_relative_edge_on_independent_walks(self):
        """Honest negative: with unrelated random walks the signal carries no relative information."""
        rel = []
        for seed in range(6):
            a = synth.gbm(600, 0.0, 0.015, seed=seed)
            b = synth.gbm(600, 0.0, 0.015, seed=seed + 100)
            rel += relative_after_entries(PairsZScore(partner=b), a, b)
        assert len(rel) >= 20 and np.mean(rel) < 0.01

    def test_profitable_round_trips_on_fast_reverting_pair(self):
        wins = []
        for seed in range(6):
            a, b = synth.cointegrated_pair(600, half_life=5.0, spread_sigma=0.02, trend_sigma=0.004, seed=seed)
            trades, _ = simulate_long_flat(PairsZScore(partner=b, exit_z=0.0), a)
            wins += [t[2] for t in trades]
        assert len(wins) >= 15 and np.mean(wins) > 0.0 and np.mean(np.array(wins) > 0) > 0.55

    def test_runs_through_engine_on_pair(self):
        a, b = synth.cointegrated_pair(500, seed=4)
        r = run_backtest(PairsZScore(partner=b), a)
        assert np.all(np.isfinite(r.equity)) and r.report.num_trades >= 1

    def test_short_window_rolling_beta_removes_the_edge(self):
        """Documented limitation: a short rolling OLS beta absorbs the spread deviations it should trade."""
        fixed, rolling = [], []
        for seed in range(5):
            a, b = synth.cointegrated_pair(500, seed=seed)
            fixed += relative_after_entries(PairsZScore(partner=b, exit_z=0.0), a, b)
            rolling += relative_after_entries(PairsZScore(partner=b, exit_z=0.0, rolling_beta=True, beta_window=60), a, b)
        assert np.mean(fixed) > 0.03 and np.mean(rolling) < 0.5 * np.mean(fixed)
