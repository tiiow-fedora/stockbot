"""Behavioural tests for RelativeStrength (dual momentum vs a benchmark partner)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.models import Action
from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.relative_strength import RelativeStrength, RelativeStrengthParams
from research.tests.helpers import decide, decisions


def pair_from_returns(ret_a, ret_b):
    """Two candle series on the same timestamps from per-bar log returns (first return ignored)."""
    a = synth.candles_from_closes(100 * np.exp(np.cumsum(ret_a)), "A", 1)
    b = synth.candles_from_closes(100 * np.exp(np.cumsum(ret_b)), "B", 2)
    return a, b


def lead_case(a_total, b_total, n=80, period=40):
    """Flat until ``period`` bars before the end, then A and B drift to the given total log returns."""
    ra, rb = np.zeros(n), np.zeros(n)
    ra[-period:], rb[-period:] = a_total / period, b_total / period
    return pair_from_returns(ra, rb)


def verdict(a_total, b_total, in_position=False, **kw):
    a, b = lead_case(a_total, b_total)
    kw.setdefault("rs_period", 40)
    return RelativeStrength(partner=b, **kw).signal(Bars.from_candles("A", a), in_position)


class TestRules:
    def test_enters_when_outperforming_and_positive(self):
        sig = verdict(0.15, 0.03)
        assert sig.kind.value == "ENTER" and sig.indicators["rs"] > 0.02

    def test_no_entry_when_peer_leads(self):
        assert verdict(0.05, 0.15).kind.value == "HOLD"

    def test_absolute_filter_blocks_least_bad(self):
        """A beats the benchmark but is itself falling: dual momentum stays out."""
        sig = verdict(-0.03, -0.15)
        assert sig.indicators["rs"] > 0.02 and sig.kind.value == "HOLD"

    def test_margin_boundary(self):
        assert verdict(0.10, 0.07, margin=0.02).kind.value == "ENTER"
        assert verdict(0.10, 0.07, margin=0.05).kind.value == "HOLD"

    def test_exit_when_peer_leads_beyond_hysteresis(self):
        assert verdict(0.05, 0.15, in_position=True, hysteresis=0.02).kind.value == "EXIT"

    def test_hysteresis_keeps_position_when_rs_slightly_negative(self):
        # rs ~ -0.01 (A +0.09 vs B +0.10); inside a 0.03 band => hold; outside a 0.0 band => exit
        assert verdict(0.09, 0.10, in_position=True, hysteresis=0.03).kind.value == "HOLD"
        assert verdict(0.09, 0.10, in_position=True, hysteresis=0.0).kind.value == "EXIT"

    def test_exit_when_own_return_clearly_negative(self):
        sig = verdict(-0.10, -0.12, in_position=True, hysteresis=0.03)
        assert sig.indicators["rs"] > 0 and sig.kind.value == "EXIT" and "absolute" in sig.reason

    def test_holds_while_still_leading(self):
        assert verdict(0.15, 0.03, in_position=True).kind.value == "HOLD"

    def test_strength_grows_with_lead(self):
        small, big = verdict(0.08, 0.03, margin=0.02), verdict(0.20, 0.03, margin=0.02)
        assert small.kind.value == big.kind.value == "ENTER"
        assert 0.5 <= small.strength < big.strength <= 1.0

    def test_period_is_measured_in_bars(self):
        """rs_period=20 sees only the second half of a move that rs_period=40 sees entirely."""
        a, b = lead_case(0.2, 0.0, period=40)
        bars = Bars.from_candles("A", a)
        s20 = RelativeStrength(partner=b, rs_period=20).signal(bars, False)
        s40 = RelativeStrength(partner=b, rs_period=40).signal(bars, False)
        assert s40.indicators["r_self"] == pytest.approx(0.2214, abs=0.01)
        assert s20.indicators["r_self"] == pytest.approx(np.expm1(0.1), abs=0.01)


class TestParams:
    @pytest.mark.parametrize("bad", [dict(rs_period=0), dict(rs_period=-5), dict(rs_period=1), dict(margin=-0.01), dict(hysteresis=-0.01)])
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValueError):
            RelativeStrengthParams(**bad)

    def test_grid_size(self):
        assert 6 <= len(RelativeStrength.param_combinations()) <= 60

    def test_partner_required_and_nonempty(self):
        with pytest.raises(ValueError):
            RelativeStrength(partner=[])
        with pytest.raises(ValueError):
            RelativeStrength()

    def test_min_bars_follows_period(self):
        _, b = synth.cointegrated_pair(50, seed=1)
        assert RelativeStrength(partner=b, rs_period=100).min_bars > RelativeStrength(partner=b, rs_period=20).min_bars


class TestPartnerHandling:
    def test_no_lookahead_on_partner_mutation(self):
        a, b = synth.cointegrated_pair(300, seed=21)
        cut = 200
        b2 = b[: cut + 1] + [replace(c, close=c.close * 0.2) for c in b[cut + 1:]]
        s1, s2 = RelativeStrength(partner=b), RelativeStrength(partner=b2)
        for i in range(s1.min_bars - 1, cut + 1, 3):
            for pos in (False, True):
                assert decide(s1, a, i, pos).action == decide(s2, a, i, pos).action, i

    def test_later_partner_data_does_matter_later(self):
        """Non-vacuity of the look-ahead test: after the cut the partner change flips signals."""
        a, b = synth.cointegrated_pair(300, seed=21)
        b2 = b[:200] + [replace(c, close=c.close * 0.2) for c in b[200:]]
        s1, s2 = RelativeStrength(partner=b), RelativeStrength(partner=b2)
        assert decide(s1, a, 205).indicators["rs"] != decide(s2, a, 205).indicators["rs"]

    def test_isolated_missing_partner_bars_are_forward_filled(self):
        a, b = synth.cointegrated_pair(300, seed=21)
        sparse = [c for i, c in enumerate(b) if i % 7 != 3]
        d = decide(RelativeStrength(partner=sparse), a, 150)
        assert d.indicators["rs"] is not None

    def test_long_gap_in_partner_holds(self):
        a, b = synth.cointegrated_pair(300, seed=21)
        gappy = b[:140] + b[160:]
        d = decide(RelativeStrength(partner=gappy), a, 150)
        assert d.action == Action.HOLD and d.indicators.get("rs") is None

    def test_disjoint_timestamps_never_trade(self):
        a, b = synth.cointegrated_pair(200, seed=21)
        far = synth.candles_from_closes([c.close for c in b], "B", 1, start=a[0].timestamp.replace(year=a[0].timestamp.year + 5))
        assert all(d.action == Action.HOLD for d in decisions(RelativeStrength(partner=far), a))


def benchmark_and_leader(n=500, seed=0, lead_drift=0.0015):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.008, n)
    leader = common + lead_drift + rng.normal(0, 0.004, n)
    bench = common + rng.normal(0, 0.003, n)
    return pair_from_returns(leader, bench)


class TestSyntheticEdge:
    def test_profits_when_instrument_persistently_leads(self):
        rets = []
        for seed in range(4):
            a, b = benchmark_and_leader(seed=seed)
            r = run_backtest(RelativeStrength(partner=b), a)
            rets.append(r.report.total_return)
        assert all(x > 0 for x in rets) and np.mean(rets) > 0.1

    def test_stays_out_when_instrument_persistently_lags(self):
        a, b = benchmark_and_leader(seed=1, lead_drift=-0.002)
        r = run_backtest(RelativeStrength(partner=b), a)
        assert r.report.avg_exposure < 0.2

    def test_avoids_a_crash_via_absolute_filter(self):
        """Both fall together: relative lead is noise but absolute momentum keeps it out of the crash."""
        n = 400
        rng = np.random.default_rng(3)
        common = np.concatenate([rng.normal(0.001, 0.006, 150), rng.normal(-0.006, 0.01, 250)])
        a, b = pair_from_returns(common + rng.normal(0, 0.002, n), common + rng.normal(0, 0.002, n))
        r = run_backtest(RelativeStrength(partner=b), a)
        assert r.report.max_drawdown < r.benchmark_report.max_drawdown
        assert r.report.total_return > r.benchmark_report.total_return

    def test_no_edge_when_relative_performance_mean_reverts(self):
        """Honest negative: if leadership reverses (cointegrated pair) chasing the leader is not profitable."""
        total = []
        for seed in range(4):
            a, b = synth.cointegrated_pair(500, half_life=10.0, seed=seed)
            total.append(run_backtest(RelativeStrength(partner=b, rs_period=20, margin=0.0, hysteresis=0.0), a).report.total_return)
        assert np.mean(total) < 0.05


class TestRobustness:
    def test_partner_list_is_copied_at_construction(self):
        a, b = synth.cointegrated_pair(200, seed=3)
        bl = list(b)
        s = RelativeStrength(partner=bl)
        before = decide(s, a, 150).indicators["rs"]
        bl.clear()
        assert decide(s, a, 150).indicators["rs"] == before

    def test_zero_price_at_lookback_bar_holds(self):
        a, b = synth.cointegrated_pair(200, seed=3)
        bars = Bars.from_candles("A", a)
        bars.close[-61] = 0.0
        sig = RelativeStrength(partner=b).signal(bars, False)
        assert sig.kind.value == "HOLD"

    @pytest.mark.parametrize("bad", [dict(margin=float("nan")), dict(hysteresis=float("nan")), dict(margin=float("inf")), dict(hysteresis=float("inf"))])
    def test_non_finite_thresholds_rejected(self, bad):
        with pytest.raises(ValueError):
            RelativeStrengthParams(**bad)
