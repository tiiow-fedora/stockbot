"""Risk-metric tests: closed-form values, invariants and degenerate inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research import metrics as m


def test_returns_and_total_return():
    eq = [100, 110, 99]
    np.testing.assert_allclose(m.to_returns(eq), [0.10, -0.10])
    assert m.total_return(eq) == pytest.approx(-0.01)
    assert m.to_returns([100]).size == 0 and math.isnan(m.total_return([100]))


def test_cagr_known_value():
    eq = np.full(253, 100.0); eq[-1] = 121.0   # 252 periods = 1 year
    assert m.cagr(eq) == pytest.approx(0.21)
    eq2 = np.full(505, 100.0); eq2[-1] = 121.0  # 2 years
    assert m.cagr(eq2) == pytest.approx(0.10)
    assert math.isnan(m.cagr([100.0])) and math.isnan(m.cagr([100.0, 0.0]))


class TestSharpe:
    def test_closed_form(self):
        r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
        expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
        assert m.sharpe_ratio(r) == pytest.approx(expected)

    def test_risk_free_reduces_sharpe(self):
        r = np.random.default_rng(0).normal(0.001, 0.01, 500)
        assert m.sharpe_ratio(r, risk_free=0.05) < m.sharpe_ratio(r)

    def test_scale_invariance_of_returns_not_of_level(self):
        r = np.random.default_rng(1).normal(0.001, 0.01, 300)
        assert m.sharpe_ratio(r * 3) == pytest.approx(m.sharpe_ratio(r))

    def test_zero_variance_is_nan_not_inf(self):
        assert math.isnan(m.sharpe_ratio(np.full(50, 0.001)))
        assert math.isnan(m.sharpe_ratio(np.zeros(50)))

    def test_too_short(self):
        assert math.isnan(m.sharpe_ratio([0.01]))

    def test_periods_per_year_scaling(self):
        r = np.random.default_rng(2).normal(0.001, 0.01, 300)
        assert m.sharpe_ratio(r, periods_per_year=12) == pytest.approx(m.sharpe_ratio(r) * math.sqrt(12 / 252))


class TestSortino:
    def test_closed_form(self):
        r = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        dd = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
        assert m.sortino_ratio(r) == pytest.approx(r.mean() / dd * math.sqrt(252))

    def test_no_downside_is_nan(self):
        assert math.isnan(m.sortino_ratio([0.01, 0.02, 0.03]))

    def test_sortino_exceeds_sharpe_when_upside_vol_dominates(self):
        r = np.array([0.05, 0.04, -0.001, 0.06, -0.002, 0.05, 0.03])
        assert m.sortino_ratio(r) > m.sharpe_ratio(r)


class TestDrawdown:
    def test_hand_values(self):
        eq = [100, 120, 90, 110, 80, 130]
        np.testing.assert_allclose(m.drawdown_series(eq), [0, 0, 0.25, 1 / 12, 1 / 3, 0])
        assert m.max_drawdown(eq) == pytest.approx(1 / 3)

    def test_monotonic_increase_has_no_drawdown(self):
        assert m.max_drawdown(np.arange(1.0, 50.0)) == 0.0
        assert m.max_drawdown_duration(np.arange(1.0, 50.0)) == 0

    def test_duration_counts_longest_underwater_run(self):
        eq = [100, 90, 95, 99, 101, 100, 99, 98, 97, 96, 105]
        assert m.max_drawdown_duration(eq) == 5   # bars 5..9 below the 101 peak; the first dip was 3 bars

    def test_empty(self):
        assert m.max_drawdown([]) == 0.0

    def test_bounded_in_unit_interval(self):
        eq = np.cumprod(1 + np.random.default_rng(3).normal(0, 0.03, 1000))
        assert 0 <= m.max_drawdown(eq) < 1

    def test_nan_equity_is_nan_not_zero_drawdown(self):
        assert math.isnan(m.max_drawdown([100, float("nan"), 90]))

    def test_ulcer_index(self):
        assert m.ulcer_index([100, 100, 100]) == 0.0
        assert m.ulcer_index([100, 90]) == pytest.approx(math.sqrt(0.5 * 0.1**2))


def test_calmar():
    eq = np.concatenate([[100.0], np.full(251, 100.0), [121.0]])
    eq[100] = 90.0
    assert m.calmar_ratio(eq) == pytest.approx(m.cagr(eq) / m.max_drawdown(eq))
    assert math.isnan(m.calmar_ratio([100, 101, 102]))   # no drawdown


class TestTailRisk:
    def test_var_cvar_ordering(self):
        r = np.random.default_rng(4).normal(0.0, 0.01, 5000)
        v, c = m.value_at_risk(r, 0.95), m.conditional_var(r, 0.95)
        assert 0.014 < v < 0.019 and c > v
        assert m.value_at_risk(r, 0.99) > v

    def test_never_negative_for_all_gains(self):
        assert m.value_at_risk([0.01, 0.02, 0.03]) == 0.0
        assert m.conditional_var([0.01, 0.02, 0.03]) == 0.0

    def test_empty(self):
        assert math.isnan(m.value_at_risk([])) and math.isnan(m.conditional_var([]))


class TestExposure:
    def test_values(self):
        eq = np.full(4, 100.0)
        e = m.exposure([0, 50, 100, 0], eq)
        assert e["time_in_market"] == 0.5 and e["avg_exposure"] == pytest.approx(0.375) and e["max_exposure"] == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            m.exposure([1, 2], [1])

    def test_empty(self):
        assert math.isnan(m.exposure([], [])["avg_exposure"])

    def test_zero_equity_safe(self):
        assert m.exposure([5.0], [0.0])["avg_exposure"] == 0.0


class TestMoreDefinitions:
    def test_sortino_uses_annual_risk_free_per_bar(self):
        r = np.array([0.01, -0.02, 0.03, -0.01])
        ex = r - 0.05 / 252
        dd = math.sqrt(np.mean(np.minimum(ex, 0) ** 2))
        assert m.sortino_ratio(r, risk_free=0.05) == pytest.approx(ex.mean() / dd * math.sqrt(252))

    def test_var_is_positive_loss_and_cvar_is_tail_mean(self):
        r = [-0.10, -0.05, 0.10, 0.20]
        assert m.value_at_risk(r, 0.75) == pytest.approx(0.0625)  # 25th pct (linear) = -0.0625
        assert m.conditional_var(r, 0.75) == pytest.approx(0.10)  # only -0.10 lies at/below it
        assert m.value_at_risk(r, 0.5) == 0.0  # median is a gain: floored, never a negative "loss"

    def test_breakeven_trades_are_not_wins(self):
        s = m.trade_stats([5.0, 0.0])
        assert s["win_rate"] == 0.5 and s["profit_factor"] == float("inf") and s["avg_loss"] == 0.0
        assert math.isnan(m.trade_stats([0.0, 0.0])["profit_factor"])


class TestTradeStats:
    def test_mixed(self):
        s = m.trade_stats([100, -50, 200, -50])
        assert s["num_trades"] == 4 and s["win_rate"] == 0.5
        assert s["profit_factor"] == pytest.approx(3.0)
        assert s["avg_trade"] == 50 and s["expectancy_ratio"] == pytest.approx(150 / 50)
        assert s["best_trade"] == 200 and s["worst_trade"] == -50

    def test_all_winners_is_inf_profit_factor(self):
        assert m.trade_stats([1, 2])["profit_factor"] == float("inf")

    def test_empty(self):
        s = m.trade_stats([])
        assert s["num_trades"] == 0 and math.isnan(s["win_rate"])

    def test_all_losers(self):
        s = m.trade_stats([-1, -2])
        assert s["profit_factor"] == 0.0 and s["win_rate"] == 0.0


class TestReport:
    def test_full_report_consistent(self):
        rng = np.random.default_rng(5)
        eq = 100_000 * np.cumprod(1 + rng.normal(0.0008, 0.01, 400))
        pv = eq * 0.6
        rep = m.compute_report(eq, pv, [10, -5, 20])
        assert rep.n_bars == 400 and rep.num_trades == 3
        assert rep.max_drawdown == pytest.approx(m.max_drawdown(eq))
        assert rep.avg_exposure == pytest.approx(0.6)
        assert rep.sharpe == pytest.approx(m.sharpe_ratio(m.to_returns(eq)))
        d = rep.as_dict()
        assert set(d) >= {"sharpe", "sortino", "max_drawdown", "avg_exposure"}
        assert "sharpe" in rep.summary()

    def test_flat_equity_report_is_safe(self):
        rep = m.compute_report(np.full(50, 100.0))
        assert rep.total_return == 0.0 and rep.max_drawdown == 0.0 and math.isnan(rep.sharpe)
        assert "n/a" in rep.summary()

    def test_report_without_optional_inputs(self):
        rep = m.compute_report([100, 101, 99, 102])
        assert math.isnan(rep.avg_exposure) and rep.num_trades == 0
