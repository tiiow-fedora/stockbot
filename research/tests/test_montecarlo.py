"""Tests for research.montecarlo: reproducibility, bounds, degenerate inputs, and
falsifiable claims (permutation power/size, block bootstrap autocorrelation, path robustness)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from research import metrics, montecarlo as mc, synth
from research.strategies.donchian_breakout import DonchianBreakout

PNLS = [500.0, -300.0, 800.0, -200.0, 150.0, -450.0, 1200.0, -100.0, 300.0, -250.0]


def ar1(n: int, phi: float, seed: int, sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def lag1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ---------------------------------------------------------------- trade bootstrap

class TestBootstrapTrades:
    def test_reproducible_by_seed_and_seeds_differ(self):
        a = mc.bootstrap_trades(PNLS, 300, seed=3).to_dict()
        b = mc.bootstrap_trades(PNLS, 300, seed=3).to_dict()
        c = mc.bootstrap_trades(PNLS, 300, seed=4).to_dict()
        assert a == b
        assert a["percentiles"] != c["percentiles"]

    def test_percentiles_ordered_and_bounds(self):
        s = mc.bootstrap_trades(PNLS, 500, seed=1)
        vals = [s.percentiles[k] for k in ("p5", "p25", "p50", "p75", "p95")]
        assert vals == sorted(vals)
        assert s.minimum <= vals[0] and vals[-1] <= s.maximum
        assert 0.0 <= s.prob_loss <= 1.0 and 0.0 <= s.prob_ruin <= 1.0
        dd = s.extras["max_drawdown"]
        assert 0.0 <= dd["min"] <= dd["max"] < 1.0

    def test_with_replacement_mean_matches_sum(self):
        s = mc.bootstrap_trades(PNLS, 4000, 100_000, seed=2)
        assert s.mean == pytest.approx(100_000 + sum(PNLS), rel=0.01)

    def test_shuffle_keeps_final_equity_but_varies_drawdown(self):
        s = mc.bootstrap_trades(PNLS, 200, 100_000, seed=2, replace=False)
        assert s.minimum == pytest.approx(s.maximum) == pytest.approx(100_000 + sum(PNLS))
        assert s.std == pytest.approx(0.0, abs=1e-9)
        assert s.extras["max_drawdown"]["max"] > s.extras["max_drawdown"]["min"]

    def test_all_winners_never_lose_or_ruin(self):
        s = mc.bootstrap_trades([100.0, 200.0, 50.0], 100, seed=0)
        assert s.prob_loss == 0.0 and s.prob_ruin == 0.0 and s.extras["max_drawdown"]["max"] == 0.0

    def test_all_losers_always_lose(self):
        s = mc.bootstrap_trades([-20_000.0, -30_000.0, -25_000.0], 100, 100_000, seed=0)
        assert s.prob_loss == 1.0 and s.prob_ruin == 1.0

    def test_zero_variance_trades(self):
        s = mc.bootstrap_trades([10.0] * 6, 50, 1000.0, seed=0)
        assert s.minimum == s.maximum == 1060.0

    def test_drawdown_matches_research_metrics(self):
        pnl = np.array(PNLS)
        draws = np.random.default_rng(5).choice(pnl, size=(5, len(pnl)))
        eq = np.concatenate([np.full((5, 1), 1000.0), 1000.0 + np.cumsum(draws, axis=1)], axis=1)
        expected = [metrics.max_drawdown(row) for row in eq]
        assert mc._max_dd_matrix(eq) == pytest.approx(expected)

    def test_to_dict_is_json_safe(self):
        d = mc.bootstrap_trades(PNLS, 50, seed=0).to_dict()
        json.dumps(d, allow_nan=False)
        z = mc.block_bootstrap_returns([0.01] * 20, 20, 5, seed=0).to_dict()  # Sharpe is NaN here
        json.dumps(z, allow_nan=False)
        assert z["extras"]["sharpe"]["mean"] is None

    @pytest.mark.parametrize("bad, match", [
        ([], "at least 2"), ([1.0], "at least 2"), ([1.0, float("nan")], "NaN"),
    ])
    def test_invalid_inputs_raise(self, bad, match):
        with pytest.raises(ValueError, match=match):
            mc.bootstrap_trades(bad, 10)

    def test_invalid_settings_raise(self):
        with pytest.raises(ValueError, match="n_sims"):
            mc.bootstrap_trades(PNLS, 0)
        with pytest.raises(ValueError, match="starting_equity"):
            mc.bootstrap_trades(PNLS, 10, starting_equity=0)
        with pytest.raises(ValueError, match="ruin_level"):
            mc.bootstrap_trades(PNLS, 10, ruin_level=1.5)


# ---------------------------------------------------------------- block bootstrap

class TestBlockBootstrap:
    def test_reproducible_and_seed_sensitive(self):
        r = ar1(200, 0.3, 1)
        a = mc.block_bootstrap_returns(r, 200, 10, seed=7).to_dict()
        assert a == mc.block_bootstrap_returns(r, 200, 10, seed=7).to_dict()
        assert a != mc.block_bootstrap_returns(r, 200, 10, seed=8).to_dict()

    def test_percentile_order_and_probability_bounds(self):
        s = mc.block_bootstrap_returns(ar1(300, 0.2, 2), 400, 8, seed=0)
        v = list(s.percentiles.values())
        assert v == sorted(v)
        assert 0 <= s.prob_loss <= 1 and 0 <= s.prob_ruin <= 1
        assert s.extras["max_drawdown"]["min"] >= 0

    def test_full_length_moving_block_returns_original_path(self):
        r = ar1(50, 0.0, 3)
        idx = mc.block_bootstrap_indices(50, 4, 50, np.random.default_rng(0), stationary=False, circular=False)
        assert (idx == np.arange(50)).all()
        s = mc.block_bootstrap_returns(r, 10, 50, seed=0, stationary=False, circular=False)
        assert s.minimum == pytest.approx(s.maximum) == pytest.approx(np.prod(1 + r) - 1)

    def test_circular_full_block_is_a_rotation(self):
        idx = mc.block_bootstrap_indices(30, 5, 30, np.random.default_rng(0), stationary=False, circular=True)
        for row in idx:
            assert (np.diff(row) % 30 == 1).all()

    def test_block_len_one_is_iid_bootstrap(self):
        """Stationary with block_len=1 starts a new block every step: indices are uniform iid,
        so the resampled series has ~zero autocorrelation even from a strongly AR(1) input."""
        r = ar1(2000, 0.8, 4)
        idx = mc.block_bootstrap_indices(len(r), 1, 1, np.random.default_rng(0))[0]
        assert abs(lag1(r[idx])) < 0.08 < abs(lag1(r))
        assert (np.diff(idx) != 1).mean() > 0.99

    def test_preserves_autocorrelation_better_than_iid(self):
        r = ar1(1500, 0.6, 5)
        rng = np.random.default_rng(1)
        blocks = np.mean([lag1(r[i]) for i in mc.block_bootstrap_indices(len(r), 30, 40, rng)])
        iid = np.mean([lag1(r[i]) for i in mc.block_bootstrap_indices(len(r), 30, 1, rng)])
        assert abs(iid) < 0.05
        assert blocks > 0.4 and blocks > iid + 0.35

    def test_stationary_mean_block_length(self):
        idx = mc.block_bootstrap_indices(5000, 1, 20, np.random.default_rng(2))[0]
        runs = np.diff(np.flatnonzero(np.r_[True, np.diff(idx) % 5000 != 1, True]))
        assert 14 < runs.mean() < 28

    def test_zero_variance_returns(self):
        s = mc.block_bootstrap_returns([0.001] * 40, 30, 5, seed=0)
        assert s.minimum == pytest.approx(s.maximum)
        assert s.extras["sharpe"]["n_valid"] == 0  # undefined, not 0/inf
        assert s.extras["max_drawdown"]["max"] == 0.0 and s.prob_loss == 0.0

    @pytest.mark.parametrize("kwargs, match", [
        (dict(returns=[]), "at least 2"),
        (dict(returns=[0.01]), "at least 2"),
        (dict(returns=[0.01, float("inf")]), "NaN or infinite"),
        (dict(returns=[0.01, -1.0]), "> -1"),
        (dict(returns=[[0.01, 0.02]]), "1-D"),
        (dict(returns=[0.01] * 10, block_len=0), "block_len"),
        (dict(returns=[0.01] * 10, block_len=11), "block_len"),
        (dict(returns=[0.01] * 10, block_len=2.5), "block_len"),
        (dict(returns=[0.01] * 10, n_sims=0), "n_sims"),
    ])
    def test_invalid_inputs_raise(self, kwargs, match):
        kwargs.setdefault("block_len", 2)
        kwargs.setdefault("n_sims", 5)
        with pytest.raises(ValueError, match=match):
            mc.block_bootstrap_returns(**kwargs)


# ---------------------------------------------------------------- permutation test

@pytest.fixture(scope="module")
def market():
    return np.random.default_rng(0).normal(0.0003, 0.01, 750)


class TestPermutationTest:
    def test_clairvoyant_position_has_p_near_zero(self, market):
        pos = (market > 0).astype(float)  # look-ahead: knows each bar's return in advance
        res = mc.permutation_test(None, pos, market, 500, seed=1)
        assert res.p_value == pytest.approx(1 / 501)
        assert res.observed > res.null_mean + 5 * res.null_std

    def test_clairvoyant_sharpe_statistic(self, market):
        pos = (market > 0).astype(float)
        assert mc.permutation_test(None, pos, market, 300, seed=1, statistic="sharpe").p_value < 0.01

    def test_random_position_is_not_significant(self, market):
        res = mc.permutation_test(None, np.random.default_rng(9).integers(0, 2, len(market)).astype(float), market, 500, seed=1)
        assert res.p_value > 0.05

    def test_size_is_controlled_over_many_random_positions(self, market):
        ps = np.array([mc.permutation_test(None, np.random.default_rng(100 + k).integers(0, 2, len(market)).astype(float),
                                           market, 200, seed=k).p_value for k in range(40)])
        assert (ps < 0.05).mean() <= 0.15
        assert 0.3 < ps.mean() < 0.7

    def test_autocorrelated_random_position_still_not_significant(self, market):
        """Slow-moving (long-hold) positions: the circular-shift null preserves that structure."""
        pos = (np.convolve(np.random.default_rng(3).normal(size=len(market)), np.ones(40) / 40, "same") > 0).astype(float)
        assert mc.permutation_test(None, pos, market, 500, seed=2).p_value > 0.05

    def test_reproducible_and_seed_sensitive(self, market):
        pos = (market > 0).astype(float)
        a = mc.permutation_test(None, pos, market, 100, seed=5)
        assert a == mc.permutation_test(None, pos, market, 100, seed=5)
        assert a.null_mean != mc.permutation_test(None, pos, market, 100, seed=6).null_mean

    def test_explicit_strategy_returns_used_as_observed(self, market):
        pos = (market > 0).astype(float)
        res = mc.permutation_test(pos * market - 0.01, pos, market, 100, seed=0)  # heavy cost drag
        assert res.p_value > 0.5

    def test_shuffle_mode_runs_and_flags_clairvoyance(self, market):
        res = mc.permutation_test(None, (market > 0).astype(float), market, 200, seed=0, mode="shuffle")
        assert res.mode == "shuffle" and res.p_value < 0.02

    def test_flat_position_gives_nan_sharpe_p(self, market):
        res = mc.permutation_test(None, np.zeros(len(market)), market, 50, seed=0, statistic="sharpe")
        assert math.isnan(res.p_value)
        assert res.to_dict()["p_value"] is None

    @pytest.mark.parametrize("kwargs, match", [
        (dict(position=[1, 0], market_returns=[0.1, 0.1]), "at least 3"),
        (dict(position=[1, 0, 1], market_returns=[0.1, 0.1, 0.1, 0.1]), "equal length"),
        (dict(position=[1, 0, 1, 0], market_returns=[0.1] * 4, statistic="x"), "statistic"),
        (dict(position=[1, 0, 1, 0], market_returns=[0.1] * 4, mode="x"), "mode"),
        (dict(position=[1, 0, 1, 0], market_returns=[0.1] * 4, min_shift=3), "min_shift"),
        (dict(position=[1, 0, 1, 0], market_returns=[0.1] * 4, strategy_returns=[0.1] * 3), "length 4"),
        (dict(position=[1, 0, 1, float("nan")], market_returns=[0.1] * 4), "NaN"),
    ])
    def test_invalid_inputs_raise(self, kwargs, match):
        kwargs.setdefault("strategy_returns", None)
        with pytest.raises(ValueError, match=match):
            mc.permutation_test(n_sims=10, seed=0, **kwargs)


# ---------------------------------------------------------------- price paths

@pytest.fixture(scope="module")
def trend_candles():
    return synth.trending(300, 0.003, 0.008, seed=1)


@pytest.fixture(scope="module")
def ou_candles():
    return synth.mean_reverting(300, 8.0, 0.012, seed=3)


class TestPathGeneration:
    def test_paths_are_valid_candles_with_original_timestamps_and_first_bar(self, trend_candles):
        for m in mc.PATH_METHODS:
            p = mc.generate_price_path(trend_candles, m, np.random.default_rng(0))
            assert len(p) == len(trend_candles) and p[0] == trend_candles[0]
            assert [c.timestamp for c in p] == [c.timestamp for c in trend_candles]
            assert all(c.high >= max(c.open, c.close) and c.low <= min(c.open, c.close) and c.volume >= 0 for c in p)

    def test_shuffle_returns_preserves_terminal_price(self, trend_candles):
        p = mc.generate_price_path(trend_candles, "shuffle_returns", np.random.default_rng(1))
        assert p[-1].close == pytest.approx(trend_candles[-1].close, rel=1e-9)
        assert [c.close for c in p] != [c.close for c in trend_candles]

    def test_volume_is_kept_from_real_bars(self, trend_candles):
        real = {c.volume for c in trend_candles}
        p = mc.generate_price_path(trend_candles, "block", np.random.default_rng(2))
        assert {c.volume for c in p[1:]} <= real

    def test_no_lookahead_in_path_construction(self, trend_candles):
        """Bars <= k depend only on the first k resampled returns / shape picks."""
        m = len(trend_candles) - 1
        rng = np.random.default_rng(0)
        lr, idx = rng.normal(0, 0.01, m), rng.integers(0, m, m)
        base = mc.build_path_candles(trend_candles, lr, idx)
        k = 120
        lr2, idx2 = lr.copy(), idx.copy()
        lr2[k:] = rng.normal(0, 0.05, m - k)
        idx2[k:] = rng.integers(0, m, m - k)
        alt = mc.build_path_candles(trend_candles, lr2, idx2)
        assert base[: k + 1] == alt[: k + 1]
        assert base[k + 1] != alt[k + 1]

    def test_bootstrap_index_uses_only_past_draws(self):
        """Stationary index t is a function of draws at positions <= t: truncating the
        'future' draws (same first-t draws via a replayed stream) cannot change the prefix."""
        n = 200
        idx_a = mc.block_bootstrap_indices(n, 1, 10, np.random.default_rng(4))[0]
        rng = np.random.default_rng(4)
        new = rng.random((1, n)) < 0.1
        starts = rng.integers(0, n, size=(1, n))
        # recompute sequentially, visiting positions in order and never looking ahead
        ref, cur = [], 0
        for t in range(n):
            cur = starts[0, t] if (t == 0 or new[0, t]) else (cur + 1) % n
            ref.append(cur)
        assert ref == list(idx_a)

    def test_invalid_method_and_lengths(self, trend_candles):
        with pytest.raises(ValueError, match="method"):
            mc.generate_price_path(trend_candles, "nope", np.random.default_rng(0))
        with pytest.raises(ValueError, match="len\\(base\\) - 1"):
            mc.build_path_candles(trend_candles, np.zeros(3), np.zeros(3, int))


class TestPricePathRobustness:
    def test_reproducible_seed_sensitive_and_prefix_stable(self, trend_candles):
        f = lambda: DonchianBreakout()  # noqa: E731
        a = mc.price_path_robustness(f, trend_candles, 4, seed=1, method="block")
        assert a.to_dict() == mc.price_path_robustness(f, trend_candles, 4, seed=1, method="block").to_dict()
        assert a.to_dict() != mc.price_path_robustness(f, trend_candles, 4, seed=2, method="block").to_dict()

    def test_sim_k_does_not_depend_on_n_sims(self):
        g2, g5 = mc.spawn_rngs(3, 2), mc.spawn_rngs(3, 5)
        assert g2[0].random() == g5[0].random() and g2[1].random() == g5[1].random()
        assert g5[0].random() != g5[1].random()

    def test_max_sims_guard_and_input_checks(self, trend_candles):
        f = lambda: DonchianBreakout()  # noqa: E731
        with pytest.raises(ValueError, match="max_sims"):
            mc.price_path_robustness(f, trend_candles, 11, max_sims=10)
        with pytest.raises(ValueError, match="candles"):
            mc.price_path_robustness(f, trend_candles[:50], 2)
        with pytest.raises(ValueError, match="method"):
            mc.price_path_robustness(f, trend_candles, 2, method="x")

    def test_fresh_strategy_instance_per_sim(self, trend_candles):
        made = []
        mc.price_path_robustness(lambda: made.append(DonchianBreakout()) or made[-1], trend_candles, 3, seed=0)
        assert len(made) == 3 + 1 and len({id(m) for m in made}) == 4  # 3 sims + the real-series run

    def test_summary_structure_and_bounds(self, trend_candles):
        s = mc.price_path_robustness(lambda: DonchianBreakout(), trend_candles, 6, seed=0, method="shuffle_returns")
        assert {"sharpe", "max_drawdown", "avg_exposure", "num_trades", "benchmark_return"} <= set(s.extras)
        for k in ("profitable", "beats_buy_hold", "no_closed_trades"):
            assert 0.0 <= s.probabilities[k] <= 1.0
        assert 0.0 <= s.extras["avg_exposure"]["min"] <= s.extras["avg_exposure"]["max"] <= 1.0
        json.dumps(s.to_dict(), allow_nan=False)

    def test_gbm_matched_trending_beats_mean_reverting_for_donchian(self, trend_candles, ou_candles):
        f = lambda: DonchianBreakout()  # noqa: E731
        tr = mc.price_path_robustness(f, trend_candles, 24, seed=5, method="gbm_matched")
        ou = mc.price_path_robustness(f, ou_candles, 24, seed=5, method="gbm_matched")
        assert tr.probabilities["profitable"] >= 0.9
        assert tr.probabilities["profitable"] > ou.probabilities["profitable"] + 0.2
        assert tr.mean > ou.mean

    def test_partner_strategy_via_closure(self, trend_candles):
        class Gated(DonchianBreakout):
            """Donchian that needs a partner's veto (partner here: always say no)."""
            def __init__(self, partner):
                super().__init__()
                self.partner = partner

            def signal(self, bars, in_position):
                sig = super().signal(bars, in_position)
                return sig if self.partner() or in_position else type(sig).hold("partner veto")

        s = mc.price_path_robustness(lambda: Gated(lambda: False), trend_candles, 3, seed=0)
        assert s.extras["avg_exposure"]["max"] == 0.0  # veto honoured on every path


# ---------------------------------------------------------------- parameter perturbation

class TestParameterPerturbation:
    def test_perturb_params_types_and_bounds(self):
        rng = np.random.default_rng(0)
        base = {"a": 3, "b": 0.5, "c": True, "d": "x", "e": 0.0}
        for _ in range(200):
            p = mc.perturb_params(base, 0.9, rng)
            assert isinstance(p["a"], int) and p["a"] >= 1 and 0.05 <= p["b"] <= 0.95
            assert p["c"] is True and p["d"] == "x" and p["e"] == 0.0

    def test_zero_noise_reproduces_base_and_full_plateau(self, trend_candles):
        s = mc.parameter_perturbation(DonchianBreakout, {"entry_period": 20, "exit_period": 10}, trend_candles, 4, rel_noise=0.0)
        assert s.probabilities["plateau_score"] == 1.0 and s.meta["n_invalid"] == 0
        assert s.minimum == pytest.approx(s.maximum) == pytest.approx(s.meta["base_objective"])

    def test_invalid_combos_skipped_and_counted(self, trend_candles):
        # exit_period must be <= entry_period: starting at the boundary, big noise breaks it often
        s = mc.parameter_perturbation(DonchianBreakout, {"entry_period": 10, "exit_period": 10}, trend_candles, 30, rel_noise=0.5, seed=1)
        assert s.meta["n_invalid"] > 0
        assert s.meta["n_invalid"] + s.meta["n_valid"] == 30 and s.n_sims == s.meta["n_valid"]

    def test_reproducible_and_plateau_bounds(self, trend_candles):
        a = mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 8, 0.3, seed=2)
        assert a.to_dict() == mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 8, 0.3, seed=2).to_dict()
        assert 0.0 <= a.probabilities["plateau_score"] <= 1.0

    def test_spiky_objective_scores_lower_than_flat_objective(self, trend_candles):
        flat = mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 6, 0.4, seed=3, objective=lambda rr: 1.0)
        assert flat.probabilities["plateau_score"] == 1.0  # constant objective: a perfect plateau
        calls = iter([1.0] + [0.0] * 100)  # base scores 1, every perturbation scores 0: a pure spike
        sp = mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 6, 0.4, seed=3, objective=lambda rr: next(calls))
        assert sp.probabilities["plateau_score"] == 0.0

    def test_negative_base_threshold_is_below_base(self, trend_candles):
        s = mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 3, 0.1, seed=0, objective=lambda rr: -1.0)
        assert s.meta["threshold"] == pytest.approx(-1.2)

    def test_invalid_inputs_raise(self, trend_candles):
        with pytest.raises(ValueError, match="rel_noise"):
            mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 3, rel_noise=-1)
        with pytest.raises(ValueError, match="keep"):
            mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 3, keep=2)
        with pytest.raises(ValueError, match="candles"):
            mc.parameter_perturbation(DonchianBreakout, {}, trend_candles[:30], 3)
        with pytest.raises(ValueError, match="NaN"):
            mc.parameter_perturbation(DonchianBreakout, {}, trend_candles, 3, objective=lambda rr: float("nan"))


# ---------------------------------------------------------------- report

def test_monte_carlo_report_structure_and_json(trend_candles):
    rep = mc.monte_carlo_report(lambda: DonchianBreakout(), trend_candles, DonchianBreakout, n_sims=50, path_sims=3, perturb_sims=3, seed=0)
    assert {"backtest", "trade_bootstrap", "return_bootstrap", "timing_permutation", "path_robustness",
            "parameter_perturbation", "text"} <= set(rep)
    assert set(rep["path_robustness"]) == {"block", "gbm_matched"}
    assert "real run" in rep["text"]
    json.dumps(rep, allow_nan=False)
