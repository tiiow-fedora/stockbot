"""Behavioural tests for VolRegimeTrend: rules on hand-built bars, parameters, synthetic edge."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.vol_regime_trend import VolRegimeTrend, VolRegimeTrendParams

SMALL = dict(trend_period=20, vol_period=10, vol_window=30)  # keeps hand-built fixtures short
OFF = dict(entry_pct=1.0, exit_pct=1.0)  # percentile gate disabled: plain SMA trend following


def bars_from_closes(closes, volume=1e6):
    closes = np.asarray(closes, float)
    return Bars("T", list(range(len(closes))), closes, closes * 1.001, closes * 0.999, closes, np.full(len(closes), volume))


def path(sigmas, drift):
    """Alternating +/- sigma_t log returns (so rolling vol tracks sigma_t exactly) plus a drift."""
    sigmas = np.asarray(sigmas, float)
    signs = np.where(np.arange(len(sigmas)) % 2 == 0, 1.0, -1.0)
    return 100.0 * np.exp(np.cumsum(drift + sigmas * signs))


CALM = path(np.linspace(0.02, 0.002, 80), 0.003)    # uptrend, vol falling -> lowest vol of the window
STORMY = path(np.linspace(0.002, 0.02, 80), 0.003)  # uptrend, vol rising  -> highest vol of the window


class TestRules:
    def test_percentile_extremes(self):
        calm = VolRegimeTrend(**SMALL).signal(bars_from_closes(CALM), False)
        stormy = VolRegimeTrend(**SMALL).signal(bars_from_closes(STORMY), False)
        assert calm.indicators["vol_pct"] <= 0.1
        assert stormy.indicators["vol_pct"] == 1.0

    def test_enters_uptrend_in_calm_regime(self):
        sig = VolRegimeTrend(**SMALL).signal(bars_from_closes(CALM), False)
        assert sig.kind.value == "ENTER" and sig.indicators["close"] > sig.indicators["sma"]
        assert sig.strength > 0.9

    def test_blocks_uptrend_when_vol_is_high(self):
        sig = VolRegimeTrend(**SMALL).signal(bars_from_closes(STORMY), False)
        assert sig.kind.value == "HOLD" and "too volatile" in sig.reason
        assert VolRegimeTrend(**SMALL, **OFF).signal(bars_from_closes(STORMY), False).kind.value == "ENTER"

    def test_no_entry_below_trend_even_if_calm(self):
        closes = path(np.linspace(0.02, 0.002, 80), -0.003)
        assert VolRegimeTrend(**SMALL).signal(bars_from_closes(closes), False).kind.value == "HOLD"

    def test_exits_on_vol_spike_while_still_above_trend(self):
        b = bars_from_closes(STORMY)
        sig = VolRegimeTrend(**SMALL).signal(b, True)
        assert sig.kind.value == "EXIT" and "risk-off" in sig.reason
        assert sig.indicators["close"] > sig.indicators["sma"]
        assert VolRegimeTrend(**SMALL).signal(b, False).kind.value == "HOLD"  # only exits when long

    def test_exits_when_trend_breaks_even_if_calm(self):
        closes = path(np.linspace(0.02, 0.002, 80), -0.003)
        sig = VolRegimeTrend(**SMALL).signal(bars_from_closes(closes), True)
        assert sig.kind.value == "EXIT" and sig.indicators["vol_pct"] <= 0.1

    def test_holds_when_trend_intact_and_vol_calm(self):
        assert VolRegimeTrend(**SMALL).signal(bars_from_closes(CALM), True).kind.value == "HOLD"

    def test_hysteresis_between_entry_and_exit_percentiles(self):
        """A vol percentile above entry_pct but below exit_pct blocks new entries yet keeps a trade."""
        pcts = []
        for end in range(60, 140):
            sig = VolRegimeTrend(**SMALL).signal(bars_from_closes(self._wave()[:end]), False)
            pcts.append(sig.indicators["vol_pct"])
        mid = next(e for e, p in zip(range(60, 140), pcts) if 0.5 < p <= 0.9)
        b = bars_from_closes(self._wave()[:mid])
        s = VolRegimeTrend(**SMALL, entry_pct=0.5, exit_pct=0.9)
        assert s.signal(b, False).kind.value == "HOLD"
        assert s.signal(b, True).kind.value == "HOLD"

    @staticmethod
    def _wave():
        return path(0.011 + 0.009 * np.sin(np.arange(140) / 9.0), 0.004)

    def test_strength_falls_as_vol_percentile_rises(self):
        s = VolRegimeTrend(**SMALL, entry_pct=0.7, exit_pct=0.95)
        seen = []
        for end in range(60, 140):
            sig = s.signal(bars_from_closes(self._wave()[:end]), False)
            if sig.kind.value == "ENTER":
                seen.append((sig.indicators["vol_pct"], sig.strength))
        seen.sort()
        assert len(seen) >= 8 and seen[0][0] < seen[-1][0]
        strengths = [st for _, st in seen]
        assert strengths == sorted(strengths, reverse=True)
        assert strengths[0] > strengths[-1] and all(0.5 <= x <= 1.0 for x in strengths)

    def test_flat_prices_zero_vol_never_enter(self):
        sig = VolRegimeTrend(**SMALL).signal(bars_from_closes([100.0] * 80), False)
        assert sig.kind.value == "HOLD"

    def test_warmup_is_hold(self):
        s = VolRegimeTrend(**SMALL)
        sig = s.signal(bars_from_closes(CALM[: SMALL["vol_period"] + SMALL["vol_window"] - 1]), False)
        assert sig.kind.value == "HOLD" and "not ready" in sig.reason


class TestParams:
    @pytest.mark.parametrize(
        "kw",
        [{"trend_period": 0}, {"trend_period": -5}, {"vol_period": 1}, {"vol_period": 0}, {"vol_window": 5},
         {"entry_pct": 0.0}, {"entry_pct": 0.9, "exit_pct": 0.8}, {"exit_pct": 1.2}],
    )
    def test_invalid_params(self, kw):
        with pytest.raises(ValueError):
            VolRegimeTrendParams(**kw)

    def test_grid_size_and_validity(self):
        combos = VolRegimeTrend.param_combinations()
        assert 6 <= len(combos) <= 60
        for c in combos:
            VolRegimeTrendParams(**c)

    def test_grid_drops_entry_above_exit(self):
        combos = VolRegimeTrend.param_combinations({"entry_pct": [0.9], "exit_pct": [0.5, 0.95]})
        assert combos == [{"entry_pct": 0.9, "exit_pct": 0.95}]

    def test_min_bars_covers_vol_history(self):
        s = VolRegimeTrend(**SMALL)
        assert s.min_bars >= SMALL["vol_period"] + SMALL["vol_window"] and s.lookback >= s.min_bars


class TestSyntheticEdge:
    def test_gate_cuts_drawdown_when_volatility_clusters(self):
        """GARCH storms: the vol gate lowers exposure and max drawdown versus the ungated SMA rule."""
        for seed in range(6):
            c = synth.garch(800, seed=seed)
            gated = run_backtest(VolRegimeTrend(), c).report
            plain = run_backtest(VolRegimeTrend(**OFF), c).report
            assert gated.avg_exposure < plain.avg_exposure
            assert gated.max_drawdown < plain.max_drawdown

    def test_gate_costs_upside_in_calm_uptrend(self):
        """Honest trade-off: in a steady trend the filter just sits out part of the move."""
        c = synth.trending(300, 0.004, 0.006, seed=1)
        gated = run_backtest(VolRegimeTrend(), c).report
        plain = run_backtest(VolRegimeTrend(**OFF), c).report
        assert 0.2 < gated.total_return < plain.total_return

    def test_misses_part_of_high_vol_recovery(self):
        c = synth.v_shape(300, seed=0)
        gated = run_backtest(VolRegimeTrend(), c).report
        plain = run_backtest(VolRegimeTrend(**OFF), c).report
        assert gated.total_return > 0 and gated.total_return < plain.total_return

    def test_stays_out_of_persistent_downtrend(self):
        r = run_backtest(VolRegimeTrend(), synth.trending(300, -0.004, 0.006, seed=2))
        assert r.report.avg_exposure < 0.1 and r.report.max_drawdown < 0.05

    def test_no_edge_in_mean_reverting_market(self):
        """Honest negative result: calm vol does not rescue an SMA rule on an OU series."""
        for seed in (0, 2):
            r = run_backtest(VolRegimeTrend(), synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=seed))
            assert r.report.total_return < 0.0


class TestReviewFixes:
    """Boundary and exactness gaps found by mutation testing."""

    def test_percentile_includes_current_bar_and_matches_definition(self):
        from research import indicators as ind

        sig = VolRegimeTrend(**SMALL).signal(bars_from_closes(CALM), False)
        vol = ind.realized_vol(CALM, SMALL["vol_period"])
        assert sig.indicators["vol_pct"] == pytest.approx(np.mean(vol[-SMALL["vol_window"]:] <= vol[-1]))
        assert sig.indicators["vol_pct"] == pytest.approx(1 / SMALL["vol_window"])  # calmest reading: only itself

    def test_exit_pct_is_strict_and_one_disables_the_vol_exit(self):
        b = bars_from_closes(STORMY)  # percentile == 1.0
        assert VolRegimeTrend(**SMALL, entry_pct=1.0, exit_pct=1.0).signal(b, True).kind.value == "HOLD"
        assert VolRegimeTrend(**SMALL, entry_pct=0.5, exit_pct=0.99).signal(b, True).kind.value == "EXIT"

    def test_close_equal_to_sma_is_neither_entry_nor_exit(self):
        flat = bars_from_closes([100.0] * 80)
        s = VolRegimeTrend(**SMALL, **OFF)
        assert s.signal(flat, False).kind.value == "HOLD"
        assert s.signal(flat, True).kind.value == "HOLD"

    def test_strength_formula(self):
        s = VolRegimeTrend(**SMALL, entry_pct=0.7, exit_pct=0.95)
        checked = 0
        for end in range(60, 140):
            sig = s.signal(bars_from_closes(TestRules._wave()[:end]), False)
            if sig.kind.value == "ENTER":
                assert sig.strength == pytest.approx(0.5 + 0.5 * (1 - sig.indicators["vol_pct"] / 0.7))
                checked += 1
        assert checked >= 8
