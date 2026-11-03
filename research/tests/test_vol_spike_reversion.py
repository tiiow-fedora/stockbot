"""Behavioural tests for VolSpikeReversion: rules on hand-built bars, params, synthetic panics."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.vol_spike_reversion import VolSpikeParams, VolSpikeReversion
from research.tests.helpers import simulate_long_flat


def bars_from_closes(closes):
    closes = np.asarray(closes, float)
    n = len(closes)
    return Bars("T", list(range(n)), closes, closes * 1.001, closes * 0.999, closes, np.full(n, 1e6))


def calm(n=60, base=100.0, wiggle=0.003):
    """Tiny alternating moves: non-zero but small realised vol."""
    return [base * (1 + wiggle * (-1) ** i) for i in range(n)]


PANIC = [0.97, 0.96, 0.97, 0.95]  # multiplicative closes steps, ~ -12% total with big daily vol


def panic_series(n_calm=60):
    c = calm(n_calm)
    for step in PANIC:
        c.append(c[-1] * step)
    return c


class TestRules:
    def test_enters_on_spike_and_drop(self):
        s = VolSpikeReversion()
        sig = s.signal(bars_from_closes(panic_series()), False)
        assert sig.kind.value == "ENTER"
        assert sig.indicators["vol_ratio"] >= s.params.spike_ratio

    def test_calm_market_holds(self):
        assert VolSpikeReversion().signal(bars_from_closes(calm(80)), False).kind.value == "HOLD"

    def test_spike_without_price_drop_holds(self):
        """High vol alone is not a panic: a violent but flat/up move must not trigger entry."""
        c = calm(60)
        for step in (1.03, 0.97, 1.03, 0.97, 1.03):
            c.append(c[-1] * step)
        sig = VolSpikeReversion().signal(bars_from_closes(c), False)
        assert sig.indicators["vol_ratio"] >= 1.6 and sig.kind.value == "HOLD"

    def test_drop_without_vol_spike_holds(self):
        """A slow, steady slide of the same size is not a fear spike (vol ratio stays low)."""
        c = list(100 * 0.9985 ** np.arange(120))  # constant -0.15%/day, zero return dispersion
        c = [x * (1 + 0.002 * (-1) ** i) for i, x in enumerate(c)]
        sig = VolSpikeReversion(drop_pct=0.05).signal(bars_from_closes(c), False)
        assert sig.indicators["vol_ratio"] < 1.6 and sig.kind.value == "HOLD"

    def test_drop_threshold_boundary(self):
        c = panic_series()
        deep = VolSpikeReversion(drop_pct=0.10).signal(bars_from_closes(c), False)
        too_deep = VolSpikeReversion(drop_pct=0.30).signal(bars_from_closes(c), False)
        assert deep.kind.value == "ENTER" and too_deep.kind.value == "HOLD"

    def test_exit_when_price_recovers_to_sma(self):
        s = VolSpikeReversion()
        c = panic_series() + [105.0]  # well above the 20-bar SMA
        assert s.signal(bars_from_closes(c), True).kind.value == "EXIT"

    def test_exit_when_vol_normalises(self):
        s = VolSpikeReversion(recover_sma=60)  # SMA far above price -> only the vol exit can fire
        base = panic_series()
        calm_tail = [base[-1] * (1 + 0.001 * (-1) ** i) for i in range(45)]
        sig = s.signal(bars_from_closes(base + calm_tail), True)
        assert sig.kind.value == "EXIT" and sig.indicators["vol_ratio"] < s.params.normal_ratio

    def test_holds_while_still_stressed(self):
        s = VolSpikeReversion()
        assert s.signal(bars_from_closes(panic_series()), True).kind.value == "HOLD"

    def test_strength_grows_with_spike_ratio(self):
        s = VolSpikeReversion()
        mild = s.signal(bars_from_closes(panic_series()), False)
        c = calm(60) + [100 * 0.9, 100 * 0.8, 100 * 0.7, 100 * 0.62]
        wild = s.signal(bars_from_closes(c), False)
        assert mild.kind.value == wild.kind.value == "ENTER"
        assert 0.5 <= mild.strength < wild.strength <= 1.0

    def test_trend_filter_blocks_panic_below_long_term_sma(self):
        # long decline into the panic -> price far below its 100-bar SMA
        down = list(np.linspace(160, 100, 110)) + PANIC
        c = down[:110]
        for step in PANIC:
            c.append(c[-1] * step)
        c = np.array(c) * np.array([1 + 0.003 * (-1) ** i for i in range(len(c))])
        assert VolSpikeReversion(trend_filter=False).signal(bars_from_closes(c), False).kind.value == "ENTER"
        assert VolSpikeReversion(trend_filter=True).signal(bars_from_closes(c), False).kind.value == "HOLD"

    def test_trend_filter_allows_panic_in_uptrend(self):
        up = list(np.linspace(60, 100, 110))
        for step in PANIC:
            up.append(up[-1] * step)
        c = np.array(up) * np.array([1 + 0.003 * (-1) ** i for i in range(len(up))])
        assert VolSpikeReversion(trend_filter=True).signal(bars_from_closes(c), False).kind.value == "ENTER"

    def test_flat_series_has_no_signal(self):
        c = np.full(100, 50.0)
        assert VolSpikeReversion().signal(bars_from_closes(c), False).kind.value == "HOLD"


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(short_vol=0), dict(long_vol=-1), dict(high_window=0), dict(recover_sma=1), dict(trend_period=0),
        dict(short_vol=40, long_vol=40), dict(drop_pct=0.0), dict(drop_pct=1.0),
        dict(normal_ratio=0.0), dict(normal_ratio=2.0, spike_ratio=1.5),
    ])
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValueError):
            VolSpikeParams(**bad)

    def test_grid_size_and_validity(self):
        combos = VolSpikeReversion.param_combinations()
        assert 6 <= len(combos) <= 60
        assert all(VolSpikeParams(**c) for c in combos)

    def test_trend_filter_raises_min_bars(self):
        assert VolSpikeReversion(trend_filter=True).min_bars > VolSpikeReversion(trend_filter=False).min_bars

    def test_higher_spike_ratio_means_fewer_trades(self):
        c = synth.v_shape(500, seed=3)
        loose, _ = simulate_long_flat(VolSpikeReversion(spike_ratio=1.3, drop_pct=0.03, normal_ratio=1.0), c)
        strict, _ = simulate_long_flat(VolSpikeReversion(spike_ratio=2.5, drop_pct=0.03, normal_ratio=1.0), c)
        assert len(strict) <= len(loose)


def panic_market(cycles=6, seed=0):
    """Calm random walk interrupted by sharp V-shaped fear spikes that fully recover."""
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(cycles):
        r += list(rng.normal(0.0004, 0.005, 60))
        r += [-0.035, -0.04, -0.03, -0.025]          # panic leg (-12%) with 5-8x normal vol
        r += list(rng.normal(0.012, 0.006, 10))      # sharp recovery
    return synth.candles_from_closes(100 * np.exp(np.cumsum(r)), "PANIC", seed)


class TestSyntheticEdge:
    def test_profits_from_v_shaped_panics(self):
        results = [run_backtest(VolSpikeReversion(), panic_market(seed=s)).report for s in range(6)]
        assert all(r.num_trades >= 3 for r in results)
        assert sum(r.total_return > 0 for r in results) >= 5
        assert np.mean([r.total_return for r in results]) > 0.03
        assert np.mean([r.win_rate for r in results]) > 0.6

    def test_does_nothing_in_calm_trending_data(self):
        c = synth.trending(300, 0.003, 0.004, seed=2)
        r = run_backtest(VolSpikeReversion(), c)
        assert r.report.num_trades <= 1 and r.report.avg_exposure < 0.05

    def test_catching_a_falling_knife_loses(self):
        """Honest negative: a persistent crash (vol stays high, price keeps falling) hurts it."""
        rng = np.random.default_rng(5)
        r = list(rng.normal(0.0004, 0.005, 80)) + list(rng.normal(-0.02, 0.025, 60))
        c = synth.candles_from_closes(100 * np.exp(np.cumsum(r)), "CRASH", 5)
        res = run_backtest(VolSpikeReversion(), c)
        assert res.report.num_trades >= 1 and res.report.total_return < 0.0
