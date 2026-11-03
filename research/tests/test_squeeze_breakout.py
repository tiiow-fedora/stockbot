"""Behavioural tests for SqueezeBreakout: rules on hand-built bars, parameters, synthetic-data claims."""

from __future__ import annotations

import numpy as np
import pytest

from research import synth
from research.base import Bars
from research.harness import run_backtest
from research.strategies.squeeze_breakout import SqueezeBreakout, SqueezeParams
from research.tests.helpers import simulate_long_flat

# Alternating +/-0.05 closes with +/-0.5 bar ranges: Bollinger std (~0.05) is far inside Keltner (ATR ~ 1).
QUIET = [100.0 + 0.05 * (-1) ** i for i in range(40)]
RAMP = list(np.linspace(60.0, 100.0, 40))  # steady trend: std of closes >> ATR, Bollinger outside Keltner, never squeezed


def mk(closes, half_range=0.5, volume=1e6):
    c = np.asarray(closes, float)
    v = np.full(len(c), volume) if np.isscalar(volume) else np.asarray(volume, float)
    return Bars("T", list(range(len(c))), c, c + half_range, c - half_range, c, v)


def kind(strategy, closes, in_position=False, **kw):
    return strategy.signal(mk(closes, **kw), in_position).kind.value


class TestRules:
    def test_enters_when_squeeze_releases_upward(self):
        s = SqueezeBreakout()
        sig = s.signal(mk(QUIET + [100.5, 101.5, 103.0, 105.0]), False)
        assert sig.kind.value == "ENTER" and sig.indicators["squeeze_now"] == 0.0

    def test_no_entry_while_squeeze_is_still_on(self):
        s = SqueezeBreakout()
        assert kind(s, QUIET + [100.5, 101.5, 103.0]) == "HOLD"
        assert kind(s, QUIET) == "HOLD"

    def test_no_entry_without_a_prior_squeeze(self):
        """Strong up-move inside an already-trending market that was never compressed."""
        assert kind(SqueezeBreakout(), RAMP + [102.0, 104.0, 106.0, 108.0]) == "HOLD"

    def test_downside_release_is_not_bought(self):
        s = SqueezeBreakout()
        assert kind(s, QUIET + [99.5, 98.5, 97.0, 95.0]) == "HOLD"

    def test_squeeze_memory_is_limited_by_lookback(self):
        """Squeeze ended 2 bars before the last bar: counts for lookback=3, not for lookback=1."""
        closes = QUIET + [100.5, 101.5, 103.0, 105.0, 107.0]
        assert kind(SqueezeBreakout(squeeze_lookback=3), closes) == "ENTER"
        assert kind(SqueezeBreakout(squeeze_lookback=1), closes) == "HOLD"

    def test_close_must_be_above_middle_band(self):
        """Squeeze released and slope positive, but price still under the mid band: no entry."""
        s = SqueezeBreakout()
        closes = QUIET + [99.0, 97.0, 95.0, 96.5]
        b = mk(closes)
        sig = s.signal(b, False)
        assert sig.indicators["close"] < sig.indicators["mid"]
        assert sig.kind.value == "HOLD"

    def test_volume_filter_blocks_weak_release(self):
        closes = QUIET + [100.5, 101.5, 103.0, 105.0]
        vol = np.full(len(closes), 1e6)
        vol[-1] = 0.5e6
        weak = Bars("T", list(range(len(closes))), *(mk(closes).open, mk(closes).high, mk(closes).low, mk(closes).close), vol)
        assert SqueezeBreakout(vol_filter=1.5).signal(weak, False).kind.value == "HOLD"
        assert SqueezeBreakout(vol_filter=0.0).signal(weak, False).kind.value == "ENTER"

    def test_strength_grows_with_momentum(self):
        s = SqueezeBreakout()
        slow = s.signal(mk(QUIET + [100.4, 101.0, 101.8, 102.8]), False)
        fast = s.signal(mk(QUIET + [100.5, 101.5, 103.0, 105.0]), False)
        assert slow.kind.value == fast.kind.value == "ENTER"
        assert 0.5 <= slow.strength < fast.strength <= 1.0

    def test_exit_when_close_drops_below_mid_minus_atr(self):
        s = SqueezeBreakout(mom_period=12)
        up = list(np.linspace(100, 120, 40))
        assert kind(s, up, in_position=True) == "HOLD"
        crash = up + [100.0]
        sig = s.signal(mk(crash), True)
        assert sig.kind.value == "EXIT" and sig.indicators["close"] < sig.indicators["mid"]

    def test_exit_on_momentum_flip_even_above_middle_band(self):
        """A slow roll-over: close stays above mid - 1 ATR but the regression slope is negative."""
        s = SqueezeBreakout(exit_atr=5.0, mom_period=8)
        closes = list(np.linspace(100, 110, 40)) + list(np.linspace(110, 108.5, 8))
        sig = s.signal(mk(closes, half_range=0.3), True)
        assert sig.indicators["slope"] < 0 and sig.indicators["close"] > sig.indicators["mid"] - 5.0 * sig.indicators["atr"]
        assert sig.kind.value == "EXIT"
        assert kind(SqueezeBreakout(exit_atr=5.0, mom_period=8), closes[:40], True, half_range=0.3) == "HOLD"

    def test_exit_ignored_when_flat(self):
        up = list(np.linspace(100, 120, 40))
        assert kind(SqueezeBreakout(), up + [100.0], in_position=False) == "HOLD"

    def test_nan_and_warmup_are_hold(self):
        s = SqueezeBreakout()
        assert kind(s, [100.0] * 5) == "HOLD"
        assert kind(s, [100.0] * 5, True) == "HOLD"
        assert kind(s, [100.0] * 60) == "HOLD"  # flat: zero ATR must not divide or signal


class TestParams:
    @pytest.mark.parametrize("bad", [
        dict(bb_period=0), dict(kc_period=-1), dict(kc_atr_period=1), dict(mom_period=0), dict(squeeze_lookback=0),
        dict(bb_std=0.0), dict(kc_mult=0.0), dict(exit_atr=-0.1), dict(vol_filter=-1.0),
    ])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            SqueezeParams(**bad)

    def test_grid_has_valid_cells(self):
        combos = SqueezeBreakout.param_combinations()
        assert 6 <= len(combos) <= 60

    def test_wider_keltner_squeezes_more_often(self):
        """kc_mult controls how easily BB fits inside KC; extremes give no / always squeeze -> no entries."""
        c = synth.range_then_breakout(500, seed=3)
        normal, _ = simulate_long_flat(SqueezeBreakout(), c)
        never, _ = simulate_long_flat(SqueezeBreakout(kc_mult=0.05), c)
        assert len(normal) >= 2 and len(never) == 0


class TestSyntheticEdge:
    def test_profits_on_range_then_breakout_cycles(self):
        c = synth.range_then_breakout(500, seed=8)
        r = run_backtest(SqueezeBreakout(), c)
        assert r.report.total_return > 0.3 and r.report.sharpe > 1.0

    def test_entries_follow_compression(self):
        """Every entry decision on a cyclic series is a fresh release: the squeeze flag is off at entry."""
        from research.tests.helpers import decide
        c = synth.range_then_breakout(500, seed=8)
        s = SqueezeBreakout()
        pos, entries = False, 0
        for i in range(s.min_bars - 1, len(c)):
            d = decide(s, c, i, pos)
            if d.action.value == "BUY" and not pos:
                pos, entries = True, entries + 1
                assert d.indicators["squeeze_now"] == 0.0
            elif d.action.value == "SELL":
                pos = False
        assert entries >= 2

    def test_stays_out_of_persistent_downtrend(self):
        c = synth.trending(300, -0.004, 0.006, seed=2)
        r = run_backtest(SqueezeBreakout(), c)
        assert r.report.avg_exposure < 0.10 and r.report.max_drawdown < 0.10

    def test_squeeze_edge_is_specific_to_compression_cycles(self):
        """Honest negative: on mean-reverting data the same rules have no edge (modest hit rate, losses)."""
        c = synth.mean_reverting(500, half_life=8.0, sigma=0.012, seed=3)
        r = run_backtest(SqueezeBreakout(), c)
        assert not (r.report.total_return > 0.1 and r.report.win_rate > 0.6)
        cyc = run_backtest(SqueezeBreakout(), synth.range_then_breakout(500, seed=8))
        assert cyc.report.total_return > r.report.total_return


class TestBoundaries:
    def test_squeeze_memory_boundary_is_exact(self):
        """The squeeze was last ON 2 bars before the decision bar: lookback=2 sees it, lookback=1 not."""
        closes = QUIET + [100.5, 101.5, 103.0, 105.0, 107.0]
        assert kind(SqueezeBreakout(squeeze_lookback=2), closes) == "ENTER"
        assert kind(SqueezeBreakout(squeeze_lookback=1), closes) == "HOLD"

    def test_volume_baseline_excludes_the_decision_bar(self):
        """Average of the 20 PRIOR bars is 1.0e6; the release bar's own 1.25e6 must not lift it."""
        closes = QUIET + [100.5, 101.5, 103.0, 105.0]
        vol = np.full(len(closes), 1e6)
        vol[-1] = 1.25e6
        b = mk(closes)
        bars = Bars("T", list(range(len(closes))), b.open, b.high, b.low, b.close, vol)
        assert SqueezeBreakout(vol_filter=1.24).signal(bars, False).kind.value == "ENTER"
        assert SqueezeBreakout(vol_filter=1.26).signal(bars, False).kind.value == "HOLD"
