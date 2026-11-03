"""Tests for the engine adapter: Bars assembly, Signal -> Decision mapping, params, grid expansion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest

from core.models import Action
from research import synth
from research.base import (
    Bars, ResearchStrategy, Signal, SignalKind, StrategyParams, clip01, expand_grid,
)
from research.tests.helpers import CASH, decide, snapshot_of
from strategy.base import Strategy


@dataclass(frozen=True)
class DummyParams(StrategyParams):
    period: int = 5
    thresh: float = 0.0

    def validate(self):
        if self.period < 1:
            raise ValueError("period >= 1")
        if self.thresh < 0:
            raise ValueError("thresh >= 0")


class Scripted(ResearchStrategy):
    """Returns whatever ``script`` says; records what it saw."""

    name = "scripted"
    family = "trend"
    description = "test double that returns a scripted Signal"
    PARAMS = DummyParams
    PARAM_GRID = {"period": [1, 3, 0, 5], "thresh": [0.0, -1.0]}

    def __init__(self, params=None, script=None, **kw):
        super().__init__(params, **kw)
        self.script = script or (lambda bars, pos: Signal.hold())
        self.seen = []

    @property
    def min_bars(self):
        return self.params.period

    def signal(self, bars, in_position):
        self.seen.append((bars, in_position))
        return self.script(bars, in_position)


@pytest.fixture
def candles():
    return synth.gbm(60, seed=1)


class TestBars:
    def test_from_history_appends_snapshot_as_last_bar(self, candles):
        snap = snapshot_of(candles[40])
        b = Bars.from_history("X", candles[:40], snap)
        assert len(b) == 41 and b.last == candles[40].close
        assert b.timestamps[-1] == candles[40].timestamp
        assert b.high[-1] == candles[40].high and b.low[-1] == candles[40].low and b.open[-1] == candles[40].open

    def test_lookback_truncates_to_most_recent(self, candles):
        b = Bars.from_history("X", candles[:40], snapshot_of(candles[40]), lookback=10)
        assert len(b) == 10 and b.timestamps[0] == candles[31].timestamp

    def test_drops_history_at_or_after_snapshot(self, candles):
        b = Bars.from_history("X", candles[:45], snapshot_of(candles[40]))
        assert len(b) == 41 and b.timestamps[-1] == candles[40].timestamp
        assert all(t <= candles[40].timestamp for t in b.timestamps)

    def test_quote_without_ohlc_uses_price(self, candles):
        from core.models import MarketSnapshot

        c = candles[40]
        snap = MarketSnapshot(symbol="X", timestamp=c.timestamp, price=123.0, volume=7)
        b = Bars.from_history("X", candles[:40], snap)
        assert b.close[-1] == b.open[-1] == b.high[-1] == b.low[-1] == 123.0 and b.volume[-1] == 7

    def test_stale_high_low_widened_to_contain_price(self, candles):
        from core.models import MarketSnapshot

        c = candles[40]
        snap = MarketSnapshot(symbol="X", timestamp=c.timestamp, price=200.0, high=150.0, low=140.0, open=145.0)
        b = Bars.from_history("X", candles[:40], snap)
        assert b.high[-1] == 200.0 and b.low[-1] == 140.0

    def test_empty_history(self, candles):
        b = Bars.from_history("X", [], snapshot_of(candles[0]))
        assert len(b) == 1

    def test_from_candles_roundtrip(self, candles):
        b = Bars.from_candles("X", candles)
        assert len(b) == len(candles) and b.close[3] == candles[3].close


class TestSignal:
    def test_constructors(self):
        assert Signal.enter(0.7, "r", rsi=3).kind is SignalKind.ENTER
        assert Signal.exit().strength == 1.0
        h = Signal.hold("why")
        assert h.kind is SignalKind.HOLD and h.strength == 0.0 and h.reason == "why"

    @pytest.mark.parametrize("x,expected", [(0.5, 0.5), (-1, 0.0), (7, 1.0), (float("nan"), 0.0), (None, 0.0)])
    def test_clip01(self, x, expected):
        assert clip01(x) == expected


class TestResolution:
    def run(self, kind, in_pos, strength=0.8):
        s = Scripted(script=lambda b, p: Signal(kind, strength, {"v": float("nan"), "w": 2}, "because"))
        return decide(s, synth.gbm(60, seed=1), 30, in_pos)

    def test_enter_when_flat_buys(self):
        d = self.run(SignalKind.ENTER, False)
        assert d.action == Action.BUY and d.confidence == 0.8 and d.scores.composite == 0.8

    def test_enter_when_long_holds(self):
        d = self.run(SignalKind.ENTER, True)
        assert d.action == Action.HOLD and d.confidence == 0.0 and d.scores.composite == 0.0

    def test_exit_when_long_sells(self):
        d = self.run(SignalKind.EXIT, True)
        assert d.action == Action.SELL and d.confidence == 0.8 and d.scores.composite == -0.8

    def test_exit_when_flat_holds(self):
        assert self.run(SignalKind.EXIT, False).action == Action.HOLD

    def test_hold_is_hold(self):
        assert self.run(SignalKind.HOLD, True).action == Action.HOLD

    def test_strength_clipped(self):
        assert self.run(SignalKind.ENTER, False, strength=5.0).confidence == 1.0

    def test_nan_indicators_become_none_and_reason_carries_name(self):
        d = self.run(SignalKind.ENTER, False)
        assert d.indicators == {"v": None, "w": 2.0}
        assert d.reason.startswith("scripted: BUY") and "because" in d.reason

    def test_decision_carries_portfolio_context(self):
        s = Scripted()
        c = synth.gbm(60, seed=1)
        d = s.generate_decision("X", snapshot_of(c[30]), c[:30], [], 12.0, 555.0, 9999.0)
        assert (d.current_position_qty, d.cash, d.portfolio_value) == (12.0, 555.0, 9999.0)
        assert d.timestamp == c[30].timestamp and d.price == c[30].close

    def test_non_signal_return_is_an_error(self):
        s = Scripted(script=lambda b, p: "BUY")
        with pytest.raises(TypeError):
            decide(s, synth.gbm(60, seed=1), 30)

    def test_warmup_never_calls_signal(self):
        s = Scripted(period=20)
        d = decide(s, synth.gbm(60, seed=1), 5)
        assert d.action == Action.HOLD and "warm-up" in d.reason and s.seen == []

    def test_position_flag_passed_through(self):
        s = Scripted()
        c = synth.gbm(60, seed=1)
        decide(s, c, 30, True)
        decide(s, c, 30, False)
        assert [p for _, p in s.seen] == [True, False]

    def test_default_lookback_bounds_bars(self):
        s = Scripted(period=5)
        assert s.lookback == 60
        decide(s, synth.gbm(300, seed=1), 250)
        assert len(s.seen[0][0]) == 60

    def test_is_engine_strategy(self):
        assert isinstance(Scripted(), Strategy)


class TestParams:
    def test_overrides_and_validation(self):
        assert Scripted(period=9).params.period == 9
        with pytest.raises(ValueError):
            Scripted(period=0)
        with pytest.raises(TypeError):
            Scripted(bogus=1)

    def test_params_object_plus_overrides(self):
        s = Scripted(DummyParams(period=4), thresh=0.5)
        assert (s.params.period, s.params.thresh) == (4, 0.5)

    def test_frozen_and_replace(self):
        p = DummyParams()
        with pytest.raises(Exception):
            p.period = 3  # type: ignore[misc]
        assert p.replace(period=7).period == 7 and p.period == 5

    def test_with_params_returns_new_instance(self):
        s = Scripted(period=3)
        t = s.with_params(period=8)
        assert t is not s and t.params.period == 8 and s.params.period == 3

    def test_wrong_param_type(self):
        @dataclass(frozen=True)
        class Other(StrategyParams):
            pass

        with pytest.raises(TypeError):
            Scripted(Other())

    def test_to_dict(self):
        assert DummyParams(period=2).to_dict() == {"period": 2, "thresh": 0.0}

    def test_param_combinations_drop_invalid(self):
        combos = Scripted.param_combinations()
        assert {"period": 0, "thresh": 0.0} not in combos
        assert all(c["thresh"] >= 0 for c in combos) and len(combos) == 3
        custom = Scripted.param_combinations({"period": [2, 4]})
        assert custom == [{"period": 2}, {"period": 4}]

    def test_repr_mentions_params(self):
        assert "period=5" in repr(Scripted())


class TestExpandGrid:
    def test_cartesian_product_order(self):
        assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
            {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]

    def test_empty_grid_is_single_empty_combo(self):
        assert expand_grid({}) == [{}]

    def test_size(self):
        assert len(expand_grid({"a": range(4), "b": range(5), "c": range(3)})) == 60


class TestLookbackEquivalence:
    """Truncating history to ``lookback`` must not change decisions when lookback covers the indicator memory."""

    def test_donchian_same_with_and_without_lookback_cap(self):
        from research.strategies.donchian_breakout import DonchianBreakout

        class Uncapped(DonchianBreakout):
            name = "donchian_uncapped_test"

            @property
            def lookback(self):
                return None

        c = synth.regime_switching(400, seed=4)
        a, b = DonchianBreakout(), Uncapped()
        for i in range(a.min_bars, 400, 3):
            for pos in (False, True):
                assert decide(a, c, i, pos).action == decide(b, c, i, pos).action
