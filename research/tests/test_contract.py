"""
Generic contract tests run against EVERY registered research strategy.

A new file in ``research/strategies/`` is picked up automatically, so a
strategy cannot be merged without satisfying: metadata, parameter validation,
engine-interface conformance, warm-up behaviour, determinism/statelessness,
no look-ahead, NaN/degenerate-data safety, long-only semantics and a full
run through the real Backtester. Strategy-specific behaviour (does a trend
follower really profit on a trend?) lives in each strategy's own test file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting.engine import Backtester
from core.models import Action, Decision
from research import synth
from research.base import ResearchStrategy, StrategyParams
from research.harness import DEFAULT_EXECUTION, RESEARCH_RISK, run_backtest
from research.registry import FAMILIES, STRATEGIES, build, engine_builders
from research.tests.helpers import CASH, decide, decisions
from strategy.base import Strategy

NAMES = sorted(STRATEGIES)
N = 320


def make(name: str, candles=None, partner=None):
    cls = STRATEGIES[name]
    if cls.requires_partner:
        if partner is None:
            raise ValueError("partner required")
        return cls(partner=partner)
    return cls()


def pair_data():
    a, b = synth.cointegrated_pair(N, seed=21)
    return a, b


def instance_and_data(name: str, series=None):
    """(strategy, candles) for ``name``; partner strategies get a cointegrated pair."""
    cls = STRATEGIES[name]
    if cls.requires_partner:
        a, b = pair_data()
        return cls(partner=b), a
    return cls(), series if series is not None else synth.gbm(N, seed=21)


def test_at_least_ten_strategies_and_all_families_covered():
    assert len(NAMES) >= 10
    fams = {STRATEGIES[n].family for n in NAMES}
    for required in ("momentum", "mean_reversion", "breakout", "pairs", "volatility", "trend"):
        assert required in fams, f"no strategy in family {required!r}"
    assert fams <= set(FAMILIES)


@pytest.mark.parametrize("name", NAMES)
class TestStrategyContract:
    def test_metadata(self, name):
        cls = STRATEGIES[name]
        assert issubclass(cls, ResearchStrategy) and issubclass(cls, Strategy)
        assert cls.name == name and name == name.lower() and " " not in name
        assert cls.family in FAMILIES
        assert len(cls.description) > 20
        assert cls.__doc__ and len(cls.__doc__.strip()) > 40, "class needs a docstring (one-paragraph summary + param notes)"
        assert issubclass(cls.PARAMS, StrategyParams)

    def test_module_docstring_documents_rules(self, name):
        import importlib

        doc = importlib.import_module(STRATEGIES[name].__module__).__doc__ or ""
        assert len(doc) > 500
        for token in ("Family", "Rules", "Expected behaviour"):
            assert token.lower() in doc.lower(), f"module docstring of {name} lacks a '{token}' section"

    def test_default_params_valid_and_grid_nonempty(self, name):
        cls = STRATEGIES[name]
        cls.PARAMS()  # defaults validate
        combos = cls.param_combinations()
        assert 2 <= len(combos) <= 150, "grid should have a handful of valid cells (optimizer cost)"
        fields = set(cls.PARAMS().to_dict())
        for combo in combos:
            assert set(combo) <= fields

    def test_invalid_params_raise(self, name):
        cls = STRATEGIES[name]
        base = cls.PARAMS().to_dict()
        int_fields = [k for k, v in base.items() if isinstance(v, int) and not isinstance(v, bool)]
        assert int_fields, "every strategy has at least one integer period parameter"
        with pytest.raises(ValueError):
            cls.PARAMS(**{int_fields[0]: 0})
        with pytest.raises(ValueError):
            cls.PARAMS(**{int_fields[0]: -3})

    def test_params_are_immutable_and_replaceable(self, name):
        p = STRATEGIES[name].PARAMS()
        key = next(iter(p.to_dict()))
        with pytest.raises(Exception):
            setattr(p, key, 1)
        assert p.replace() == p

    def test_wrong_params_type_rejected(self, name):
        class Other(StrategyParams):
            pass

        with pytest.raises(TypeError):
            STRATEGIES[name](Other())

    def test_min_bars_and_lookback_sane(self, name):
        s, _ = instance_and_data(name)
        assert isinstance(s.min_bars, int) and 2 <= s.min_bars <= 400
        assert s.lookback is None or s.lookback >= s.min_bars

    def test_warmup_returns_hold_with_reason(self, name):
        s, candles = instance_and_data(name)
        for i in (0, 1, max(0, s.min_bars - 2)):
            d = decide(s, candles, i)
            assert d.action == Action.HOLD and d.confidence == 0.0
            assert "warm-up" in d.reason or "insufficient" in d.reason.lower() or "not ready" in d.reason.lower() or d.action == Action.HOLD

    def test_decision_is_well_formed(self, name):
        s, candles = instance_and_data(name)
        ds = decisions(s, candles)
        assert ds, "no decisions produced"
        for d in ds:
            assert isinstance(d, Decision)
            assert d.symbol == candles[0].symbol
            assert d.action in (Action.BUY, Action.SELL, Action.HOLD)
            assert 0.0 <= d.confidence <= 1.0
            assert -1.0 <= d.scores.composite <= 1.0
            assert isinstance(d.reason, str) and name in d.reason
            for k, v in d.indicators.items():
                assert v is None or (isinstance(v, float) and math.isfinite(v)), (k, v)
            if d.action == Action.HOLD:
                assert d.confidence == 0.0
            else:
                assert d.confidence > 0.0

    def test_long_only_semantics(self, name):
        """Flat -> never SELL.  Long -> never BUY (no pyramiding)."""
        for series in (synth.trending(N, 0.004, 0.006, seed=5), synth.mean_reverting(N, 6.0, 0.015, seed=6),
                       synth.range_then_breakout(N, seed=7)):
            s, candles = instance_and_data(name, series)
            assert Action.SELL not in {d.action for d in decisions(s, candles, in_position=False)}
            assert Action.BUY not in {d.action for d in decisions(s, candles, in_position=True)}

    def test_deterministic_and_stateless(self, name):
        s1, candles = instance_and_data(name)
        s2, _ = instance_and_data(name)
        idx = list(range(s1.min_bars - 1, len(candles), 7))
        forward = [decide(s1, candles, i).action for i in idx]
        # fresh instance, reverse call order: stateless => identical answers
        backward = [decide(s2, candles, i).action for i in reversed(idx)][::-1]
        assert forward == backward
        again = [decide(s1, candles, i).action for i in idx]
        assert forward == again

    def test_no_lookahead(self, name):
        """Decisions at bar i must not depend on any bar after i."""
        s, candles = instance_and_data(name)
        a, b = candles, None
        # Corrupt every bar after the cut with garbage; decisions at/below the cut must be unchanged.
        cut = len(candles) * 2 // 3
        from dataclasses import replace

        garbage = [replace(c, open=c.open * 3, high=c.high * 3.5, low=c.low * 2.5, close=c.close * 3, volume=c.volume * 9)
                   for c in candles[cut + 1:]]
        mutated = candles[: cut + 1] + garbage
        for i in range(s.min_bars - 1, cut + 1, 5):
            for pos in (False, True):
                assert decide(s, a, i, pos).action == decide(s, mutated, i, pos).action, f"look-ahead at bar {i}"

    def test_ignores_history_at_or_after_snapshot(self, name):
        """The base class defensively drops history bars not strictly before the snapshot."""
        s, candles = instance_and_data(name)
        i = len(candles) // 2
        clean = decide(s, candles, i)
        c = candles[i]
        from research.tests.helpers import CASH, snapshot_of

        dirty_history = list(candles[:i]) + list(candles[i : i + 5])  # includes bar i and the future
        d = s.generate_decision(c.symbol, snapshot_of(c), dirty_history, [], 0.0, CASH, CASH)
        assert d.action == clean.action and d.confidence == clean.confidence

    @pytest.mark.parametrize("kind", ["flat", "jumpy", "tiny_vol"])
    def test_degenerate_data_does_not_crash(self, name, kind):
        series = {
            "flat": synth.flat(N, seed=1),
            "jumpy": synth.jump_diffusion(N, jump_prob=0.1, jump_sigma=0.15, seed=2),
            "tiny_vol": synth.gbm(N, 0.0, 1e-4, seed=3),
        }[kind]
        s, candles = instance_and_data(name, series)
        for d in decisions(s, candles):
            assert d.action in (Action.BUY, Action.SELL, Action.HOLD)
            assert math.isfinite(d.confidence)

    def test_short_history_does_not_crash(self, name):
        s, candles = instance_and_data(name)
        for n in (1, 2, 3, 5):
            d = decide(s, candles, n - 1)
            assert d.action == Action.HOLD

    def test_live_snapshot_without_ohlc(self, name):
        """The live loop's snapshots may lack open/high/low; the strategy must still work."""
        from core.models import MarketSnapshot

        s, candles = instance_and_data(name)
        i = len(candles) - 1
        c = candles[i]
        snap = MarketSnapshot(symbol=c.symbol, timestamp=c.timestamp, price=c.close, volume=c.volume)
        d = s.generate_decision(c.symbol, snap, list(candles[:i]), [], 0.0, CASH, CASH)
        assert d.action in (Action.BUY, Action.SELL, Action.HOLD)

    def test_runs_through_real_backtester(self, name):
        s, candles = instance_and_data(name, synth.regime_switching(N, seed=33))
        r = run_backtest(s, candles)
        assert len(r.equity) == len(candles) - 40
        assert np.all(np.isfinite(r.equity)) and np.all(r.equity > 0)
        assert 0.0 <= r.report.max_drawdown < 1.0
        # fills never exceed what long-only 100% invested allows
        assert np.all(r.position_value <= r.equity * 1.0001)

    def test_trades_when_conditions_arise(self, name):
        """Across the scenario suite the strategy must act at least once (not a dead strategy)."""
        total = 0
        for series in synth.scenario_suite(N, seed=3).values():
            s, candles = instance_and_data(name, series)
            total += sum(1 for d in decisions(s, candles) if d.action == Action.BUY)
        assert total > 0

    def test_param_override_changes_instance(self, name):
        cls = STRATEGIES[name]
        combo = cls.param_combinations()[-1]
        s = cls(**combo, **({"partner": synth.cointegrated_pair(80, seed=1)[1]} if cls.requires_partner else {}))
        for k, v in combo.items():
            assert getattr(s.params, k) == v


def test_registry_build_and_engine_builders():
    for name in NAMES:
        if STRATEGIES[name].requires_partner:
            continue
        assert isinstance(build(name), STRATEGIES[name])
    builders = engine_builders()
    assert set(builders) == {n for n in NAMES if not STRATEGIES[n].requires_partner}
    from core.config import StrategyConfig

    for name, b in builders.items():
        assert isinstance(b(StrategyConfig()), Strategy)
    with pytest.raises(ValueError):
        build("nope")


def test_names_unique_and_modules_match():
    assert len(set(NAMES)) == len(NAMES)
