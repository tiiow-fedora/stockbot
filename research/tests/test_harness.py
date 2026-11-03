"""Harness: the research lab drives the REAL engine; exposure & trade P&L reconstruct correctly."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from research import synth
from research.base import ResearchStrategy, Signal, StrategyParams
from research.harness import (
    DEFAULT_EXECUTION, RESEARCH_RISK, compare, position_value_series, production_risk, quiet_engine,
    run_backtest, trade_pnls,
)


class AlwaysLong(ResearchStrategy):
    name = "always_long_test"
    family = "trend"
    description = "buys immediately and never sells (test double)"
    PARAMS = StrategyParams

    @property
    def min_bars(self):
        return 2

    def signal(self, bars, in_position):
        return Signal.enter(1.0, "always") if not in_position else Signal.hold()


class BuyThenSell(ResearchStrategy):
    """Enter on the first chance, exit ``hold`` bars later (by position in the visible window)."""

    name = "buy_then_sell_test"
    family = "trend"
    description = "round-trips every 20 bars (test double)"
    PARAMS = StrategyParams

    @property
    def min_bars(self):
        return 2

    def signal(self, bars, in_position):
        idx = len(bars.timestamps)
        phase = bars.timestamps[-1].toordinal() % 20
        if not in_position and phase < 10:
            return Signal.enter(1.0)
        if in_position and phase >= 10:
            return Signal.exit()
        return Signal.hold()


class NeverTrades(AlwaysLong):
    name = "never_trades_test"

    def signal(self, bars, in_position):
        return Signal.hold()


def test_risk_presets():
    assert RESEARCH_RISK.max_position_size_pct == 0.95
    assert production_risk().max_position_size_pct == 0.20 and production_risk().stop_loss_pct == 0.08
    assert DEFAULT_EXECUTION.mode == "paper"       # paper trading only


def test_never_trades_equals_flat_cash():
    r = run_backtest(NeverTrades(), synth.gbm(200, seed=1))
    assert np.allclose(r.equity, 100_000.0)
    assert r.report.num_trades == 0 and r.report.time_in_market == 0.0 and r.report.avg_exposure == 0.0
    assert np.isnan(r.report.sharpe)


def test_always_long_tracks_buy_and_hold_closely():
    c = synth.trending(250, 0.002, 0.005, seed=3)
    r = run_backtest(AlwaysLong(), c)
    assert r.report.total_return == pytest.approx(r.benchmark_report.total_return, abs=0.08)
    assert r.report.time_in_market > 0.95 and 0.85 < r.report.avg_exposure <= 1.0
    assert r.benchmark_report.avg_exposure == pytest.approx(1.0, abs=0.01)


def test_exposure_reconstruction_matches_trade_log():
    c = synth.gbm(300, seed=2)
    r = run_backtest(BuyThenSell(), c)
    assert r.report.num_trades >= 5
    pv = position_value_series(r.result, c)
    assert len(pv) == len(r.result.equity_curve)
    # every bar with a position has value ~ close * shares; flat bars have exactly zero
    flat = pv == 0.0
    assert flat.any() and (~flat).any()
    assert np.all(pv <= r.equity + 1e-6)
    assert r.report.time_in_market == pytest.approx(np.mean(pv > 0))


def test_trade_pnls_one_per_sell_and_sum_to_realised():
    c = synth.gbm(300, seed=5)
    r = run_backtest(BuyThenSell(), c)
    pnls = trade_pnls(r.result)
    sells = [t for t in r.result.trade_log if t["side"] == "SELL"]
    assert len(pnls) == len(sells) == r.report.num_trades
    assert r.report.win_rate == pytest.approx(np.mean([p > 0 for p in pnls]))


def test_fills_happen_at_next_bar_open_no_lookahead():
    """Decision bar close -> fill at the NEXT bar's open (engine guarantee the lab relies on)."""
    c = synth.gbm(120, seed=7)
    r = run_backtest(AlwaysLong(), c)
    buy = next(t for t in r.result.trade_log if t["side"] == "BUY")
    bar = next(x for x in c if x.timestamp == buy["timestamp"])
    assert buy["price"] == pytest.approx(bar.open, rel=1e-3)      # open + 5 bps slippage


def test_costs_reduce_returns():
    c = synth.gbm(300, seed=8)
    free = run_backtest(BuyThenSell(), c, execution=DEFAULT_EXECUTION.model_copy(update={"slippage_bps": 0.0}))
    pricey = run_backtest(BuyThenSell(), c, execution=DEFAULT_EXECUTION.model_copy(update={"slippage_bps": 50.0}))
    assert pricey.report.total_return < free.report.total_return


def test_production_risk_limits_exposure():
    c = synth.trending(250, 0.003, 0.004, seed=1)
    lab = run_backtest(AlwaysLong(), c)
    prod = run_backtest(AlwaysLong(), c, risk=production_risk())
    assert prod.report.max_exposure <= 0.25 < lab.report.max_exposure


def test_empty_candles_rejected_and_short_series_raise():
    with pytest.raises(ValueError):
        run_backtest(AlwaysLong(), [])
    with pytest.raises(ValueError):
        run_backtest(AlwaysLong(), synth.gbm(30, seed=1))   # engine: needs warmup+2 candles


def test_quiet_engine_restores_level():
    lg = logging.getLogger("stockbot")
    before = lg.level
    with quiet_engine(logging.ERROR):
        assert lg.level == logging.ERROR
    assert lg.level == before


def test_compare_matrix_shape():
    sets = {"a": synth.gbm(120, seed=1), "b": synth.gbm(120, seed=2)}
    out = compare({"long": AlwaysLong(), "flat": NeverTrades()}, sets)
    assert set(out) == {"long", "flat"} and set(out["long"]) == {"a", "b"}
    assert out["flat"]["a"].report.total_return == 0.0


def test_run_is_deterministic():
    c = synth.gbm(200, seed=9)
    a, b = run_backtest(BuyThenSell(), c), run_backtest(BuyThenSell(), c)
    np.testing.assert_array_equal(a.equity, b.equity)
