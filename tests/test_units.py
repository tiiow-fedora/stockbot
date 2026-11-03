"""Regression tests for unit mismatches (volume / price / quantity / percent)
across signal generation. The original production bug: a snapshot's volume
was a single trade's size (~100 shares) compared against daily volumes in the
millions, so every cycle read as a -100% volume collapse and BUYs were
suppressed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from core.models import Candle, MarketSnapshot, Order, OrderSide
from data.market_calendar import project_full_day_volume, session_fraction_elapsed
from data.market_data import completed_bars
from execution.paper import PaperExecutor
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager
from strategy.mean_reversion import MeanReversionStrategy
from strategy.scoring import ScoringEngine
from strategy.technical import TechnicalStrategy
from tests.conftest import make_candles

NOW = datetime(2026, 11, 3, 20, 0, tzinfo=timezone.utc)


def _snap(candles, volume, price=None):
    last = candles[-1]
    return MarketSnapshot(
        symbol="TEST", timestamp=last.timestamp + timedelta(days=1),
        price=price if price is not None else last.close, volume=volume,
    )


def _strategies():
    cfg = StrategyConfig()
    eng = ScoringEngine(cfg.weights, cfg.buy_threshold, cfg.sell_threshold)
    return [TechnicalStrategy(cfg, eng), MeanReversionStrategy(cfg, eng)]


def _decide(strategy, history, snap):
    return strategy.generate_decision("TEST", snap, history, [], 0.0, 100_000.0, 100_000.0)


@pytest.mark.parametrize("strategy", _strategies(), ids=["technical", "mean_reversion"])
def test_same_unit_volume_is_not_a_collapse(strategy):
    history = make_candles(n=60)  # ~1.0M shares/day
    d = _decide(strategy, history, _snap(history, volume=1_040_000))
    assert abs(d.indicators["volume_change"]) < 0.1


@pytest.mark.parametrize("strategy", _strategies(), ids=["technical", "mean_reversion"])
def test_trade_size_volume_would_read_as_collapse(strategy):
    """Documents the original bug's signature: share-count-of-one-trade vs
    daily volume gives ~-100%. Providers must never produce this."""
    history = make_candles(n=60)
    d = _decide(strategy, history, _snap(history, volume=100))
    assert d.indicators["volume_change"] < -0.99


@pytest.mark.parametrize("strategy", _strategies(), ids=["technical", "mean_reversion"])
def test_zero_volume_means_unavailable_not_collapse(strategy):
    history = make_candles(n=60)
    d = _decide(strategy, history, _snap(history, volume=0))
    assert d.indicators["volume_change"] is None
    assert d.scores.volume == 0.0


def test_volume_does_not_flip_oversold_buy_to_hold():
    """A deep dip scores BUY for mean reversion with same-unit volume."""
    history = make_candles(n=60, daily_return=-0.01)
    strat = _strategies()[1]
    d = _decide(strat, history, _snap(history, volume=1_040_000))
    assert d.action.value == "BUY"


def test_price_unit_mismatch_is_visible_in_indicators():
    """A snapshot priced in cents against dollar history is a 100x jump; the
    strategies pass it through (they can't know) but momentum exposes it, so
    providers must be dollars. Guard: MarketSnapshot/Candle reject <= 0 only."""
    history = make_candles(n=60, start_price=100.0)
    d = _decide(_strategies()[0], history, _snap(history, 1_000_000, price=history[-1].close * 100))
    assert d.indicators["momentum"] > 50  # unmistakably absurd


# -- live-loop bar handling ------------------------------------------------

def test_completed_bars_excludes_todays_forming_daily_bar():
    today_bar_ts = datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc)
    candles = [
        Candle("T", today_bar_ts - timedelta(days=2), 10, 11, 9, 10, 1_000_000),
        Candle("T", today_bar_ts - timedelta(days=1), 10, 11, 9, 10, 1_000_000),
        Candle("T", today_bar_ts, 10, 11, 9, 10, 400_000),  # partial, forming
    ]
    out = completed_bars(candles, NOW, "1Day")
    assert [c.volume for c in out] == [1_000_000, 1_000_000]


def test_completed_bars_intraday_uses_strict_timestamp():
    candles = make_candles(n=3)
    out = completed_bars(candles, candles[1].timestamp, "1Min")
    assert len(out) == 1


def test_session_fraction_and_projection():
    # 13:45 ET (EST, Nov 3 2026 is after DST ends Nov 1) = 18:45 UTC: 255/390 min elapsed
    t = datetime(2026, 11, 3, 18, 45, tzinfo=timezone.utc)
    assert session_fraction_elapsed(t) == pytest.approx(255 / 390)
    assert project_full_day_volume(650_000, t) == int(650_000 / (255 / 390))


def test_projection_is_identity_when_session_over_or_weekend():
    after_close = datetime(2026, 11, 3, 22, 0, tzinfo=timezone.utc)
    saturday = datetime(2026, 11, 7, 17, 0, tzinfo=timezone.utc)
    assert project_full_day_volume(5_000_000, after_close) == 5_000_000
    assert project_full_day_volume(5_000_000, saturday) == 5_000_000


def test_projection_caps_early_session_blowup():
    just_open = datetime(2026, 11, 3, 14, 31, tzinfo=timezone.utc)  # 9:31 ET
    assert project_full_day_volume(100_000, just_open) == 1_000_000  # capped at 10x


# -- quantity / money units ---------------------------------------------------

def test_sized_quantity_is_whole_shares_and_value_within_limit():
    rm = RiskManager(RiskConfig())
    pf = Portfolio(100_000.0)
    qty = rm.size_position("AAPL", 187.42, pf, {"AAPL": 187.42}, confidence=0.9)
    assert qty == int(qty) > 0
    assert qty * 187.42 <= 100_000 * 0.20 + 1e-6


def test_fill_cash_change_equals_quantity_times_price_not_dollars():
    ex = PaperExecutor(ExecutionConfig(slippage_bps=0.0))
    pf = Portfolio(10_000.0)
    order = Order("AAPL", OrderSide.BUY, 10, timestamp=NOW)
    pf.apply_fill(ex.execute(order, 100.0, NOW))
    assert pf.cash == pytest.approx(9_000.0)
    assert pf.position_qty("AAPL") == 10


def test_slippage_bps_is_basis_points_not_percent():
    ex = PaperExecutor(ExecutionConfig(slippage_bps=5.0))
    fill = ex.execute(Order("AAPL", OrderSide.BUY, 1, timestamp=NOW), 100.0, NOW)
    assert fill.fill_price == pytest.approx(100.05)  # 5 bps = 0.05%, not 5%


# -- config percent/fraction units ---------------------------------------------

@pytest.mark.parametrize("field", ["max_position_size_pct", "stop_loss_pct", "take_profit_pct",
                                   "max_portfolio_exposure_pct", "min_cash_reserve_pct"])
def test_risk_pct_fields_reject_whole_percent_values(field):
    with pytest.raises(ValidationError):
        RiskConfig(**{field: 8})  # meant 8%, would be 800%


def test_execution_config_rejects_unit_slips():
    with pytest.raises(ValidationError):
        ExecutionConfig(commission_pct=5)  # meant 5%? it's a fraction
    with pytest.raises(ValidationError):
        ExecutionConfig(slippage_bps=10_000)
    with pytest.raises(ValidationError):
        ExecutionConfig(commission_per_trade=-1)


def test_default_config_file_loads():
    from core.config import load_config
    assert load_config().risk.stop_loss_pct == 0.08


def test_alpaca_snapshot_projects_partial_today_volume(monkeypatch):
    """Mocked client (no network): today's forming bar volume is projected."""
    import data.providers.alpaca as alp
    from data.providers.alpaca import AlpacaMarketDataProvider

    fixed_now = datetime(2026, 11, 3, 18, 45, tzinfo=timezone.utc)  # 255/390 into session

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(alp, "datetime", FakeDT)
    prov = AlpacaMarketDataProvider.__new__(AlpacaMarketDataProvider)
    bar = Candle("T", datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc), 10, 11, 9, 10.5, 650_000)
    monkeypatch.setattr(prov, "get_historical_candles", lambda *a, **k: [bar], raising=False)
    snap = prov.get_snapshot("T")
    assert snap.volume == int(650_000 / (255 / 390))
    assert snap.price == 10.5
