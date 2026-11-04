"""Property-style fuzz on the numeric core. Aimed at the bug class that has
bitten this project before: a unit/scale mismatch that silently suppresses
every BUY (the sizer says 0 shares, or the risk gate vetoes what the sizer
proposed) with no error anywhere."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import pytest

from analysis import indicators as ind
from core.config import RiskConfig
from core.models import Action, Order, OrderSide
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager

NOW = datetime(2024, 3, 12, 15, tzinfo=timezone.utc)


def rand_series(rng, n):
    kind = rng.choice(["walk", "flat", "spiky", "tiny", "huge"])
    p = {"tiny": 0.01, "huge": 5e5}.get(kind, rng.uniform(5, 500))
    out = []
    for _ in range(n):
        if kind == "walk":
            p *= 1 + rng.gauss(0, 0.02)
        elif kind == "spiky":
            p *= rng.choice([0.5, 1, 1, 2])
        p = max(p, 1e-4)
        out.append(p)
    return out


@pytest.mark.parametrize("seed", range(40))
def test_indicators_are_finite_and_in_range_on_arbitrary_series(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 5, 14, 15, 30, 60, 120])
    closes = rand_series(rng, n)
    vols = [rng.choice([0, 0, 1, 10, 10**6, 10**9]) for _ in range(n)]
    for name, series in {
        "sma": ind.sma(closes, 10), "ema": ind.ema(closes, 12), "rsi": ind.rsi(closes, 14),
        "mom": ind.momentum(closes, 10), "vol": ind.volume_change(vols, 10), "volat": ind.volatility(closes, 20),
    }.items():
        assert len(series) == n, name
        for v in series:
            assert v is None or math.isfinite(v), (name, v)
    for v in ind.rsi(closes, 14):
        assert v is None or 0.0 <= v <= 100.0


@pytest.mark.parametrize("seed", range(60))
def test_strategies_emit_valid_decisions_on_arbitrary_series(seed):
    from core.config import StrategyConfig
    from core.models import Candle, MarketSnapshot
    from strategy.registry import build_strategy_by_name

    rng = random.Random(seed)
    n = rng.choice([0, 3, 34, 35, 36, 80, 200])
    closes = rand_series(rng, n + 1)
    from datetime import timedelta

    candles = [Candle("X", NOW - timedelta(days=n - i + 1), c, c * 1.01, c * 0.99, c, rng.choice([0, 1000, 10**7]))
               for i, c in enumerate(closes[:-1])]
    snap = MarketSnapshot("X", NOW, closes[-1], rng.choice([0, 500, 10**7]))
    for name in ("technical", "mean_reversion"):
        s = build_strategy_by_name(name, StrategyConfig(name=name))
        d = s.generate_decision("X", snap, candles, [], 0.0, 100_000.0, 100_000.0)
        assert d.action in Action
        assert -1.0 <= d.scores.composite <= 1.0 and math.isfinite(d.confidence)
        if n + 1 < 35:
            assert d.action == Action.HOLD       # never trades on too little history


@pytest.mark.parametrize("seed", range(300))
def test_whatever_the_sizer_proposes_the_risk_gate_must_accept(seed):
    """If size_position says 'buy N' and evaluate_order then says no, the BUY is
    silently lost. Allowed vetoes are only the ones the sizer cannot know about
    cheaply (open-position count, daily trade count)."""
    rng = random.Random(seed)
    cfg = RiskConfig(
        max_position_size_pct=rng.choice([0.05, 0.1, 0.2, 0.5, 1.0]),
        max_portfolio_exposure_pct=rng.choice([0.3, 0.5, 0.8, 1.0]),
        min_cash_reserve_pct=rng.choice([0.0, 0.05, 0.2]),
        max_open_positions=rng.choice([2, 8, 50]),
    )
    rm = RiskManager(cfg)
    pf = Portfolio(rng.choice([1_000.0, 25_000.0, 100_000.0, 5e6]))
    prices = {s: round(rng.uniform(2, 900), 2) for s in "ABCDEFG"}
    # build a random starting book through legal fills
    from core.models import Fill

    for s in rng.sample(list(prices), rng.randint(0, 5)):
        q = float(rng.randint(1, 40))
        try:
            pf.apply_fill(Fill(Order(s, OrderSide.BUY, q), prices[s], q, NOW))
        except Exception:
            pass
    sym = rng.choice(list(prices))
    qty = rm.size_position(sym, prices[sym], pf, prices, rng.uniform(0, 1))
    if qty <= 0:
        return
    decision = rm.evaluate_order(Order(sym, OrderSide.BUY, qty, timestamp=NOW), pf, prices, now=NOW)
    allowed = {"max_open_positions", "max_trade_frequency"}
    assert decision.approved or decision.reason in allowed, (
        f"sizer proposed {qty} {sym}@{prices[sym]} but risk gate said {decision.reason}; "
        f"cfg={cfg} cash={pf.cash} positions={ {k: v.quantity for k, v in pf.positions.items()} }"
    )


def test_near_the_exposure_cap_a_buy_is_scaled_down_not_vetoed():
    """BUG-17 (fixed): with default limits and 70% already invested, the sizer
    used to propose a full 20%-of-portfolio order that the 80% exposure cap then
    rejected, so the bot silently stopped buying ~10% short of its own limit."""
    from core.models import Fill

    rm, pf = RiskManager(RiskConfig()), Portfolio(100_000.0)
    prices = {s: 100.0 for s in "ABCDEFG"}
    for s in "ABC":                                   # 3 x 20% = 60%
        pf.apply_fill(Fill(Order(s, OrderSide.BUY, 200), 100.0, 200.0, NOW))
    pf.apply_fill(Fill(Order("D", OrderSide.BUY, 100), 100.0, 100.0, NOW))   # +10% -> 70%
    qty = rm.size_position("E", 100.0, pf, prices, confidence=1.0)
    assert 0 < qty <= 100                              # only ~10% of room left under the 80% cap
    assert rm.evaluate_order(Order("E", OrderSide.BUY, qty, timestamp=NOW), pf, prices, now=NOW).approved


# -- config unit mistakes must fail loudly, not silently disable trading ------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_cash_reserve_pct": 5},        # "5%" typed as 5 -> every BUY sizes to 0
        {"max_position_size_pct": 20},      # "20%" typed as 20 -> no cap
        {"max_portfolio_exposure_pct": 80},
        {"stop_loss_pct": 8},               # never fires
        {"stop_loss_pct": 0},
        {"max_position_size_pct": -0.2},
    ],
)
def test_risk_config_rejects_percent_vs_fraction_mistakes(kwargs):
    """BUG-18 (fixed): RiskConfig accepted any number. Writing percents instead
    of fractions silently zeroed or uncapped trading."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RiskConfig(**kwargs)


def test_shipped_config_and_defaults_still_validate():
    from core.config import ExecutionConfig, load_config

    cfg = load_config()
    assert 0 < cfg.risk.min_cash_reserve_pct < 1
    ExecutionConfig(slippage_bps=5)
    with pytest.raises(Exception):
        ExecutionConfig(slippage_bps=-5)
    with pytest.raises(Exception):
        ExecutionConfig(commission_pct=1.5)


# -- backtester under hostile series ----------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("strategy", ["technical", "mean_reversion"])
def test_backtester_survives_gappy_jumpy_series(seed, strategy):
    from datetime import timedelta

    from backtesting.engine import Backtester
    from core.config import ExecutionConfig, StrategyConfig
    from core.models import Candle
    from strategy.registry import build_strategy_by_name

    rng = random.Random(seed)
    price, ts, candles = 100.0, datetime(2023, 1, 2, tzinfo=timezone.utc), []
    for _ in range(rng.choice([45, 90, 250])):
        move = rng.gauss(0, 0.02) + (rng.choice([-0.35, 0.4]) if rng.random() < 0.03 else 0)
        o = price * (1 + rng.gauss(0, 0.01))
        c = max(0.05, price * (1 + move))
        hi, lo = max(o, c) * 1.01, min(o, c) * 0.99
        candles.append(Candle("X", ts, o, hi, lo, c, rng.choice([0, 10**5, 10**7])))
        price = c
        ts += timedelta(days=rng.choice([1, 1, 1, 1, 3, 7]))      # weekends / multi-day holes
    bt = Backtester(build_strategy_by_name(strategy, StrategyConfig(name=strategy)), RiskConfig(),
                    ExecutionConfig(slippage_bps=rng.choice([0, 5, 50]), commission_per_trade=rng.choice([0, 1])),
                    warmup_bars=40)
    res = bt.run("X", candles)
    assert math.isfinite(res.total_return_pct) and res.ending_balance > 0
    assert all(v > 0 for _, v in res.equity_curve)
