from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backtest import (
    BacktestHarness,
    compute_trade_stats,
    generate_scenario,
    load_report,
    max_drawdown_detail,
    save_report,
)
from backtest.__main__ import main as cli_main
from backtest.loaders import load_csv_series, synthetic_series
from backtest.scenarios import ALL_SCENARIOS
from backtest.web import create_app
from core.config import StrategyConfig
from core.models import PortfolioSnapshot
from database.database import DatabaseManager
from strategy.registry import build_strategy_by_name
from datetime import datetime, timezone


def _harness():
    return BacktestHarness(build_strategy_by_name("technical", StrategyConfig()), "technical")


def test_scenarios_are_deterministic_and_valid():
    for name in ALL_SCENARIOS:
        a = generate_scenario(name, n=80, seed=3)
        b = generate_scenario(name, n=80, seed=3)
        assert a == b and len(a) == 80
        assert all(c.timestamp.weekday() < 5 for c in a)
    assert generate_scenario("bull", n=80, seed=3) != generate_scenario("bull", n=80, seed=4)
    with pytest.raises(ValueError):
        generate_scenario("nope")


def test_scenario_regimes_trend_as_named():
    bull = generate_scenario("bull", n=250)
    bear = generate_scenario("bear", n=250)
    assert bull[-1].close > bull[0].close
    assert bear[-1].close < bear[0].close


def test_trade_stats():
    log = [
        {"side": "BUY", "realized_pnl": None},
        {"side": "SELL", "realized_pnl": 100.0},
        {"side": "SELL", "realized_pnl": -40.0},
        {"side": "SELL", "realized_pnl": 20.0},
    ]
    s = compute_trade_stats(log)
    assert (s.wins, s.losses, s.num_round_trips) == (2, 1, 3)
    assert s.win_rate == pytest.approx(2 / 3)
    assert s.profit_factor == pytest.approx(3.0)
    assert s.net_pnl == pytest.approx(80.0)
    assert s.worst_trade == -40.0


def test_trade_stats_empty_and_no_losses():
    s = compute_trade_stats([])
    assert s.num_round_trips == 0 and s.win_rate == 0.0 and s.profit_factor is None
    assert compute_trade_stats([{"side": "SELL", "realized_pnl": 5.0}]).profit_factor is None


def test_max_drawdown_detail():
    t = [datetime(2024, 1, d, tzinfo=timezone.utc) for d in range(1, 6)]
    d = max_drawdown_detail(list(zip(t, [100, 120, 60, 90, 130])))
    assert d["max_drawdown_pct"] == pytest.approx(0.5)
    assert d["max_drawdown_abs"] == 60
    assert d["peak_time"] == t[1] and d["trough_time"] == t[2]
    assert max_drawdown_detail([])["max_drawdown_pct"] == 0.0


def test_harness_report_is_consistent():
    series = synthetic_series(["bull", "crash"], n=200)
    res = _harness().run(series, name="t")
    r = res.report
    assert len(r.per_symbol) == 2
    assert r.starting_balance == 200_000.0
    assert r.total_pnl == pytest.approx(sum(s["pnl"] for s in r.per_symbol))
    assert 0.0 <= r.max_drawdown_pct <= 1.0
    assert 0.0 <= r.win_rate <= 1.0
    assert r.num_trades == sum(s["num_trades"] for s in r.per_symbol)
    assert r.equity_curve and r.trades == sorted(r.trades, key=lambda t: t["timestamp"])


def test_harness_deterministic():
    a = _harness().run(synthetic_series(["sideways"], n=150)).report
    b = _harness().run(synthetic_series(["sideways"], n=150)).report
    assert (a.total_pnl, a.num_trades, a.equity_curve) == (b.total_pnl, b.num_trades, b.equity_curve)


def test_harness_rejects_empty_and_short_series():
    with pytest.raises(ValueError):
        _harness().run({})
    with pytest.raises(ValueError):
        _harness().run({"x": generate_scenario("bull", n=10)})


def test_report_roundtrip(tmp_path):
    rep = _harness().run(synthetic_series(["bull"], n=120), name="my run/1").report
    path = save_report(rep, str(tmp_path))
    assert path.parent == tmp_path and "/" not in path.name.replace(str(tmp_path), "")
    assert load_report(str(path)) == rep


def test_csv_loader_and_cli(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    candles = generate_scenario("bull", symbol="ABC", n=120)
    with open(csv_dir / "ABC.csv", "w") as fh:
        fh.write("timestamp,open,high,low,close,volume\n")
        for c in candles:
            fh.write(f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}\n")
    series = load_csv_series(str(csv_dir), ["ABC"])
    assert len(series["ABC"]) == 120
    out = tmp_path / "out"
    rc = cli_main(["--csv-dir", str(csv_dir), "--symbols", "ABC", "--out", str(out), "--name", "csvrun"])
    assert rc == 0 and (out / "csvrun.json").exists()


def test_cli_synthetic(tmp_path, capsys):
    assert cli_main(["--synthetic", "--scenarios", "bull", "--bars", "120", "--out", str(tmp_path)]) == 0
    assert "saved" in capsys.readouterr().out


# --- dashboard -------------------------------------------------------------

@pytest.fixture
def dash(tmp_path):
    reports = tmp_path / "reports"
    save_report(_harness().run(synthetic_series(["bull"], n=120), name="alpha").report, str(reports))
    db_url = f"sqlite:///{tmp_path}/d.db"
    return TestClient(create_app(str(reports), db_url)), db_url, tmp_path


def test_dashboard_pages_and_reports(dash):
    client, _, _ = dash
    assert "StockBot" in client.get("/").text
    listing = client.get("/api/reports").json()
    assert [r["id"] for r in listing] == ["alpha"]
    detail = client.get("/api/reports/alpha").json()
    assert detail["name"] == "alpha" and detail["equity_curve"]


def test_dashboard_unknown_and_traversal_404(dash):
    client, _, tmp_path = dash
    (tmp_path / "secret.json").write_text("{}")
    assert client.get("/api/reports/missing").status_code == 404
    assert client.get("/api/reports/..%2Fsecret").status_code == 404
    assert client.get("/api/reports/%2e%2e/secret").status_code == 404


def test_dashboard_is_read_only(dash):
    client, _, _ = dash
    for path in ("/", "/api/reports", "/api/positions", "/api/reports/alpha"):
        assert client.post(path).status_code == 405
        assert client.delete(path).status_code == 405


def test_dashboard_positions_from_db(dash):
    client, db_url, _ = dash
    assert client.get("/api/positions").json()["positions"] == {}
    db = DatabaseManager(db_url)
    db.save_portfolio_snapshot(
        PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash=90_000.0,
            positions_value=10_000.0,
            total_value=100_000.0,
            positions={"AAPL": {"quantity": 50, "avg_entry_price": 190.0, "market_value": 10_000.0, "unrealized_pnl": 500.0}},
        )
    )
    p = client.get("/api/positions").json()
    assert p["positions"]["AAPL"]["quantity"] == 50 and p["total_value"] == 100_000.0


def test_dashboard_skips_corrupt_reports(dash):
    client, _, tmp_path = dash
    (tmp_path / "reports" / "bad.json").write_text("{not json")
    assert [r["id"] for r in client.get("/api/reports").json()] == ["alpha"]
