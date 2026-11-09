from __future__ import annotations

import json
import logging
import signal
import threading
import urllib.error
import urllib.request

import pytest
from pydantic import ValidationError

from core.config import AppConfig, DataConfig, ExecutionConfig, RiskConfig, StrategyConfig
from observability import metrics
from observability.health import HealthState, start_server
from observability.jsonlog import JsonFormatter, configure_logging
from observability.shutdown import GracefulShutdown
from observability.validation import ConfigError, ensure_valid, validate_config

SYNTH = dict(data=DataConfig(provider="synthetic"))
GOOD_ENV = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}


def errs(config, env=None):
    return {i.path for i in validate_config(config, env or {}) if i.level == "error"}


# ---------------------------------------------------------------- validation
def test_default_synthetic_config_is_valid():
    assert errs(AppConfig(**SYNTH)) == set()


def test_shipped_config_is_valid():
    from core.config import load_config
    assert errs(load_config(), GOOD_ENV) == set()


def test_percent_typo_is_caught():
    # RiskConfig now rejects non-fraction values at construction time (units
    # audit); validate_config remains the backstop for configs built another way.
    with pytest.raises(ValidationError, match="max_position_size_pct"):
        RiskConfig(max_position_size_pct=20)
    cfg = AppConfig(risk=RiskConfig(), **SYNTH)
    object.__setattr__(cfg.risk, "max_position_size_pct", 20)
    assert "risk.max_position_size_pct" in errs(cfg)


def test_unknown_modes_and_providers():
    cfg = AppConfig(execution=ExecutionConfig(mode="live"), data=DataConfig(provider="ibkr"))
    assert {"execution.mode", "data.provider"} <= errs(cfg)


def test_unknown_strategy_and_threshold_order():
    cfg = AppConfig(strategy=StrategyConfig(name="nope", buy_threshold=0.1, sell_threshold=0.2), **SYNTH)
    assert {"strategy.name", "strategy.sell_threshold"} <= errs(cfg)


def test_bad_symbols():
    assert "symbols" in errs(AppConfig(symbols=["aapl"], **SYNTH))
    assert "symbols" in errs(AppConfig(symbols=[], **SYNTH))


def test_alpaca_requires_credentials_but_not_when_present():
    cfg = AppConfig(data=DataConfig(provider="alpaca"))
    assert "env" in errs(cfg, {})
    assert "env" not in errs(cfg, GOOD_ENV)


def test_dashboard_mode_does_not_require_credentials():
    """The read-only dashboard is deployed without broker keys (regression:
    it crash-looped with exit 2 on a config that mirrors to Alpaca)."""
    cfg = AppConfig(
        data=DataConfig(provider="alpaca"),
        execution=ExecutionConfig(mirror_to_alpaca=True),
    )
    issues = validate_config(cfg, {}, require_credentials=False)
    assert "env" not in {i.path for i in issues if i.level == "error"}
    ensure_valid(cfg, {}, require_credentials=False)
    # The paper-only guard still applies without credentials.
    live = {"ALPACA_PAPER": "false"}
    assert "env.ALPACA_PAPER" in {
        i.path for i in validate_config(cfg, live, require_credentials=False)
    }
    with pytest.raises(ConfigError):
        ensure_valid(cfg, {})


def test_live_alpaca_endpoint_is_rejected():
    env = {**GOOD_ENV, "APCA_API_BASE_URL": "https://api.alpaca.markets"}
    assert "env.APCA_API_BASE_URL" in errs(AppConfig(**SYNTH), env)
    env["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
    assert "env.APCA_API_BASE_URL" not in errs(AppConfig(**SYNTH), env)
    assert "env.ALPACA_PAPER" in errs(AppConfig(**SYNTH), {"ALPACA_PAPER": "false"})


def test_mirror_requires_paper_mode():
    cfg = AppConfig(execution=ExecutionConfig(mode="manual", mirror_to_alpaca=True), **SYNTH)
    assert "execution.mirror_to_alpaca" in errs(cfg, GOOD_ENV)


def test_ensure_valid_raises_with_all_issues():
    # AppConfig itself now rejects a negative balance at construction (PR #6),
    # so bypass that to exercise validate_config as the backstop.
    with pytest.raises(ValidationError):
        AppConfig(symbols=[], starting_balance=-1, **SYNTH)
    cfg = AppConfig(symbols=[], **SYNTH)
    object.__setattr__(cfg, "starting_balance", -1)
    with pytest.raises(ConfigError) as ei:
        ensure_valid(cfg, {})
    assert len(ei.value.issues) >= 2


def test_ensure_valid_returns_warnings():
    cfg = AppConfig(risk=RiskConfig(stop_loss_pct=0.3, take_profit_pct=0.2), **SYNTH)
    assert any(i.level == "warning" for i in ensure_valid(cfg, {}))


# ------------------------------------------------------------------- logging
def _record(msg="hi", **extra):
    r = logging.LogRecord("stockbot.x", logging.INFO, __file__, 1, msg, (), None)
    r.__dict__.update(extra)
    return r


def test_json_formatter_basic_and_extras():
    out = json.loads(JsonFormatter().format(_record("hello %s" % "w", symbol="AAPL", qty=3)))
    assert out["msg"] == "hello w" and out["level"] == "INFO" and out["symbol"] == "AAPL" and out["qty"] == 3
    assert out["ts"].endswith("+00:00")


def test_json_formatter_redacts_secrets_and_handles_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        r = _record(api_key="abc123")
        r.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(r))
    assert out["api_key"] == "***" and "ValueError: boom" in out["exc"]


def test_json_formatter_never_raises_on_odd_values():
    json.loads(JsonFormatter().format(_record(obj=object())))


def test_configure_logging_is_idempotent_and_switches_format(capsys):
    configure_logging("INFO", json_format=True)
    configure_logging("INFO", json_format=True)
    root = logging.getLogger("stockbot")
    assert len(root.handlers) == 1
    logging.getLogger("stockbot.t").info("x", extra={"symbol": "MSFT"})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["symbol"] == "MSFT"
    configure_logging("INFO", json_format=False)


# ------------------------------------------------------------------- metrics
def _val(name, labels=None):
    return metrics.REGISTRY.get_sample_value(name, labels or {})


def test_metrics_record_and_render():
    before = _val("stockbot_orders_total", {"side": "buy", "result": "filled"}) or 0
    metrics.record_order("buy", "filled")
    assert _val("stockbot_orders_total", {"side": "buy", "result": "filled"}) == before + 1
    metrics.record_portfolio(101.0, 50.0, 51.0, 2)
    assert _val("stockbot_portfolio_value_dollars") == 101.0
    body, ctype = metrics.render()
    assert b"stockbot_portfolio_value_dollars 101.0" in body and "text/plain" in ctype


def test_rejection_reason_label_is_bounded():
    metrics.record_order("buy", "rejected", "max_position_size (limit 20000.00, wanted 31234.5)")
    assert _val("stockbot_risk_rejections_total", {"reason": "max_position_size"}) >= 1


def test_failure_then_success_resets_gauge():
    metrics.record_failure(3)
    assert _val("stockbot_consecutive_failures") == 3
    metrics.record_cycle("ok", 0.2)
    assert _val("stockbot_consecutive_failures") == 0


# -------------------------------------------------------------------- health
class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_liveness_goes_stale():
    c = Clock()
    s = HealthState(stale_after=100, clock=c)
    assert s.liveness()[0]
    c.t += 101
    assert not s.liveness()[0]
    s.beat()
    assert s.liveness()[0]


def test_readiness_needs_first_beat_and_db():
    c = Clock()
    ok_db = {"v": True}
    s = HealthState(stale_after=100, db_check=lambda: ok_db["v"], clock=c)
    assert not s.readiness()[0]
    s.beat()
    assert s.readiness()[0]
    ok_db["v"] = False
    assert s.readiness()[1]["database"] == "down" and not s.readiness()[0]
    ok_db["v"] = True
    s.shutting_down = True
    assert not s.readiness()[0]


def test_readiness_db_exception_is_not_ready():
    def boom():
        raise RuntimeError
    s = HealthState(stale_after=1, db_check=boom)
    s.beat()
    assert not s.readiness()[0]


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_http_endpoints():
    state = HealthState(stale_after=100, db_check=lambda: True)
    server = start_server(state, 0, host="127.0.0.1")
    port = server.server_address[1]
    try:
        assert _get(port, "/healthz")[0] == 200
        assert _get(port, "/readyz")[0] == 503
        state.beat()
        assert _get(port, "/readyz")[0] == 200
        code, body = _get(port, "/metrics")
        assert code == 200 and "stockbot_cycles_total" in body
        assert _get(port, "/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ shutdown
def test_shutdown_flag_and_sleep_interrupt():
    g = GracefulShutdown()
    assert not g.requested
    with pytest.raises(KeyboardInterrupt):
        with g.interruptible():
            g._handle(signal.SIGTERM, None)
    assert g.requested
    with pytest.raises(KeyboardInterrupt):  # already requested: never start sleeping
        with g.interruptible():
            pass


def test_signal_mid_cycle_only_sets_flag():
    g = GracefulShutdown()
    g._handle(signal.SIGTERM, None)  # not sleeping -> no raise
    assert g.requested


def test_real_sigterm_wakes_sleeping_loop():
    import os
    import time
    g = GracefulShutdown().install()
    try:
        threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            with g.interruptible():
                time.sleep(10)
        assert time.monotonic() - start < 5 and g.requested
    finally:
        g.uninstall()


def test_install_restores_previous_handlers():
    before = signal.getsignal(signal.SIGTERM)
    g = GracefulShutdown().install()
    assert signal.getsignal(signal.SIGTERM) != before
    g.uninstall()
    assert signal.getsignal(signal.SIGTERM) == before
