from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import main
from core.config import AppConfig, DatabaseConfig, DataConfig
from core.models import MarketSnapshot
from data.market_calendar import AlwaysOpenCalendar
from database.database import DatabaseManager
from portfolio.portfolio import Portfolio

NOW = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_build_market_calendar_auto_picks_alpaca_when_provider_is_alpaca(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    from data.market_calendar import AlpacaMarketCalendar

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeClient)

    config = AppConfig(data=DataConfig(provider="alpaca", market_calendar="auto"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlpacaMarketCalendar)


def test_build_market_calendar_auto_picks_always_open_for_synthetic():
    config = AppConfig(data=DataConfig(provider="synthetic", market_calendar="auto"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlwaysOpenCalendar)


def test_build_market_calendar_explicit_override_bypasses_auto():
    config = AppConfig(data=DataConfig(provider="alpaca", market_calendar="always_open"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlwaysOpenCalendar)


class _FakeMarketData:
    def __init__(self, prices):
        self._prices = prices

    def get_snapshot(self, symbol):
        return MarketSnapshot(symbol=symbol, timestamp=NOW, price=self._prices[symbol], volume=1000)


class _FarFutureCalendar:
    def next_open(self, now):
        return now + timedelta(hours=10)  # longer than max_closed_market_heartbeat_seconds


class _SoonCalendar:
    def next_open(self, now):
        return now + timedelta(minutes=5)  # shorter than the cap


def test_heartbeat_while_closed_marks_portfolio_without_trading(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, AlwaysOpenCalendar(), NOW
    )

    history = db.get_portfolio_history(limit=10)
    assert len(history) == 1
    assert history[0]["total_value"] == pytest.approx(100_000.0)  # marked, but nothing traded
    assert sleep_seconds > 0


def test_heartbeat_caps_sleep_duration_at_the_configured_maximum(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, _FarFutureCalendar(), NOW
    )

    assert sleep_seconds == config.max_closed_market_heartbeat_seconds


def test_heartbeat_does_not_cap_a_short_wait(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, _SoonCalendar(), NOW
    )

    assert sleep_seconds == pytest.approx(300, abs=1)  # 5 minutes, well under the cap


def test_heartbeat_survives_a_data_outage(tmp_path):
    """If the data provider is unreachable while the market's closed, the
    heartbeat must not crash the live loop - it should just skip marking
    that symbol and keep going."""

    class _BrokenMarketData:
        def get_snapshot(self, symbol):
            raise RuntimeError("simulated outage")

    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _BrokenMarketData(), portfolio, AlwaysOpenCalendar(), NOW
    )
    assert sleep_seconds > 0  # did not raise


class _FlakyCalendar:
    """Fails `failures` times, then raises KeyboardInterrupt so the loop
    under test exits instead of spinning forever."""

    def __init__(self, failures: int):
        self._failures = failures
        self.calls = 0

    def is_open(self, now):
        self.calls += 1
        if self.calls <= self._failures:
            raise ConnectionError("simulated DNS failure")
        raise KeyboardInterrupt


def _live_loop_config(tmp_path) -> AppConfig:
    return AppConfig(
        symbols=["AAPL"],
        log_file=None,
        data=DataConfig(provider="synthetic", market_calendar="always_open"),
        database=DatabaseConfig(url=f"sqlite:///{tmp_path}/test.db"),
    )


def test_live_loop_survives_a_transient_outage_instead_of_exiting(tmp_path, monkeypatch):
    """A failed network call inside the loop must not escape run_live_loop.

    Regression test: the loop's only handler used to be `except
    KeyboardInterrupt`, so one failed calendar.is_open() - an unguarded
    network call made every iteration - propagated straight out of main()
    and killed the process. Docker's restart policy hid it behind 336
    restarts in production before anyone noticed."""
    calendar = _FlakyCalendar(failures=3)
    monkeypatch.setattr(main, "build_market_calendar", lambda config: calendar)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)

    main.run_live_loop(_live_loop_config(tmp_path))  # must return, not raise

    assert calendar.calls == 4  # 3 failures retried, then the clean exit


def test_live_loop_backs_off_exponentially_up_to_the_cap(tmp_path, monkeypatch):
    """Retries must back off rather than hammer a service that's already
    struggling, and must stop doubling at the configured ceiling."""
    calendar = _FlakyCalendar(failures=8)
    monkeypatch.setattr(main, "build_market_calendar", lambda config: calendar)

    sleeps: list = []
    monkeypatch.setattr(main.time, "sleep", lambda seconds: sleeps.append(seconds))

    main.run_live_loop(_live_loop_config(tmp_path))

    base, cap = main.RETRY_BACKOFF_BASE_SECONDS, main.RETRY_BACKOFF_MAX_SECONDS
    assert sleeps == [base, base * 2, base * 4, base * 8, base * 16, cap, cap, cap]


def test_live_loop_resets_backoff_after_a_success(tmp_path, monkeypatch):
    """A recovered connection must not stay penalised by earlier failures."""

    class _RecoveringCalendar:
        def __init__(self):
            self.calls = 0

        def is_open(self, now):
            self.calls += 1
            if self.calls in (1, 2):
                raise ConnectionError("simulated DNS failure")
            if self.calls == 3:
                return False  # recovered - closed-market heartbeat runs
            if self.calls == 4:
                raise ConnectionError("simulated DNS failure")
            raise KeyboardInterrupt

        def next_open(self, now):
            return now + timedelta(minutes=5)

    calendar = _RecoveringCalendar()
    monkeypatch.setattr(main, "build_market_calendar", lambda config: calendar)

    sleeps: list = []
    monkeypatch.setattr(main.time, "sleep", lambda seconds: sleeps.append(seconds))

    main.run_live_loop(_live_loop_config(tmp_path))

    base = main.RETRY_BACKOFF_BASE_SECONDS
    # two failures, the heartbeat's own sleep, then a failure back at base
    assert sleeps[0] == base
    assert sleeps[1] == base * 2
    assert sleeps[3] == base


def test_live_loop_still_stops_on_keyboard_interrupt(tmp_path, monkeypatch):
    """The retry wrapper must not swallow Ctrl-C."""
    calendar = _FlakyCalendar(failures=0)
    monkeypatch.setattr(main, "build_market_calendar", lambda config: calendar)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)

    main.run_live_loop(_live_loop_config(tmp_path))

    assert calendar.calls == 1  # exited immediately, no retry


def test_live_loop_exits_cleanly_when_shutdown_requested_mid_cycle(tmp_path, monkeypatch):
    """SIGTERM during a cycle must finish that cycle and exit; not loop on."""
    import observability.shutdown as sd

    holder = {}
    orig_init = sd.GracefulShutdown.__init__

    def spy(self):
        orig_init(self)
        holder["g"] = self

    monkeypatch.setattr(sd.GracefulShutdown, "__init__", spy)
    monkeypatch.setenv("METRICS_PORT", "0")

    class _Cal:
        def is_open(self, now):
            holder["g"].request()  # signal arrives during the iteration
            return True

    cycles = []
    monkeypatch.setattr(main, "build_market_calendar", lambda config: _Cal())
    monkeypatch.setattr(main, "_run_cycle", lambda *a, **k: cycles.append(1))
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    main.run_live_loop(_live_loop_config(tmp_path))
    assert len(cycles) == 1
