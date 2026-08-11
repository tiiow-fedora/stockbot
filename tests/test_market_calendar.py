from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.base import MarketDataError
from data.market_calendar import (
    AlpacaMarketCalendar,
    AlwaysOpenCalendar,
    USEquityMarketCalendar,
    build_market_calendar,
)

# All in July 2024 (EDT = UTC-4, no DST ambiguity):
#   2024-07-09 = Tuesday, 2024-07-12 = Friday,
#   2024-07-13 = Saturday, 2024-07-15 = Monday.
TUE_MID_SESSION = datetime(2024, 7, 9, 14, 0, tzinfo=timezone.utc)   # 10:00 ET - open
TUE_BEFORE_OPEN = datetime(2024, 7, 9, 13, 29, tzinfo=timezone.utc)  # 09:29 ET - closed
TUE_AT_CLOSE = datetime(2024, 7, 9, 20, 0, tzinfo=timezone.utc)      # 16:00 ET - closed (exclusive)
TUE_JUST_BEFORE_CLOSE = datetime(2024, 7, 9, 19, 59, tzinfo=timezone.utc)  # 15:59 ET - open
TUE_AFTER_CLOSE = datetime(2024, 7, 9, 21, 0, tzinfo=timezone.utc)   # 17:00 ET - closed
SAT_MIDDAY = datetime(2024, 7, 13, 15, 0, tzinfo=timezone.utc)
FRI_AFTER_CLOSE = datetime(2024, 7, 12, 21, 0, tzinfo=timezone.utc)  # 17:00 ET Friday
MON_OPEN_UTC = datetime(2024, 7, 15, 13, 30, tzinfo=timezone.utc)    # 09:30 ET Monday


def test_always_open_calendar_is_always_open():
    cal = AlwaysOpenCalendar()
    assert cal.is_open(TUE_BEFORE_OPEN) is True
    assert cal.is_open(SAT_MIDDAY) is True


def test_always_open_next_open_is_the_timestamp_itself():
    cal = AlwaysOpenCalendar()
    assert cal.next_open(SAT_MIDDAY) == SAT_MIDDAY


def test_us_equity_calendar_open_during_session():
    cal = USEquityMarketCalendar()
    assert cal.is_open(TUE_MID_SESSION) is True
    assert cal.is_open(TUE_JUST_BEFORE_CLOSE) is True


def test_us_equity_calendar_closed_before_open():
    cal = USEquityMarketCalendar()
    assert cal.is_open(TUE_BEFORE_OPEN) is False


def test_us_equity_calendar_closed_at_and_after_close():
    cal = USEquityMarketCalendar()
    assert cal.is_open(TUE_AT_CLOSE) is False
    assert cal.is_open(TUE_AFTER_CLOSE) is False


def test_us_equity_calendar_closed_on_weekends():
    cal = USEquityMarketCalendar()
    assert cal.is_open(SAT_MIDDAY) is False


def test_us_equity_next_open_same_day_if_before_open():
    cal = USEquityMarketCalendar()
    expected = datetime(2024, 7, 9, 13, 30, tzinfo=timezone.utc)  # 09:30 ET same Tuesday
    assert cal.next_open(TUE_BEFORE_OPEN) == expected


def test_us_equity_next_open_next_day_if_after_close():
    cal = USEquityMarketCalendar()
    expected = datetime(2024, 7, 10, 13, 30, tzinfo=timezone.utc)  # 09:30 ET Wednesday
    assert cal.next_open(TUE_AFTER_CLOSE) == expected


def test_us_equity_next_open_skips_weekend_from_friday_close():
    cal = USEquityMarketCalendar()
    assert cal.next_open(FRI_AFTER_CLOSE) == MON_OPEN_UTC


def test_us_equity_next_open_skips_weekend_from_saturday():
    cal = USEquityMarketCalendar()
    assert cal.next_open(SAT_MIDDAY) == MON_OPEN_UTC


def test_build_market_calendar_always_open():
    assert isinstance(build_market_calendar("always_open"), AlwaysOpenCalendar)


def test_build_market_calendar_us_equity():
    assert isinstance(build_market_calendar("us_equity"), USEquityMarketCalendar)


def test_build_market_calendar_unknown_raises():
    with pytest.raises(ValueError):
        build_market_calendar("not_a_real_calendar")


def test_alpaca_market_calendar_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(MarketDataError, match="ALPACA_API_KEY"):
        AlpacaMarketCalendar()


class _FakeClock:
    def __init__(self, is_open, next_open):
        self.is_open = is_open
        self.next_open = next_open


class _FakeTradingClientWithClock:
    def __init__(self, *args, **kwargs):
        self._clock = _FakeClock(is_open=False, next_open=MON_OPEN_UTC)

    def get_clock(self):
        return self._clock


def test_alpaca_market_calendar_reflects_the_clock(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClientWithClock)

    cal = AlpacaMarketCalendar()
    assert cal.is_open(TUE_MID_SESSION) is False  # ignores the argument - reflects the live clock
    assert cal.next_open(TUE_MID_SESSION) == MON_OPEN_UTC
