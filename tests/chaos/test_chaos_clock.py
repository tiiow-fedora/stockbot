"""Clock skew, DST, day rollover, non-UTC timestamps."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from core.models import Fill, Order, OrderSide
from data.market_calendar import USEquityMarketCalendar
from tests.chaos.bot import Bot, patch_clock
from tests.chaos.conftest import dip_candles, make_config

ET = ZoneInfo("America/New_York")
TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


# -- calendar around DST / weekends ----------------------------------------------------

@pytest.mark.parametrize(
    "ts,expected",
    [
        (utc(2024, 3, 8, 14, 29), False),    # Fri EST, 09:29 ET
        (utc(2024, 3, 8, 14, 30), True),     # Fri EST, 09:30 ET
        (utc(2024, 3, 8, 20, 59), True),     # 15:59 ET
        (utc(2024, 3, 8, 21, 0), False),     # 16:00 ET closed
        (utc(2024, 3, 11, 13, 29), False),   # Mon after spring-forward: 09:29 EDT
        (utc(2024, 3, 11, 13, 30), True),    # 09:30 EDT == 13:30Z (NOT 14:30Z)
        (utc(2024, 11, 1, 13, 30), True),    # Fri, still EDT
        (utc(2024, 11, 4, 14, 29), False),   # Mon after fall-back: 09:29 EST
        (utc(2024, 11, 4, 14, 30), True),
        (utc(2024, 3, 9, 15, 0), False),     # Saturday
        (utc(2024, 3, 10, 15, 0), False),    # Sunday (DST changeover day)
    ],
)
def test_us_equity_calendar_across_dst(ts, expected):
    assert USEquityMarketCalendar().is_open(ts) is expected


@pytest.mark.parametrize(
    "now,expected_open",
    [
        (utc(2024, 3, 8, 21, 0), utc(2024, 3, 11, 13, 30)),    # Fri close -> Mon open (crosses DST)
        (utc(2024, 3, 9, 12, 0), utc(2024, 3, 11, 13, 30)),    # Saturday
        (utc(2024, 3, 10, 12, 0), utc(2024, 3, 11, 13, 30)),   # Sunday
        (utc(2024, 3, 12, 12, 0), utc(2024, 3, 12, 13, 30)),   # Tue pre-open -> same day
        (utc(2024, 11, 1, 21, 0), utc(2024, 11, 4, 14, 30)),   # Fri -> Mon across fall-back
    ],
)
def test_next_open_across_dst(now, expected_open):
    assert USEquityMarketCalendar().next_open(now) == expected_open


def test_next_open_is_always_in_the_future_and_always_open(  ):
    cal = USEquityMarketCalendar()
    t = utc(2024, 1, 1)
    for _ in range(24 * 60):     # every 17 minutes for ~17 days, spanning two DST regimes
        t += timedelta(minutes=17)
        n = cal.next_open(t)
        assert n > t or cal.is_open(t) and n >= t
        assert cal.is_open(n) and not cal.is_open(n - timedelta(minutes=1))


def test_a_regular_session_always_falls_on_one_utc_date():
    """count_trades_today buckets by UTC date; that is only safe because a
    whole US session never straddles UTC midnight. Verify for the whole year
    (both DST regimes)."""
    cal = USEquityMarketCalendar()
    d = date(2024, 1, 1)
    while d.year == 2024:
        if d.weekday() < 5:
            o = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)
            c = datetime(d.year, d.month, d.day, 15, 59, tzinfo=ET).astimezone(timezone.utc)
            assert cal.is_open(o) and cal.is_open(c) and o.date() == c.date(), d
        d += timedelta(days=1)


# -- skew against the broker clock ----------------------------------------------------------

@pytest.mark.parametrize("skew", [timedelta(minutes=-10), timedelta(minutes=10), timedelta(hours=3), timedelta(hours=-3)])
def test_trading_permission_follows_the_broker_clock_not_the_local_one(broker, db_url, monkeypatch, skew):
    """With the Alpaca calendar, is_open comes from the broker, so a wrong local
    clock must not make the bot trade while the broker says closed."""
    patch_clock(monkeypatch, broker.clock)
    broker.clock.local_skew = skew
    broker.market_open = False
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url))
    assert bot.is_open() is False


@pytest.mark.xfail(strict=True, reason="BUG-11: no local-vs-broker clock sanity check; a skewed machine clock stamps "
                                       "fills/decisions and buckets daily limits with the wrong time (README)")
def test_fill_timestamps_stay_close_to_broker_time_under_large_skew(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.clock.local_skew = timedelta(hours=26)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url))
    bot.cycle()
    for t in bot.db.get_recent_trades():
        ts = datetime.fromisoformat(t["timestamp"])
        assert abs(ts - broker.clock.true_now) < timedelta(minutes=5)


def test_backwards_clock_step_does_not_crash_or_double_trade(broker, db_url, monkeypatch):
    patch_clock(monkeypatch, broker.clock)
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    bot = Bot(make_config(db_url, max_trades_per_symbol_per_day=1))
    bot.cycle()
    qty = bot.portfolio.position_qty("AAPL")
    broker.clock.local_skew = timedelta(minutes=-45)     # NTP steps the clock back
    for _ in range(3):
        bot.cycle()
    assert bot.portfolio.position_qty("AAPL") == qty


def test_daily_trade_limit_rolls_over_at_utc_midnight(db):
    def fill(ts, qty=1):
        o = Order("AAPL", OrderSide.BUY, qty, timestamp=ts)
        db.save_fill(Fill(o, 100.0, qty, ts), db.save_order(o))

    fill(utc(2024, 3, 12, 23, 59, 59))
    fill(utc(2024, 3, 13, 0, 0, 1))
    assert db.count_trades_today("AAPL", utc(2024, 3, 12, 12)) == 1
    assert db.count_trades_today("AAPL", utc(2024, 3, 13, 12)) == 1


@pytest.mark.xfail(strict=True, reason="BUG-12 (latent): DB stores tz-aware non-UTC timestamps as naive wall-clock "
                                       "without converting to UTC, so count_trades_today mis-buckets them")
def test_non_utc_aware_timestamps_are_normalised_before_storage(db):
    et_evening = datetime(2024, 3, 12, 20, 30, tzinfo=ET)       # == 2024-03-13 00:30Z
    o = Order("AAPL", OrderSide.BUY, 1, timestamp=et_evening)
    db.save_fill(Fill(o, 100.0, 1, et_evening), db.save_order(o))
    assert db.count_trades_today("AAPL", utc(2024, 3, 13, 12)) == 1
