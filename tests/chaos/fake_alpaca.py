"""
Fake Alpaca PAPER broker + market-data simulator for chaos tests.

Nothing in here touches the network. It implements just enough of the
alpaca-py surface that StockBot actually uses:

  TradingClient            -> FakeTradingClient   (submit_order, get_clock,
                                                   get_order_by_id, get_orders,
                                                   get_all_positions, get_account,
                                                   cancel_order_by_id)
  StockHistoricalDataClient-> FakeDataClient      (get_stock_bars)

State lives in one `FakeAlpacaBroker` so a test can inspect "what Alpaca
thinks happened" and compare it with "what StockBot thinks happened".

Fault injection (deterministic - queued per API method):

  broker.fail_next("submit_order", Timeout())              # raise, nothing happened
  broker.fail_next("submit_order", Timeout(applied=True))  # AMBIGUOUS: order was
                                                           # accepted, response lost
  broker.fail_next("submit_order", HTTPError(504))
  broker.fail_next("get_clock", HTTPError(503))
  broker.reject_next("insufficient buying power")          # 403 style rejection
  broker.fill_mode = PartialFills(0.5)                     # fill only half
  broker.fill_mode = NoFills()                             # accepted but never fills
  broker.hang_next("submit_order")                         # call "never returns"

Safety: building a FakeTradingClient with paper != True raises
LiveTradingAttempted, so any code path that ever dropped `paper=True`
fails these tests loudly.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional

import requests
from alpaca.common.exceptions import APIError


class LiveTradingAttempted(AssertionError):
    """Raised if anything tries to build a non-paper trading client."""


class HangError(Exception):
    """Stands in for 'the socket never answers'. alpaca-py sets NO request
    timeout, so in production this would block the live loop forever; the
    fake raises instead so a test can observe that the call would hang."""


# --------------------------------------------------------------------------
# Fault descriptions
# --------------------------------------------------------------------------

@dataclass
class Timeout:
    """requests ReadTimeout. applied=True models the nasty case: the
    broker processed the request but the response never arrived."""
    applied: bool = False


@dataclass
class HTTPError:
    status: int = 500
    message: str = "server error"
    applied: bool = False


@dataclass
class ConnectionDrop:
    applied: bool = False


class FillMode:
    def fill_qty(self, order: "FakeOrder", broker: "FakeAlpacaBroker") -> float:
        return order.qty


class FullFills(FillMode):
    pass


class PartialFills(FillMode):
    def __init__(self, fraction: float):
        self.fraction = fraction

    def fill_qty(self, order, broker):
        return float(int(order.qty * self.fraction))  # whole shares, like equities


class NoFills(FillMode):
    """Order is accepted and sits 'new' (e.g. market closed / illiquid)."""

    def fill_qty(self, order, broker):
        return 0.0


# --------------------------------------------------------------------------
# Broker state
# --------------------------------------------------------------------------

@dataclass
class FakeOrder:
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: str            # "buy" / "sell"
    qty: float
    status: str = "new"  # new, partially_filled, filled, canceled, rejected
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    reject_reason: Optional[str] = None
    submitted_at: Optional[datetime] = None

    def as_alpaca(self) -> SimpleNamespace:
        return SimpleNamespace(
            id=self.id, client_order_id=self.client_order_id, symbol=self.symbol,
            side=self.side, qty=str(self.qty), status=self.status,
            filled_qty=str(self.filled_qty), filled_avg_price=self.filled_avg_price,
            submitted_at=self.submitted_at,
        )


class FakeClock:
    """Controllable 'true time'. `local_skew` is how far the BOT's machine
    clock is from the broker's (positive = bot is ahead)."""

    def __init__(self, start: datetime):
        assert start.tzinfo is not None
        self.true_now = start
        self.local_skew = timedelta(0)

    def advance(self, **kw) -> None:
        self.true_now += timedelta(**kw)

    def local_now(self) -> datetime:
        return self.true_now + self.local_skew


class FakeAlpacaBroker:
    def __init__(self, starting_cash: float = 100_000.0, clock: Optional[FakeClock] = None, seed: int = 0):
        # default: Tue 2024-03-12 11:00 ET (market open)
        self.clock = clock or FakeClock(datetime(2024, 3, 12, 15, 0, tzinfo=timezone.utc))
        self.cash = starting_cash
        self.positions: Dict[str, Dict[str, float]] = {}   # symbol -> {"qty", "avg"}
        self.orders: Dict[str, FakeOrder] = {}
        self.by_client_id: Dict[str, str] = {}
        self.prices: Dict[str, float] = {}
        self.bars: Dict[str, List[SimpleNamespace]] = {}
        self.market_open = True
        self.fill_mode: FillMode = FullFills()
        self.rng = random.Random(seed)
        self.calls: List[tuple] = []
        self._faults: Dict[str, Deque[Any]] = defaultdict(deque)
        self._ids = itertools.count(1)
        self.live_attempts = 0

    # -- fault queueing -------------------------------------------------------

    def fail_next(self, method: str, fault: Any, times: int = 1) -> None:
        for _ in range(times):
            self._faults[method].append(fault)

    def reject_next(self, reason: str = "insufficient buying power", times: int = 1) -> None:
        for _ in range(times):
            self._faults["submit_order"].append(("reject", reason))

    def hang_next(self, method: str) -> None:
        self._faults[method].append(HangError("call never returned (no timeout configured)"))

    def calls_to(self, method: str) -> List[tuple]:
        return [c for c in self.calls if c[0] == method]

    def _maybe_fault(self, method: str):
        """Pops a queued fault. Non-applied faults raise right here. Applied
        faults / rejections are returned so the caller can do its side
        effects first and raise afterwards."""
        q = self._faults[method]
        if not q:
            return None
        fault = q.popleft()
        if isinstance(fault, HangError):
            raise fault
        if isinstance(fault, tuple) and fault[0] == "reject":
            return fault
        if getattr(fault, "applied", False):
            return fault
        self._raise(fault)

    @staticmethod
    def _raise(fault):
        if isinstance(fault, Timeout):
            raise requests.exceptions.ReadTimeout("fake read timeout")
        if isinstance(fault, ConnectionDrop):
            raise requests.exceptions.ConnectionError("fake connection reset")
        if isinstance(fault, HTTPError):
            raise _api_error(fault.status, fault.message)
        raise fault

    # -- trading API ---------------------------------------------------------

    def submit_order(self, request) -> SimpleNamespace:
        self.calls.append(("submit_order", request.symbol, request.qty, getattr(request, "client_order_id", None)))
        fault = self._maybe_fault("submit_order")
        if isinstance(fault, tuple):  # explicit rejection
            order = self._new_order(request)
            order.status, order.reject_reason = "rejected", fault[1]
            raise _api_error(403, fault[1])

        cid = getattr(request, "client_order_id", None)
        if cid and cid in self.by_client_id:
            raise _api_error(422, "client_order_id must be unique")

        order = self._new_order(request)
        self._process(order)
        if fault is not None:   # applied-then-lost response
            self._raise(fault)
        return order.as_alpaca()

    def _new_order(self, request) -> FakeOrder:
        side = getattr(request.side, "value", str(request.side)).lower()
        order = FakeOrder(
            id=f"ord-{next(self._ids)}",
            client_order_id=getattr(request, "client_order_id", None),
            symbol=request.symbol, side=side, qty=float(request.qty),
            submitted_at=self.clock.true_now,
        )
        self.orders[order.id] = order
        if order.client_order_id:
            self.by_client_id[order.client_order_id] = order.id
        return order

    def _process(self, order: FakeOrder) -> None:
        if not self.market_open:
            return  # stays "new" until the market opens
        price = self.prices.get(order.symbol)
        if price is None:
            order.status, order.reject_reason = "rejected", "asset not found"
            return
        if order.side == "buy":
            affordable = int(self.cash // price)
            if affordable < 1:
                order.status, order.reject_reason = "rejected", "insufficient buying power"
                return
            max_qty = min(order.qty, affordable)
        else:
            held = self.positions.get(order.symbol, {}).get("qty", 0.0)
            if held < order.qty:
                order.status, order.reject_reason = "rejected", "insufficient qty available for order"
                return
            max_qty = order.qty
        qty = min(self.fill_mode.fill_qty(order, self), max_qty)
        if qty > 0:
            self._apply_fill(order, qty, price)

    def _apply_fill(self, order: FakeOrder, qty: float, price: float) -> None:
        if order.side == "buy":
            self.cash -= qty * price
            pos = self.positions.setdefault(order.symbol, {"qty": 0.0, "avg": 0.0})
            total = pos["qty"] + qty
            pos["avg"] = (pos["avg"] * pos["qty"] + price * qty) / total
            pos["qty"] = total
        else:
            self.cash += qty * price
            pos = self.positions[order.symbol]
            pos["qty"] -= qty
            if pos["qty"] <= 1e-9:
                del self.positions[order.symbol]
        prev = order.filled_qty
        order.filled_avg_price = ((order.filled_avg_price or 0.0) * prev + price * qty) / (prev + qty)
        order.filled_qty = prev + qty
        order.status = "filled" if order.filled_qty >= order.qty else "partially_filled"

    def settle(self) -> None:
        """Passage of time at the broker: liquidity arrives and every open
        order fills its remainder (once the market is open)."""
        if not self.market_open:
            return
        for order in list(self.orders.values()):
            if order.status not in ("new", "partially_filled"):
                continue
            remaining = order.qty - order.filled_qty
            price = self.prices.get(order.symbol)
            if not price or remaining <= 0:
                continue
            cap = int(self.cash // price) if order.side == "buy" else self.positions.get(order.symbol, {}).get("qty", 0.0)
            qty = min(remaining, cap)
            if qty > 0:
                self._apply_fill(order, qty, price)

    def get_order_by_id(self, order_id: str):
        self.calls.append(("get_order_by_id", order_id))
        self._maybe_fault("get_order_by_id")
        return self.orders[order_id].as_alpaca()

    def get_orders(self, *a, **k):
        self.calls.append(("get_orders",))
        self._maybe_fault("get_orders")
        return [o.as_alpaca() for o in self.orders.values()]

    def cancel_order_by_id(self, order_id: str):
        self.calls.append(("cancel_order_by_id", order_id))
        self._maybe_fault("cancel_order_by_id")
        o = self.orders[order_id]
        if o.status in ("new", "partially_filled"):
            o.status = "canceled"

    def get_all_positions(self):
        self.calls.append(("get_all_positions",))
        self._maybe_fault("get_all_positions")
        return [SimpleNamespace(symbol=s, qty=str(p["qty"]), avg_entry_price=str(p["avg"]))
                for s, p in self.positions.items()]

    def get_account(self):
        self.calls.append(("get_account",))
        self._maybe_fault("get_account")
        return SimpleNamespace(cash=str(self.cash), buying_power=str(self.cash), status="ACTIVE")

    def get_clock(self):
        self.calls.append(("get_clock",))
        self._maybe_fault("get_clock")
        now = self.clock.true_now
        return SimpleNamespace(
            timestamp=now, is_open=self.market_open,
            next_open=now + timedelta(hours=17), next_close=now + timedelta(hours=5),
        )

    # -- market data API --------------------------------------------------------

    def get_stock_bars(self, request) -> "FakeBarSet":
        self.calls.append(("get_stock_bars", request.symbol_or_symbols))
        self._maybe_fault("get_stock_bars")
        sym = request.symbol_or_symbols
        sym = sym[0] if isinstance(sym, (list, tuple)) else sym
        # alpaca-py's request model normalises start/end to NAIVE UTC.
        def _utc(t):
            return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t
        bars = [b for b in self.bars.get(sym, []) if _utc(request.start) <= b.timestamp <= _utc(request.end)]
        return FakeBarSet({sym: bars} if bars else {})

    def set_bars_from_candles(self, symbol: str, candles) -> None:
        self.bars[symbol] = [
            SimpleNamespace(symbol=symbol, timestamp=c.timestamp, open=c.open, high=c.high,
                            low=c.low, close=c.close, volume=c.volume)
            for c in candles
        ]
        if candles:
            self.prices[symbol] = candles[-1].close


def _api_error(status: int, message: str) -> APIError:
    resp = requests.Response()
    resp.status_code = status
    return APIError(message, requests.exceptions.HTTPError(response=resp))


class FakeBarSet:
    """Mimics alpaca-py BarSet: indexing a symbol with no data raises KeyError."""

    def __init__(self, data: Dict[str, list]):
        self.data = data

    def __getitem__(self, symbol):
        return self.data[symbol]


# --------------------------------------------------------------------------
# Client facades (what alpaca-py constructors get replaced with)
# --------------------------------------------------------------------------

class FakeTradingClient:
    def __init__(self, broker: FakeAlpacaBroker, paper: bool = True):
        if paper is not True:
            broker.live_attempts += 1
            raise LiveTradingAttempted("TradingClient built with paper != True")
        self._b = broker

    def __getattr__(self, name):
        return getattr(self._b, name)


class FakeDataClient:
    def __init__(self, broker: FakeAlpacaBroker):
        self._b = broker

    def get_stock_bars(self, request):
        return self._b.get_stock_bars(request)


def install(monkeypatch, broker: FakeAlpacaBroker) -> FakeAlpacaBroker:
    """Replace alpaca-py's network clients with fakes bound to `broker`."""
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    monkeypatch.setattr(
        "alpaca.trading.client.TradingClient",
        lambda api_key=None, secret_key=None, paper=True, **k: FakeTradingClient(broker, paper),
    )
    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient",
        lambda *a, **k: FakeDataClient(broker),
    )
    return broker
