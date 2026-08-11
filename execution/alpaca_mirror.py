"""
AlpacaMirrorExecutor: optional visualization aid, NOT a replacement for
the local paper simulator.

The local `PaperExecutor` remains the ONLY source of truth for this
project's portfolio accounting - its Fill is what `portfolio.apply_fill()`
uses, exactly as if this class didn't exist. This class additionally
best-effort mirrors the same order to your own Alpaca PAPER account
purely so you can see the same activity on Alpaca's dashboard too.

Hard safety guarantees:
  - `paper=True` is hard-coded when constructing Alpaca's TradingClient.
    This class can never submit to a live trading account.
  - Mirroring is fire-and-forget: if it fails for any reason (market
    closed, network error, your Alpaca paper account doesn't hold the
    position a SELL needs, etc.) the error is logged and swallowed - the
    local fill that already happened is never affected or rolled back.

Known limitation: because mirroring is best-effort, the local simulation
and your Alpaca paper account can drift out of sync over time (e.g. a
BUY mirror fails once, and a later SELL mirror then fails too because
Alpaca never held the shares locally simulated). That's an accepted
tradeoff for a purely cosmetic feature - the local ledger is always the
authoritative one for backtesting, risk decisions, and the dashboard.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from core.logging import get_logger
from core.models import Fill, Order, OrderSide
from execution.base import ExecutionError, OrderExecutor
from execution.paper import PaperExecutor

logger = get_logger("execution.alpaca_mirror")


def _get_alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ExecutionError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set - required to mirror "
            "orders to Alpaca's paper account. Set them in .env, or set "
            "execution.mirror_to_alpaca: false in config.yaml to disable mirroring."
        )
    return api_key, secret_key


class AlpacaMirrorExecutor(OrderExecutor):
    def __init__(self, inner: PaperExecutor):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - only hit if extra not installed
            raise ExecutionError(
                "alpaca-py is not installed. Run `pip install alpaca-py` "
                "or set execution.mirror_to_alpaca: false in config.yaml."
            ) from exc

        self._inner = inner
        api_key, secret_key = _get_alpaca_credentials()
        self._client = TradingClient(api_key, secret_key, paper=True)

    def execute(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        fill = self._inner.execute(order, market_price, timestamp)
        self._mirror(order)
        return fill

    def _mirror(self, order: Order) -> None:
        try:
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide
            from alpaca.trading.enums import TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            alpaca_order = self._client.submit_order(request)
            logger.info(
                "Mirrored %s %s qty=%s to Alpaca paper account (order id=%s)",
                order.side.value, order.symbol, order.quantity, alpaca_order.id,
            )
        except Exception as exc:
            # Mirroring is cosmetic - never allowed to affect the
            # authoritative local simulation, which already completed.
            logger.warning(
                "Failed to mirror %s %s qty=%s to Alpaca paper account "
                "(local simulation is unaffected): %s",
                order.side.value, order.symbol, order.quantity, exc,
            )
