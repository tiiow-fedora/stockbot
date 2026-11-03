"""
Base class that adapts research strategies to the engine's ``Strategy`` interface.

The engine's contract (``strategy/base.py``) is a single method,
``generate_decision(symbol, snapshot, history, news, position, cash, value)
-> Decision``. The engine is long-only and does its own position sizing, so
a research strategy only has to answer one question per bar:

    "Given the bars I can see and whether I am long, should I ENTER, EXIT or HOLD?"

That question is the abstract :meth:`ResearchStrategy.signal`. Everything
else - assembling numpy arrays from ``history + snapshot``, the warm-up
check, NaN safety, turning the answer into a fully populated
``core.models.Decision`` (``BUY``/``SELL``/``HOLD`` + confidence + score
breakdown + indicator dict + reason) - lives here, once, so each concrete
strategy file is just its maths.

Design rules every strategy follows (enforced by ``research/tests``):

* **Stateless.** ``signal`` is a pure function of the visible bars and the
  ``in_position`` flag; the strategy object holds only immutable params.
  That makes backtests, walk-forward windows and Monte-Carlo replays
  independent and reproducible, and means a restarted live process behaves
  exactly like one that never stopped.
* **No look-ahead.** The last element of every array is the decision bar
  (the snapshot); nothing later exists. History bars at/after the snapshot
  timestamp are dropped defensively.
* **Bounded cost.** Only the most recent ``lookback`` bars are materialised,
  so a decision costs O(lookback), not O(history), on long backtests.
* **Long-only.** ``ENTER`` while already long and ``EXIT`` while flat both
  become ``HOLD`` (no pyramiding, no shorting) - matching the engine.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar

import numpy as np

from core.models import Action, Candle, Decision, MarketSnapshot, NewsItem, ScoreBreakdown
from strategy.base import Strategy

P = TypeVar("P", bound="StrategyParams")


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyParams:
    """Immutable, validated parameter container. Subclass per strategy.

    Subclasses are ``@dataclass(frozen=True)`` and override :meth:`validate`
    to raise ``ValueError`` on nonsense (negative periods, fast >= slow, ...).
    Validation runs on construction, so an invalid parameter set can never
    reach a backtest - the optimizer relies on this to skip invalid grid
    cells instead of crashing.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:  # pragma: no cover - default is permissive
        return None

    def replace(self: P, **changes: Any) -> P:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a ``{param: [values]}`` grid -> list of override dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


# --------------------------------------------------------------------------
# Bars: the numpy view of "everything visible at decision time"
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Bars:
    """Parallel OHLCV arrays, oldest -> newest. ``close[-1]`` is the decision bar."""

    symbol: str
    timestamps: Sequence[datetime]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    @property
    def last(self) -> float:
        return float(self.close[-1])

    @classmethod
    def from_candles(cls, symbol: str, candles: Sequence[Candle]) -> "Bars":
        """Convenience for tests / notebooks: build Bars straight from candles."""
        return cls(
            symbol=symbol,
            timestamps=[c.timestamp for c in candles],
            open=np.array([c.open for c in candles], dtype=float),
            high=np.array([c.high for c in candles], dtype=float),
            low=np.array([c.low for c in candles], dtype=float),
            close=np.array([c.close for c in candles], dtype=float),
            volume=np.array([c.volume for c in candles], dtype=float),
        )

    @classmethod
    def from_history(
        cls, symbol: str, history: Sequence[Candle], snapshot: MarketSnapshot, lookback: Optional[int] = None
    ) -> "Bars":
        """History (strictly before the snapshot) + the snapshot as the final bar."""
        hist = list(history)
        # Defensive no-look-ahead guard: callers promise this, we verify cheaply.
        while hist and hist[-1].timestamp >= snapshot.timestamp:
            hist.pop()
        if lookback is not None and lookback > 1:
            hist = hist[-(lookback - 1):]
        price = float(snapshot.price)
        s_open = snapshot.open if snapshot.open is not None else price
        s_high = snapshot.high if snapshot.high is not None else max(price, s_open)
        s_low = snapshot.low if snapshot.low is not None else min(price, s_open)
        # In a live quote ``price`` can sit outside a stale H/L; widen so H >= C >= L.
        s_high, s_low = max(s_high, price), min(s_low, price)
        return cls(
            symbol=symbol,
            timestamps=[c.timestamp for c in hist] + [snapshot.timestamp],
            open=np.array([c.open for c in hist] + [s_open], dtype=float),
            high=np.array([c.high for c in hist] + [s_high], dtype=float),
            low=np.array([c.low for c in hist] + [s_low], dtype=float),
            close=np.array([c.close for c in hist] + [price], dtype=float),
            volume=np.array([c.volume for c in hist] + [float(snapshot.volume or 0)], dtype=float),
        )


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------

class SignalKind(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A strategy's answer for one bar.

    ``strength`` in [0, 1] becomes the Decision's ``confidence``, which the
    engine's RiskManager uses to scale position size (0.5x .. 1x of the
    allowed size). ``indicators`` is logged on the Decision for explainability.
    """

    kind: SignalKind
    strength: float = 1.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def enter(cls, strength: float = 1.0, reason: str = "", **indicators: Any) -> "Signal":
        return cls(SignalKind.ENTER, strength, indicators, reason)

    @classmethod
    def exit(cls, strength: float = 1.0, reason: str = "", **indicators: Any) -> "Signal":
        return cls(SignalKind.EXIT, strength, indicators, reason)

    @classmethod
    def hold(cls, reason: str = "", **indicators: Any) -> "Signal":
        return cls(SignalKind.HOLD, 0.0, indicators, reason)


def clip01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, float(x)))


def _clean_indicator(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------
# ResearchStrategy
# --------------------------------------------------------------------------

class ResearchStrategy(Strategy):
    """Adapter from ``signal(bars, in_position)`` to the engine's ``Strategy``.

    Subclass contract
    -----------------
    * set class attributes ``name`` (registry key, snake_case), ``family``
      (one of ``momentum | trend | mean_reversion | breakout | pairs |
      volatility``), ``description`` (one sentence), ``PARAMS`` (the
      ``StrategyParams`` subclass) and ``PARAM_GRID`` (default optimizer grid,
      ``{param: [values]}`` - small: the optimizer takes the Cartesian product).
    * implement :meth:`signal`, :attr:`min_bars` and (optionally) :attr:`lookback`.
    """

    name: ClassVar[str] = ""
    family: ClassVar[str] = ""
    description: ClassVar[str] = ""
    PARAMS: ClassVar[Type[StrategyParams]] = StrategyParams
    PARAM_GRID: ClassVar[Dict[str, Sequence[Any]]] = {}
    #: True for strategies that need a second instrument (pairs / relative strength).
    #: They take a ``partner=<List[Candle]>`` constructor kwarg and may only ever read
    #: partner bars whose timestamp is <= the decision bar's timestamp.
    requires_partner: ClassVar[bool] = False

    def __init__(self, params: Optional[StrategyParams] = None, **overrides: Any):
        if params is None:
            params = self.PARAMS(**overrides)
        elif overrides:
            params = params.replace(**overrides)
        if not isinstance(params, self.PARAMS):
            raise TypeError(f"{type(self).__name__} expects {self.PARAMS.__name__}, got {type(params).__name__}")
        self.params = params

    # -- to implement --------------------------------------------------

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Bars (including the decision bar) needed before the first real signal."""

    @property
    def lookback(self) -> Optional[int]:
        """Max bars to materialise per decision (None = all). Default: ``4 * min_bars``
        - enough for recursive indicators (EMA/Wilder) to forget their seed."""
        return max(4 * self.min_bars, 60)

    @abstractmethod
    def signal(self, bars: Bars, in_position: bool) -> Signal:
        """Decide ENTER / EXIT / HOLD from ``bars`` (last bar = now)."""

    # -- helpers for subclasses / optimizer ----------------------------

    @classmethod
    def param_combinations(cls, grid: Optional[Mapping[str, Sequence[Any]]] = None) -> List[Dict[str, Any]]:
        """Valid override dicts from ``grid`` (default ``PARAM_GRID``); combos whose
        params fail validation (e.g. fast >= slow) are silently dropped."""
        out = []
        for combo in expand_grid(cls.PARAM_GRID if grid is None else grid):
            try:
                cls.PARAMS(**combo)
            except (ValueError, TypeError):
                continue
            out.append(combo)
        return out

    def with_params(self, **changes: Any) -> "ResearchStrategy":
        return type(self)(self.params.replace(**changes))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"

    # -- Strategy interface ---------------------------------------------

    def generate_decision(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        history: List[Candle],
        news: List[NewsItem],
        current_position_qty: float,
        cash: float,
        portfolio_value: float,
    ) -> Decision:
        in_position = current_position_qty > 0
        bars = Bars.from_history(symbol, history, snapshot, self.lookback)

        if len(bars) < self.min_bars:
            sig = Signal.hold(f"warm-up: {len(bars)}/{self.min_bars} bars")
        else:
            sig = self.signal(bars, in_position)
            if not isinstance(sig, Signal):
                raise TypeError(f"{type(self).__name__}.signal must return Signal, got {type(sig).__name__}")

        action, score, confidence = self._resolve(sig, in_position)
        reason = f"{self.name}: {action.value} - {sig.reason}" if sig.reason else f"{self.name}: {action.value}"
        breakdown = ScoreBreakdown(
            technical=score, momentum=0.0, volume=0.0, risk=0.0, sentiment=None, composite=score,
            weights={"signal": 1.0},
        )
        return Decision(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            price=snapshot.price,
            action=action,
            confidence=confidence,
            scores=breakdown,
            indicators={k: _clean_indicator(v) for k, v in sig.indicators.items()},
            current_position_qty=current_position_qty,
            cash=cash,
            portfolio_value=portfolio_value,
            reason=reason,
        )

    @staticmethod
    def _resolve(sig: Signal, in_position: bool):
        strength = clip01(sig.strength)
        if sig.kind is SignalKind.ENTER and not in_position:
            return Action.BUY, strength, strength
        if sig.kind is SignalKind.EXIT and in_position:
            return Action.SELL, -strength, strength
        return Action.HOLD, 0.0, 0.0
