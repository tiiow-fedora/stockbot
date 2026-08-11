"""
Technical indicators.

Every function is a pure function over a plain sequence of floats (no
pandas/DB/network dependency) and returns a list the same length as the
input, with `None` in every position where there isn't yet enough history
to compute a value. This makes "not enough data yet" an explicit,
impossible-to-ignore value rather than a silently wrong number - callers
must check for None before trading on an indicator.

All functions only ever look backward from index i, never forward, which
is what makes them safe to use inside the backtester without look-ahead
bias.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Simple moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    values = list(values)
    out: List[Optional[float]] = [None] * len(values)
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        if i >= period:
            running_sum -= values[i - period]
        if i >= period - 1:
            out[i] = running_sum / period
    return out


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Exponential moving average, seeded with an SMA over the first `period` values."""
    if period < 1:
        raise ValueError("period must be >= 1")
    values = list(values)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out

    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index using Wilder's smoothing method."""
    values = list(values)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out

    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains[i] = max(diff, 0.0)
        losses[i] = max(-diff, 0.0)

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)

    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Returns (macd_line, signal_line, histogram), each the same length as `values`."""
    values = list(values)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)

    macd_line: List[Optional[float]] = [
        (f - s) if (f is not None and s is not None) else None for f, s in zip(ema_fast, ema_slow)
    ]

    first_valid = next((i for i, v in enumerate(macd_line) if v is not None), None)
    signal_line: List[Optional[float]] = [None] * len(values)
    if first_valid is not None:
        tail = [v for v in macd_line[first_valid:]]  # type: ignore[misc]
        sig_tail = ema(tail, signal)
        for i, v in enumerate(sig_tail):
            signal_line[first_valid + i] = v

    histogram: List[Optional[float]] = [
        (m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def momentum(values: Sequence[float], period: int = 10) -> List[Optional[float]]:
    """Percent change vs. `period` bars ago."""
    values = list(values)
    out: List[Optional[float]] = [None] * len(values)
    for i in range(period, len(values)):
        prev = values[i - period]
        if prev != 0:
            out[i] = (values[i] - prev) / prev
    return out


def volume_change(volumes: Sequence[float], period: int = 10) -> List[Optional[float]]:
    """Current volume vs. the average volume of the preceding `period` bars."""
    volumes = list(volumes)
    out: List[Optional[float]] = [None] * len(volumes)
    for i in range(period, len(volumes)):
        window = volumes[i - period : i]
        avg = sum(window) / period
        if avg > 0:
            out[i] = (volumes[i] - avg) / avg
    return out


def volatility(values: Sequence[float], period: int = 20) -> List[Optional[float]]:
    """Standard deviation of simple returns over a trailing `period`-bar window."""
    values = list(values)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period + 1:
        return out

    returns: List[float] = [None] * len(values)  # type: ignore[list-item]
    for i in range(1, len(values)):
        prev = values[i - 1]
        returns[i] = (values[i] - prev) / prev if prev != 0 else 0.0

    for i in range(period, len(values)):
        window = [r for r in returns[i - period + 1 : i + 1] if r is not None]
        if len(window) == period:
            out[i] = float(np.std(window))
    return out
