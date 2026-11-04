"""
Vectorised (numpy) indicators used by the research strategies.

Conventions - identical to ``analysis/indicators.py`` where they overlap:

* Every function takes 1-D ``numpy`` arrays and returns arrays of the SAME
  length. Positions without enough history hold ``NaN`` (the numpy
  analogue of the engine's ``None``), so "not enough data" can never be
  mistaken for a real value. Strategies must treat NaN as "no signal".
* Value ``i`` only ever depends on inputs ``<= i`` (no look-ahead). This is
  asserted by ``research/tests/test_indicators.py`` by truncating inputs and
  checking that previously computed values do not change.
* Where a channel is used for breakouts (``donchian``) the default is to
  EXCLUDE the current bar, so "close > 20-day high" is a meaningful test.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

Array = np.ndarray


def _as_float(a) -> Array:
    return np.asarray(a, dtype=float)


def _nan_like(a: Array) -> Array:
    return np.full(len(a), np.nan)


# --------------------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------------------

def _rolling(a: Array, window: int, fn) -> Array:
    if window < 1:
        raise ValueError("window must be >= 1")
    a = _as_float(a)
    out = _nan_like(a)
    if len(a) >= window:
        out[window - 1 :] = fn(sliding_window_view(a, window), axis=1)
    return out


def rolling_mean(a: Array, window: int) -> Array:
    return _rolling(a, window, np.mean)


def rolling_std(a: Array, window: int, ddof: int = 0) -> Array:
    return _rolling(a, window, lambda w, axis: np.std(w, axis=axis, ddof=ddof))


def rolling_max(a: Array, window: int) -> Array:
    return _rolling(a, window, np.max)


def rolling_min(a: Array, window: int) -> Array:
    return _rolling(a, window, np.min)


def shift(a: Array, n: int = 1) -> Array:
    """Delay ``a`` by ``n`` bars (``out[i] = a[i-n]``), NaN-filled at the front."""
    a = _as_float(a)
    out = _nan_like(a)
    if n == 0:
        return a.copy()
    if n < 0:
        raise ValueError("shift only supports n >= 0 (negative would look ahead)")
    if n < len(a):
        out[n:] = a[:-n]
    return out


def zscore(a: Array, window: int) -> Array:
    """(a - rolling_mean) / rolling_std; NaN where std is 0 or history is short."""
    mean = rolling_mean(a, window)
    std = rolling_std(a, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (_as_float(a) - mean) / std
    z[~np.isfinite(z)] = np.nan
    return z


def pct_change(a: Array, period: int = 1) -> Array:
    """Simple return over ``period`` bars."""
    a = _as_float(a)
    out = _nan_like(a)
    if period < 1:
        raise ValueError("period must be >= 1")
    if len(a) > period:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = a[period:] / a[:-period] - 1.0
    return out


def log_returns(close: Array) -> Array:
    close = _as_float(close)
    out = _nan_like(close)
    if len(close) > 1:
        out[1:] = np.log(close[1:] / close[:-1])
    return out


def realized_vol(close: Array, window: int, annualize: int = 0) -> Array:
    """Rolling std of log returns; multiply by sqrt(annualize) if annualize > 0."""
    r = log_returns(close)
    out = np.full(len(r), np.nan)
    if len(r) > window:
        out[window:] = rolling_std(r[1:], window, ddof=1)[window - 1 :]
    if annualize:
        out = out * np.sqrt(annualize)
    return out


def linreg_slope(a: Array, window: int) -> Array:
    """Slope of an OLS line fitted to each trailing ``window`` of ``a``."""
    a = _as_float(a)
    out = _nan_like(a)
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(a) >= window:
        x = np.arange(window, dtype=float)
        x = x - x.mean()
        w = sliding_window_view(a, window)
        out[window - 1 :] = (w - w.mean(axis=1, keepdims=True)) @ x / (x @ x)
    return out


def rolling_beta(y: Array, x: Array, window: int) -> Array:
    """Rolling OLS slope of ``y`` on ``x`` (hedge ratio for pairs trading)."""
    y, x = _as_float(y), _as_float(x)
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    out = _nan_like(y)
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(y) >= window:
        wy, wx = sliding_window_view(y, window), sliding_window_view(x, window)
        dy = wy - wy.mean(axis=1, keepdims=True)
        dx = wx - wx.mean(axis=1, keepdims=True)
        var = (dx * dx).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (dx * dy).sum(axis=1) / var
        beta[var == 0] = np.nan
        out[window - 1 :] = beta
    return out


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------

def ema(a: Array, period: int) -> Array:
    """EMA seeded with the SMA of the first ``period`` values (matches analysis.indicators.ema)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    a = _as_float(a)
    out = _nan_like(a)
    if len(a) < period:
        return out
    k = 2.0 / (period + 1)
    prev = float(np.mean(a[:period]))
    out[period - 1] = prev
    for i in range(period, len(a)):
        prev = a[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def wilder(a: Array, period: int) -> Array:
    """Wilder's smoothing (RMA): EMA with alpha = 1/period, SMA-seeded.
    ``a`` may start with NaNs (e.g. true range's first bar); seeding begins at
    the first finite value."""
    if period < 1:
        raise ValueError("period must be >= 1")
    a = _as_float(a)
    out = _nan_like(a)
    finite = np.flatnonzero(np.isfinite(a))
    if len(finite) == 0:
        return out
    start = finite[0]
    if len(a) - start < period:
        return out
    prev = float(np.mean(a[start : start + period]))
    out[start + period - 1] = prev
    for i in range(start + period, len(a)):
        prev = (prev * (period - 1) + a[i]) / period
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# Range / volatility
# --------------------------------------------------------------------------

def true_range(high: Array, low: Array, close: Array) -> Array:
    """max(H-L, |H-prevC|, |L-prevC|); first bar is NaN (no previous close)."""
    high, low, close = _as_float(high), _as_float(low), _as_float(close)
    out = _nan_like(close)
    if len(close) > 1:
        prev_close = close[:-1]
        out[1:] = np.maximum.reduce(
            [high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)]
        )
    return out


def atr(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """Average True Range with Wilder smoothing."""
    return wilder(true_range(high, low, close), period)


def bollinger(close: Array, period: int = 20, num_std: float = 2.0) -> Tuple[Array, Array, Array]:
    """Returns (middle, upper, lower)."""
    mid = rolling_mean(close, period)
    sd = rolling_std(close, period)
    return mid, mid + num_std * sd, mid - num_std * sd


def keltner(
    high: Array, low: Array, close: Array, ema_period: int = 20, atr_period: int = 10, mult: float = 1.5
) -> Tuple[Array, Array, Array]:
    """Returns (middle, upper, lower): EMA(close) +/- mult * ATR."""
    mid = ema(close, ema_period)
    a = atr(high, low, close, atr_period)
    return mid, mid + mult * a, mid - mult * a


def donchian(high: Array, low: Array, period: int = 20, exclude_current: bool = True) -> Tuple[Array, Array]:
    """Returns (upper, lower): highest high / lowest low of the last ``period`` bars.
    With ``exclude_current`` (default) the window is the ``period`` bars BEFORE
    bar i, so ``close[i] > upper[i]`` is a genuine breakout test."""
    upper, lower = rolling_max(high, period), rolling_min(low, period)
    if exclude_current:
        return shift(upper, 1), shift(lower, 1)
    return upper, lower


# --------------------------------------------------------------------------
# Oscillators / trend strength
# --------------------------------------------------------------------------

def rsi(close: Array, period: int = 14) -> Array:
    """Wilder RSI in [0, 100]; 100 when there were no losses in the window."""
    close = _as_float(close)
    out = _nan_like(close)
    if len(close) <= period:
        return out
    diff = np.diff(close)
    gains = np.concatenate([[np.nan], np.maximum(diff, 0.0)])
    losses = np.concatenate([[np.nan], np.maximum(-diff, 0.0)])
    avg_gain, avg_loss = wilder(gains, period), wilder(losses, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out[(avg_loss == 0) & np.isfinite(avg_gain)] = 100.0
    out[~np.isfinite(avg_gain) | ~np.isfinite(avg_loss)] = np.nan
    return out


def stochastic(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """%K in [0, 100]; NaN when the window has zero range."""
    hh, ll = rolling_max(high, period), rolling_min(low, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (_as_float(close) - ll) / (hh - ll)
    k[~np.isfinite(k)] = np.nan
    return k


def ibs(high: Array, low: Array, close: Array) -> Array:
    """Internal bar strength: (C-L)/(H-L) in [0, 1]; NaN on zero-range bars."""
    high, low, close = _as_float(high), _as_float(low), _as_float(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (close - low) / (high - low)
    out[~np.isfinite(out)] = np.nan
    return out


def macd(close: Array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Array, Array, Array]:
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = _nan_like(line)
    finite = np.flatnonzero(np.isfinite(line))
    if len(finite):
        s = finite[0]
        sig[s:] = ema(line[s:], signal)
    return line, sig, line - sig


def adx(high: Array, low: Array, close: Array, period: int = 14) -> Tuple[Array, Array, Array]:
    """Wilder's ADX. Returns (adx, plus_di, minus_di), each in [0, 100]."""
    high, low, close = _as_float(high), _as_float(low), _as_float(close)
    n = len(close)
    nan = _nan_like(close)
    if n < 2:
        return nan, nan.copy(), nan.copy()
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr_s = wilder(true_range(high, low, close), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder(plus_dm, period) / tr_s
        minus_di = 100.0 * wilder(minus_dm, period) / tr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[~np.isfinite(dx)] = np.where(np.isfinite(plus_di) & np.isfinite(minus_di), 0.0, np.nan)[~np.isfinite(dx)]
    return wilder(dx, period), plus_di, minus_di


def supertrend(
    high: Array, low: Array, close: Array, period: int = 10, mult: float = 3.0
) -> Tuple[Array, Array]:
    """SuperTrend. Returns (line, direction) with direction +1 (uptrend, line
    below price) or -1 (downtrend, line above price); NaN during warm-up."""
    high, low, close = _as_float(high), _as_float(low), _as_float(close)
    n = len(close)
    a = atr(high, low, close, period)
    mid = (high + low) / 2.0
    basic_upper, basic_lower = mid + mult * a, mid - mult * a
    line, direction = _nan_like(close), _nan_like(close)
    first = np.flatnonzero(np.isfinite(a))
    if len(first) == 0:
        return line, direction
    s = first[0]
    final_upper, final_lower = basic_upper[s], basic_lower[s]
    d = 1.0 if close[s] >= final_lower else -1.0
    line[s], direction[s] = (final_lower if d > 0 else final_upper), d
    for i in range(s + 1, n):
        # Bands only tighten while price stays on the same side of the previous band.
        final_upper = basic_upper[i] if (basic_upper[i] < final_upper or close[i - 1] > final_upper) else final_upper
        final_lower = basic_lower[i] if (basic_lower[i] > final_lower or close[i - 1] < final_lower) else final_lower
        if d > 0 and close[i] < final_lower:
            d = -1.0
        elif d < 0 and close[i] > final_upper:
            d = 1.0
        line[i], direction[i] = (final_lower if d > 0 else final_upper), d
    return line, direction


def obv(close: Array, volume: Array) -> Array:
    """On-balance volume."""
    close, volume = _as_float(close), _as_float(volume)
    sign = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(sign * volume)


def last_finite(*values: float) -> bool:
    """True iff every value is a finite number (NaN/None -> False)."""
    for v in values:
        if v is None or not np.isfinite(v):
            return False
    return True
