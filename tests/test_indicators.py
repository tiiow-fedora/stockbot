from __future__ import annotations

import math

import pytest

from analysis import indicators


def test_sma_basic():
    values = [1, 2, 3, 4, 5, 6]
    result = indicators.sma(values, 3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[5] == pytest.approx(5.0)


def test_sma_rejects_invalid_period():
    with pytest.raises(ValueError):
        indicators.sma([1, 2, 3], 0)


def test_ema_seeds_with_sma_then_diverges():
    values = [10, 11, 12, 13, 14, 15, 20]
    result = indicators.ema(values, 3)
    assert result[0] is None and result[1] is None
    assert result[2] == pytest.approx(sum(values[:3]) / 3)
    # EMA should react more to the final jump than SMA would.
    assert result[-1] > indicators.sma(values, 3)[-1]


def test_rsi_all_gains_is_100():
    values = [10 + i for i in range(20)]  # strictly increasing
    result = indicators.rsi(values, period=14)
    assert result[14] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    values = [30 - i for i in range(20)]  # strictly decreasing
    result = indicators.rsi(values, period=14)
    assert result[14] == pytest.approx(0.0)


def test_rsi_insufficient_history_returns_none():
    values = [1, 2, 3]
    result = indicators.rsi(values, period=14)
    assert all(v is None for v in result)


def test_macd_returns_none_until_slow_ema_ready():
    values = [100 + i * 0.5 for i in range(40)]
    macd_line, signal_line, hist = indicators.macd(values, fast=12, slow=26, signal=9)
    assert macd_line[24] is None  # slow EMA (period 26) not seeded yet
    assert macd_line[25] is not None
    assert len(macd_line) == len(values) == len(signal_line) == len(hist)


def test_momentum_percent_change():
    values = [100, 100, 100, 100, 100, 110]
    result = indicators.momentum(values, period=5)
    assert result[5] == pytest.approx(0.10)


def test_volume_change_relative_to_average():
    volumes = [1000] * 10 + [2000]
    result = indicators.volume_change(volumes, period=10)
    assert result[10] == pytest.approx(1.0)  # double the trailing average


def test_volatility_zero_for_constant_prices():
    values = [100.0] * 30
    result = indicators.volatility(values, period=20)
    assert result[20] == pytest.approx(0.0, abs=1e-9)


def test_volatility_nonzero_for_noisy_prices():
    values = [100, 105, 98, 110, 95, 108, 100, 112, 90, 115] * 3
    result = indicators.volatility(values, period=20)
    assert result[20] is not None
    assert result[20] > 0
