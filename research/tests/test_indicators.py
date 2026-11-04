"""Indicator correctness: hand-computed values, cross-checks against the engine's own
list-based indicators, NaN warm-up conventions, and a generic no-look-ahead property."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import indicators as eng
from research import indicators as ind
from research import synth


@pytest.fixture(scope="module")
def ohlc():
    c = synth.gbm(300, seed=5)
    return (np.array([x.open for x in c]), np.array([x.high for x in c]),
            np.array([x.low for x in c]), np.array([x.close for x in c]), np.array([x.volume for x in c], float))


def nan_eq(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern differs"
    m = ~np.isnan(a)
    np.testing.assert_allclose(a[m], b[m], rtol=tol, atol=tol)


def to_nan(lst):
    return np.array([np.nan if v is None else v for v in lst], float)


class TestAgainstEngineIndicators:
    """Where the engine already has an indicator, ours must agree to float precision."""

    def test_sma(self, ohlc):
        nan_eq(ind.rolling_mean(ohlc[3], 20), to_nan(eng.sma(ohlc[3], 20)))

    def test_ema(self, ohlc):
        nan_eq(ind.ema(ohlc[3], 12), to_nan(eng.ema(ohlc[3], 12)))

    def test_rsi(self, ohlc):
        nan_eq(ind.rsi(ohlc[3], 14), to_nan(eng.rsi(ohlc[3], 14)), tol=1e-7)

    def test_macd(self, ohlc):
        ours = ind.macd(ohlc[3], 12, 26, 9)
        theirs = eng.macd(ohlc[3], 12, 26, 9)
        for a, b in zip(ours, theirs):
            nan_eq(a, to_nan(b), tol=1e-7)


class TestRolling:
    def test_rolling_mean_std_hand_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(ind.rolling_mean(x, 3)[2:], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(ind.rolling_std(x, 3)[2:], [np.std([1, 2, 3])] * 3)
        assert np.isnan(ind.rolling_mean(x, 3)[:2]).all()

    def test_window_longer_than_data_is_all_nan(self):
        assert np.isnan(ind.rolling_max(np.arange(3.0), 5)).all()

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            ind.rolling_mean(np.arange(5.0), 0)

    def test_shift_delays_and_forbids_negative(self):
        x = np.arange(5.0)
        np.testing.assert_array_equal(ind.shift(x, 2)[2:], [0, 1, 2])
        assert np.isnan(ind.shift(x, 2)[:2]).all()
        with pytest.raises(ValueError):
            ind.shift(x, -1)

    def test_zscore_of_constant_is_nan_not_inf(self):
        z = ind.zscore(np.full(30, 5.0), 10)
        assert np.isnan(z).all()

    def test_zscore_known_value(self):
        x = np.array([1.0, 2.0, 3.0])
        assert ind.zscore(x, 3)[-1] == pytest.approx((3 - 2) / np.std(x))

    def test_pct_change_and_log_returns(self):
        x = np.array([100.0, 110.0, 121.0])
        np.testing.assert_allclose(ind.pct_change(x)[1:], [0.1, 0.1])
        np.testing.assert_allclose(ind.log_returns(x)[1:], np.log([1.1, 1.1]))

    def test_realized_vol_matches_manual(self, ohlc):
        c = ohlc[3]
        v = ind.realized_vol(c, 20)
        lr = np.diff(np.log(c))
        assert v[100] == pytest.approx(np.std(lr[100 - 20:100], ddof=1))
        assert np.isnan(v[:20]).all() and np.isfinite(v[20:]).all()

    def test_linreg_slope_recovers_line(self):
        x = 3.0 * np.arange(50) + 7
        np.testing.assert_allclose(ind.linreg_slope(x, 10)[9:], 3.0)

    def test_rolling_beta_recovers_hedge_ratio(self):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(size=200))
        y = 2.5 * x + rng.normal(scale=0.01, size=200)
        b = ind.rolling_beta(y, x, 60)
        assert np.isnan(b[:59]).all()
        np.testing.assert_allclose(b[59:], 2.5, atol=0.05)

    def test_rolling_beta_constant_x_is_nan(self):
        assert np.isnan(ind.rolling_beta(np.arange(30.0), np.ones(30), 10)).all()

    def test_rolling_beta_length_mismatch(self):
        with pytest.raises(ValueError):
            ind.rolling_beta(np.arange(5.0), np.arange(4.0), 3)


class TestRangeIndicators:
    def test_true_range_hand_values(self):
        h, l, c = np.array([10, 12, 11.0]), np.array([9, 10, 8.0]), np.array([9.5, 11, 9.0])
        tr = ind.true_range(h, l, c)
        assert np.isnan(tr[0])
        assert tr[1] == max(2, abs(12 - 9.5), abs(10 - 9.5))   # 2.5
        assert tr[2] == max(3, abs(11 - 11), abs(8 - 11))      # 3

    def test_atr_positive_and_wilder_recursion(self, ohlc):
        _, h, l, c, _ = ohlc
        a = ind.atr(h, l, c, 14)
        assert np.isnan(a[:14]).all() and (a[14:] > 0).all()
        tr = ind.true_range(h, l, c)
        assert a[20] == pytest.approx((a[19] * 13 + tr[20]) / 14)

    def test_bollinger_symmetry(self, ohlc):
        mid, up, lo = ind.bollinger(ohlc[3], 20, 2.0)
        m = ~np.isnan(mid)
        np.testing.assert_allclose((up - mid)[m], (mid - lo)[m])
        assert (up[m] >= mid[m]).all()

    def test_keltner_channel_ordering(self, ohlc):
        _, h, l, c, _ = ohlc
        mid, up, lo = ind.keltner(h, l, c)
        m = ~np.isnan(up)
        assert (up[m] > mid[m]).all() and (lo[m] < mid[m]).all()

    def test_donchian_excludes_current_bar(self):
        h = np.array([1, 2, 3, 10.0]); l = h - 1
        up, _ = ind.donchian(h, l, 3, exclude_current=True)
        assert up[3] == 3.0  # highest of PREVIOUS 3 bars, not 10
        up2, _ = ind.donchian(h, l, 3, exclude_current=False)
        assert up2[3] == 10.0

    def test_donchian_breakout_meaningful(self, ohlc):
        _, h, l, c, _ = ohlc
        up, dn = ind.donchian(h, l, 20)
        assert (up[~np.isnan(up)] >= dn[~np.isnan(dn)]).all()


class TestOscillators:
    def test_rsi_bounds_and_extremes(self, ohlc):
        r = ind.rsi(ohlc[3], 14)
        assert np.nanmin(r) >= 0 and np.nanmax(r) <= 100
        assert ind.rsi(np.arange(1.0, 40.0), 14)[-1] == 100.0     # only gains
        assert ind.rsi(np.arange(40.0, 1.0, -1.0), 14)[-1] == pytest.approx(0.0)  # only losses

    def test_rsi_short_input_all_nan(self):
        assert np.isnan(ind.rsi(np.arange(5.0), 14)).all()

    def test_stochastic_bounds(self, ohlc):
        _, h, l, c, _ = ohlc
        k = ind.stochastic(h, l, c, 14)
        assert np.nanmin(k) >= 0 and np.nanmax(k) <= 100

    def test_ibs_known_and_degenerate(self):
        assert ind.ibs(np.array([10.0]), np.array([8.0]), np.array([9.5]))[0] == pytest.approx(0.75)
        assert np.isnan(ind.ibs(np.array([5.0]), np.array([5.0]), np.array([5.0]))[0])

    def test_macd_histogram_identity(self, ohlc):
        line, sig, hist = ind.macd(ohlc[3])
        m = ~np.isnan(hist)
        np.testing.assert_allclose(hist[m], (line - sig)[m])

    def test_adx_bounds_and_trend_vs_range(self):
        up = synth.trending(200, 0.01, 0.002, seed=1)
        rng = synth.mean_reverting(200, 4.0, 0.01, seed=1)
        def adx_last(c):
            h = np.array([x.high for x in c]); l = np.array([x.low for x in c]); cl = np.array([x.close for x in c])
            a, pdi, mdi = ind.adx(h, l, cl, 14)
            assert 0 <= np.nanmin(a) and np.nanmax(a) <= 100
            return a[-1], pdi[-1], mdi[-1]
        a_up, pdi, mdi = adx_last(up)
        a_rng, _, _ = adx_last(rng)
        assert a_up > 40 and pdi > mdi
        assert a_up > a_rng

    def test_supertrend_direction_follows_trend(self):
        for sign in (1, -1):
            c = synth.trending(200, 0.01 * sign, 0.003, seed=2)
            h = np.array([x.high for x in c]); l = np.array([x.low for x in c]); cl = np.array([x.close for x in c])
            line, d = ind.supertrend(h, l, cl, 10, 3.0)
            assert d[-1] == sign
            assert (line[-1] < cl[-1]) if sign > 0 else (line[-1] > cl[-1])

    def test_obv_signs(self):
        c = np.array([1.0, 2.0, 1.5, 1.5]); v = np.array([10.0, 10, 10, 10])
        np.testing.assert_array_equal(ind.obv(c, v), [0, 10, 0, 0])

    def test_last_finite(self):
        assert ind.last_finite(1.0, 2.0) and not ind.last_finite(1.0, float("nan")) and not ind.last_finite(None)


@pytest.mark.parametrize("fn", [
    lambda o, h, l, c, v: ind.rolling_mean(c, 20),
    lambda o, h, l, c, v: ind.rolling_std(c, 20),
    lambda o, h, l, c, v: ind.ema(c, 12),
    lambda o, h, l, c, v: ind.rsi(c, 14),
    lambda o, h, l, c, v: ind.atr(h, l, c, 14),
    lambda o, h, l, c, v: ind.adx(h, l, c, 14)[0],
    lambda o, h, l, c, v: ind.macd(c)[2],
    lambda o, h, l, c, v: ind.bollinger(c, 20)[1],
    lambda o, h, l, c, v: ind.keltner(h, l, c)[1],
    lambda o, h, l, c, v: ind.donchian(h, l, 20)[0],
    lambda o, h, l, c, v: ind.stochastic(h, l, c, 14),
    lambda o, h, l, c, v: ind.supertrend(h, l, c, 10, 3.0)[1],
    lambda o, h, l, c, v: ind.zscore(c, 20),
    lambda o, h, l, c, v: ind.realized_vol(c, 20),
    lambda o, h, l, c, v: ind.linreg_slope(c, 20),
    lambda o, h, l, c, v: ind.obv(c, v),
], ids=lambda f: "")
def test_no_lookahead_property(fn, ohlc):
    """Truncating the inputs must not change any value already computed."""
    full = fn(*ohlc)
    for cut in (80, 150, 233):
        part = fn(*[a[:cut] for a in ohlc])
        nan_eq(full[:cut], part, tol=1e-9)
