"""The synthetic generators must be valid, deterministic, and actually have the structure they claim."""

from __future__ import annotations

import numpy as np
import pytest

from core.models import Candle
from research import synth


def closes(c):
    return np.array([x.close for x in c])


def lret(c):
    return np.diff(np.log(closes(c)))


GENERATORS = {
    "gbm": lambda s: synth.gbm(200, seed=s),
    "trending": lambda s: synth.trending(200, seed=s),
    "ar1": lambda s: synth.ar1_momentum(200, seed=s),
    "ou": lambda s: synth.mean_reverting(200, seed=s),
    "regime": lambda s: synth.regime_switching(200, seed=s),
    "garch": lambda s: synth.garch(200, seed=s),
    "jump": lambda s: synth.jump_diffusion(200, seed=s),
    "breakout": lambda s: synth.range_then_breakout(200, seed=s),
    "cyclical": lambda s: synth.cyclical(200, seed=s),
    "v": lambda s: synth.v_shape(200, seed=s),
    "flat": lambda s: synth.flat(200, seed=s),
}


@pytest.mark.parametrize("name", GENERATORS)
class TestEveryGenerator:
    def test_valid_candles(self, name):
        c = GENERATORS[name](1)
        assert len(c) == 200 and all(isinstance(x, Candle) for x in c)
        for x in c:
            assert x.high >= max(x.open, x.close) - 1e-9 and x.low <= min(x.open, x.close) + 1e-9
            assert x.low > 0 and x.volume >= 1 and x.timestamp.tzinfo is not None

    def test_deterministic_and_seed_sensitive(self, name):
        a, b, c = GENERATORS[name](1), GENERATORS[name](1), GENERATORS[name](2)
        assert [x.close for x in a] == [x.close for x in b]
        assert [x.close for x in a] != [x.close for x in c]

    def test_timestamps_are_increasing_business_days(self, name):
        ts = [x.timestamp for x in GENERATORS[name](1)]
        assert all(t2 > t1 for t1, t2 in zip(ts, ts[1:]))
        assert all(t.weekday() < 5 for t in ts)


def test_business_days_skips_weekends():
    days = synth.business_days(10)
    assert len(days) == 10 and all(d.weekday() < 5 for d in days)
    assert (days[2] - days[1]).days == 1


def test_candles_from_closes_rejects_bad_input():
    with pytest.raises(ValueError):
        synth.candles_from_closes([100.0, -1.0])
    with pytest.raises(ValueError):
        synth.candles_from_closes([100.0, float("nan")])


def test_trending_direction_and_gbm_null():
    assert closes(synth.trending(300, 0.004, 0.006, seed=1))[-1] > 200
    assert closes(synth.trending(300, -0.004, 0.006, seed=1))[-1] < 50
    # GBM sample mean log-return approx mu - sigma^2/2 within sampling error
    r = np.concatenate([lret(synth.gbm(2000, 0.0, 0.01, seed=s)) for s in range(5)])
    assert abs(r.mean() - (-0.5 * 0.01**2)) < 3 * 0.01 / np.sqrt(len(r))


@pytest.mark.parametrize("phi", [0.4, -0.4])
def test_ar1_autocorrelation_sign_and_size(phi):
    ac = []
    for s in range(6):
        r = lret(synth.ar1_momentum(1500, phi=phi, mu=0.0, seed=s))
        ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    assert np.mean(ac) == pytest.approx(phi, abs=0.06)


def test_ar1_rejects_nonstationary_phi():
    with pytest.raises(ValueError):
        synth.ar1_momentum(100, phi=1.0)


def test_mean_reverting_half_life():
    """AR(1) coefficient of log price must be ~ exp(-ln2/half_life)."""
    hl = 8.0
    coefs = []
    for s in range(6):
        x = np.log(closes(synth.mean_reverting(2000, half_life=hl, seed=s)))
        coefs.append(np.polyfit(x[:-1], x[1:], 1)[0])
    assert np.mean(coefs) == pytest.approx(np.exp(-np.log(2) / hl), abs=0.03)
    with pytest.raises(ValueError):
        synth.mean_reverting(100, half_life=0)


def test_regime_switching_labels_and_dispersion():
    c, labels = synth.regime_switching(3000, seed=3, return_regimes=True)
    assert len(labels) == len(c) and set(labels) <= set(synth.REGIMES)
    r = lret(c)
    lab = np.array(labels[1:])
    if (lab == "bear").sum() > 50 and (lab == "bull").sum() > 50:
        assert r[lab == "bear"].std() > r[lab == "bull"].std()


def test_garch_shows_volatility_clustering():
    r = lret(synth.garch(4000, seed=1))
    ac = np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1]
    iid = lret(synth.gbm(4000, seed=1))
    assert ac > 0.1 and ac > np.corrcoef(np.abs(iid[:-1]), np.abs(iid[1:]))[0, 1] + 0.05
    with pytest.raises(ValueError):
        synth.garch(100, alpha=0.5, beta=0.6)


def _excess_kurtosis(r):
    r = r - r.mean()
    return float(np.mean(r**4) / np.mean(r**2) ** 2 - 3.0)


def test_jump_diffusion_has_fat_tails():
    jumpy = np.concatenate([lret(synth.jump_diffusion(1500, seed=s)) for s in range(4)])
    calm = np.concatenate([lret(synth.gbm(1500, seed=s)) for s in range(4)])
    assert _excess_kurtosis(jumpy) > 2.0 > _excess_kurtosis(calm) + 1.0


def test_range_then_breakout_structure():
    c = synth.range_then_breakout(600, cycle=60, range_frac=0.5, seed=2)
    r = lret(c)
    pos = (np.arange(1, 600) % 60)
    in_range, in_break = np.abs(r[pos < 30]), np.abs(r[pos >= 30])
    assert in_break.mean() > 2 * in_range.mean()
    vol = np.array([x.volume for x in c], float)[1:]
    assert vol[pos >= 30].mean() > 1.5 * vol[pos < 30].mean()


def test_cyclical_has_expected_period():
    x = np.log(closes(synth.cyclical(800, period=40, amplitude=0.1, noise=0.001, seed=1)))
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    assert 1 / freqs[spec[1:].argmax() + 1] == pytest.approx(40, rel=0.1)


def test_v_shape_falls_then_rises():
    c = closes(synth.v_shape(300, crash_frac=0.4, seed=1))
    k = int(300 * 0.4)
    assert c[k] < c[0] * 0.7 and c[-1] > c[k] * 1.5


def test_flat_is_almost_constant():
    c = closes(synth.flat(300, price=50.0, seed=1))
    assert abs(c[-1] / c[0] - 1) < 0.01 and lret(synth.flat(300)).std() < 1e-4


class TestCointegratedPair:
    def test_shared_timestamps_and_distinct_symbols(self):
        a, b = synth.cointegrated_pair(300, seed=1)
        assert [x.timestamp for x in a] == [x.timestamp for x in b]
        assert a[0].symbol != b[0].symbol

    def test_spread_is_stationary_while_levels_are_not(self):
        """ADF-lite: AR(1) coefficient of the spread is well below 1; of each leg's log price it is ~1."""
        rho_spread, rho_leg = [], []
        for s in range(6):
            a, b = synth.cointegrated_pair(1500, beta=1.0, half_life=8.0, seed=s)
            la, lb = np.log(closes(a)), np.log(closes(b))
            spread = la - lb
            rho_spread.append(np.polyfit(spread[:-1], spread[1:], 1)[0])
            rho_leg.append(np.polyfit(la[:-1], la[1:], 1)[0])
        assert np.mean(rho_spread) < 0.95 < np.mean(rho_leg)

    def test_beta_respected(self):
        a, b = synth.cointegrated_pair(2000, beta=1.5, half_life=5.0, seed=3)
        la, lb = np.log(closes(a)), np.log(closes(b))
        slope = np.polyfit(lb, la, 1)[0]
        assert slope == pytest.approx(1.5, abs=0.15)

    def test_invalid_half_life(self):
        with pytest.raises(ValueError):
            synth.cointegrated_pair(100, half_life=-1)


def test_scenario_suite_complete_and_distinct():
    s = synth.scenario_suite(250, seed=0)
    assert len(s) >= 10 and all(len(v) == 250 for v in s.values())
    assert len({tuple(closes(v)[:5]) for v in s.values()}) == len(s)
    assert closes(s["uptrend"])[-1] > closes(s["uptrend"])[0] > closes(s["downtrend"])[-1]
