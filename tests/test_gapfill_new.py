# tests/test_gapfill_new.py
# AR(1) residual gap fill and the ensemble Kalman smoother in core/gapfill.py.

import numpy as np
import pytest

from core.models import simulate
from core.gapfill import gapfill_ar1, gapfill_enkf, gapfill_p50, gapfill_snapped
from core.gapfill import _fit_ar1


def _case(model="GR4J"):
    """The calibrated model runs 30 per cent low; the 'observed' series is that
    run scaled up by 1.3 plus light noise. A persistent multiplicative bias is
    exactly what a rainfall-perturbation ensemble spans and what the residual
    autocorrelation carries, so any method that uses the surrounding
    observations should recover most of it inside a gap."""
    rng = np.random.default_rng(3)
    n = 3500
    doy = np.arange(n) % 365
    rain = np.where(rng.random(n) < 0.4, rng.gamma(0.9, 10.0, n), 0.0).astype(float)
    pet = np.clip(3.2 - 2.7 * np.cos(2 * np.pi * (doy - 15) / 365)
                  + rng.normal(0, 0.3, n), 0.1, None)
    if model == "GR4J":
        p_cal = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
    else:
        p_cal = {"INSC": 1.8, "COEFF": 240.0, "SQ": 2.2, "SMSC": 350.0,
                 "SUB": 0.35, "CRAK": 0.45, "K": 0.06}
    q50 = simulate(rain, pet, p_cal, model=model)                 # 30 per cent low
    truth = np.clip(q50 * (1.3 + rng.normal(0.0, 0.03, n)), 0.0, None)

    q_obs = truth.copy()
    deleted = np.zeros(n, bool)
    pos = 400
    for L in (7, 21, 60):
        q_obs[pos:pos + L] = np.nan
        deleted[pos:pos + L] = True
        pos += L + 250
    return rain, pet, p_cal, q_obs, truth, deleted, q50


def test_ar1_fills_and_beats_median():
    rain, pet, p_cal, q_obs, truth, deleted, q50 = _case()

    ar1 = gapfill_ar1(q_obs, q50)
    bm = gapfill_p50(q_obs, q50)
    assert np.isfinite(ar1).all()
    assert np.array_equal(ar1[np.isfinite(q_obs)], q_obs[np.isfinite(q_obs)])
    r_ar1 = np.sqrt(np.mean((ar1[deleted] - truth[deleted]) ** 2))
    r_bm = np.sqrt(np.mean((bm[deleted] - truth[deleted]) ** 2))
    assert r_ar1 < r_bm


def test_ar1_phi_one_is_snapped():
    # equivalence holds on gaps short enough that the snapped taper stays
    # inactive (everywhere within SNAP_TAPER_EDGE_DAYS of an edge): the 7- and
    # 21-day gaps here, not the 60-day one.
    rain, pet, p_cal, q_obs, truth, deleted, q50 = _case()
    import core.gapfill as gf
    orig = gf._fit_ar1
    gf._fit_ar1 = lambda r: (float(np.nanmean(q_obs - q50)), 1.0)
    try:
        got = gapfill_ar1(q_obs, q50)
    finally:
        gf._fit_ar1 = orig
    short = deleted.copy()
    for gap in gf.identify_gaps(q_obs):
        s, e = gap["start_idx"], gap["end_idx"]
        if (e - s + 1) > 2 * gf.SNAP_TAPER_EDGE_DAYS:
            short[s:e + 1] = False
    assert short.sum() > 0
    np.testing.assert_allclose(got[short], gapfill_snapped(q_obs, q50)[short],
                               atol=1e-6)


def test_snapped_caps_and_tapers_long_gaps():
    rain, pet, p_cal, q_obs, truth, deleted, q50 = _case()
    q_obs = truth.copy()
    s, L = 800, 130
    q_obs[s:s + L] = np.nan
    q_obs[s - 1] = truth[s - 1] + 50.0 * q50[s - 1]        # absurd pre-gap residual
    filled = gapfill_snapped(q_obs, q50)

    assert np.isfinite(filled).all()
    mid = slice(s + L // 2 - 5, s + L // 2 + 5)
    assert np.all(filled[mid] < 2.0 * q50[mid] + 1e-9)     # deep interior reverts to the median
    assert np.all(filled[mid] > 0.4 * q50[mid])


@pytest.mark.parametrize("model", ["GR4J", "SIMHYD"])
def test_enkf_fills_non_negative_and_helps(model):
    rain, pet, p_cal, q_obs, truth, deleted, q50 = _case(model)
    filled = gapfill_enkf(q_obs, rain, pet, p_cal, model=model,
                          n_ensemble=48, seed=1)
    assert np.isfinite(filled).all() and np.all(filled >= 0.0)
    assert np.array_equal(filled[np.isfinite(q_obs)], q_obs[np.isfinite(q_obs)])
    open_loop = simulate(rain, pet, p_cal, model=model)
    r_enkf = np.sqrt(np.mean((filled[deleted] - truth[deleted]) ** 2))
    r_open = np.sqrt(np.mean((open_loop[deleted] - truth[deleted]) ** 2))
    assert r_enkf < r_open


def test_enkf_return_spread():
    rain, pet, p_cal, q_obs, truth, deleted, q50 = _case()
    filled, spread = gapfill_enkf(q_obs, rain, pet, p_cal, model="GR4J",
                                  n_ensemble=16, seed=0, return_spread=True)
    assert np.all(np.isfinite(spread[deleted]))
    assert np.all(np.isnan(spread[np.isfinite(q_obs)]))
