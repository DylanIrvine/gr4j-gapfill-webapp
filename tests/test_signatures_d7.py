# tests/test_signatures_d7.py
# The dual baseflow feature (D7): Lyne-Hollick columns get an _LH suffix for
# every model, and SIMHYD's own runoff split is carried alongside under _SIMHYD
# when supplied.

import numpy as np
import pandas as pd
import pytest

from core.baseflow import lyne_hollick
from core.models import simhyd_components, simulate
from core.signatures import build_all_products, build_daily_frame
from tests._reference_params import SIMHYD_PARAM_SETS


@pytest.fixture
def daily_frame_with_components(forcing):
    rain, pet = forcing
    dates = pd.date_range("2016-01-01", periods=len(rain), freq="D")
    params = SIMHYD_PARAM_SETS[0]

    q = simulate(rain, pet, params, model="SIMHYD")
    q_obs = q.copy()
    q_obs[400:460] = np.nan
    q_fill = pd.Series(q_obs).interpolate(limit_direction="both").to_numpy()

    bf = lyne_hollick(q_fill)["baseflow"]
    comp = simhyd_components(rain, pet, params)

    frame = build_daily_frame(dates, q_fill, np.isnan(q_obs).astype(int),
                              area_km2=200.0, start_month=9,
                              baseflow_mmd=bf, ctf_flag=(q_fill <= 0.0),
                              model_components=comp)
    return frame, comp


def test_lyne_hollick_columns_are_suffixed(daily_frame_with_components):
    frame, _ = daily_frame_with_components
    for col in ("Qbase_LH_mmd", "Qbase_LH_MLd", "Qquick_LH_MLd"):
        assert col in frame.columns
    # the old unsuffixed names must be gone
    for col in ("Qbase_mmd", "Qbase_MLd", "Qquick_MLd"):
        assert col not in frame.columns


def test_simhyd_component_columns_present_and_consistent(daily_frame_with_components):
    frame, comp = daily_frame_with_components
    np.testing.assert_allclose(frame["Qsurface_SIMHYD_mmd"], comp["surface"])
    np.testing.assert_allclose(frame["Qinterflow_SIMHYD_mmd"], comp["interflow"])
    np.testing.assert_allclose(frame["Qbase_SIMHYD_mmd"], comp["baseflow"])
    np.testing.assert_allclose(
        frame["Qtotal_SIMHYD_mmd"],
        frame["Qsurface_SIMHYD_mmd"] + frame["Qinterflow_SIMHYD_mmd"]
        + frame["Qbase_SIMHYD_mmd"], atol=1e-12)
    # ML/d columns are mm/d * area
    np.testing.assert_allclose(frame["Qbase_SIMHYD_MLd"],
                               frame["Qbase_SIMHYD_mmd"] * 200.0)


def test_products_include_both_baseflow_tables(daily_frame_with_components):
    frame, _ = daily_frame_with_components
    products = build_all_products(frame, 200.0, start_month=9)
    assert "annual_baseflow" in products            # Lyne-Hollick
    assert "annual_baseflow_simhyd" in products     # SIMHYD model
    combined = products["daily_baseflow"]
    assert "Qbase_LH_MLd" in combined.columns
    assert "Qbase_SIMHYD_MLd" in combined.columns


def test_frame_without_components_has_no_simhyd_columns(forcing):
    rain, pet = forcing
    dates = pd.date_range("2016-01-01", periods=len(rain), freq="D")
    q = simulate(rain, pet, {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}, model="GR4J")
    bf = lyne_hollick(q)["baseflow"]
    frame = build_daily_frame(dates, q, np.zeros(len(q)), area_km2=100.0,
                              start_month=1, baseflow_mmd=bf)
    assert "Qbase_LH_MLd" in frame.columns
    assert not any("SIMHYD" in c for c in frame.columns)
    products = build_all_products(frame, 100.0)
    assert "annual_baseflow" in products
    assert "annual_baseflow_simhyd" not in products
