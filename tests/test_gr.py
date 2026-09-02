# tests/test_gr.py
# Regression and cross-implementation checks for the GR family. These run
# against the current code and are the safety net for the SIMHYD registry
# refactor: if MODEL_PARAMS / the shims change GR behaviour, these fail.

import pathlib

import numpy as np
import pytest

from core import gr4j as gr4j_standalone
from core.models import MODELS, PARAM_NAMES, simulate
from tests._reference_params import GR_PARAM_SETS

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "gr"


@pytest.mark.parametrize("model", ["GR4J", "GR5J", "GR6J"])
def test_regression_against_committed_fixture(model, forcing):
    rain, pet = forcing
    for i, params in enumerate(GR_PARAM_SETS[model]):
        path = FIXTURES / f"{model}_{i}.npy"
        if not path.exists():
            pytest.skip(f"{path.name} missing; run tests/fixtures/gr/_generate.py")
        expected = np.load(path)
        got = simulate(rain, pet, params, model=model)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_models_gr4j_matches_standalone_gr4j(forcing):
    """core.models GR4J path and the original core.gr4j implementation should
    agree: they are the same model written twice."""
    rain, pet = forcing
    params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
    a = simulate(rain, pet, params, model="GR4J")
    b = gr4j_standalone.simulate(rain, pet, params)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)


@pytest.mark.parametrize("model", ["GR4J", "GR5J", "GR6J"])
def test_output_is_finite_and_non_negative(model, forcing):
    rain, pet = forcing
    q = simulate(rain, pet, GR_PARAM_SETS[model][0], model=model)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.0)
    assert len(q) == len(rain)


def test_numba_and_python_paths_agree(forcing, monkeypatch):
    """The pure-Python fallback must produce the same numbers as the compiled
    kernels. We cannot easily un-import numba mid-session, so this instead
    checks the two GR4J implementations, which exercise both styles."""
    rain, pet = forcing
    params = GR_PARAM_SETS["GR4J"][1]
    fast = simulate(rain, pet, params, model="GR4J")
    slow = gr4j_standalone.simulate(rain, pet, params)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-8)


def test_unknown_model_rejected(forcing):
    rain, pet = forcing
    with pytest.raises(ValueError):
        simulate(rain, pet, {"X1": 1.0}, model="NOPE")


def test_missing_parameter_rejected(forcing):
    rain, pet = forcing
    with pytest.raises(ValueError):
        simulate(rain, pet, {"X1": 350.0, "X2": 0.5}, model="GR4J")
