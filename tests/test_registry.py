# tests/test_registry.py
# The MODEL_PARAMS nested registry is the source of truth; the flat PARAM_*
# dictionaries are derived shims. These tests hold that relationship in place so
# a future model cannot silently break callers that still use the flat views.

import pytest

from core.models import (MODELS, MODEL_PARAMS, MODEL_INFO, PARAM_NAMES,
                         PARAM_BOUNDS, PARAM_LABELS, PARAM_ROUNDING, MODEL_NOTES,
                         STRICTLY_POSITIVE_PARAMS, _SIMULATORS)


def test_models_tuple_matches_registry():
    assert MODELS == tuple(MODEL_PARAMS)
    assert set(MODELS) == set(MODEL_INFO) == set(_SIMULATORS)


@pytest.mark.parametrize("model", MODELS)
def test_each_model_is_internally_consistent(model):
    specs = MODEL_PARAMS[model]
    info = MODEL_INFO[model]
    assert info.n_params == len(specs)
    assert PARAM_NAMES[model] == tuple(specs)
    for name, spec in specs.items():
        assert spec.bounds[0] < spec.bounds[1]
        assert spec.bounds[0] <= spec.default <= spec.bounds[1]
        assert spec.label.strip() != ""


def test_no_parameter_name_collision_across_models():
    """The flat shims are keyed by bare parameter name, so a name shared by two
    models must mean the same thing in both (same spec)."""
    seen = {}
    for model, specs in MODEL_PARAMS.items():
        for name, spec in specs.items():
            if name in seen:
                assert seen[name] == spec, (
                    f"{name} has different specs in {model} and elsewhere; the flat "
                    f"PARAM_* shims cannot represent that")
            else:
                seen[name] = spec


def test_flat_shims_round_trip():
    for model, specs in MODEL_PARAMS.items():
        for name, spec in specs.items():
            assert PARAM_BOUNDS[name] == spec.bounds
            assert PARAM_LABELS[name] == spec.label
            assert PARAM_ROUNDING[name] == spec.rounding
        assert MODEL_NOTES[model] == MODEL_INFO[model].notes


def test_strictly_positive_params_exist_and_have_positive_bounds():
    for name in STRICTLY_POSITIVE_PARAMS:
        assert name in PARAM_BOUNDS
        assert PARAM_BOUNDS[name][0] > 0.0


@pytest.mark.parametrize("model", MODELS)
def test_diagram_descriptor_shape(model):
    image, caption, notes = MODEL_INFO[model].diagram
    assert image.endswith(".png")
    assert caption and notes
