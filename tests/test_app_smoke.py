# tests/test_app_smoke.py
# End-to-end smoke of app.py through Streamlit's AppTest: run the whole script
# with a stubbed CSV upload, select each model, calibrate on a tiny budget, and
# check the script does not raise and the expected sections render. This is the
# guard that the SIMHYD wiring (dropdown, registry-driven UI, D7 baseflow) holds
# together, not just the core functions.

import io

import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

from core.models import simulate
from tests.conftest import _synthetic_forcing


class _FakeUpload(io.BytesIO):
    """Enough of Streamlit's UploadedFile for app.py: a readable buffer with
    .name and .size."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def _demo_csv_bytes(model, params, n_days=6 * 365, seed=7):
    rain, pet = _synthetic_forcing(n_days=n_days)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    truth = simulate(rain, pet, params, model=model)
    rng = np.random.default_rng(seed)
    flow = np.clip(truth * (1 + rng.normal(0, 0.08, n_days)), 0.0, None)
    flow[800:860] = np.nan          # an interior gap
    flow[-40:] = np.nan             # a recent gap
    df = pd.DataFrame({
        "date": dates.strftime("%d/%m/%Y"),
        "rain_mm": rain.round(4),
        "pet_mm": pet.round(4),
        "flow_mmd": [("" if np.isnan(v) else round(v, 5)) for v in flow],
    })
    return df.to_csv(index=False).encode()


def _run_app(monkeypatch, model, params, calibrate=False):
    # a fresh buffer per call, since Streamlit re-reads the upload on every rerun
    data = _demo_csv_bytes(model, params)
    name = f"{model.lower()}_demo.csv"
    monkeypatch.setattr("streamlit.file_uploader",
                        lambda *a, **k: _FakeUpload(data, name))

    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"{model}: script raised on first run"

    def _set(label, value):
        for sb in at.selectbox:
            if sb.label == label:
                sb.set_value(value)
                return
        raise AssertionError(f"no selectbox {label!r}")

    # the column selectors exist first; the model selector only appears once the
    # columns parse, so this is two rerun stages
    _set("Date Column", "date")
    _set("Rain Column", "rain_mm")
    _set("PET Column", "pet_mm")
    _set("Flow Column", "flow_mmd")
    at.run()
    assert not at.exception, f"{model}: script raised after column selection"

    _set("Hydrological Model", model)
    at.run()
    assert not at.exception, f"{model}: script raised after model selection"

    return at


@pytest.mark.parametrize("model,params", [
    ("GR4J", {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}),
    ("SIMHYD", {"INSC": 1.8, "COEFF": 240.0, "SQ": 2.2, "SMSC": 350.0,
                "SUB": 0.35, "CRAK": 0.45, "K": 0.06}),
])
def test_app_runs_for_model(monkeypatch, model, params):
    at = _run_app(monkeypatch, model, params)

    text = " ".join(md.value for md in at.markdown)
    assert "Model Selection" in " ".join(sh.value for sh in at.subheader)
    # the capability-driven note for the model should be on the page
    if model == "SIMHYD":
        assert "SIMHYD" in text or "infiltration" in text.lower()


def test_app_calibrates_and_gapfills_simhyd(monkeypatch):
    params = {"INSC": 1.8, "COEFF": 240.0, "SQ": 2.2, "SMSC": 350.0,
              "SUB": 0.35, "CRAK": 0.45, "K": 0.06}
    at = _run_app(monkeypatch, "SIMHYD", params)

    # smallest possible calibration budget
    for ni in at.number_input:
        if ni.label == "Maximum Iterations":
            ni.set_value(2)
        elif ni.label == "Population Size":
            ni.set_value(4)
        elif ni.label == "Warm-up Days":
            ni.set_value(120)
    at.run()
    assert not at.exception

    for btn in at.button:
        if btn.label == "Calibrate":
            btn.click()
            break
    at.run()
    assert not at.exception, "calibration run raised"

    # the analysis toggle, then a full render with baseflow separation
    for cb in at.checkbox:
        if "Baseflow separation" in cb.label or cb.label.startswith("Baseflow"):
            cb.set_value(True)
    at.run()
    assert not at.exception, "analysis render raised"

    subs = " ".join(sh.value for sh in at.subheader)
    assert "Lyne" in subs                       # LH panel present
    assert "SIMHYD model components" in subs     # D7 second panel present
