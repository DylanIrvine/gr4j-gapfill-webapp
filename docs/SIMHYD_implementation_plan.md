# Adding SIMHYD to HydroSTITCH — implementation plan

Status: **in progress on branch `feature/simhyd`**. Decisions D1–D8 / Q1–Q4 settled.

### Progress — all phases landed on `feature/simhyd`, 44 tests green

| Phase | State |
|---|---|
| 1. Test scaffold (`tests/`, `pytest.ini`, `requirements-dev.txt`, `.gitignore`) | **done** |
| 2. `core/models.py` registry + `_simhyd_loop` + entry points; `calibration.py` guard | **done** |
| 3. airGR / hydromad numeric cross-check fixtures | **skipped for now** (Dylan may run later); the in-repo hydromad transcription `tests/_simhyd_reference.py` pins the arithmetic |
| 4. `app.py` wiring (registry-driven UI, capability blocks, copy) | **done** |
| 5. D7 — dual baseflow (LH + SIMHYD components), combined CSV, in-zip README | **done** |
| 6. `docs/simhyd_structure.svg` + rendered `.png` draft | **done — draft only; Dylan to finalise** |
| 7. End-to-end pass with SIMHYD selected | **done** — `tests/test_app_smoke.py` drives the whole Streamlit script (upload → columns → model → calibrate → gapfill → both baseflow panels) for GR4J and SIMHYD via `AppTest` |

### Test inventory (`tests/`, run with the Anaconda Python)

- `test_gr.py` — GR4J/5J/6J regression fixtures; `core.models` GR4J vs standalone `core/gr4j.py`
- `test_simhyd.py` — kernel vs `_simhyd_reference.py`; component sum; water balance; regression fixture; `SMSC>0` guard
- `test_registry.py` — flat shims round-trip `MODEL_PARAMS`; no cross-model name collision; `MODEL_INFO` consistency
- `test_signatures_d7.py` — `_LH` rename; `_SIMHYD` component columns; `annual_baseflow` + `annual_baseflow_simhyd` products
- `test_app_smoke.py` — full-script `AppTest` runs for GR4J and SIMHYD, incl. a tiny SIMHYD calibration + analysis render

### Not bumped: `CAL_SCHEMA`

D7 as built computes the SIMHYD split at render time from `cal['best_params']`; it adds no keys to the stored `cal` dict, and `PARAM_NAMES[cal_model]` already handles the model-specific parameter set. A stored GR calibration stays valid. Left at 6.

### Soil-store overflow — a switch (added on request)

hydromad discards soil-store overflow (`REC += SMS - SMSC` after `SMS = SMSC`); Chiew et al.
(2009) Fig. 2 routes it into groundwater. Both are selectable:

- `_simhyd_loop(..., overflow_to_gw)` bool kernel arg; `simulate_simhyd`, `simhyd_components`,
  `simulate(..., simhyd_overflow_to_gw=False)`, `calibrate_gr(..., simhyd_overflow_to_gw=False)`
  thread it explicitly (no module-level state).
- `core.models.SIMHYD_OVERFLOW_CHOICES` — `{label: flag}` for the UI.
- `app.py`: a **Soil-store overflow** selectbox under Model Selection, shown only for SIMHYD,
  default = hydromad. Part of `data_key`; stored as `cal['simhyd_overflow_to_gw']`; carried
  into the ensemble, `q_cal`, `simhyd_components`, and the results-package README.
  `CAL_SCHEMA` 6 → 7.
- Default is hydromad (`False`), so committed fixtures and prior behaviour are unchanged.

### Diagrams

Final figures supplied by Dylan on `main` and wired into `MODEL_INFO`:
`docs/GR4J-6J_v1.1_w_background.png` (GR4J/5J/6J) and `docs/SIMHYD_v1.png` (SIMHYD).
The old `docs/gr_structures.png` is no longer referenced; Claude's draft
`docs/simhyd_structure.{svg,png}` was removed.

### Local run

`streamlit run app.py` with `C:\Users\dirvine\AppData\Local\anaconda3\python.exe`. A `.claude/launch.json` (port 8533) is included for the in-editor browser preview; its Python path is machine-specific.

### Background — hydromad's soil-store overflow

hydromad's SIMHYD (both its default C path `src/simhyd.cpp` and the R fallback)
**discards soil-store overflow** instead of routing it to groundwater:

```
if (SMS > SMSC) { SMS = SMSC; REC = REC + SMS - SMSC; }   // second line adds zero
```

Chiew et al. (2009) Fig. 2 sends that overflow to the groundwater store. Both behaviours
are now selectable — see "Soil-store overflow — a switch" above. Default = hydromad.

### Local environment (discovered 2026-09-03)

| Tool | Location (neither is on `PATH`) |
|---|---|
| Python 3.13.9 | `C:\Users\dirvine\AppData\Local\anaconda3\python.exe` — has numpy, pandas, scipy, streamlit, scikit-learn, matplotlib, numba 0.62.1 |
| R 4.5.3 | `C:\Program Files\R\R-4.5.3\bin\Rscript.exe` — `devtools`/`remotes` present, **no Rtools**, **hydromad not installed** |

`airGR` is on CRAN and ships a Windows binary (no compiler needed). `hydromad` is
GitHub-only with C sources, so installing it may require Rtools.

---

## 1. Decisions (settled)

| # | Decision | Choice |
|---|---|---|
| D1 | Numerical reference for validation | `hydromad::simhyd` (R), its canonical/default formulation. Kernel keeps the variant-sensitive choices (soil-ET form, interflow basis) as named flags so alternative variants can be switched in later without restructuring. Cite Chiew et al. (2002). |
| Q1 | Variant handling | Implement the common formulation now; add variant switches later only if the alternatives prove commonly used. |
| Q2 | Test suite | These are the **first committed tests** for the repo. Cover **all** models (GR4J/5J/6J + SIMHYD), not just SIMHYD. |
| Q3 | CSV headers | `_LH` rename confirmed for all models: `Qbase_LH_mmd`, `Qbase_SIMHYD_mmd`, etc. |
| Q4 | Reference fixtures | Claude generates them. Primary: transcribe hydromad's published SIMHYD source as an in-repo naive reference (no install dependency). Secondary: attempt a real `hydromad` install for an independent numeric cross-check; Dylan can run it if the compile fails. |
| D2 | Model scope | 7-parameter SIMHYD. **No** Muskingum channel routing (can be added later behind a checkbox). |
| D3 | Parameter registry | Introduce nested `MODEL_PARAMS`; keep the current flat `PARAM_*` dicts as **derived shims** so existing imports keep working and rollback stays cheap. |
| D4 | Model-specific UI copy | Data-driven `MODEL_INFO` capability descriptor; the ~8 GR-specific `if` blocks in `app.py` read from it. |
| D5 | Warm-up / initial stores | `INT=0`, `SMS=0.5·SMSC`, `GW=0`. `min_warmup_days`: GR6J 1095, others 365. The `warmup_days` widget is already model-agnostic; only advisory copy changes. |
| D6 | Diagram | Claude drafts `docs/simhyd_structure.svg` in the GR figure's visual language + a generator script; **Dylan produces the final figure**. Show logic via `MODEL_INFO[model].diagram`. |
| D7 | SIMHYD internal baseflow | Included now. When model is SIMHYD **and** baseflow separation is ticked: two clearly-labelled figures (Lyne–Hollick filter vs SIMHYD model components) and one combined CSV with method-flagged column headers. |
| D8 | Naming | Soften "GR4J, GR5J and GR6J" copy to "the GR family and SIMHYD". Repo name unchanged (README already brands it HydroSTITCH). |

---

## 2. What does *not* change

The core contract is `forcing (precip, pet, mm/d, complete) → simulate(...) → q_sim (mm/d)`.
SIMHYD honours it exactly, so these are untouched:

- `calibrate_gr` differential evolution, Latin-hypercube refinement, behavioural set —
  all sized off `len(names)`.
- Composite / transformed criteria, KGE bias resolution (`core/metrics.py`).
- Behavioural ensemble construction, all three gap-fill methods (`core/gapfill.py`).
- Signatures, indices, FDC, recession analysis, water-year products (`core/signatures.py`,
  `core/indices.py`, `core/evaluation.py`).
- `st.session_state['cal']` storage and run history — keyed by `PARAM_NAMES[cal_model]`
  dynamically.
- The model dropdown wiring — adding `'SIMHYD'` to `MODELS` makes it appear.

Lyne–Hollick baseflow separation (`core/baseflow.py`) is also unchanged as a function;
D7 adds a *second, parallel* separation for SIMHYD only.

---

## 3. New model code — `core/models.py`

### 3.1 Registry

```python
MODELS = ('GR4J', 'GR5J', 'GR6J', 'SIMHYD')

@dataclass(frozen=True)
class ParamSpec:
    bounds:   tuple[float, float]
    label:    str          # carries units in parentheses, matching existing PARAM_LABELS
    units:    str
    rounding: int
    default:  float
    typical:  str          # human string, e.g. "50 to 500"

MODEL_PARAMS: dict[str, dict[str, ParamSpec]] = {
    'GR4J':   {...},       # X1..X4   — values lifted verbatim from current PARAM_* dicts
    'GR5J':   {...},       # X1..X5
    'GR6J':   {...},       # X1..X6
    'SIMHYD': {
        'INSC':  ParamSpec((0.0,   5.0),  'Interception Store Capacity (mm)',     'mm', 2, 1.0,   '0.5 to 5'),
        'COEFF': ParamSpec((0.0, 400.0),  'Maximum Infiltration Loss (mm)',       'mm', 1, 200.0, '50 to 400'),
        'SQ':    ParamSpec((0.0,  10.0),  'Infiltration Loss Exponent (-)',       '-',  2, 2.0,   '0 to 6'),
        'SMSC':  ParamSpec((1.0, 1000.0), 'Soil Moisture Store Capacity (mm)',    'mm', 1, 300.0, '50 to 500'),
        'SUB':   ParamSpec((0.0,   1.0),  'Interflow Coefficient (-)',            '-',  3, 0.5,   '0 to 1'),
        'CRAK':  ParamSpec((0.0,   1.0),  'Groundwater Recharge Coefficient (-)', '-',  3, 0.5,   '0 to 1'),
        'K':     ParamSpec((0.003, 0.3),  'Baseflow Linear Recession (1/d)',      '1/d',4, 0.1,   '0.01 to 0.3'),
    },
}

# --- derived shims: keep the current flat API so calibration.py and app.py imports
#     do not all have to change in one commit, and so reverting SIMHYD is a small diff.
PARAM_NAMES    = {m: tuple(p)                         for m, p in MODEL_PARAMS.items()}
PARAM_BOUNDS   = {n: s.bounds  for m in MODEL_PARAMS for n, s in MODEL_PARAMS[m].items()}
PARAM_LABELS   = {n: s.label   for m in MODEL_PARAMS for n, s in MODEL_PARAMS[m].items()}
PARAM_ROUNDING = {n: s.rounding for m in MODEL_PARAMS for n, s in MODEL_PARAMS[m].items()}
```

Flat shims are safe today because SIMHYD's names (`INSC`…) don't collide with `X1..X6`.
That constraint is documented next to the shim. `app.py`'s hardcoded `PARAM_DEFAULTS`
(L125) and the `UNITS` / `TYPICAL` dicts in the model expander (L938–940) are **deleted**
and read from `MODEL_PARAMS` instead.

### 3.2 Capability descriptor (D4)

```python
@dataclass(frozen=True)
class ModelInfo:
    n_params:                int
    can_produce_zero_flow:   bool
    has_exchange_threshold:  bool   # the X2*(R/X3 - X5) caption; True for GR5J/GR6J
    min_warmup_days:         int
    provides_components:     bool   # True for SIMHYD -> enables D7 figures/CSV
    notes:                   str    # replaces MODEL_NOTES[model]
    diagram:                 tuple[str, str, str]   # (image_path, caption, markdown_notes)

MODEL_INFO = {
    'GR4J':   ModelInfo(4, True,  False, 365,  False, "...", (GR_PNG, GR_CAPTION, GR_NOTES)),
    'GR5J':   ModelInfo(5, True,  True,  365,  False, "...", (GR_PNG, GR_CAPTION, GR_NOTES)),
    'GR6J':   ModelInfo(6, False, True,  1095, False, "...", (GR_PNG, GR_CAPTION, GR_NOTES)),
    'SIMHYD': ModelInfo(7, True,  False, 365,  True,  "...", ('docs/simhyd_structure.png',
                                                             SIMHYD_CAPTION, SIMHYD_NOTES)),
}
```

`MODEL_NOTES` stays as a derived `{m: MODEL_INFO[m].notes}` shim.

### 3.3 Simulation kernel

numba `@njit(cache=True)` day loop in the existing style (see `_gr4j_loop` etc.):

```python
@njit(cache=True)
def _simhyd_loop(precip, pet, insc, coeff, sq, smsc, sub, crak, k):
    n = precip.shape[0]
    total     = np.empty(n)
    surface   = np.empty(n)   # infiltration-excess runoff
    interflow = np.empty(n)
    baseflow  = np.empty(n)

    intercept_store = 0.0
    sms = 0.5 * smsc
    gw  = 0.0

    for i in range(n):
        p, ep = precip[i], pet[i]

        imax        = min(insc, ep)
        intercepted = min(imax, p)
        throughfall = p - intercepted

        smf     = sms / smsc
        inf_cap = coeff * np.exp(-sq * smf)
        infil   = min(throughfall, inf_cap)
        infil_excess = throughfall - infil

        inter   = sub  * smf * infil
        recharge = crak * smf * (infil - inter)
        soil_in = infil - inter - recharge

        pot_soil_et = ep - intercepted
        soil_et = min(pot_soil_et, 10.0 * smf)          # <-- variant-dependent, pin in D1
        soil_et = min(soil_et, sms + soil_in)

        sms += soil_in - soil_et
        if sms > smsc:
            recharge += sms - smsc
            sms = smsc
        if sms < 0.0:
            sms = 0.0

        gw += recharge
        bf  = k * gw
        gw -= bf

        surface[i]   = infil_excess
        interflow[i] = inter
        baseflow[i]  = bf
        total[i]     = infil_excess + inter + bf if (infil_excess + inter + bf) > 0.0 else 0.0

    return total, surface, interflow, baseflow


def simulate_simhyd(precip, pet, params):
    args = (float(params[p]) for p in PARAM_NAMES['SIMHYD'])
    precip, pet = _forcing(precip, pet)
    total, _, _, _ = _simhyd_loop(precip, pet, *args)
    return total


def simhyd_components(precip, pet, params):
    """Total, surface, interflow and baseflow as mm/d arrays. Used only by the
    D7 analysis panel; the calibration path uses simulate() and never sees this."""
    args = [float(params[p]) for p in PARAM_NAMES['SIMHYD']]
    precip, pet = _forcing(precip, pet)
    total, surface, interflow, baseflow = _simhyd_loop(precip, pet, *args)
    return {'total': total, 'surface': surface, 'interflow': interflow, 'baseflow': baseflow}


_SIMULATORS = {'GR4J': simulate_gr4j, 'GR5J': simulate_gr5j,
               'GR6J': simulate_gr6j, 'SIMHYD': simulate_simhyd}
```

`simulate()` dispatcher: add the `SIMHYD` positivity guard alongside the existing GR6J
one (`SMSC > 0`, `0 < K < 1`), or better, generalise to a per-`ParamSpec` check that
lower bounds respect the spec.

### 3.4 Pure-Python parity

The `njit` no-op fallback already covers this; `_simhyd_loop` is written in the same
scalar style as the GR kernels so it runs identically without numba, only slower.

---

## 4. Test suite (first committed tests — Q2)

Set up `tests/` with `pytest`, add `pytest` to `requirements.txt` (or a
`requirements-dev.txt`). Run with the Anaconda Python above. Cover all four models.

### 4.1 `tests/test_simhyd.py`
- `tests/_simhyd_reference.py` — a deliberately naive, pure-Python transcription of
  hydromad's SIMHYD (`sim.R` / `simhyd.c`), no numba, written for readability not speed.
  Header cites the exact hydromad source revision transcribed.
- Assert `_simhyd_loop` (numba path **and** njit-disabled path) matches the reference to
  < 1e-9 mm/d on several parameter sets over a multi-year synthetic record.
- Water balance: `sum(P) == sum(Qsim) + sum(actual_ET) + ΔS_interception + ΔS_soil + ΔS_gw`
  to floating-point tolerance.
- Components: `total == surface + interflow + baseflow` before the `>0` clip.
- Monotonicity / bounds sanity: `SMS ∈ [0, SMSC]`, all stores finite, `Qsim ≥ 0`.
- If a real `hydromad` install succeeds: an extra test comparing against CSV fixtures in
  `tests/fixtures/simhyd/` (skipped with a clear reason if the fixtures are absent).

### 4.2 `tests/test_gr.py`
- **Cross-implementation**: `core.gr4j.simulate` vs `core.models.simulate(..., 'GR4J')`
  on shared inputs — they should already agree; locks that in.
- **Regression fixtures**: current `core.models` output for GR4J/5J/6J on a synthetic
  record saved to `tests/fixtures/gr/`; test asserts future runs still match. Guards
  against accidental breakage during the registry refactor.
- **airGR cross-check** (`tests/test_against_airgr.py`): an `Rscript` helper
  (`tests/r/gen_airgr_fixtures.R`) installs `airGR` from CRAN and writes reference
  `Qsim` for GR4J/5J/6J. Test loads the fixtures and asserts a tight match; `skipif`
  when the fixtures / R are unavailable so CI without R still passes.

### 4.3 `tests/test_registry.py`
- Every `model in MODELS` has a `MODEL_PARAMS` entry, a `MODEL_INFO` entry, a
  `_SIMULATORS` entry, and consistent `n_params`.
- The derived flat shims (`PARAM_BOUNDS` etc.) round-trip against `MODEL_PARAMS`.
- No parameter-name collision across models (the assumption the flat shims rely on).

---

## 5. `app.py` changes

### 5.1 Mechanical / copy (D8)
- L2 header comment; L741–742, L763–764 intro copy → "the GR family and SIMHYD".

### 5.2 Registry-driven (D3)
- Delete `PARAM_DEFAULTS` (L125); read `MODEL_PARAMS[model][name].default`.
- Model expander (L918–958): `UNITS` / `TYPICAL` dicts deleted; table built from
  `MODEL_PARAMS[model]`. Image + notes from `MODEL_INFO[model].diagram`.
- Manual-sim inputs (L985–991) and bounds UI (L1197–1216) already loop `param_names`;
  the X6-specific bound check (L1213) generalises to "lower bound must exceed the
  spec's lower bound where the spec is strictly positive".

### 5.3 Capability-driven (D4) — replace these blocks
| Line | Now | Becomes |
|---|---|---|
| L961 | `if model == 'GR6J' and zero_fraction > 0.05` | `if not MODEL_INFO[model].can_produce_zero_flow and ...` |
| L967 | `if model in ('GR5J', 'GR6J')` exchange caption | `if MODEL_INFO[model].has_exchange_threshold` |
| L1068 | GR6J untransformed-criterion warning | `if not MODEL_INFO[model].can_produce_zero_flow and ...` (keep GR6J wording keyed to model) |
| L1075 | `warmup_days < GR6J_MIN_WARMUP` | `warmup_days < MODEL_INFO[model].min_warmup_days`; drop the `GR6J_MIN_WARMUP` constant |
| L1624 | GR6J equifinality caveat | show when `MODEL_INFO[model].n_params >= 6` (covers GR6J + SIMHYD) |
| L1767 | "necessary for GR6J" CTF help text | reword generically; mention GR6J and note SIMHYD *can* reach zero |
| L1907 | `if cal_model == 'GR6J' and ctf_threshold == 0.0` | `if not MODEL_INFO[cal_model].can_produce_zero_flow and ...` |

### 5.4 `run_model` cache
`run_model(rain, pet, model, param_values)` (L501) is generic already — no change.

### 5.5 CAL schema
Storage (L1297) is keyed dynamically, but D7 adds keys (`q_simhyd_components`, BFI-model).
Bump `CAL_SCHEMA` 6 → 7.

---

## 6. D7 — SIMHYD model baseflow alongside Lyne–Hollick

### 6.1 Concept (state this in the UI)
- **Lyne–Hollick** is signal separation applied to `q_gapfilled` (observed spliced with
  the behavioural median in gaps).
- **SIMHYD components** come from a pure model run of the **calibrated best parameter
  set** over the whole record: `simhyd_components(rain, pet, best_params)`.
- They are different quantities on different series. The panel must say so.

### 6.2 UI (in the `if show_analysis:` block, ~L1771–1844)
- Keep the existing LH controls and figure. Retitle its figure/subheader
  **"Baseflow separation — Lyne–Hollick digital filter"**.
- When `MODEL_INFO[cal_model].provides_components`: add a second figure
  **"Baseflow separation — SIMHYD model components"**: stacked area of
  surface / interflow / baseflow (mm/d), log-y, hold-out shading as elsewhere.
- Metrics row: show both `BFI (Lyne–Hollick)` and `BFI (SIMHYD model)` where available.
- `simhyd_components` result cached like `run_baseflow` (`@st.cache_data`), keyed on
  `best_params` + forcing.

### 6.3 Combined CSV (D7 — "headers flag which is which")
`build_daily_frame` (`core/signatures.py` L85) gains an optional
`model_components: dict | None` argument. LH columns are **renamed** with an `_LH`
suffix for every model (Q3); SIMHYD component columns are added only when components
are supplied:

| Column | Meaning |
|---|---|
| `Qbase_LH_mmd`, `Qbase_LH_MLd`, `Qquick_LH_MLd` | rename of current `Qbase_mmd` / `Qbase_MLd` / `Qquick_MLd` — Lyne–Hollick filter on the gap-filled series |
| `Qbase_SIMHYD_mmd`, `Qbase_SIMHYD_MLd` | SIMHYD groundwater store outflow |
| `Qsurface_SIMHYD_mmd`, `Qsurface_SIMHYD_MLd` | infiltration-excess runoff |
| `Qinterflow_SIMHYD_mmd`, `Qinterflow_SIMHYD_MLd` | interflow + saturation excess |
| `Qtotal_SIMHYD_mmd` | model total (diagnostic; differs from `Q_mmd`, which is the gap-filled series) |

> The `_LH` rename touches the download schema for **all** models. Note it in the
> in-zip README (`app.py` ~L2370) and the changelog.

- `annual_baseflow` product (`core/signatures.py` L210) keys off `Qbase_MLd`; update to
  `Qbase_LH_MLd`. Add a parallel `annual_baseflow_simhyd` product (same shape) when
  components are present. Trend inputs (`app.py` L2134) merge whichever exist.
- In-zip README: add two lines distinguishing the LH filter from the SIMHYD components.

### 6.4 Not in v1
- No P5–P95 band on the SIMHYD baseflow (best-fit line only). The ensemble is already
  simulated; adding a component band later is cheap. Noted, deferred.

---

## 7. The diagram (D6)

- Claude drafts `docs/simhyd_structure.svg`: three stores (interception, soil moisture,
  groundwater) as boxes; labelled flux arrows — rainfall, throughfall, interception loss,
  infiltration, infiltration-excess runoff, interflow, recharge, soil ET, baseflow;
  same box/arrow/typography style as `docs/gr_structures.png`.
- Ship a generator script (`docs/make_simhyd_diagram.py` or the SVG itself) so the figure
  is reproducible.
- Dylan replaces it with the final version; filename `docs/simhyd_structure.png` stays
  stable so `MODEL_INFO` needs no edit.
- Expander at `app.py` L918 already becomes `MODEL_INFO[model].diagram`-driven in §5.2.

---

## 8. Rollback (D3 concern)

1. Primary: implement on branch `feature/simhyd`; `main` untouched until merge.
2. In-code: the flat `PARAM_*` / `MODEL_NOTES` shims mean reverting = drop the `SIMHYD`
   entries from `MODEL_PARAMS` / `MODEL_INFO` / `_SIMULATORS` + the `simhyd_*` functions.
   No caller rewrites to undo.
3. `core/gr4j.py` (the original standalone pure-Python GR4J) is not touched.

---

## 9. Phasing

Work on branch `feature/simhyd`.

1. **Test scaffold**: `tests/` + `pytest`, `conftest.py` with a synthetic forcing
   fixture. `tests/test_gr.py` (cross-impl + regression fixtures) green against the
   **current** code, before any refactor — this is the safety net.
2. `core/models.py`: `MODEL_PARAMS` + derived shims, `ModelInfo`/`MODEL_INFO`,
   `_simhyd_loop`, `simulate_simhyd`, `simhyd_components`, dispatcher + positivity guard.
   `tests/test_registry.py` + `tests/test_simhyd.py` green.
3. `tests/r/gen_airgr_fixtures.R` + `tests/test_against_airgr.py` (skip if no R).
   Attempt `hydromad` install; add the numeric SIMHYD cross-check if it succeeds.
4. `app.py`: registry-driven expander/inputs/bounds; capability-driven warning blocks;
   dropdown falls out for free; copy edits; `CAL_SCHEMA` → 7.
5. D7: `build_daily_frame` signature + `_LH`/`_SIMHYD` columns, second figure, dual BFI,
   `annual_baseflow_simhyd`, in-zip README.
6. `docs/simhyd_structure.svg` draft → Dylan finalises.
7. End-to-end pass with SIMHYD selected on a real catchment: manual sim → calibrate →
   ensemble → gap-fill → both baseflow separations → workbook + zip export.

---

## 10. Resolved follow-ups

1. **Soil-ET form** — match hydromad's canonical `simhyd` exactly, pinned from its
   published source; variant-sensitive lines kept behind named flags for later switches
   (Q1). Transcription recorded in `tests/_simhyd_reference.py`.
2. **`tests/` layout** — none exists; this is the first committed suite, covering all
   four models (Q2).
3. **CSV `_LH` rename** — confirmed for all models (Q3).
4. **hydromad access** — Claude generates fixtures: primary path is the in-repo
   transcription (no install); secondary is a real `hydromad` install for a numeric
   cross-check, with Dylan as fallback if the C compile needs Rtools (Q4).
