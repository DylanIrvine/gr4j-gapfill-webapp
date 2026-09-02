# core/models.py
# GR4J, GR5J and GR6J daily lumped rainfall-runoff models.
#
# Transcribed from the Fortran in the airGR R package (frun_GR4J.f90,
# frun_GR5J.f90, frun_GR6J.f90, utils_D.f90) and verified against that code
# compiled and run on the same inputs. See tests/test_against_airgr.py.
#
# References
#   Perrin, C., Michel, C., Andreassian, V. (2003). Improvement of a
#     parsimonious model for streamflow simulation. Journal of Hydrology
#     279(1), 275-289.
#   Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T., Andreassian, V.
#     (2011). A downward structural sensitivity analysis of hydrological models
#     to improve low-flow simulation. Journal of Hydrology 411(1-2), 66-76.
#   Coron, L., Thirel, G., Delaigue, O., Perrin, C., Andreassian, V. (2017).
#     The suite of lumped GR hydrological models in an R package. Environmental
#     Modelling and Software 94, 166-171.
#
# Three structural differences between the models are easy to get wrong, so
# they are stated explicitly here:
#
#   1. GR4J and GR6J split effective rainfall 90/10 BEFORE convolution and use
#      two unit hydrographs, UH1 with base X4 and UH2 with base 2*X4. GR5J
#      routes the whole of PR through a single hydrograph of base 2*X4 and
#      splits 90/10 AFTER convolution. GR5J is not GR4J with a different
#      exchange term.
#   2. The exchange term is applied twice in GR4J and GR5J (routing store and
#      direct branch) but three times in GR6J (routing store, exponential
#      store and direct branch).
#   3. The GR6J exponential store is not bounded below at zero. Allowing it to
#      go negative is what lets it sustain a very low, slowly decaying baseflow
#      indefinitely.

# %%
import math
from dataclasses import dataclass

import numpy as np

# Optional acceleration. The GR models are sequential day loops, so they cannot
# be vectorised, and in pure Python a single GR6J calibration runs tens of
# millions of interpreted iterations. numba compiles the loop to machine code
# and typically gives one to two orders of magnitude.
#
# It is deliberately optional and deliberately NOT in requirements.txt. numba
# pins a supported numpy range, and forcing it into a deployment can drag numpy
# backwards and break something else. Where numba is absent the pure Python
# fallback runs, producing bit-identical results, only slower.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op stand-in so the decorators below work without numba."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator

# %% model registry
#
# MODEL_PARAMS is the single source of truth for what parameters a model has and
# how each one is presented. The flat PARAM_NAMES / PARAM_BOUNDS / PARAM_LABELS /
# PARAM_ROUNDING / MODEL_NOTES dictionaries below are DERIVED from it, and kept
# only so that existing callers (core/calibration.py, app.py) do not all have to
# change at once. Reverting a model is then just deleting its MODEL_PARAMS and
# MODEL_INFO entries plus its simulator.
#
# The flat dictionaries are keyed by bare parameter name, which is only safe
# while parameter names do not collide across models. GR4J..GR6J share X1..X6 by
# design; SIMHYD's names are disjoint. A registry test enforces the no-collision
# rule so a future model cannot quietly break the shims.


@dataclass(frozen=True)
class ParamSpec:
    """Everything the UI and the optimiser need to know about one parameter."""
    bounds: tuple      # (lower, upper) calibration bounds
    label: str         # human label, units in parentheses to match the old PARAM_LABELS
    units: str         # units alone, for the parameter table
    rounding: int      # decimal places for display and de-duplication
    default: float     # starting value for manual simulation
    typical: str       # human range string, e.g. '100 to 800'


# GR bounds: X1 to X4 follow the ranges already in use in this app. X5 is
# dimensionless and airGR's own calibration grid spans roughly -4 to 4. X6 is a
# depletion coefficient in mm and must be strictly positive, since it divides the
# store level.
_GR_PARAMS = {
    'X1': ParamSpec((1.0, 3000.0), 'Production Store Capacity (mm)', 'mm', 1, 500.0, '100 to 800'),
    'X2': ParamSpec((-25.0, 5.0), 'Groundwater Exchange Coefficient (mm/d)', 'mm/d', 2, 0.0, '-5 to 3'),
    'X3': ParamSpec((1.0, 1000.0), 'Routing Store Capacity (mm)', 'mm', 1, 100.0, '20 to 500'),
    'X4': ParamSpec((0.5, 20.0), 'Unit Hydrograph Time Base (days)', 'd', 2, 2.0, '1 to 10'),
    'X5': ParamSpec((-4.0, 4.0), 'Groundwater Exchange Threshold (-)', '-', 3, 0.0, '-1 to 1'),
    'X6': ParamSpec((0.01, 100.0), 'Exponential Store Depletion Coefficient (mm)', 'mm', 2, 10.0, '1 to 60'),
}

# SIMHYD (Chiew et al. 2002), the seven-parameter form without Muskingum channel
# routing. Bounds follow the ranges in common use for Australian catchments.
# SMSC divides the soil store and K divides the groundwater store, so both must
# stay strictly positive.
_SIMHYD_PARAMS = {
    'INSC': ParamSpec((0.0, 5.0), 'Interception Store Capacity (mm)', 'mm', 2, 1.0, '0.5 to 5'),
    'COEFF': ParamSpec((0.0, 400.0), 'Maximum Infiltration Loss (mm)', 'mm', 1, 200.0, '50 to 400'),
    'SQ': ParamSpec((0.0, 10.0), 'Infiltration Loss Exponent (-)', '-', 2, 2.0, '0 to 6'),
    'SMSC': ParamSpec((1.0, 1000.0), 'Soil Moisture Store Capacity (mm)', 'mm', 1, 300.0, '50 to 500'),
    'SUB': ParamSpec((0.0, 1.0), 'Interflow Coefficient (-)', '-', 3, 0.5, '0 to 1'),
    'CRAK': ParamSpec((0.0, 1.0), 'Groundwater Recharge Coefficient (-)', '-', 3, 0.5, '0 to 1'),
    'K': ParamSpec((0.003, 0.3), 'Baseflow Linear Recession (1/d)', '1/d', 4, 0.1, '0.01 to 0.3'),
}

MODEL_PARAMS = {
    'GR4J': {k: _GR_PARAMS[k] for k in ('X1', 'X2', 'X3', 'X4')},
    'GR5J': {k: _GR_PARAMS[k] for k in ('X1', 'X2', 'X3', 'X4', 'X5')},
    'GR6J': {k: _GR_PARAMS[k] for k in ('X1', 'X2', 'X3', 'X4', 'X5', 'X6')},
    'SIMHYD': dict(_SIMHYD_PARAMS),
}

# Parameters the model divides by, so their lower bound must stay strictly
# positive. Checked in simulate() and by the calibration bounds guard. K only
# multiplies (BAS = K * GW), so it is bounded away from zero by choice, not
# necessity, and is not in this set.
STRICTLY_POSITIVE_PARAMS = {'X6', 'SMSC'}


_GR_DIAGRAM_NOTES = """
**Three differences that are easy to misread.**

- **Routing split.** GR4J and GR6J split effective rainfall 90/10 *before*
  convolution and use two unit hydrographs, of base X4 and 2·X4. GR5J routes all
  of Pr through a single hydrograph of base 2·X4 and splits *after*. GR5J is not
  GR4J with a modified exchange term.
- **Exchange applications.** F is applied twice in GR4J and GR5J, three times in
  GR6J. The same X2 moves substantially more water in GR6J.
- **Exponential store.** R2 is not bounded below at zero. That is what sustains a
  slow recession indefinitely, and it is why GR6J cannot produce exactly zero flow.
"""

_SIMHYD_DIAGRAM_NOTES = """
**How SIMHYD differs in shape from the GR models.**

- **No unit hydrograph.** Runoff is the sum of three paths computed each day:
  infiltration-excess overland flow, interflow (and saturation excess), and
  baseflow drained from a linear groundwater store at rate K. There is no
  convolution and no X4-equivalent time base.
- **Three explicit stores.** Interception (INSC), soil moisture (SMSC) and
  groundwater, updated in that order every day. Infiltration capacity falls as
  the soil store fills, through COEFF·exp(−SQ·S/SMSC).
- **No atmospheric exchange term.** Unlike GR's F, SIMHYD has no gain or loss of
  water across the catchment boundary; the water balance closes on P, ET and
  storage change.
- **Baseflow is a model output.** The groundwater store's outflow is an explicit
  baseflow series, so it can be compared directly against a digital-filter
  separation of the observed hydrograph.
"""

_GR_DIAGRAM = (
    'docs/GR4J-6J_v1.1_w_background.png',
    'Production module shared by all three models, and the routing differences '
    'between GR4J, GR5J and GR6J. Structure after Perrin et al. (2003), '
    'Le Moine (2008) and Pushpalatha et al. (2011).',
    _GR_DIAGRAM_NOTES,
)

_SIMHYD_DIAGRAM = (
    'docs/SIMHYD_v1.png',
    'The three SIMHYD stores — interception, soil moisture and groundwater — and '
    'the runoff paths between them. Structure after Chiew et al. (2009).',
    _SIMHYD_DIAGRAM_NOTES,
)


@dataclass(frozen=True)
class ModelInfo:
    """Capability descriptor. app.py reads these instead of hard-coding model names."""
    n_params: int
    can_produce_zero_flow: bool      # False for GR6J: the exponential store never reaches zero
    has_exchange_threshold: bool     # True for GR5J/GR6J: exchange is X2*(R/X3 - X5)
    provides_components: bool         # True for SIMHYD: surface / interflow / baseflow split
    min_warmup_days: int             # advisory floor; 0 means no warm-up warning for this model
    warmup_note: str                 # body of that warning
    low_flow_criterion_note: str     # shown for a single untransformed criterion; '' shows nothing
    notes: str                       # one-paragraph summary, was MODEL_NOTES[model]
    diagram: tuple                   # (image_path, caption, structure_markdown)


MODEL_INFO = {
    'GR4J': ModelInfo(
        n_params=4, can_produce_zero_flow=True, has_exchange_threshold=False,
        provides_components=False, min_warmup_days=0, warmup_note='',
        low_flow_criterion_note='',
        notes='Four parameters. Exchange is X2*(R/X3)^3.5, which can only ever take '
              'the sign of X2, so the catchment either always gains or always loses water.',
        diagram=_GR_DIAGRAM,
    ),
    'GR5J': ModelInfo(
        n_params=5, can_produce_zero_flow=True, has_exchange_threshold=True,
        provides_components=False, min_warmup_days=0, warmup_note='',
        low_flow_criterion_note='',
        notes='Adds X5, a threshold in the exchange term X2*(R/X3 - X5), so exchange can '
              'reverse direction as the routing store fills and empties. Note that GR5J '
              'uses a single unit hydrograph of base 2*X4 rather than the two hydrographs '
              'of GR4J.',
        diagram=_GR_DIAGRAM,
    ),
    'GR6J': ModelInfo(
        n_params=6, can_produce_zero_flow=False, has_exchange_threshold=True,
        provides_components=False, min_warmup_days=1095,
        warmup_note='The GR6J exponential store equilibrates slowly and is initialised at '
                    'zero. With only {warmup_days} warm-up days the calibration may be '
                    'fitting the spin-up rather than the catchment. At least {min_days} days '
                    'is safer.',
        low_flow_criterion_note='GR6J adds X6 specifically to control low flows, but an '
                                'untransformed criterion is almost entirely determined by '
                                'peak flows, so X6 will be poorly constrained. Consider the '
                                'logarithmic or inverse transformation.',
        notes='Adds X5 and X6. Forty percent of the slow branch is diverted into an '
              'exponential store that is not bounded below at zero, which sustains long, '
              'slowly decaying recessions. Intended for groundwater-fed baseflow. Note '
              'that it cannot produce exactly zero flow, so it is a poor choice for '
              'catchments that dry out completely.',
        diagram=_GR_DIAGRAM,
    ),
    'SIMHYD': ModelInfo(
        n_params=7, can_produce_zero_flow=True, has_exchange_threshold=False,
        provides_components=True, min_warmup_days=365,
        warmup_note='The SIMHYD groundwater store is initialised at zero and fills only '
                    'through recharge, so it equilibrates over months to years. With only '
                    '{warmup_days} warm-up days the calibration may be fitting the spin-up '
                    'rather than the catchment. At least {min_days} days is safer.',
        low_flow_criterion_note='SIMHYD baseflow is controlled by K and the groundwater '
                                'store, which an untransformed criterion barely sees. If '
                                'low flows matter, use the logarithmic or inverse '
                                'transformation.',
        notes='Seven parameters, three stores (interception, soil moisture, groundwater). '
              'Daily runoff is the sum of infiltration-excess flow, interflow and a linear '
              'baseflow, with no unit hydrograph and no atmospheric exchange term. Baseflow '
              'is an explicit model output. Can produce exactly zero flow.',
        diagram=_SIMHYD_DIAGRAM,
    ),
}


# %% derived flat views (compatibility shims — see the note above)
MODELS = tuple(MODEL_PARAMS)

PARAM_NAMES = {model: tuple(specs) for model, specs in MODEL_PARAMS.items()}

PARAM_BOUNDS = {name: spec.bounds
                for specs in MODEL_PARAMS.values() for name, spec in specs.items()}

PARAM_LABELS = {name: spec.label
                for specs in MODEL_PARAMS.values() for name, spec in specs.items()}

PARAM_ROUNDING = {name: spec.rounding
                  for specs in MODEL_PARAMS.values() for name, spec in specs.items()}

MODEL_NOTES = {model: info.notes for model, info in MODEL_INFO.items()}


# %% unit hydrograph ordinates
def _ss1(t, x4, d=2.5):
    """S-curve for UH1, cumulative to time t."""
    if t <= 0.0:
        return 0.0
    if t < x4:
        return (t / x4) ** d
    return 1.0


def _ss2(t, x4, d=2.5):
    """S-curve for UH2, cumulative to time t."""
    if t <= 0.0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** d
    if t < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** d
    return 1.0


def _ordinates(x4):
    """UH1 and UH2 ordinates as successive differences of the S-curves."""
    n1 = int(math.ceil(x4))
    n2 = int(math.ceil(2.0 * x4))
    o1 = [_ss1(t, x4) - _ss1(t - 1, x4) for t in range(1, n1 + 1)]
    o2 = [_ss2(t, x4) - _ss2(t - 1, x4) for t in range(1, n2 + 1)]
    return o1, o2


# %% compiled kernels
# One kernel per model, each running the whole time loop on scalars and arrays
# so numba can compile it. The arithmetic is identical to the reference
# implementation; only the container types changed, from Python lists to numpy
# arrays, because numba handles arrays and does not handle lists efficiently.


@njit(cache=True)
def _production(s, p, e, x1):
    """Production store update. Returns the new level and effective rainfall."""
    if p <= e:
        en = e - p
        ws = en / x1
        if ws > 13.0:
            ws = 13.0
        tws = np.tanh(ws)
        sr = s / x1
        er = s * (2.0 - sr) * tws / (1.0 + (1.0 - sr) * tws)
        s = s - er
        pr = 0.0
    else:
        pn = p - e
        ws = pn / x1
        if ws > 13.0:
            ws = 13.0
        tws = np.tanh(ws)
        sr = s / x1
        ps = x1 * (1.0 - sr * sr) * tws / (1.0 + sr * tws)
        s = s + ps
        pr = pn - ps

    if s < 0.0:
        s = 0.0

    # percolation, with (9/4)^4 = 25.62890625
    sr4 = (s / x1) ** 4
    perc = s * (1.0 - 1.0 / (1.0 + sr4 / 25.62890625) ** 0.25)
    s = s - perc

    return s, pr + perc


@njit(cache=True)
def _routing_out(r, x3):
    return r * (1.0 - 1.0 / (1.0 + (r / x3) ** 4) ** 0.25)


@njit(cache=True)
def _exponential_out(level, x6):
    ar = level / x6
    if ar > 33.0:
        ar = 33.0
    elif ar < -33.0:
        ar = -33.0

    if ar > 7.0:
        return level + x6 / np.exp(ar)
    if ar < -7.0:
        return x6 * np.exp(ar)
    return x6 * np.log(np.exp(ar) + 1.0)


@njit(cache=True)
def _gr4j_loop(precip, pet, x1, x2, x3, x4, o1, o2):
    n = precip.shape[0]
    n1, n2 = o1.shape[0], o2.shape[0]

    uh1 = np.zeros(n1)
    uh2 = np.zeros(n2)
    out = np.empty(n)

    s = 0.3 * x1
    r = 0.5 * x3

    for i in range(n):
        s, pr = _production(s, precip[i], pet[i], x1)

        # split before convolution, 90 per cent to UH1 and 10 per cent to UH2
        pruh1 = pr * 0.9
        pruh2 = pr * 0.1

        for j in range(n1 - 1):
            uh1[j] = uh1[j + 1] + o1[j] * pruh1
        uh1[n1 - 1] = o1[n1 - 1] * pruh1

        for j in range(n2 - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pruh2
        uh2[n2 - 1] = o2[n2 - 1] * pruh2

        exch = x2 * (r / x3) ** 3.5

        r = r + uh1[0] + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_out(r, x3)
        r = r - qr

        qd = uh2[0] + exch
        if qd < 0.0:
            qd = 0.0

        total = qr + qd
        out[i] = total if total > 0.0 else 0.0

    return out


@njit(cache=True)
def _gr5j_loop(precip, pet, x1, x2, x3, x4, x5, o2):
    n = precip.shape[0]
    n2 = o2.shape[0]

    uh2 = np.zeros(n2)
    out = np.empty(n)

    s = 0.3 * x1
    r = 0.5 * x3

    for i in range(n):
        s, pr = _production(s, precip[i], pet[i], x1)

        # GR5J routes the WHOLE of PR through one hydrograph of base 2*X4
        for j in range(n2 - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pr
        uh2[n2 - 1] = o2[n2 - 1] * pr

        # and splits 90/10 AFTER convolution
        q9 = uh2[0] * 0.9
        q1 = uh2[0] * 0.1

        exch = x2 * (r / x3 - x5)

        r = r + q9 + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_out(r, x3)
        r = r - qr

        qd = q1 + exch
        if qd < 0.0:
            qd = 0.0

        total = qr + qd
        out[i] = total if total > 0.0 else 0.0

    return out


@njit(cache=True)
def _gr6j_loop(precip, pet, x1, x2, x3, x4, x5, x6, o1, o2):
    n = precip.shape[0]
    n1, n2 = o1.shape[0], o2.shape[0]

    uh1 = np.zeros(n1)
    uh2 = np.zeros(n2)
    out = np.empty(n)

    s = 0.3 * x1
    r = 0.5 * x3
    r_exp = 0.0

    for i in range(n):
        s, pr = _production(s, precip[i], pet[i], x1)

        pruh1 = pr * 0.9
        pruh2 = pr * 0.1

        for j in range(n1 - 1):
            uh1[j] = uh1[j + 1] + o1[j] * pruh1
        uh1[n1 - 1] = o1[n1 - 1] * pruh1

        for j in range(n2 - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pruh2
        uh2[n2 - 1] = o2[n2 - 1] * pruh2

        exch = x2 * (r / x3 - x5)

        # the exchange term is applied THREE times in GR6J
        r = r + 0.6 * uh1[0] + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_out(r, x3)
        r = r - qr

        # the exponential store is NOT clipped at zero
        r_exp = r_exp + 0.4 * uh1[0] + exch
        qr_exp = _exponential_out(r_exp, x6)
        r_exp = r_exp - qr_exp

        qd = uh2[0] + exch
        if qd < 0.0:
            qd = 0.0

        total = qr + qd + qr_exp
        out[i] = total if total > 0.0 else 0.0

    return out


# %% SIMHYD kernel
# Transcribed line for line from hydromad's simhyd (src/simhyd.cpp and the
# pure-R fallback in R/simhyd.R, hydromad commit fetched 2026-09-03), itself
# after Chiew et al. (2009). The C and R paths in hydromad are identical and
# with overflow_to_gw False this reproduces them exactly, so a numeric
# cross-check against a hydromad run should match to floating point.
#
# Soil-store overflow variant (overflow_to_gw):
#
#   hydromad writes
#       if (SMS > SMSC) { SMS = SMSC; REC = REC + SMS - SMSC; }
#   and the second line adds SMSC - SMSC = 0, so the excess above SMSC is
#   discarded. Chiew et al. (2009) Figure 2 instead routes that excess into the
#   groundwater store.
#
#     overflow_to_gw = False  ->  hydromad behaviour, excess discarded (default)
#     overflow_to_gw = True   ->  Chiew et al. 2009, excess added to recharge
#
#   The two agree exactly whenever the soil store never fills, which is the
#   common case; they diverge on wet catchments with a small SMSC.
#
# Other preserved hydromad choices:
#   * Neither store is clipped at zero and the total is not clipped at zero.
#     With 0 <= K < 1 and REC >= 0 the groundwater store cannot go negative, so
#     BAS >= 0 and U >= 0 across the usable parameter range. The soil store can
#     go slightly negative only for SMSC below about 10 mm, where 10*SMS/SMSC
#     soil ET can exceed the inflow; the optimiser scores that region poorly and
#     leaves it. simulate_simhyd applies a defensive max(0, .) that is a no-op in
#     the valid domain, so the app-wide "flow is non-negative" invariant holds.
#   * etmult (hydromad's maxT-to-PET multiplier) is fixed at 1.0: this app is
#     given real potential evapotranspiration in mm/d, so no conversion applies.

# Overflow-variant choices, for the UI. Maps a label to the overflow_to_gw flag.
SIMHYD_OVERFLOW_CHOICES = {
    'hydromad (soil-store overflow discarded)': False,
    'Chiew et al. 2009 (overflow recharges groundwater)': True,
}
SIMHYD_OVERFLOW_DEFAULT = False


@njit(cache=True)
def _simhyd_loop(precip, pet, insc, coeff, sq, smsc, sub, crak, k, overflow_to_gw):
    n = precip.shape[0]

    total = np.empty(n)
    surface = np.empty(n)      # IRUN, infiltration-excess (direct) runoff
    interflow = np.empty(n)    # SRUN, saturation excess and interflow
    baseflow = np.empty(n)     # BAS, linear groundwater store outflow

    sms = 0.5 * smsc           # SMSt0 = 0.5, soil moisture store
    gw = 0.0                   # GWt0 = 0, groundwater store

    for i in range(n):
        p = precip[i]
        e = pet[i]

        imax = insc if insc < e else e
        intercepted = imax if imax < p else p
        inr = p - intercepted

        infil_cap = coeff * np.exp(-sq * sms / smsc)
        rmo = infil_cap if infil_cap < inr else inr

        irun = inr - rmo
        srun = sub * sms / smsc * rmo
        rec = crak * sms / smsc * (rmo - srun)
        smf = rmo - srun - rec

        pot = e - intercepted
        et_cap = 10.0 * sms / smsc
        et = et_cap if et_cap < pot else pot

        sms = sms + smf - et
        if sms > smsc:
            if overflow_to_gw:
                rec = rec + sms - smsc     # Chiew et al. 2009: excess recharges GW
            sms = smsc
            # hydromad default (overflow_to_gw False): excess above SMSC is lost

        bas = k * gw
        gw = gw + rec - bas

        surface[i] = irun
        interflow[i] = srun
        baseflow[i] = bas
        total[i] = irun + srun + bas

    return total, surface, interflow, baseflow


# %% model entry points
def _forcing(precip, pet):
    return (np.ascontiguousarray(precip, dtype=np.float64),
            np.ascontiguousarray(pet, dtype=np.float64))


def simulate_gr4j(precip, pet, params):
    x1, x2, x3, x4 = (float(params[k]) for k in PARAM_NAMES['GR4J'])
    o1, o2 = _ordinates(x4)
    precip, pet = _forcing(precip, pet)
    return _gr4j_loop(precip, pet, x1, x2, x3, x4,
                      np.asarray(o1, dtype=np.float64), np.asarray(o2, dtype=np.float64))


def simulate_gr5j(precip, pet, params):
    x1, x2, x3, x4, x5 = (float(params[k]) for k in PARAM_NAMES['GR5J'])
    _, o2 = _ordinates(x4)
    precip, pet = _forcing(precip, pet)
    return _gr5j_loop(precip, pet, x1, x2, x3, x4, x5,
                      np.asarray(o2, dtype=np.float64))


def simulate_gr6j(precip, pet, params):
    x1, x2, x3, x4, x5, x6 = (float(params[k]) for k in PARAM_NAMES['GR6J'])
    o1, o2 = _ordinates(x4)
    precip, pet = _forcing(precip, pet)
    return _gr6j_loop(precip, pet, x1, x2, x3, x4, x5, x6,
                      np.asarray(o1, dtype=np.float64), np.asarray(o2, dtype=np.float64))


def _simhyd_args(params):
    return tuple(float(params[k]) for k in PARAM_NAMES['SIMHYD'])


def simulate_simhyd(precip, pet, params, overflow_to_gw=SIMHYD_OVERFLOW_DEFAULT):
    """SIMHYD daily runoff in mm/d.

    Returns the total only, to satisfy the simulate() contract. The defensive
    max(0, .) is a no-op for 0 <= K < 1 (see the kernel comment) and only exists
    so callers can keep assuming non-negative flow.

    overflow_to_gw selects the soil-store-overflow variant; see the kernel
    comment. False (default) matches hydromad.
    """
    precip, pet = _forcing(precip, pet)
    total, _, _, _ = _simhyd_loop(precip, pet, *_simhyd_args(params), bool(overflow_to_gw))
    return np.maximum(total, 0.0)


def simhyd_components(precip, pet, params, overflow_to_gw=SIMHYD_OVERFLOW_DEFAULT):
    """SIMHYD runoff split into its three paths, each an mm/d array.

    Keys: 'total', 'surface' (infiltration excess), 'interflow' (saturation
    excess and interflow) and 'baseflow' (groundwater store outflow). Used only
    by the hydrological-analysis panel; the calibration path calls simulate()
    and never touches this.

    overflow_to_gw must match whatever was used for the calibration.
    """
    precip, pet = _forcing(precip, pet)
    total, surface, interflow, baseflow = _simhyd_loop(
        precip, pet, *_simhyd_args(params), bool(overflow_to_gw))
    return {'total': np.maximum(total, 0.0), 'surface': surface,
            'interflow': interflow, 'baseflow': baseflow}


# %% dispatcher
_SIMULATORS = {'GR4J': simulate_gr4j, 'GR5J': simulate_gr5j, 'GR6J': simulate_gr6j,
               'SIMHYD': simulate_simhyd}


def simulate(precip, pet, params, model='GR4J',
             simhyd_overflow_to_gw=SIMHYD_OVERFLOW_DEFAULT):
    """Run the named model.

    Parameters
    ----------
    precip, pet : array-like, mm/d, must be complete
    params : dict or sequence of the parameters for the chosen model
    model : one of MODELS ('GR4J', 'GR5J', 'GR6J', 'SIMHYD')
    simhyd_overflow_to_gw : SIMHYD soil-store-overflow variant, ignored for the
        GR models. False (default) matches hydromad; True follows Chiew et al.
        (2009). It must be the same for a calibration and everything computed
        from that calibration.

    Returns
    -------
    numpy array of simulated runoff in mm/d
    """
    if model not in _SIMULATORS:
        raise ValueError(f'Unknown model {model!r}. Choose from {MODELS}.')

    names = PARAM_NAMES[model]

    if not isinstance(params, dict):
        params = dict(zip(names, params))

    missing = [k for k in names if k not in params]
    if missing:
        raise ValueError(f'{model} requires {missing} in addition to what was supplied.')

    for name in STRICTLY_POSITIVE_PARAMS:
        if name in names and params[name] <= 0.0:
            raise ValueError(f'{name} must be strictly positive, the model divides by it.')

    if model == 'SIMHYD':
        return simulate_simhyd(precip, pet, params, overflow_to_gw=simhyd_overflow_to_gw)

    return _SIMULATORS[model](precip, pet, params)
