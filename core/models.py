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
import numpy as np

# %% model registry
MODELS = ('GR4J', 'GR5J', 'GR6J')

PARAM_NAMES = {
    'GR4J': ('X1', 'X2', 'X3', 'X4'),
    'GR5J': ('X1', 'X2', 'X3', 'X4', 'X5'),
    'GR6J': ('X1', 'X2', 'X3', 'X4', 'X5', 'X6'),
}

# Calibration bounds. X1 to X4 follow the ranges already in use in this app.
# X5 is dimensionless and airGR's own calibration grid spans roughly -4 to 4.
# X6 is a depletion coefficient in mm and must be strictly positive, since it
# divides the store level.
PARAM_BOUNDS = {
    'X1': (1.0, 3000.0),
    'X2': (-25.0, 5.0),
    'X3': (1.0, 1000.0),
    'X4': (0.5, 20.0),
    'X5': (-4.0, 4.0),
    'X6': (0.01, 100.0),
}

PARAM_LABELS = {
    'X1': 'Production Store Capacity (mm)',
    'X2': 'Groundwater Exchange Coefficient (mm/d)',
    'X3': 'Routing Store Capacity (mm)',
    'X4': 'Unit Hydrograph Time Base (days)',
    'X5': 'Groundwater Exchange Threshold (-)',
    'X6': 'Exponential Store Depletion Coefficient (mm)',
}

PARAM_ROUNDING = {'X1': 1, 'X2': 2, 'X3': 1, 'X4': 2, 'X5': 3, 'X6': 2}

MODEL_NOTES = {
    'GR4J': 'Four parameters. Exchange is X2*(R/X3)^3.5, which can only ever take '
            'the sign of X2, so the catchment either always gains or always loses water.',
    'GR5J': 'Adds X5, a threshold in the exchange term X2*(R/X3 - X5), so exchange can '
            'reverse direction as the routing store fills and empties. Note that GR5J '
            'uses a single unit hydrograph of base 2*X4 rather than the two hydrographs '
            'of GR4J.',
    'GR6J': 'Adds X5 and X6. Forty percent of the slow branch is diverted into an '
            'exponential store that is not bounded below at zero, which sustains long, '
            'slowly decaying recessions. Intended for groundwater-fed baseflow. Note '
            'that it cannot produce exactly zero flow, so it is a poor choice for '
            'catchments that dry out completely.',
}


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


def _production(s, p, e, x1):
    """Production store update. Returns (new S, PR, actual ET)."""
    if p <= e:
        en = e - p
        ws = min(en / x1, 13.0)
        tws = math.tanh(ws)
        sr = s / x1
        er = s * (2.0 - sr) * tws / (1.0 + (1.0 - sr) * tws)
        s = s - er
        pr = 0.0
        ae = er + p
    else:
        pn = p - e
        ws = min(pn / x1, 13.0)
        tws = math.tanh(ws)
        sr = s / x1
        ps = x1 * (1.0 - sr * sr) * tws / (1.0 + sr * tws)
        s = s + ps
        pr = pn - ps
        ae = e

    if s < 0.0:
        s = 0.0

    # percolation, with (9/4)^4 = 25.62890625
    sr = (s / x1) ** 4
    perc = s * (1.0 - 1.0 / (1.0 + sr / 25.62890625) ** 0.25)
    s = s - perc

    return s, pr + perc, ae


def _routing_outflow(r, x3):
    """Outflow from the non-linear routing store."""
    return r * (1.0 - 1.0 / (1.0 + (r / x3) ** 4) ** 0.25)


def _exponential_outflow(level, x6):
    """Outflow from the GR6J exponential store, with airGR's numerical guards."""
    ar = level / x6
    ar = max(-33.0, min(33.0, ar))

    if ar > 7.0:
        return level + x6 / math.exp(ar)
    if ar < -7.0:
        return x6 * math.exp(ar)
    return x6 * math.log(math.exp(ar) + 1.0)


# %% GR4J
def simulate_gr4j(precip, pet, params):
    x1, x2, x3, x4 = (params[k] for k in PARAM_NAMES['GR4J'])

    o1, o2 = _ordinates(x4)
    uh1 = [0.0] * len(o1)
    uh2 = [0.0] * len(o2)

    s = 0.3 * x1
    r = 0.5 * x3

    n = len(precip)
    out = np.empty(n)

    for i in range(n):
        s, pr, _ = _production(s, precip[i], pet[i], x1)

        # split before convolution, 90 percent to UH1 and 10 percent to UH2
        pruh1 = pr * 0.9
        pruh2 = pr * 0.1

        for j in range(len(o1) - 1):
            uh1[j] = uh1[j + 1] + o1[j] * pruh1
        uh1[-1] = o1[-1] * pruh1

        for j in range(len(o2) - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pruh2
        uh2[-1] = o2[-1] * pruh2

        exch = x2 * (r / x3) ** 3.5

        r = r + uh1[0] + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_outflow(r, x3)
        r = r - qr

        qd = max(0.0, uh2[0] + exch)

        out[i] = max(0.0, qr + qd)

    return out


# %% GR5J
def simulate_gr5j(precip, pet, params):
    x1, x2, x3, x4, x5 = (params[k] for k in PARAM_NAMES['GR5J'])

    # GR5J uses UH2 only, with the whole of PR routed through it
    _, o2 = _ordinates(x4)
    uh2 = [0.0] * len(o2)

    s = 0.3 * x1
    r = 0.5 * x3

    n = len(precip)
    out = np.empty(n)

    for i in range(n):
        s, pr, _ = _production(s, precip[i], pet[i], x1)

        for j in range(len(o2) - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pr
        uh2[-1] = o2[-1] * pr

        # split after convolution
        q9 = uh2[0] * 0.9
        q1 = uh2[0] * 0.1

        exch = x2 * (r / x3 - x5)

        r = r + q9 + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_outflow(r, x3)
        r = r - qr

        qd = max(0.0, q1 + exch)

        out[i] = max(0.0, qr + qd)

    return out


# %% GR6J
def simulate_gr6j(precip, pet, params):
    x1, x2, x3, x4, x5, x6 = (params[k] for k in PARAM_NAMES['GR6J'])

    o1, o2 = _ordinates(x4)
    uh1 = [0.0] * len(o1)
    uh2 = [0.0] * len(o2)

    s = 0.3 * x1
    r = 0.5 * x3
    r_exp = 0.0

    n = len(precip)
    out = np.empty(n)

    for i in range(n):
        s, pr, _ = _production(s, precip[i], pet[i], x1)

        pruh1 = pr * 0.9
        pruh2 = pr * 0.1

        for j in range(len(o1) - 1):
            uh1[j] = uh1[j + 1] + o1[j] * pruh1
        uh1[-1] = o1[-1] * pruh1

        for j in range(len(o2) - 1):
            uh2[j] = uh2[j + 1] + o2[j] * pruh2
        uh2[-1] = o2[-1] * pruh2

        exch = x2 * (r / x3 - x5)

        # 60 percent of the UH1 output to the routing store
        r = r + 0.6 * uh1[0] + exch
        if r < 0.0:
            r = 0.0

        qr = _routing_outflow(r, x3)
        r = r - qr

        # 40 percent to the exponential store, which is NOT clipped at zero
        r_exp = r_exp + 0.4 * uh1[0] + exch
        qr_exp = _exponential_outflow(r_exp, x6)
        r_exp = r_exp - qr_exp

        qd = max(0.0, uh2[0] + exch)

        out[i] = max(0.0, qr + qd + qr_exp)

    return out


# %% dispatcher
_SIMULATORS = {'GR4J': simulate_gr4j, 'GR5J': simulate_gr5j, 'GR6J': simulate_gr6j}


def simulate(precip, pet, params, model='GR4J'):
    """Run the named GR model.

    Parameters
    ----------
    precip, pet : array-like, mm/d, must be complete
    params : dict or sequence of the parameters for the chosen model
    model : 'GR4J', 'GR5J' or 'GR6J'

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

    if model == 'GR6J' and params['X6'] <= 0.0:
        raise ValueError('X6 must be strictly positive, it divides the store level.')

    return _SIMULATORS[model](precip, pet, params)
