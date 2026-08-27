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
