# core/gapfill.py
# Gap filling of daily streamflow.
#
# Five methods. The first four work on the residual series (observed minus the
# behavioural ensemble median) rather than on flow directly, so the filled
# values inherit the shape of the modelled hydrograph and are anchored to the
# observed record either side of the gap:
#
#   gapfill_p50               insert the behavioural median
#   gapfill_snapped           linear interpolation of the residual across the gap
#   gapfill_gaussian_process  a Gaussian process posterior over the residual
#   gapfill_ar1               an AR(1) process on the residual, run from both
#                             anchors towards the gap interior and blended
#
# The fifth needs the model itself:
#
#   gapfill_enkf              a fixed-interval ensemble Kalman smoother: an
#                             ensemble of model runs with perturbed forcing,
#                             each gap updated with the observations straddling
#                             it through the ensemble cross-covariance

# %%
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from core.models import simulate

# %% Gaussian process configuration
# The exact GP solve is O(n^3) in time and O(n^2) in memory, so a global fit on
# a multi-decade daily record is not tractable. Two things make it tractable
# without changing the method:
#
#   1. Hyperparameters are fitted once, on a bounded sample made of contiguous
#      blocks spread across the record. Blocks rather than a thinned sample, so
#      that short-lag structure is still visible to the optimiser.
#   2. Each gap is then solved exactly, using only observations within a few
#      length scales either side. The RBF correlation at 4 length scales is
#      about 3e-4, so points beyond the window carry no usable information and
#      the windowed posterior matches the full-record posterior to well within
#      the residual noise.
#
# WINDOW_MAX_POINTS caps the per-solve kernel matrix. At 1500 points that is
# about 18 MB per copy, and scikit-learn holds several during the Cholesky, so
# roughly 50 to 60 MB transient. Raising it grows that quadratically, which
# matters on a memory-limited container.

HYPER_MAX_POINTS = 1000       # cap on training points for the hyperparameter fit
HYPER_BLOCK_DAYS = 250        # length of each contiguous block in that sample
WINDOW_LENGTH_SCALES = 4.0    # window half-width, in fitted length scales
WINDOW_MIN_DAYS = 120         # floor on window half-width
WINDOW_MAX_DAYS = 1500        # ceiling on window half-width
WINDOW_MAX_POINTS = 1500      # cap on training points per local solve
MIN_TRAIN_POINTS = 10         # below this, fall back to the snapped method

# %% AR(1) residual configuration
AR1_MIN_POINTS = 30           # residual pairs needed to estimate phi

# %% ensemble Kalman smoother configuration
ENKF_N_ENSEMBLE = 60          # model runs; runtime is linear in this
ENKF_RAIN_CV = 0.25           # coefficient of variation of the rainfall multiplier
ENKF_RAIN_AR = 0.4            # lag-1 correlation of the rainfall multiplier
ENKF_PET_CV = 0.10            # coefficient of variation of the PET multiplier
ENKF_OBS_ERR_FRAC = 0.12      # streamflow observation error, as a fraction of the flow
ENKF_OBS_FLOOR_FRAC = 0.02    # plus a floor, as a fraction of the mean observed flow
ENKF_WINDOW_DAYS = 180        # observations this many days either side of a gap inform it
ENKF_MAX_OBS = 400            # cap on the observation vector per gap, for the solve


# %% shared helpers
def _as_float(q):
    return np.asarray(q, dtype=float)


def identify_gaps(q_obs):
    """Identify runs of missing values in a series.

    Parameters
    ----------
    q_obs : array-like

    Returns
    -------
    list of dict, one per gap, with start_idx, end_idx, length_days and flags
    indicating whether observed data exist either side.
    """
    q_obs = _as_float(q_obs)
    is_gap = ~np.isfinite(q_obs)
    transitions = np.diff(np.concatenate(([0], is_gap.astype(int), [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1

    return [{'gap_id': int(i),
             'start_idx': int(s),
             'end_idx': int(e),
             'length_days': int(e - s + 1),
             'has_left_anchor': bool(s > 0),
             'has_right_anchor': bool(e < len(q_obs) - 1)}
            for i, (s, e) in enumerate(zip(starts, ends), start=1)]


def clip_negative(q_filled, q_obs):
    """Clip gap filled values at zero and report how many were clipped.

    Residual-based methods can drive filled flows below zero during recessions.
    Observed values are never modified, so genuine negative observations remain
    visible as a data quality issue rather than being silently corrected.

    Returns
    -------
    (clipped series, number of values clipped)
    """
    q_filled = _as_float(q_filled).copy()
    filled = ~np.isfinite(_as_float(q_obs))
    below = filled & np.isfinite(q_filled) & (q_filled < 0.0)
    q_filled[below] = 0.0
    return q_filled, int(below.sum())


# %% method 1: behavioural median
def gapfill_p50(q_obs, q50):
    """Insert the behavioural median directly into each gap.

    Simple and unbiased with respect to the ensemble, but the filled series will
    step at the gap edges wherever the model has a persistent local bias.
    """
    q_obs = _as_float(q_obs)
    gapfilled = q_obs.copy()
    missing = ~np.isfinite(q_obs)
    gapfilled[missing] = _as_float(q50)[missing]
    return gapfilled


# %% method 2: endpoint snapped residuals
def gapfill_snapped(q_obs, q50):
    """Linearly interpolate the residual across each gap and add it to the median.

    This removes the step at the gap edges. Beyond the first and last
    observation the residual is held constant, since there is nothing to
    interpolate between.
    """
    q_obs = _as_float(q_obs)
    q50 = _as_float(q50)
    gapfilled = q_obs.copy()

    residual_interp = (pd.Series(q_obs - q50)
                       .interpolate(method='linear', limit_direction='both')
                       .to_numpy())

    missing = ~np.isfinite(q_obs)
    gapfilled[missing] = q50[missing] + residual_interp[missing]
    return gapfilled


# %% method 3: Gaussian process residuals
def _hyperparameter_sample(obs_idx, n_max=HYPER_MAX_POINTS, block=HYPER_BLOCK_DAYS):
    """Contiguous blocks of observations spread evenly across the record."""
    if len(obs_idx) <= n_max:
        return obs_idx

    n_blocks = max(1, int(np.ceil(n_max / block)))
    block = min(block, len(obs_idx))
    starts = np.unique(np.linspace(0, len(obs_idx) - block, n_blocks).astype(int))
    sample = np.concatenate([obs_idx[s:s + block] for s in starts])
    return np.unique(sample)[:n_max]


def _fit_hyperparameters(x, y, seed=0):
    """Fit RBF amplitude, length scale and noise on a bounded training sample."""
    kernel = (1.0 * RBF(length_scale=30.0, length_scale_bounds=(2.0, 2000.0))
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e2)))

    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=1, random_state=seed)
    gp.fit(x, y)
    return gp.kernel_


def _length_scale(kernel, default=100.0):
    for name, value in kernel.get_params().items():
        if name.endswith('length_scale') and np.isscalar(value):
            return float(value)
    return default


def _cluster_gaps(gaps, window):
    """Merge gaps separated by less than one window into a single solve."""
    clusters = []
    for gap in gaps:
        start, end = gap['start_idx'], gap['end_idx']
        if clusters and (start - clusters[-1][1]) <= window:
            clusters[-1] = (clusters[-1][0], end)
        else:
            clusters.append((start, end))
    return clusters


def gapfill_gaussian_process(q_obs, q50, seed=0):
    """Fill gaps with a Gaussian process posterior over the residual series.

    Hyperparameters are fitted once on a bounded sample, then each gap (or
    cluster of nearby gaps) is solved exactly against the observations inside a
    local window. Runtime scales with the number of gaps rather than the length
    of the record, and peak memory is set by WINDOW_MAX_POINTS rather than by
    the record length.

    Far from any observation the posterior reverts to the mean residual, so
    unanchored gaps at the start or end of a record degrade gracefully towards
    the behavioural median rather than extrapolating a trend.
    """
    q_obs = _as_float(q_obs)
    q50 = _as_float(q50)
    gapfilled = q_obs.copy()

    missing = ~np.isfinite(q_obs)
    if not missing.any():
        return gapfilled

    residuals = q_obs - q50
    observed = np.isfinite(residuals)

    if observed.sum() < MIN_TRAIN_POINTS:
        return gapfill_snapped(q_obs, q50)

    # standardise once, globally, so the fitted hyperparameters mean the same
    # thing in every local window
    mu = float(np.mean(residuals[observed]))
    sd = float(np.std(residuals[observed]))
    if not np.isfinite(sd) or sd <= 0.0:
        sd = 1.0
    y = (residuals - mu) / sd

    obs_idx = np.where(observed)[0]
    hyper_idx = _hyperparameter_sample(obs_idx)

    kernel = _fit_hyperparameters(hyper_idx.reshape(-1, 1).astype(float),
                                  y[hyper_idx], seed=seed)

    window = int(np.clip(WINDOW_LENGTH_SCALES * _length_scale(kernel),
                         WINDOW_MIN_DAYS, WINDOW_MAX_DAYS))

    for start, end in _cluster_gaps(identify_gaps(q_obs), window):

        lo = max(0, start - window)
        hi = min(len(q_obs) - 1, end + window)
        train = obs_idx[(obs_idx >= lo) & (obs_idx <= hi)]

        if len(train) > WINDOW_MAX_POINTS:
            centre = 0.5 * (start + end)
            train = np.sort(train[np.argsort(np.abs(train - centre))[:WINDOW_MAX_POINTS]])

        if len(train) < MIN_TRAIN_POINTS:
            continue

        target = np.arange(start, end + 1)
        target = target[missing[target]]
        if len(target) == 0:
            continue

        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
        gp.fit(train.reshape(-1, 1).astype(float), y[train])

        prediction = gp.predict(target.reshape(-1, 1).astype(float))
        gapfilled[target] = q50[target] + (prediction * sd + mu)

        del gp

    # anything left unfilled (too few local observations) falls back to the median
    unfilled = ~np.isfinite(gapfilled)
    gapfilled[unfilled] = q50[unfilled]

    return gapfilled


# %% method 4: AR(1) residuals
def _fit_ar1(residuals):
    """Mean and lag-1 autocorrelation of a residual series that contains gaps.

    Only consecutive-day pairs where both residuals exist contribute to the
    autocorrelation, so the estimate is not contaminated by the discontinuities
    at gap edges.
    """
    residuals = _as_float(residuals)
    finite = residuals[np.isfinite(residuals)]
    if finite.size < AR1_MIN_POINTS:
        return None

    mu = float(np.mean(finite))
    centred = residuals - mu

    a, b = centred[:-1], centred[1:]
    pair = np.isfinite(a) & np.isfinite(b)
    if pair.sum() < AR1_MIN_POINTS:
        return None

    c0 = float(np.mean(centred[np.isfinite(centred)] ** 2))
    c1 = float(np.mean(a[pair] * b[pair]))
    if c0 <= 0:
        return None

    phi = float(np.clip(c1 / c0, 0.0, 0.999))
    return mu, phi


def gapfill_ar1(q_obs, q50):
    """Fill gaps with an AR(1) process on the residual, blended from both anchors.

    The residual r = observed - behavioural median is modelled as
    r(t) = mu + phi * (r(t-1) - mu). Its noise-free expectation is propagated
    forward from the last observation before the gap and backward from the first
    observation after it; the two predictions are blended linearly across the
    gap, so the fill is anchored at both edges and decays towards the mean
    residual in the interior of a long gap. phi is the lag-1 autocorrelation of
    the observed residual series, so the decay rate is set by the data rather
    than assumed. With phi close to 1 this reduces to the snapped method; with
    phi small it reduces to the behavioural median.

    An unanchored gap at the start or end of the record uses the single
    available anchor. Too few residuals to estimate phi falls back to snapped.
    """
    q_obs = _as_float(q_obs)
    q50 = _as_float(q50)
    gapfilled = q_obs.copy()

    missing = ~np.isfinite(q_obs)
    if not missing.any():
        return gapfilled

    residuals = q_obs - q50
    fit = _fit_ar1(residuals)
    if fit is None:
        return gapfill_snapped(q_obs, q50)
    mu, phi = fit

    n = len(q_obs)
    for gap in identify_gaps(q_obs):
        s, e = gap['start_idx'], gap['end_idx']
        length = e - s + 1
        k = np.arange(1, length + 1)                     # 1..L within the gap

        left = residuals[s - 1] if s > 0 and np.isfinite(residuals[s - 1]) else None
        right = residuals[e + 1] if e < n - 1 and np.isfinite(residuals[e + 1]) else None

        forward = mu + phi ** k * (left - mu) if left is not None else None
        backward = mu + phi ** (length + 1 - k) * (right - mu) if right is not None else None

        if forward is not None and backward is not None:
            w = k / (length + 1)                          # 0 at the left edge, 1 at the right
            r_gap = (1.0 - w) * forward + w * backward
        elif forward is not None:
            r_gap = forward
        elif backward is not None:
            r_gap = backward
        else:
            r_gap = np.full(length, mu)

        gapfilled[s:e + 1] = q50[s:e + 1] + r_gap

    return gapfilled


# %% method 5: ensemble Kalman smoother
def _perturbed_forcing(precip, pet, n_ensemble, rain_cv, rain_ar, pet_cv, seed):
    """Ensembles of rainfall and PET.

    Rainfall gets a temporally correlated lognormal multiplier (an AR(1) in log
    space), so a wet or dry bias persists over a spell rather than averaging out
    day to day. PET gets a small independent multiplicative perturbation. Both
    multipliers are mean-one so the ensemble is unbiased with respect to the
    supplied forcing.
    """
    rng = np.random.default_rng(seed)
    n = len(precip)

    log_sd = np.sqrt(np.log(1.0 + rain_cv ** 2))
    innov_sd = log_sd * np.sqrt(1.0 - rain_ar ** 2)

    rain_ens = np.empty((n_ensemble, n))
    pet_ens = np.empty((n_ensemble, n))
    for m in range(n_ensemble):
        x = np.empty(n)
        x[0] = rng.normal(0.0, log_sd)
        noise = rng.normal(0.0, innov_sd, n)
        for t in range(1, n):
            x[t] = rain_ar * x[t - 1] + noise[t]
        multiplier = np.exp(x - 0.5 * log_sd ** 2)        # mean-one lognormal
        rain_ens[m] = np.clip(precip * multiplier, 0.0, None)
        pet_ens[m] = np.clip(pet * (1.0 + rng.normal(0.0, pet_cv, n)), 0.0, None)

    return rain_ens, pet_ens


def gapfill_enkf(q_obs, precip, pet, params, model='GR4J', *,
                 n_ensemble=ENKF_N_ENSEMBLE, rain_cv=ENKF_RAIN_CV,
                 rain_ar=ENKF_RAIN_AR, pet_cv=ENKF_PET_CV,
                 obs_err_frac=ENKF_OBS_ERR_FRAC, obs_floor_frac=ENKF_OBS_FLOOR_FRAC,
                 window_days=ENKF_WINDOW_DAYS, max_obs=ENKF_MAX_OBS, seed=0,
                 simhyd_overflow_to_gw=False, return_spread=False):
    """Fill gaps with a fixed-interval ensemble Kalman smoother.

    An ensemble of runs of the calibrated model is generated by perturbing the
    rainfall (a temporally correlated lognormal multiplier) and PET, so the
    ensemble spread carries input uncertainty. Each gap is then updated with the
    observed flows in a window straddling it, using the ensemble cross-covariance
    between the gap-day flows and the windowed observations:

        q_gap  <-  q_gap_mean  +  Cov(q_gap, q_win) [Cov(q_win) + R]^-1 (y_win - q_win_mean)

    R is a heteroscedastic observation-error variance, a fraction of the flow
    plus a floor. The correction on a gap day scales with its ensemble
    covariance to the observations, which is largest next to the observed record,
    so the fill is drawn towards the data from the gap start (pre-gap
    observations dominate), from the gap end (post-gap observations dominate),
    and blends across the interior. Both sides of the gap inform the fill.

    This is a forcing-driven smoother: it does not sequentially update the
    model's internal stores. For offline single-gauge gap filling with a
    complete forcing record that is the appropriate simplification, and it keeps
    the method to one pass over simulate().

    Parameters
    ----------
    q_obs : array-like, mm/d, may contain NaN
    precip, pet : array-like, mm/d, complete, the same length as q_obs
    params : dict, one calibrated parameter set for `model`
    model : one of 'GR4J', 'GR5J', 'GR6J', 'SIMHYD'
    return_spread : if True, also return the ensemble standard deviation on the
        gap days (NaN elsewhere) as a second array

    Returns
    -------
    filled series (and, if return_spread, the gap-day standard deviation)
    """
    q_obs = _as_float(q_obs)
    precip = _as_float(precip)
    pet = _as_float(pet)
    gapfilled = q_obs.copy()
    spread = np.full(len(q_obs), np.nan)

    missing = ~np.isfinite(q_obs)
    if not missing.any():
        return (gapfilled, spread) if return_spread else gapfilled

    n = len(q_obs)
    observed = np.where(np.isfinite(q_obs))[0]
    if observed.size < MIN_TRAIN_POINTS:
        filled = gapfill_snapped(q_obs, np.full(n, np.nanmean(q_obs)))
        return (filled, spread) if return_spread else filled

    rain_ens, pet_ens = _perturbed_forcing(precip, pet, n_ensemble, rain_cv,
                                           rain_ar, pet_cv, seed)

    ensemble = np.empty((n_ensemble, n))
    for m in range(n_ensemble):
        ensemble[m] = simulate(rain_ens[m], pet_ens[m], params, model=model,
                               simhyd_overflow_to_gw=simhyd_overflow_to_gw)

    mean = ensemble.mean(axis=0)
    anomaly = ensemble - mean                             # n_ensemble x n
    denom = n_ensemble - 1

    mean_flow = float(np.mean(q_obs[observed]))
    floor_var = (obs_floor_frac * mean_flow) ** 2

    for gap in identify_gaps(q_obs):
        s, e = gap['start_idx'], gap['end_idx']
        gidx = np.arange(s, e + 1)
        gidx = gidx[missing[gidx]]
        if gidx.size == 0:
            continue

        lo, hi = s - window_days, e + window_days
        widx = observed[(observed >= lo) & (observed <= hi)]
        if widx.size == 0:
            gapfilled[gidx] = mean[gidx]
            continue
        if widx.size > max_obs:
            centre = 0.5 * (s + e)
            widx = np.sort(widx[np.argsort(np.abs(widx - centre))[:max_obs]])

        ha = anomaly[:, widx]                             # n_ensemble x n_obs
        ga = anomaly[:, gidx]                             # n_ensemble x n_gap

        pyy = ha.T @ ha / denom
        r_diag = (obs_err_frac * np.maximum(q_obs[widx], 0.0)) ** 2 + floor_var
        pyy[np.diag_indices_from(pyy)] += r_diag

        innovation = q_obs[widx] - mean[widx]
        gain_rhs = np.linalg.solve(pyy, innovation)       # n_obs
        update = (ga.T @ ha / denom) @ gain_rhs           # n_gap

        gapfilled[gidx] = mean[gidx] + update
        spread[gidx] = ensemble[:, gidx].std(axis=0)

    unfilled = ~np.isfinite(gapfilled)
    gapfilled[unfilled] = mean[unfilled]

    return (gapfilled, spread) if return_spread else gapfilled
