# core/gapfill.py
# Gap filling of daily streamflow using a behavioural GR4J ensemble.
#
# All three methods work on the residual series (observed minus behavioural
# median) rather than on flow directly, so the filled values inherit the shape
# of the modelled hydrograph and are anchored to the observed record either side
# of the gap.

# %%
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# %% Gaussian process configuration
# The exact GP solve is O(n^3) in the number of training points, so a global fit
# on a multi-decade daily record is not tractable. Two things make it tractable
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

HYPER_MAX_POINTS = 1000       # cap on training points for the hyperparameter fit
HYPER_BLOCK_DAYS = 250        # length of each contiguous block in that sample
WINDOW_LENGTH_SCALES = 4.0    # window half-width, in fitted length scales
WINDOW_MIN_DAYS = 120         # floor on window half-width
WINDOW_MAX_DAYS = 1500        # ceiling on window half-width
WINDOW_MAX_POINTS = 2500      # cap on training points per local solve
MIN_TRAIN_POINTS = 10         # below this, fall back to the snapped method


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
    of the record.

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

    # anything left unfilled (too few local observations) falls back to the median
    unfilled = ~np.isfinite(gapfilled)
    gapfilled[unfilled] = q50[unfilled]

    return gapfilled
