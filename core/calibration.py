# core/calibration.py
# Differential evolution calibration of GR4J, with a behavioural archive
# assembled from the objective function evaluations.
#
# No module-level mutable state. Everything the calibration accumulates lives
# inside calibrate_gr4j, so concurrent sessions on a deployed app cannot write
# into each other's archive.

# %%
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from core.gr4j import simulate
from core.metrics import kge, nse

# %% configuration
PARAM_NAMES = ('X1', 'X2', 'X3', 'X4')

PARAM_BOUNDS = {'X1': (1.0, 3000.0),
                'X2': (-25.0, 5.0),
                'X3': (1.0, 1000.0),
                'X4': (0.5, 20.0)}

PARAM_ROUNDING = {'X1': 1, 'X2': 2, 'X3': 1, 'X4': 2}

MAX_BEHAVIOURAL_MODELS = 200
MIN_FITTING_DAYS = 100
PENALTY = 1e9


# %%
def _objective_function(params, precip, pet, q_obs, warmup_days, objective, archive):
    """Negative objective score, with every valid evaluation appended to archive."""
    param_dict = dict(zip(PARAM_NAMES, [float(p) for p in params]))

    q_sim = simulate(precip, pet, param_dict)

    mask = np.isfinite(q_obs) & np.isfinite(q_sim)
    if warmup_days > 0:
        mask[:warmup_days] = False

    if mask.sum() < MIN_FITTING_DAYS:
        return PENALTY

    if objective == 'NSE':
        score = nse(q_obs[mask], q_sim[mask])
    else:
        score = kge(q_obs[mask], q_sim[mask])

    if not np.isfinite(score):
        return PENALTY

    archive.append({**param_dict, 'Score': float(score)})

    return -score


# %%
def _empty_archive():
    return pd.DataFrame(columns=list(PARAM_NAMES) + ['Score'])


# %%
def calibrate_gr4j(precip, pet, q_obs, warmup_days=730, objective='KGE', maxiter=25,
                   popsize=12, behavioural_delta=0.05, seed=1, progress_callback=None):
    """Calibrate GR4J and return the best parameter set plus a behavioural archive.

    Parameters
    ----------
    precip, pet : array-like, mm/d, must be complete
    q_obs : array-like, mm/d, may contain NaN
    warmup_days : int, days excluded from the objective at the start of the record
    objective : 'KGE' or 'NSE'
    maxiter, popsize : differential evolution settings
    behavioural_delta : models scoring within this distance of the best are retained
    seed : int, fixes the differential evolution random state for reproducibility
    progress_callback : optional callable, called once per generation as
        callback(params_dict, convergence)

    Returns
    -------
    dict with keys best_params, best_score, behavioural_df

    Note on interpretation
    ---------------------
    The behavioural set is drawn from the differential evolution trajectory, not
    from a random sample of parameter space. The population concentrates near the
    optimum as the search proceeds, so this is not a GLUE sample and the
    resulting ensemble spread should be read as a local sensitivity band around
    the optimum, not as a calibrated predictive uncertainty.
    """
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)

    archive = []
    bounds = [PARAM_BOUNDS[name] for name in PARAM_NAMES]

    def _callback(xk, convergence=None):
        progress_callback(dict(zip(PARAM_NAMES, [float(v) for v in xk])), convergence)
        return False

    result = differential_evolution(
        _objective_function,
        bounds=bounds,
        args=(precip, pet, q_obs, warmup_days, objective, archive),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=_callback if progress_callback is not None else None,
    )

    best_score = float(-result.fun)
    best_params = dict(zip(PARAM_NAMES, [float(v) for v in result.x]))
    best_params['ObjectiveValue'] = best_score

    if len(archive) == 0:
        return {'best_params': best_params, 'best_score': best_score,
                'behavioural_df': _empty_archive()}

    archive_df = pd.DataFrame(archive)
    archive_df = archive_df[np.isfinite(archive_df['Score'])]
    archive_df = archive_df.sort_values('Score', ascending=False)

    behavioural_df = archive_df[archive_df['Score'] >= best_score - behavioural_delta].copy()

    # round first, so numerically identical parameter sets collapse to one entry
    for name, decimals in PARAM_ROUNDING.items():
        behavioural_df[name] = behavioural_df[name].round(decimals)

    behavioural_df = behavioural_df.drop_duplicates(subset=list(PARAM_NAMES))
    behavioural_df = behavioural_df.head(MAX_BEHAVIOURAL_MODELS).reset_index(drop=True)

    return {'best_params': best_params, 'best_score': best_score,
            'behavioural_df': behavioural_df}
