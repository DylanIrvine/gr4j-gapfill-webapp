# core/calibration.py
# Differential evolution calibration of the GR family, with a behavioural
# archive assembled from the objective function evaluations.
#
# No module-level mutable state. Everything the calibration accumulates lives
# inside calibrate_gr, so concurrent sessions on a deployed app cannot write
# into each other's archive.

# %%
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from core.models import simulate, PARAM_NAMES, PARAM_BOUNDS, PARAM_ROUNDING
from core.metrics import score, epsilon_from_obs

# %% configuration
MAX_BEHAVIOURAL_MODELS = 200
MIN_FITTING_DAYS = 100
PENALTY = 1e9


# %%
def _objective_function(params, precip, pet, q_obs, warmup_days, metric,
                        transform_kind, epsilon, model, names, archive):
    """Negative criterion, with every valid evaluation appended to archive."""
    param_dict = dict(zip(names, [float(p) for p in params]))

    q_sim = simulate(precip, pet, param_dict, model=model)

    mask = np.isfinite(q_obs) & np.isfinite(q_sim)
    if warmup_days > 0:
        mask[:warmup_days] = False

    if mask.sum() < MIN_FITTING_DAYS:
        return PENALTY

    value = score(q_obs[mask], q_sim[mask], metric=metric,
                  transform_kind=transform_kind, epsilon=epsilon)

    if not np.isfinite(value):
        return PENALTY

    archive.append({**param_dict, 'Score': float(value)})

    return -value


# %%
def calibrate_gr(precip, pet, q_obs, model='GR4J', warmup_days=730, metric='KGE',
                 transform_kind='none', maxiter=25, popsize=12,
                 behavioural_delta=0.05, seed=1, progress_callback=None):
    """Calibrate a GR model and return the best parameter set plus an archive.

    Parameters
    ----------
    precip, pet : array-like, mm/d, must be complete
    q_obs : array-like, mm/d, may contain NaN
    model : 'GR4J', 'GR5J' or 'GR6J'
    warmup_days : int, days excluded from the objective at the start of the record
    metric : 'KGE' or 'NSE'
    transform_kind : 'none', 'sqrt', 'log' or 'inverse', applied to both series
        before the criterion is computed
    maxiter, popsize : differential evolution settings. Note that scipy sizes the
        population as popsize times the number of parameters, so GR6J costs half
        again as much per generation as GR4J.
    behavioural_delta : models scoring within this distance of the best are retained
    seed : int, fixes the differential evolution random state for reproducibility
    progress_callback : optional callable, called once per generation as
        callback(params_dict, convergence)

    Returns
    -------
    dict with keys best_params, best_score, behavioural_df, model, epsilon

    Note on interpretation
    ---------------------
    The behavioural set is drawn from the differential evolution trajectory, not
    from a random sample of parameter space. The population concentrates near the
    optimum as the search proceeds, so this is not a GLUE sample and the
    resulting ensemble spread should be read as a local sensitivity band around
    the optimum, not as a calibrated predictive uncertainty. Adding parameters
    widens the region of near-equivalent performance, so this caveat matters more
    for GR6J than for GR4J.
    """
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)

    names = PARAM_NAMES[model]
    bounds = [PARAM_BOUNDS[name] for name in names]

    # the transform offset is fixed once from the observations, so every
    # candidate parameter set is scored against exactly the same criterion
    epsilon = epsilon_from_obs(q_obs)

    archive = []

    def _callback(xk, convergence=None):
        progress_callback(dict(zip(names, [float(v) for v in xk])), convergence)
        return False

    result = differential_evolution(
        _objective_function,
        bounds=bounds,
        args=(precip, pet, q_obs, warmup_days, metric, transform_kind, epsilon,
              model, names, archive),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=_callback if progress_callback is not None else None,
    )

    best_score = float(-result.fun)
    best_params = dict(zip(names, [float(v) for v in result.x]))
    best_params['ObjectiveValue'] = best_score

    if len(archive) == 0:
        empty = pd.DataFrame(columns=list(names) + ['Score'])
        return {'best_params': best_params, 'best_score': best_score,
                'behavioural_df': empty, 'model': model, 'epsilon': epsilon}

    archive_df = pd.DataFrame(archive)
    archive_df = archive_df[np.isfinite(archive_df['Score'])]
    archive_df = archive_df.sort_values('Score', ascending=False)

    behavioural_df = archive_df[archive_df['Score'] >= best_score - behavioural_delta].copy()

    # round first, so numerically identical parameter sets collapse to one entry
    for name in names:
        behavioural_df[name] = behavioural_df[name].round(PARAM_ROUNDING[name])

    behavioural_df = behavioural_df.drop_duplicates(subset=list(names))
    behavioural_df = behavioural_df.head(MAX_BEHAVIOURAL_MODELS).reset_index(drop=True)

    return {'best_params': best_params, 'best_score': best_score,
            'behavioural_df': behavioural_df, 'model': model, 'epsilon': epsilon}


# %% backwards compatible alias
def calibrate_gr4j(precip, pet, q_obs, warmup_days=730, objective='KGE', **kwargs):
    """Deprecated. Retained so older calls keep working; use calibrate_gr."""
    return calibrate_gr(precip, pet, q_obs, model='GR4J', warmup_days=warmup_days,
                        metric=objective, **kwargs)
