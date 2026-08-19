#%%
import numpy as np

from scipy.optimize import differential_evolution

from core.gr4j import simulate
from core.metrics import kge
from core.metrics import nse


#%%
def objective_function(
        params,
        precip,
        pet,
        q_obs,
        warmup_days=730,
        objective='KGE'):

    q_sim = simulate(
        precip,
        pet,
        params
    )

    mask = (
        np.isfinite(q_obs)
    )

    if warmup_days > 0:

        mask[:warmup_days] = False

    if mask.sum() < 100:

        return 1e9

    q_obs_fit = q_obs[mask]
    q_sim_fit = q_sim[mask]

    if objective == 'KGE':

        score = kge(
            q_obs_fit,
            q_sim_fit
        )

    elif objective == 'NSE':

        score = nse(
            q_obs_fit,
            q_sim_fit
        )

    else:

        score = kge(
            q_obs_fit,
            q_sim_fit
        )

    return -score
