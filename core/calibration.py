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

#%%
def calibrate_gr4j(
        precip,
        pet,
        q_obs,
        warmup_days=730,
        objective='KGE',
        maxiter=25,
        popsize=12):

    bounds = [

        (1.0, 3000.0),   # X1

        (-20.0, 5.0),    # X2

        (1.0, 1000.0),   # X3

        (0.5, 20.0)      # X4
    ]

    result = differential_evolution(

        objective_function,

        bounds=bounds,

        args=(

            precip,
            pet,
            q_obs,
            warmup_days,
            objective
        ),

        maxiter=maxiter,

        popsize=popsize,

        seed=1
    )

    params = {

        'X1': result.x[0],

        'X2': result.x[1],

        'X3': result.x[2],

        'X4': result.x[3]
    }

    return params
