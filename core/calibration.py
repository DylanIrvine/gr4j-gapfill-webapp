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

    mask = np.isfinite(q_obs)

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
def calibration_callback(
        xk,
        convergence):

    global calibration_progress

    calibration_progress['generation'] += 1

    calibration_progress['params'] = {

        'X1': xk[0],
        'X2': xk[1],
        'X3': xk[2],
        'X4': xk[3]

    }

    calibration_progress['convergence'] = convergence

    return False


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

        (1.0, 3000.0),

        (-20.0, 5.0),

        (1.0, 1000.0),

        (0.5, 20.0)

    ]

    global calibration_progress

    calibration_progress = {

        'generation': 0,

        'params': None,

        'convergence': np.nan

    }

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

        seed=1,

        callback=calibration_callback

    )

    params = {

        'X1': result.x[0],

        'X2': result.x[1],

        'X3': result.x[2],

        'X4': result.x[3],

        'ObjectiveValue': -result.fun

    }

    return params
