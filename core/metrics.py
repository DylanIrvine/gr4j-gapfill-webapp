#%%
import numpy as np


#%%
def nse(obs, sim):

    mask = (
        np.isfinite(obs) &
        np.isfinite(sim)
    )

    obs = np.asarray(obs)[mask]
    sim = np.asarray(sim)[mask]

    if len(obs) < 2:
        return np.nan

    return (
        1
        - np.sum((obs - sim) ** 2)
        / np.sum((obs - np.mean(obs)) ** 2)
    )


#%%
def kge(obs, sim):

    mask = (
        np.isfinite(obs) &
        np.isfinite(sim)
    )

    obs = np.asarray(obs)[mask]
    sim = np.asarray(sim)[mask]

    if len(obs) < 2:
        return np.nan

    r = np.corrcoef(
        obs,
        sim
    )[0, 1]

    alpha = (
        np.std(sim)
        /
        np.std(obs)
    )

    beta = (
        np.mean(sim)
        /
        np.mean(obs)
    )

    return (
        1
        -
        np.sqrt(
            (r - 1) ** 2
            +
            (alpha - 1) ** 2
            +
            (beta - 1) ** 2
        )
    )
