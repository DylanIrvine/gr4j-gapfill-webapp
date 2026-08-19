#%%
import numpy as np


#%%
def cumecs_to_mmd(q, area_km2):

    q = np.asarray(q, dtype=float)

    return (
        q
        * 86400.0
        / (area_km2 * 1e6)
        * 1000.0
    )


#%%
def mmd_to_cumecs(q, area_km2):

    q = np.asarray(q, dtype=float)

    return (
        q
        * area_km2
        * 1000.0
        / 86400.0
    )


#%%
def mld_to_mmd(q, area_km2):

    q = np.asarray(q, dtype=float)

    return (
        q
        / area_km2
    )


#%%
def mmd_to_mld(q, area_km2):

    q = np.asarray(q, dtype=float)

    return (
        q
        * area_km2
    )
