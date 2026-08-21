#%%
import numpy as np
#%%
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel
)

#%%
def gapfill_p50(
        q_obs,
        q50):

    gapfilled = np.array(
        q_obs,
        copy=True
    )

    missing = ~np.isfinite(q_obs)

    gapfilled[missing] = q50[missing]

    return gapfilled

#%%
def gapfill_snapped(
        q_obs,
        q50):

    gapfilled = np.array(
        q_obs,
        copy=True
    )

    residuals = (
        q_obs
        -
        q50
    )

    residual_series = pd.Series(
        residuals
    )

    residual_interp = (
        residual_series
        .interpolate(
            method='linear',
            limit_direction='both'
        )
        .to_numpy()
    )

    missing = ~np.isfinite(q_obs)

    gapfilled[missing] = (
        q50[missing]
        +
        residual_interp[missing]
    )

    return gapfilled

#%%
def gapfill_gaussian_process(
        q_obs,
        q50):

    gapfilled = np.array(
        q_obs,
        copy=True
    )

    residuals = (
        q_obs
        -
        q50
    )

    mask = np.isfinite(
        residuals
    )

    x_train = np.where(
        mask
    )[0].reshape(-1, 1)

    y_train = residuals[
        mask
    ]

    x_all = np.arange(
        len(q_obs)
    ).reshape(-1, 1)

    kernel = (
        50.0**2
        * RBF(
            length_scale=100.0
        )
        +
        WhiteKernel(
            noise_level=0.1
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3
    )

    gp.fit(
        x_train,
        y_train
    )

    residual_pred = gp.predict(
        x_all
    )

    missing = ~np.isfinite(
        q_obs
    )

    gapfilled[missing] = (
        q50[missing]
        +
        residual_pred[missing]
    )

    return gapfilled

#%%
def identify_gaps(q_obs):
    '''
    Identify gaps in an observed flow series.

    Parameters
    ----------
    q_obs : array-like

    Returns
    -------
    list of dictionaries
    '''

    q_obs = np.asarray(q_obs, dtype=float)

    is_gap = np.isnan(q_obs)

    transitions = np.diff(
        np.concatenate(
            ([0], is_gap.astype(int), [0])
        )
    )

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1

    gaps = []

    for gap_id, (start, end) in enumerate(
            zip(starts, ends),
            start=1):

        gaps.append({
            'gap_id': int(gap_id),
            'start_idx': int(start),
            'end_idx': int(end),
            'length_days': int(end - start + 1)
        })

    return gaps
