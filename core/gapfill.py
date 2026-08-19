#%%
import numpy as np


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
            'gap_id': gap_id,
            'start_idx': start,
            'end_idx': end,
            'length_days': end - start + 1
        })

    return gaps
