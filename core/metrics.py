# core/metrics.py
# Efficiency criteria, with optional flow transformations.
#
# Why transformations matter here. KGE and NSE on untransformed flow are
# dominated by the largest values, because both are built on squared errors. On
# a wet-dry tropical river where flood peaks are two or three orders of
# magnitude above dry season baseflow, essentially the entire objective is
# determined by a few weeks a year. A parameter that only affects recession
# behaviour, such as GR6J's X6, is then effectively unconstrained.
#
# Transforming the flows before computing the criterion changes which part of
# the hydrograph the calibration is sensitive to:
#
#   none     squared errors on flow, dominated by peaks
#   sqrt     intermediate, a common general-purpose compromise
#   log      emphasises the middle and lower range
#   inverse  strongly emphasises low flows; recommended for low-flow
#            evaluation by Pushpalatha et al. (2012)
#
# Reference
#   Pushpalatha, R., Perrin, C., Le Moine, N., Andreassian, V. (2012). A review
#   of efficiency criteria suitable for evaluating low-flow simulations.
#   Journal of Hydrology 420-421, 171-182.

# %%
import numpy as np

TRANSFORMS = ('none', 'sqrt', 'log', 'inverse')

TRANSFORM_LABELS = {
    'none': 'None (weights high flows)',
    'sqrt': 'Square root (balanced)',
    'log': 'Logarithmic (weights mid and low flows)',
    'inverse': 'Inverse (weights low flows)',
}

METRICS = ('KGE', 'NSE')

# Pushpalatha et al. (2012) use one hundredth of the mean observed flow as the
# offset that keeps log and inverse transforms finite at zero flow. The offset
# is derived from the observations only, and the same value is applied to both
# series, so the transformation cannot favour the simulation.
EPSILON_FRACTION = 0.01


# %%
def _finite_pair(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


# %%
def epsilon_from_obs(obs, fraction=EPSILON_FRACTION):
    """Offset for the log and inverse transforms, from the observed series."""
    obs = np.asarray(obs, dtype=float)
    finite = obs[np.isfinite(obs)]

    if finite.size == 0:
        return 1e-6

    mean_obs = float(np.mean(finite))
    epsilon = fraction * mean_obs

    # guard against an all-zero or negative-mean record
    return epsilon if epsilon > 0 else 1e-6


# %%
def transform(q, kind='none', epsilon=1e-6):
    """Apply a flow transformation. Negative values are clipped to zero first."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, None)

    if kind == 'none':
        return q
    if kind == 'sqrt':
        return np.sqrt(q)
    if kind == 'log':
        return np.log(q + epsilon)
    if kind == 'inverse':
        return 1.0 / (q + epsilon)

    raise ValueError(f'Unknown transform {kind!r}. Choose from {TRANSFORMS}.')


# %%
def nse(obs, sim):
    obs, sim = _finite_pair(obs, sim)

    if len(obs) < 2:
        return np.nan

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator <= 0:
        return np.nan

    return 1.0 - np.sum((obs - sim) ** 2) / denominator


# %%
def kge(obs, sim):
    obs, sim = _finite_pair(obs, sim)

    if len(obs) < 2:
        return np.nan

    sd_obs, sd_sim = np.std(obs), np.std(sim)
    mean_obs, mean_sim = np.mean(obs), np.mean(sim)

    if sd_obs <= 0 or mean_obs == 0 or sd_sim <= 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sd_sim / sd_obs
    beta = mean_sim / mean_obs

    if not np.isfinite(r):
        return np.nan

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


# %%
def score(obs, sim, metric='KGE', transform_kind='none', epsilon=None):
    """Efficiency criterion computed on optionally transformed flows.

    The transformation is applied to both series with the same offset before the
    criterion is evaluated, so the result is KGE or NSE of the transformed
    variable, not of flow. Values are not comparable across transformations.
    """
    if epsilon is None:
        epsilon = epsilon_from_obs(obs)

    obs_t = transform(obs, transform_kind, epsilon)
    sim_t = transform(sim, transform_kind, epsilon)

    if metric == 'NSE':
        return nse(obs_t, sim_t)
    if metric == 'KGE':
        return kge(obs_t, sim_t)

    raise ValueError(f'Unknown metric {metric!r}. Choose from {METRICS}.')


# %%
def criterion_label(metric, transform_kind):
    """Short name for reporting, e.g. KGE(1/Q)."""
    suffix = {'none': 'Q', 'sqrt': 'sqrt(Q)', 'log': 'log(Q)', 'inverse': '1/Q'}
    return f'{metric}({suffix[transform_kind]})'
