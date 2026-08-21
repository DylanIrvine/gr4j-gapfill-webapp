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


# %% composite criterion
# A single transformation forces a choice between the high-flow and the low-flow
# end of the hydrograph. Averaging the same criterion under two transformations
# asks the optimiser to do reasonably well at both instead of trading one for
# the other, and sweeping the weight from 0 to 1 traces out the trade-off curve.
#
# Caveat to state in any write-up: a weighted mean of two efficiency criteria is
# a pragmatic calibration choice, not a likelihood. It makes no coherent
# statistical assumption about the residual structure, and it sits in the
# territory McInerney and colleagues describe as objective function
# inconsistency, where the implicit error assumptions of the criterion do not
# match the actual residual errors. Describe it as a calibration target, not as
# an inference.

COMPOSITE_TRANSFORMS = ('none', 'log')


def composite_score(obs, sim, metric='KGE', weight=0.5,
                    transforms=COMPOSITE_TRANSFORMS, epsilon=None):
    """Weighted mean of one criterion evaluated under two transformations.

    weight applies to the first transformation, 1 - weight to the second, so
    weight = 1 reduces to the first alone and weight = 0 to the second alone.
    Returns NaN if either component is not finite, so a parameter set that
    breaks one end of the hydrograph cannot be rescued by the other.
    """
    if epsilon is None:
        epsilon = epsilon_from_obs(obs)

    first = score(obs, sim, metric=metric, transform_kind=transforms[0], epsilon=epsilon)
    second = score(obs, sim, metric=metric, transform_kind=transforms[1], epsilon=epsilon)

    if not (np.isfinite(first) and np.isfinite(second)):
        return np.nan

    w = float(np.clip(weight, 0.0, 1.0))
    return w * first + (1.0 - w) * second


def composite_label(metric, weight, transforms=COMPOSITE_TRANSFORMS):
    """Short name for reporting, e.g. 0.50*KGE(Q) + 0.50*KGE(log(Q))."""
    w = float(np.clip(weight, 0.0, 1.0))
    return (f'{w:.2f}*{criterion_label(metric, transforms[0])} + '
            f'{1.0 - w:.2f}*{criterion_label(metric, transforms[1])}')
