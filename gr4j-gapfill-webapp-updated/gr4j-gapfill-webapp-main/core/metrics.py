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


# %% the bias term
# KGE's bias component is beta = mean_sim / mean_obs, a ratio of means. That is
# well defined for discharge, which is strictly positive, and it is NOT well
# defined for a signed variable. Log-transformed flow is signed: ln(Q + eps)
# passes through zero when Q + eps passes through 1, so mean_obs can be
# arbitrarily close to zero and beta explodes.
#
# The effect is large. A uniform five per cent overestimate yields KGE(log Q)
# between approximately 0.37 and 0.99 depending only on the units in which flow
# is expressed, because a change of units moves the mean of the log series
# through zero. NSE on the same series is unaffected, having no ratio-of-means
# term.
#
# Two bias formulations are therefore offered:
#
#   'ratio'         beta = mu_sim / mu_obs
#                   the standard KGE of Gupta et al. (2009). Use it on
#                   untransformed, square root or inverse flow, all of which are
#                   strictly positive. Comparable with the published literature.
#
#   'standardised'  beta = 1 + (mu_sim - mu_obs) / sd_obs
#                   the bias expressed in standard deviations rather than as a
#                   ratio. Well defined for any variable, reduces to the same
#                   behaviour when the mean is comfortably away from zero, and
#                   is the sensible choice for a signed transform.
#
# The default is 'ratio' so results stay comparable with everything published,
# and kge_bias_is_unstable() flags when that choice cannot be trusted.

BIAS_FORMS = ('auto', 'ratio', 'standardised')

# Transformations that can produce a negative value, for which a ratio of means
# is not a meaningful bias measure.
SIGNED_TRANSFORMS = ('log',)

# below this ratio of |mean| to standard deviation, the ratio form is unreliable
BIAS_STABILITY_LIMIT = 0.5


def resolve_kge_bias(transform_kind, kge_bias='auto'):
    """Which bias form to use for a given transformation.

    'auto' selects 'standardised' for transformations that produce a signed
    variable and 'ratio' otherwise. This is the default because the alternative
    is an option that appears to work and does not: KGE on log-transformed flow
    with a ratio bias returns a score determined partly by the units the flow is
    expressed in.

    The consequence is that a KGE(log Q) value from this package is not directly
    comparable with one computed elsewhere using the standard formula. The bias
    form is recorded alongside every reported value so the difference is visible
    rather than implicit. Pass kge_bias='ratio' to reproduce the standard
    formula, accepting its instability.
    """
    if kge_bias not in BIAS_FORMS:
        raise ValueError(f'Unknown bias form {kge_bias!r}. Choose from {BIAS_FORMS}.')

    if kge_bias != 'auto':
        return kge_bias

    return 'standardised' if transform_kind in SIGNED_TRANSFORMS else 'ratio'


def kge_bias_is_unstable(values, limit=BIAS_STABILITY_LIMIT):
    """True when the mean of a series is too close to zero for a ratio bias.

    Pass the TRANSFORMED observations. Returns True when |mean| is small
    relative to the standard deviation, which is when beta = mu_sim / mu_obs
    stops being meaningful.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return False

    sd = float(np.std(values))
    if sd <= 0:
        return False

    return abs(float(np.mean(values))) / sd < limit


# %%
def kge(obs, sim, bias='ratio'):
    """Kling-Gupta efficiency.

    bias selects the formulation of the bias component; see BIAS_FORMS above.
    Use 'standardised' for any transformed variable that can take a negative
    value, which among the transforms here means the logarithm.
    """
    obs, sim = _finite_pair(obs, sim)

    if len(obs) < 2:
        return np.nan

    sd_obs, sd_sim = np.std(obs), np.std(sim)
    mean_obs, mean_sim = np.mean(obs), np.mean(sim)

    if sd_obs <= 0 or sd_sim <= 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    if not np.isfinite(r):
        return np.nan

    alpha = sd_sim / sd_obs

    if bias == 'standardised':
        beta = 1.0 + (mean_sim - mean_obs) / sd_obs
    elif bias == 'ratio':
        if mean_obs == 0:
            return np.nan
        beta = mean_sim / mean_obs
    else:
        raise ValueError(f'Unknown bias form {bias!r}. Choose from {BIAS_FORMS}.')

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


# %%
def score(obs, sim, metric='KGE', transform_kind='none', epsilon=None,
          kge_bias='auto'):
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
        return kge(obs_t, sim_t, bias=resolve_kge_bias(transform_kind, kge_bias))

    raise ValueError(f'Unknown metric {metric!r}. Choose from {METRICS}.')


# %%
def criterion_label(metric, transform_kind, kge_bias='auto'):
    """Short name for reporting, e.g. KGE(1/Q).

    A KGE computed with the standardised bias is marked with an asterisk, so a
    value that is not directly comparable with the standard formula is never
    reported as though it were.
    """
    suffix = {'none': 'Q', 'sqrt': 'sqrt(Q)', 'log': 'log(Q)', 'inverse': '1/Q'}
    name = f'{metric}({suffix[transform_kind]})'

    if metric == 'KGE' and resolve_kge_bias(transform_kind, kge_bias) == 'standardised':
        name = f'{metric}*({suffix[transform_kind]})'

    return name


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
                    transforms=COMPOSITE_TRANSFORMS, epsilon=None,
                    kge_bias='auto'):
    """Weighted mean of one criterion evaluated under two transformations.

    weight applies to the first transformation, 1 - weight to the second, so
    weight = 1 reduces to the first alone and weight = 0 to the second alone.
    Returns NaN if either component is not finite, so a parameter set that
    breaks one end of the hydrograph cannot be rescued by the other.
    """
    if epsilon is None:
        epsilon = epsilon_from_obs(obs)

    first = score(obs, sim, metric=metric, transform_kind=transforms[0], epsilon=epsilon,
                  kge_bias=kge_bias)
    second = score(obs, sim, metric=metric, transform_kind=transforms[1], epsilon=epsilon,
                   kge_bias=kge_bias)

    if not (np.isfinite(first) and np.isfinite(second)):
        return np.nan

    w = float(np.clip(weight, 0.0, 1.0))
    return w * first + (1.0 - w) * second


def composite_label(metric, weight, transforms=COMPOSITE_TRANSFORMS, kge_bias='auto'):
    """Short name for reporting, e.g. 0.50*KGE(Q) + 0.50*KGE*(log(Q))."""
    w = float(np.clip(weight, 0.0, 1.0))
    return (f'{w:.2f}*{criterion_label(metric, transforms[0], kge_bias)} + '
            f'{1.0 - w:.2f}*{criterion_label(metric, transforms[1], kge_bias)}')
