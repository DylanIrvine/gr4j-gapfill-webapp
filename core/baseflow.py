# core/baseflow.py
# Lyne and Hollick recursive digital filter, with alpha optionally derived from
# the observed recession behaviour rather than taken from the literature.
#
# References
#   Lyne, V., Hollick, M. (1979). Stochastic time-variable rainfall-runoff
#     modelling. Institute of Engineers Australia National Conference, 89-93.
#   Nathan, R.J., McMahon, T.A. (1990). Evaluation of automated techniques for
#     base flow and recession analysis. Water Resources Research 26(7), 1465-1473.
#
# On the choice of alpha
# ----------------------
# The conventional value of 0.925 comes from Nathan and McMahon, who chose it
# because the resulting separations resembled manual separations on Australian
# catchments. It was not derived from recession theory. The filter coefficient
# and the linear reservoir recession constant are related but they are not the
# same quantity, so deriving alpha from the observed recession is a change of
# definition rather than a refinement of the original method. It is defensible,
# and arguably better suited to a catchment with an unusual recession, but any
# write-up should report both values and say which was used.

# %%
import numpy as np
import pandas as pd

# %% defaults
DEFAULT_ALPHA = 0.925
DEFAULT_PASSES = 3
DEFAULT_REFLECT = 30

# recession extraction
RECESSION_MIN_LENGTH = 5      # minimum consecutive falling days to use a segment
RECESSION_SKIP_DAYS = 2       # days dropped after each peak, to exclude quickflow
RECESSION_QUANTILE = 0.5      # quantile of the daily flow ratio taken as alpha
RECESSION_MIN_FLOW = 0.0      # flows at or below this are excluded
RECESSION_MAX_RATIO = 0.999   # days declining by less than this are not informative


# %%
def _as_float(q):
    return np.asarray(q, dtype=float)


def _filter_pass(series, alpha, reverse=False):
    """One pass of the Lyne and Hollick filter over a baseflow series.

    Returns the updated baseflow. The quickflow component is discarded here
    because each pass filters the baseflow left by the previous pass, which is
    what makes the three-pass version a three-pass version.
    """
    x = series[::-1] if reverse else series

    quick = np.zeros(len(x))
    quick[0] = x[0]

    for i in range(1, len(x)):
        quick[i] = alpha * quick[i - 1] + 0.5 * (1.0 + alpha) * (x[i] - x[i - 1])

    base = np.where(quick > 0.0, x - quick, x)
    return base[::-1] if reverse else base


def _separate_block(q, alpha, passes, n_reflect):
    """Filter one complete block of data. Returns (baseflow, BFI)."""
    q = _as_float(q)

    # reflect at both ends so the filter has run-in either side
    reflected = np.concatenate([q[n_reflect:0:-1], q, q[-2:-n_reflect - 2:-1]])

    base = _filter_pass(reflected, alpha)

    for _ in range((passes - 1) // 2):
        base = _filter_pass(_filter_pass(base, alpha, reverse=True), alpha)

    base = base[n_reflect:len(base) - n_reflect]
    base = np.clip(base, 0.0, None)
    base = np.minimum(base, q)

    total = float(np.sum(q))
    bfi = float(np.sum(base) / total) if total > 0 else np.nan

    return base, bfi


def _valid_runs(q, min_length):
    """Runs of consecutive finite values longer than min_length."""
    finite = np.isfinite(q).astype(int)
    transitions = np.diff(np.concatenate(([0], finite, [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) > min_length]


# %%
def lyne_hollick(q, alpha=DEFAULT_ALPHA, passes=DEFAULT_PASSES,
                 n_reflect=DEFAULT_REFLECT):
    """Separate baseflow from a daily flow series.

    Handles gaps by filtering each run of consecutive finite values separately,
    which is the only defensible treatment: the filter is recursive, so carrying
    it across a gap would propagate a state that has no basis in observation.

    Returns
    -------
    dict with baseflow, quickflow, bfi, alpha, fraction_used, n_blocks
    """
    if passes % 2 == 0 or passes < 3:
        raise ValueError('passes must be odd and at least 3.')
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f'alpha must satisfy 0 <= alpha < 1, got {alpha}.')

    q = _as_float(q)

    if len(q) <= n_reflect:
        raise ValueError(f'The series must be longer than n_reflect ({n_reflect}).')

    baseflow = np.full(len(q), np.nan)

    if np.isfinite(q).all():
        baseflow, bfi = _separate_block(q, alpha, passes, n_reflect)
        fraction_used = 1.0
        n_blocks = 1
    else:
        # NOTE: runs of VALID data, not runs of gaps. Getting this the wrong way
        # round silently separates the missing blocks and ignores the observed
        # ones, which is a failure mode that produces plausible-looking output.
        blocks = _valid_runs(q, n_reflect)

        weights, block_bfi = [], []

        for start, end in blocks:
            block = q[start:end + 1]
            base, bfi = _separate_block(block, alpha, passes, n_reflect)
            baseflow[start:end + 1] = base
            weights.append(len(block))
            block_bfi.append(bfi)

        n_finite = int(np.isfinite(q).sum())

        if weights:
            bfi = float(np.average(block_bfi, weights=weights))
            fraction_used = float(np.sum(weights) / n_finite) if n_finite else 0.0
        else:
            bfi, fraction_used = np.nan, 0.0

        n_blocks = len(blocks)

    quickflow = q - baseflow

    return {'baseflow': baseflow, 'quickflow': quickflow, 'bfi': bfi,
            'alpha': float(alpha), 'fraction_used': fraction_used,
            'n_blocks': n_blocks}


# %%
def recession_alpha(q, min_length=RECESSION_MIN_LENGTH, skip_days=RECESSION_SKIP_DAYS,
                    quantile=RECESSION_QUANTILE, min_flow=RECESSION_MIN_FLOW,
                    max_ratio=RECESSION_MAX_RATIO):
    """Derive a filter coefficient from the falling limbs of the hydrograph.

    Consecutive strictly decreasing days are treated as recession segments. The
    first skip_days of each segment are discarded, because flow immediately
    after a peak is still dominated by quickflow and would bias the ratio low.
    Alpha is then the chosen quantile of the daily ratio Q(t)/Q(t-1) across all
    retained recession days.

    Days declining by less than max_ratio are excluded. Without that guard, a
    recession that has flattened onto a baseflow floor contributes a long tail
    of ratios arbitrarily close to 1, biasing the estimate upward. On a
    single-component synthetic recession with a decay constant of 0.85, omitting
    the guard raises the estimate to approximately 0.97.

    Interpretation caveat
    ---------------------
    A hydrograph mixes a fast quickflow recession with a slow baseflow
    recession, so the ratios are drawn from a mixture and the estimate depends
    on which part of that mixture is sampled. The quartiles are returned
    alongside alpha for this reason: a wide interquartile range indicates the
    estimate is not well defined for the catchment and that the choice of
    quantile materially affects the result. Both should be reported.

    Returns
    -------
    dict with alpha, n_segments, n_ratios, and the ratio quartiles for
    diagnostics. alpha is NaN when too few recession days are available.
    """
    q = _as_float(q)

    usable = np.isfinite(q) & (q > min_flow)
    falling = np.zeros(len(q), dtype=bool)
    falling[1:] = usable[1:] & usable[:-1] & (q[1:] < q[:-1])

    transitions = np.diff(np.concatenate(([0], falling.astype(int), [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1

    ratios = []
    n_segments = 0

    for start, end in zip(starts, ends):
        if (end - start + 1) < min_length:
            continue

        n_segments += 1
        first = start + skip_days

        if first > end:
            continue

        idx = np.arange(first, end + 1)
        ratio = q[idx] / q[idx - 1]
        ratios.append(ratio[(ratio > 0.0) & (ratio <= max_ratio)])

    if not ratios:
        return {'alpha': np.nan, 'n_segments': n_segments, 'n_ratios': 0,
                'q25': np.nan, 'q50': np.nan, 'q75': np.nan}

    ratios = np.concatenate(ratios)

    if len(ratios) < 10:
        return {'alpha': np.nan, 'n_segments': n_segments, 'n_ratios': int(len(ratios)),
                'q25': np.nan, 'q50': np.nan, 'q75': np.nan}

    return {'alpha': float(np.quantile(ratios, quantile)),
            'n_segments': n_segments,
            'n_ratios': int(len(ratios)),
            'q25': float(np.quantile(ratios, 0.25)),
            'q50': float(np.quantile(ratios, 0.50)),
            'q75': float(np.quantile(ratios, 0.75))}


# %%
def cease_to_flow(q, threshold=0.0):
    """Boolean array marking days at or below the cease-to-flow threshold.

    A threshold of exactly zero is the strict definition. A small positive
    threshold is often more useful in practice, because a gauge reading a few
    litres per second is not a flowing river, and because GR6J cannot produce
    exactly zero flow by construction, so modelled cease-to-flow days will never
    be detected at a threshold of zero.
    """
    q = _as_float(q)
    return np.isfinite(q) & (q <= threshold)


# %% recession analysis
RECESSION_BINS = 20
RECESSION_ENVELOPE_QUANTILE = 0.05


def recession_analysis(q, min_length=RECESSION_MIN_LENGTH, skip_days=RECESSION_SKIP_DAYS,
                       min_flow=RECESSION_MIN_FLOW, n_bins=RECESSION_BINS,
                       envelope_quantile=RECESSION_ENVELOPE_QUANTILE):
    """Brutsaert and Nieber recession analysis, -dQ/dt against Q.

    Plotting the rate of recession against the flow at which it occurs collapses
    every recession in the record onto one relationship, -dQ/dt = a Q^b. The
    lower envelope of that cloud is taken to represent drainage from storage
    with no residual quickflow and no evapotranspiration, so its slope and
    intercept are an aquifer drainage signature rather than an event property.

    The exponent b is diagnostic. b close to 1 corresponds to a linear
    reservoir, the assumption underlying an exponential recession and the
    constant filter coefficient of Lyne and Hollick. b near 1.5 corresponds to
    the Boussinesq late-time solution for a horizontal unconfined aquifer, and b
    near 3 to the early-time solution. Where b departs markedly from 1, a single
    recession constant is a poor description of the catchment, which bears
    directly on the interpretation of a derived filter coefficient.

    When b is close to 1, the equivalent daily recession ratio is exp(-a), which
    is directly comparable with the filter coefficient.

    Returns
    -------
    dict with a, b, alpha_equivalent, n_points, r_squared, and the point cloud
    (q_mid, dqdt) plus the fitted envelope for plotting.
    """
    q = np.asarray(q, dtype=float)

    usable = np.isfinite(q) & (q > min_flow)
    falling = np.zeros(len(q), dtype=bool)
    falling[1:] = usable[1:] & usable[:-1] & (q[1:] < q[:-1])

    transitions = np.diff(np.concatenate(([0], falling.astype(int), [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1

    q_mid, dqdt = [], []
    n_segments = 0

    for start, end in zip(starts, ends):
        if (end - start + 1) < min_length:
            continue

        n_segments += 1
        first = start + skip_days
        if first > end:
            continue

        idx = np.arange(first, end + 1)
        q_mid.append(0.5 * (q[idx] + q[idx - 1]))
        dqdt.append(q[idx - 1] - q[idx])          # positive, since flow is falling

    if not q_mid:
        return {'a': np.nan, 'b': np.nan, 'alpha_equivalent': np.nan,
                'n_points': 0, 'n_segments': n_segments, 'r_squared': np.nan,
                'q_mid': np.array([]), 'dqdt': np.array([]),
                'envelope_q': np.array([]), 'envelope_dqdt': np.array([])}

    q_mid = np.concatenate(q_mid)
    dqdt = np.concatenate(dqdt)

    valid = (q_mid > 0) & (dqdt > 0) & np.isfinite(q_mid) & np.isfinite(dqdt)
    q_mid, dqdt = q_mid[valid], dqdt[valid]

    if q_mid.size < 20:
        return {'a': np.nan, 'b': np.nan, 'alpha_equivalent': np.nan,
                'n_points': int(q_mid.size), 'n_segments': n_segments,
                'r_squared': np.nan, 'q_mid': q_mid, 'dqdt': dqdt,
                'envelope_q': np.array([]), 'envelope_dqdt': np.array([])}

    # lower envelope: a low quantile of dQ/dt within logarithmic flow bins
    edges = np.geomspace(q_mid.min(), q_mid.max(), n_bins + 1)
    bin_index = np.clip(np.digitize(q_mid, edges) - 1, 0, n_bins - 1)

    envelope_q, envelope_dqdt = [], []
    for b_index in range(n_bins):
        selection = bin_index == b_index
        if selection.sum() < 5:
            continue
        envelope_q.append(float(np.median(q_mid[selection])))
        envelope_dqdt.append(float(np.quantile(dqdt[selection], envelope_quantile)))

    envelope_q = np.array(envelope_q)
    envelope_dqdt = np.array(envelope_dqdt)

    positive = envelope_dqdt > 0
    envelope_q, envelope_dqdt = envelope_q[positive], envelope_dqdt[positive]

    if envelope_q.size < 4:
        return {'a': np.nan, 'b': np.nan, 'alpha_equivalent': np.nan,
                'n_points': int(q_mid.size), 'n_segments': n_segments,
                'r_squared': np.nan, 'q_mid': q_mid, 'dqdt': dqdt,
                'envelope_q': envelope_q, 'envelope_dqdt': envelope_dqdt}

    log_q = np.log(envelope_q)
    log_d = np.log(envelope_dqdt)
    slope, intercept = np.polyfit(log_q, log_d, 1)

    predicted = slope * log_q + intercept
    residual = log_d - predicted
    total = log_d - log_d.mean()
    r_squared = float(1.0 - np.sum(residual ** 2) / np.sum(total ** 2)) if np.any(total) else np.nan

    a = float(np.exp(intercept))
    b = float(slope)
    alpha_equivalent = float(np.exp(-a)) if abs(b - 1.0) < 0.25 else np.nan

    return {'a': a, 'b': b, 'alpha_equivalent': alpha_equivalent,
            'n_points': int(q_mid.size), 'n_segments': n_segments,
            'r_squared': r_squared, 'q_mid': q_mid, 'dqdt': dqdt,
            'envelope_q': envelope_q, 'envelope_dqdt': envelope_dqdt}
