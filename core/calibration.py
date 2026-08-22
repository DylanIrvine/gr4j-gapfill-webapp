# core/calibration.py
# Differential evolution calibration of the GR family, with a behavioural
# archive assembled from the objective function evaluations.
#
# No module-level mutable state. Everything the calibration accumulates lives
# inside calibrate_gr, so concurrent sessions on a deployed app cannot write
# into each other's archive.

# %%
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from core.models import simulate, PARAM_NAMES, PARAM_BOUNDS, PARAM_ROUNDING
from core.metrics import score, composite_score, epsilon_from_obs

# %% configuration
MAX_BEHAVIOURAL_MODELS = 200

# Minimum width of the local sampling box along any one parameter, as a
# fraction of that parameter's full range. Stops the sample inheriting a
# collapse from the search.
MIN_REFINE_SPAN = 0.03
MIN_FITTING_DAYS = 100
PENALTY = 1e9


# %%
def _objective_function(params, precip, pet, q_obs, warmup_days, metric,
                        transform_kind, composite_weight, epsilon, model, names,
                        archive):
    """Negative criterion, with every valid evaluation appended to archive."""
    param_dict = dict(zip(names, [float(p) for p in params]))

    q_sim = simulate(precip, pet, param_dict, model=model)

    mask = np.isfinite(q_obs) & np.isfinite(q_sim)
    if warmup_days > 0:
        mask[:warmup_days] = False

    if mask.sum() < MIN_FITTING_DAYS:
        return PENALTY

    if composite_weight is None:
        value = score(q_obs[mask], q_sim[mask], metric=metric,
                      transform_kind=transform_kind, epsilon=epsilon)
    else:
        value = composite_score(q_obs[mask], q_sim[mask], metric=metric,
                                weight=composite_weight, epsilon=epsilon)

    if not np.isfinite(value):
        return PENALTY

    archive.append({**param_dict, 'Score': float(value)})

    return -value


# %%
def _behavioural_frame(rows, best_score, delta, names):
    """Rows scoring within delta of the best, sorted best first, as a DataFrame."""
    if not rows:
        return pd.DataFrame(columns=list(names) + ['Score'])

    frame = pd.DataFrame(rows)
    frame = frame[np.isfinite(frame['Score'])]
    frame = frame.sort_values('Score', ascending=False)
    return frame[frame['Score'] >= best_score - delta].copy()


# %%
def calibrate_gr(precip, pet, q_obs, model='GR4J', warmup_days=730, metric='KGE',
                 transform_kind='none', composite_weight=None, maxiter=25, popsize=12,
                 behavioural_delta=0.05, seed=1, bounds=None, refine_sample=0,
                 refine_scale=0.15, progress_callback=None):
    """Calibrate a GR model and return the best parameter set plus an archive.

    Parameters
    ----------
    precip, pet : array-like, mm/d, must be complete
    q_obs : array-like, mm/d, may contain NaN
    model : 'GR4J', 'GR5J' or 'GR6J'
    warmup_days : int, days excluded from the objective at the start of the record
    metric : 'KGE' or 'NSE'
    transform_kind : 'none', 'sqrt', 'log' or 'inverse', applied to both series
        before the criterion is computed. Ignored when composite_weight is set.
    composite_weight : optional float in [0, 1]. When set, the criterion becomes
        a weighted mean of the metric under the untransformed and logarithmic
        transformations, with this weight on the untransformed component. This
        asks the optimiser to perform at both ends of the hydrograph rather than
        trading one for the other. It is a pragmatic calibration target and not
        a likelihood, so report it as such.
    maxiter, popsize : differential evolution settings. Note that scipy sizes the
        population as popsize times the number of parameters, so GR6J costs half
        again as much per generation as GR4J.
    behavioural_delta : models scoring within this distance of the best are retained
    seed : int, fixes the differential evolution random state for reproducibility
    bounds : optional dict of parameter name to (lower, upper). Defaults to
        PARAM_BOUNDS. Widening a bound gives the optimiser more room to
        compensate for data problems; narrowing one suppresses the bound-hit
        warning rather than fixing whatever caused it. Report a constrained run
        alongside the unconstrained one rather than in place of it.
    refine_sample : int. When greater than zero, a Latin hypercube sample of this
        many parameter sets is drawn from a box around the optimum after the
        search converges, and the behavioural set is built from that sample
        rather than from the search trajectory. This matters more than it looks.
        The differential evolution archive is a record of where the optimiser
        travelled, not a sample of anything, so density in it reflects where the
        population happened to linger. A Latin hypercube over the local region
        IS a sample, which makes the resulting spread defensible as a parameter
        sensitivity analysis rather than an artefact of the search path.
    refine_scale : float. Margin added to each side of the box, as a fraction of
        the spread the search found for that parameter. The box itself comes from
        the search, not from the bounds.
    progress_callback : optional callable, called once per generation as
        callback(params_dict, convergence)

    Returns
    -------
    dict with keys best_params, best_score, behavioural_df, model, epsilon,
    bounds, seed, composite_weight

    Note on interpretation
    ---------------------
    The behavioural set is drawn from the differential evolution trajectory, not
    from a random sample of parameter space. The population concentrates near the
    optimum as the search proceeds, so this is not a GLUE sample and the
    resulting ensemble spread should be read as a local sensitivity band around
    the optimum, not as a calibrated predictive uncertainty. Adding parameters
    widens the region of near-equivalent performance, so this caveat matters more
    for GR6J than for GR4J.
    """
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)

    names = PARAM_NAMES[model]

    if bounds is None:
        bounds = {name: PARAM_BOUNDS[name] for name in names}

    missing = [name for name in names if name not in bounds]
    if missing:
        raise ValueError(f'{model} needs bounds for {missing}.')

    bounds = {name: (float(bounds[name][0]), float(bounds[name][1])) for name in names}

    for name in names:
        lo, hi = bounds[name]
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f'Bounds for {name} must satisfy lower < upper, got {lo} to {hi}.')

    if model == 'GR6J' and bounds['X6'][0] <= 0.0:
        raise ValueError('The lower bound on X6 must be strictly positive, it divides the '
                         'store level.')

    bounds_list = [bounds[name] for name in names]

    if composite_weight is not None:
        composite_weight = float(composite_weight)
        if not (0.0 <= composite_weight <= 1.0):
            raise ValueError('composite_weight must lie between 0 and 1, got '
                             f'{composite_weight}.')

    # the transform offset is fixed once from the observations, so every
    # candidate parameter set is scored against exactly the same criterion
    epsilon = epsilon_from_obs(q_obs)

    archive = []

    def _callback(xk, convergence=None):
        progress_callback(dict(zip(names, [float(v) for v in xk])), convergence)
        return False

    result = differential_evolution(
        _objective_function,
        bounds=bounds_list,
        args=(precip, pet, q_obs, warmup_days, metric, transform_kind,
              composite_weight, epsilon, model, names, archive),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=_callback if progress_callback is not None else None,
    )

    best_score = float(-result.fun)
    best_params = dict(zip(names, [float(v) for v in result.x]))
    best_params['ObjectiveValue'] = best_score

    # Local Latin hypercube refinement.
    #
    # The sampling box is taken from where the SEARCH found behavioural sets,
    # expanded by a margin, rather than from a fixed fraction of the parameter
    # bounds. A fixed fraction cannot work: 15 per cent of X1's range is 450 mm,
    # so the box would span most of the plausible parameter space and almost
    # nothing sampled in it would be behavioural. The division of labour is the
    # point. Differential evolution is good at locating the behavioural region
    # and bad at sampling it, because its density reflects the path taken. A
    # Latin hypercube is the reverse. Using each for what it is good at gives a
    # behavioural set that is an actual sample of an actual region.
    sample_rows = []
    sampled = False

    trajectory_df = _behavioural_frame(archive, best_score, behavioural_delta, names)

    if refine_sample and refine_sample > 0 and len(trajectory_df) >= 5:
        lower = np.empty(len(names))
        upper = np.empty(len(names))

        for i, name in enumerate(names):
            values = trajectory_df[name].to_numpy(dtype=float)
            low, high = float(values.min()), float(values.max())
            span = high - low

            # A parameter the search collapsed onto, typically because it ran
            # into a bound, has a spread near zero. Taking the box straight from
            # that spread makes the hypercube degenerate along that axis: every
            # sampled point gets the same value and the histogram becomes a
            # single bar. The sample would then be faithfully reporting the
            # search's collapse rather than testing it. A minimum width forces
            # the sample to explore the parameter regardless of what the search
            # did, which is the whole reason for sampling in the first place.
            minimum_span = MIN_REFINE_SPAN * (bounds[name][1] - bounds[name][0])
            span = max(span, minimum_span)

            centre = 0.5 * (low + high)
            half_width = 0.5 * span + refine_scale * span

            lower[i] = max(bounds[name][0], centre - half_width)
            upper[i] = min(bounds[name][1], centre + half_width)

            # a bound-limited parameter still needs somewhere to go, so push the
            # box inward from the bound rather than collapsing against it
            if upper[i] - lower[i] < minimum_span:
                if lower[i] <= bounds[name][0]:
                    upper[i] = min(bounds[name][1], bounds[name][0] + minimum_span)
                    lower[i] = bounds[name][0]
                else:
                    lower[i] = max(bounds[name][0], bounds[name][1] - minimum_span)
                    upper[i] = bounds[name][1]

            if upper[i] <= lower[i]:
                lower[i], upper[i] = bounds[name]

        engine = qmc.LatinHypercube(d=len(names), seed=seed)
        candidates = qmc.scale(engine.random(int(refine_sample)), lower, upper)

        mark = len(archive)
        for candidate in candidates:
            _objective_function(candidate, precip, pet, q_obs, warmup_days, metric,
                                transform_kind, composite_weight, epsilon, model,
                                names, archive)
        sample_rows = archive[mark:]

        # the sample can find a better set than the polished optimum
        for entry in sample_rows:
            if entry['Score'] > best_score:
                best_score = float(entry['Score'])
                best_params = {name: float(entry[name]) for name in names}
                best_params['ObjectiveValue'] = best_score

    if len(archive) == 0:
        empty = pd.DataFrame(columns=list(names) + ['Score'])
        return {'best_params': best_params, 'best_score': best_score,
                'behavioural_df': empty, 'model': model, 'epsilon': epsilon,
                'bounds': bounds, 'seed': int(seed),
                'composite_weight': composite_weight,
                'behavioural_source': 'differential evolution trajectory',
                'n_sampled': 0, 'refine_scale': float(refine_scale)}

    # When a local sample was drawn, the behavioural set comes from the sample
    # only. Mixing it with the search trajectory would reintroduce exactly the
    # unsampled density the refinement exists to remove.
    behavioural_df = _behavioural_frame(archive, best_score, behavioural_delta, names)

    if sample_rows:
        sample_df = _behavioural_frame(sample_rows, best_score, behavioural_delta, names)
        if len(sample_df) >= 10:
            behavioural_df = sample_df
            sampled = True

    # round first, so numerically identical parameter sets collapse to one entry
    for name in names:
        behavioural_df[name] = behavioural_df[name].round(PARAM_ROUNDING[name])

    behavioural_df = behavioural_df.drop_duplicates(subset=list(names))
    behavioural_df = behavioural_df.head(MAX_BEHAVIOURAL_MODELS).reset_index(drop=True)

    return {'best_params': best_params, 'best_score': best_score,
            'behavioural_df': behavioural_df, 'model': model, 'epsilon': epsilon,
            'bounds': bounds, 'seed': int(seed),
            'composite_weight': composite_weight,
            'behavioural_source': 'local Latin hypercube sample' if sampled
                                  else 'differential evolution trajectory',
            'n_sampled': len(sample_rows), 'refine_scale': float(refine_scale)}


# %% backwards compatible alias
def calibrate_gr4j(precip, pet, q_obs, warmup_days=730, objective='KGE', **kwargs):
    """Deprecated. Retained so older calls keep working; use calibrate_gr."""
    return calibrate_gr(precip, pet, q_obs, model='GR4J', warmup_days=warmup_days,
                        metric=objective, **kwargs)
