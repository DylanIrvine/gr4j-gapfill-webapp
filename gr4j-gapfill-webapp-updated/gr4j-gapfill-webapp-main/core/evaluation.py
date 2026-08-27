# core/evaluation.py
# Signature-based evaluation of calibrated models against the observed record.
#
# Why this exists
# ---------------
# A single efficiency score measures the distance between simulation and
# observation under one weighting. It does not indicate whether the model
# reproduces the shape of the flow duration curve, the proportion of flow
# arriving as baseflow, the flashiness of the hydrograph, the seasonality of the
# regime, or the number of days on which flow ceases.
#
# Those properties are what downstream analyses rely on. A model may score 0.9
# on KGE while misestimating the baseflow index by a third. Evaluating against a
# set of signatures rather than a single score follows the diagnostic approach
# of Gupta et al. (2008) and the FARE framework of Euser et al. (2013).
#
# Everything here is computed on the days where an observation exists, so the
# comparison is like for like and gap filled values never enter the evaluation.
#
# References
#   Gupta, H.V., Wagener, T., Liu, Y. (2008). Reconciling theory with
#     observations: elements of a diagnostic approach to model evaluation.
#     Hydrological Processes 22, 3802-3813.
#   Euser, T., Winsemius, H.C., Hrachowitz, M., Fenicia, F., Uhlenbrook, S.,
#     Savenije, H.H.G. (2013). A framework to assess the realism of model
#     structures using hydrological signatures. HESS 17, 1893-1912.

# %%
import numpy as np
import pandas as pd

from core.metrics import (kge, nse, score, transform, epsilon_from_obs,
                      resolve_kge_bias, criterion_label)
from core.baseflow import lyne_hollick
from core.indices import (colwell_indices, seasonality, fdc_indices,
                          richards_baker_index, coefficient_of_variation)

# %% which signatures are reported, and in what order
SIGNATURE_ORDER = [
    'Mean flow (mm/d)',
    'Median flow (mm/d)',
    'Q10, exceeded 10% of time (mm/d)',
    'Q50, exceeded 50% of time (mm/d)',
    'Q90, exceeded 90% of time (mm/d)',
    'Maximum daily flow (mm/d)',
    'Flow duration curve slope (33-66%)',
    'Q5 to Q95 ratio',
    'Baseflow index',
    'Richards-Baker flashiness index',
    'Coefficient of variation of daily flow',
    'Seasonality strength (0-1)',
    'Flow-weighted mean day of year',
    'Colwell predictability (P)',
    'Colwell constancy (C)',
    'Colwell contingency (M)',
    'Zero-flow fraction',
    'Days at or below cease-to-flow threshold',
]


# %%
def _exceeded(values, percent):
    """Flow exceeded the given percentage of the time."""
    return float(np.percentile(values, 100.0 - percent))


def _signatures(dates, q, alpha, passes, n_reflect, ctf_threshold):
    """Every signature for one flow series, as a dict."""
    q = np.asarray(q, dtype=float)

    colwell = colwell_indices(dates, q)
    season = seasonality(dates, q)
    fdc = fdc_indices(q)

    try:
        separation = lyne_hollick(q, alpha=alpha, passes=passes, n_reflect=n_reflect)
        bfi = separation['bfi']
    except (ValueError, IndexError):
        bfi = np.nan

    return {
        'Mean flow (mm/d)': float(np.mean(q)),
        'Median flow (mm/d)': float(np.median(q)),
        'Q10, exceeded 10% of time (mm/d)': _exceeded(q, 10.0),
        'Q50, exceeded 50% of time (mm/d)': _exceeded(q, 50.0),
        'Q90, exceeded 90% of time (mm/d)': _exceeded(q, 90.0),
        'Maximum daily flow (mm/d)': float(np.max(q)),
        'Flow duration curve slope (33-66%)': fdc['fdc_slope'],
        'Q5 to Q95 ratio': fdc['q5_q95_ratio'],
        'Baseflow index': bfi,
        'Richards-Baker flashiness index': richards_baker_index(q),
        'Coefficient of variation of daily flow': coefficient_of_variation(q),
        'Seasonality strength (0-1)': season['strength'],
        'Flow-weighted mean day of year': season['mean_day'],
        'Colwell predictability (P)': colwell['predictability'],
        'Colwell constancy (C)': colwell['constancy'],
        'Colwell contingency (M)': colwell['contingency'],
        'Zero-flow fraction': fdc['zero_flow_fraction'],
        'Days at or below cease-to-flow threshold': float(np.sum(q <= ctf_threshold)),
    }


# %%
def efficiency_table(q_obs, simulations, warmup_days=0, kge_bias='auto'):
    """Efficiency criteria under every transformation, for each simulation.

    KGE is reported with its three components, because a composite of 0.85 built
    from a correlation of 0.95 and a variability ratio of 1.3 describes a very
    different failure from one built from a correlation of 0.87 and a
    variability ratio of 1.0, and the composite alone cannot tell them apart.
    """
    q_obs = np.asarray(q_obs, dtype=float)

    mask = np.isfinite(q_obs)
    if warmup_days > 0:
        mask[:warmup_days] = False

    epsilon = epsilon_from_obs(q_obs[mask])
    rows = []

    for label, q_sim in simulations.items():
        q_sim = np.asarray(q_sim, dtype=float)
        pair = mask & np.isfinite(q_sim)
        obs, sim = q_obs[pair], q_sim[pair]

        entry = {'Model': label, 'N days': int(pair.sum())}

        for kind in ('none', 'sqrt', 'log', 'inverse'):
            entry[criterion_label('NSE', kind)] = score(obs, sim, 'NSE', kind, epsilon)
            entry[criterion_label('KGE', kind, kge_bias)] = score(
                obs, sim, 'KGE', kind, epsilon, kge_bias=kge_bias)

        entry['KGE bias form'] = resolve_kge_bias('log', kge_bias)

        # KGE components on untransformed flow
        if obs.size > 2 and np.std(obs) > 0 and np.std(sim) > 0:
            entry['KGE r'] = float(np.corrcoef(obs, sim)[0, 1])
            entry['KGE alpha (variability)'] = float(np.std(sim) / np.std(obs))
            entry['KGE beta (bias)'] = float(np.mean(sim) / np.mean(obs))
        else:
            entry['KGE r'] = entry['KGE alpha (variability)'] = np.nan
            entry['KGE beta (bias)'] = np.nan

        rows.append(entry)

    return pd.DataFrame(rows)


# %%
def signature_report(dates, q_obs, simulations, alpha=0.925, passes=3,
                     n_reflect=30, ctf_threshold=0.0, warmup_days=0):
    """Compare observed and simulated signatures on the observed days only.

    Returns one row per signature with the observed value, the value from each
    simulation, and the percentage error of each. Percentage error is relative
    to the observed value and is left blank where that value is zero or where
    the signature is a date, since a percentage error on a day of year is
    meaningless.
    """
    dates = pd.DatetimeIndex(dates)
    q_obs = np.asarray(q_obs, dtype=float)

    mask = np.isfinite(q_obs)
    if warmup_days > 0:
        mask[:warmup_days] = False

    if mask.sum() < 365:
        return pd.DataFrame()

    obs_dates = dates[mask]
    observed = _signatures(obs_dates, q_obs[mask], alpha, passes, n_reflect,
                           ctf_threshold)

    table = {'Signature': SIGNATURE_ORDER,
             'Observed': [observed[k] for k in SIGNATURE_ORDER]}

    no_percentage = {'Flow-weighted mean day of year'}

    for label, q_sim in simulations.items():
        q_sim = np.asarray(q_sim, dtype=float)
        simulated = _signatures(obs_dates, q_sim[mask], alpha, passes, n_reflect,
                                ctf_threshold)

        table[label] = [simulated[k] for k in SIGNATURE_ORDER]

        errors = []
        for k in SIGNATURE_ORDER:
            o, s = observed[k], simulated[k]
            if k in no_percentage or not np.isfinite(o) or o == 0:
                errors.append(np.nan)
            else:
                errors.append(100.0 * (s - o) / o)
        table[f'{label} error %'] = errors

    return pd.DataFrame(table)


# %%
def worst_signatures(report, threshold=25.0):
    """Signatures reproduced worst, for a plain-language summary.

    Returns the signature names where any simulation is out by more than the
    threshold percentage, which is the list worth putting in a results section
    rather than the full table.
    """
    if report.empty:
        return []

    error_columns = [c for c in report.columns if c.endswith('error %')]
    if not error_columns:
        return []

    worst = report[['Signature'] + error_columns].copy()
    worst['MaxAbsError'] = worst[error_columns].abs().max(axis=1)
    worst = worst[worst['MaxAbsError'] > threshold]

    return worst.sort_values('MaxAbsError', ascending=False)['Signature'].tolist()
