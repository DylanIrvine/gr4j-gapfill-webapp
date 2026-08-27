# core/indices.py
# Hydrological signature indices computed from a daily flow record.
#
# The metrics implemented here are selected both for their descriptive value
# and because they are sensitive to how a record was gap filled and to which
# model produced the filled values. That sensitivity is itself informative and
# should be reported alongside the metric.
#
# References
#   Colwell, R.K. (1974). Predictability, constancy and contingency of periodic
#     phenomena. Ecology 55(5), 1148-1153.
#   Brutsaert, W., Nieber, J.L. (1977). Regionalized drought flow hydrographs
#     from a mature glaciated plateau. Water Resources Research 13(3), 637-643.
#   Baker, D.B., Richards, R.P., Loftus, T.T., Kramer, J.W. (2004). A new
#     flashiness index. J. American Water Resources Association 40(2), 503-522.
#   Sankarasubramanian, A., Vogel, R.M., Limbrunner, J.F. (2001). Climate
#     elasticity of streamflow in the United States. WRR 37(6), 1771-1781.
#   Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's
#     tau. J. American Statistical Association 63, 1379-1389.
#   Liebmann, B., Marengo, J.A. (2001). Interannual variability of the rainy
#     season and rainfall in the Brazilian Amazon Basin. J. Climate 14, 4308-4318.

# %%
import numpy as np
import pandas as pd
from scipy import stats

DAYS_IN_YEAR = 365.25


# %% helpers
def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _runs(flag):
    """Start and end indices of each run of True."""
    flag = np.asarray(flag).astype(int)
    transitions = np.diff(np.concatenate(([0], flag, [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# %% cease to flow spells
def cease_to_flow_spells(dates, ctf_flag, filled_flag=None):
    """Every continuous no-flow spell, not just the annual day count.

    A water year with 90 no-flow days in one block is a very different river
    from one with 90 days spread across thirty short events. Refuge pool
    persistence, fish passage and stygofauna habitat all depend on the length of
    the longest continuous spell, which the annual count cannot express.
    """
    dates = pd.DatetimeIndex(dates)
    ctf = np.asarray(ctf_flag).astype(bool)
    filled = (np.zeros(len(ctf), dtype=bool) if filled_flag is None
              else np.asarray(filled_flag).astype(bool))

    rows = []
    for start, end in _runs(ctf):
        rows.append({'SpellStart': dates[start],
                     'SpellEnd': dates[end],
                     'LengthDays': int(end - start + 1),
                     'FilledDays': int(filled[start:end + 1].sum()),
                     'PercentFilled': 100.0 * float(filled[start:end + 1].mean()),
                     'TouchesRecordStart': bool(start == 0),
                     'TouchesRecordEnd': bool(end == len(ctf) - 1)})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def annual_cease_to_flow_spells(spells, water_year_of, complete_years):
    """Spell statistics per water year, keyed on the year each spell STARTS in.

    A spell that straddles a boundary is attributed to the year it began, which
    keeps each spell whole rather than splitting it at an arbitrary date.
    """
    if spells.empty:
        return pd.DataFrame()

    spells = spells.copy()
    spells['WaterYear'] = [water_year_of(d) for d in spells['SpellStart']]
    spells = spells[spells['WaterYear'].isin(complete_years)]

    if spells.empty:
        return pd.DataFrame()

    rows = []
    for year, group in spells.groupby('WaterYear'):
        longest = group.loc[group['LengthDays'].idxmax()]
        rows.append({'WaterYear': int(year),
                     'Spells': int(len(group)),
                     'TotalNoFlowDays': int(group['LengthDays'].sum()),
                     'LongestSpellDays': int(longest['LengthDays']),
                     'LongestSpellStart': longest['SpellStart'],
                     'FirstCessation': group['SpellStart'].min(),
                     'LastResumption': group['SpellEnd'].max(),
                     'PercentFilledInSpells': 100.0 * float(
                         group['FilledDays'].sum() / group['LengthDays'].sum())})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


# %% Colwell predictability
def colwell_indices(dates, q, n_classes=11, period='month'):
    """Colwell's predictability, split into constancy and contingency.

    Predictability P measures how well flow can be anticipated from the date
    alone. It decomposes into constancy C, the tendency for flow to remain in
    the same state, and contingency M, the tendency for flow to differ
    reliably between times of year, with P = C + M.

    Monsoonal regimes typically combine low constancy with high contingency:
    highly variable, but variable on a seasonal schedule. Separating the two
    components is more informative about regime character than a measure of
    variability alone.

    Flow classes are logarithmic, with zero and near-zero flows given their own
    class so that an intermittent river is not forced into a log bin.
    """
    dates = pd.DatetimeIndex(dates)
    q = np.asarray(q, dtype=float)

    usable = np.isfinite(q)
    q, dates = q[usable], dates[usable]

    if len(q) < 365:
        return {'predictability': np.nan, 'constancy': np.nan,
                'contingency': np.nan, 'n_classes': n_classes}

    if period == 'month':
        time_index = dates.month.to_numpy() - 1
        n_periods = 12
    else:
        time_index = np.minimum(((dates.dayofyear - 1) // 7), 51)
        n_periods = 52

    positive = q[q > 0]
    if positive.size == 0:
        return {'predictability': np.nan, 'constancy': np.nan,
                'contingency': np.nan, 'n_classes': n_classes}

    # Flow classes are powers of two either side of the record mean. They must
    # be anchored to an absolute scale rather than stretched across the observed
    # range: rescaling to the range would spread a perfectly constant river
    # across every class and report it as unpredictable. Class 0 is no flow.
    # The half-power offset puts the mean at the centre of a class rather than
    # on a boundary. Without it a near-constant series straddles an edge and is
    # reported as split across two classes.
    mean_flow = float(np.mean(positive))
    half = (n_classes - 2) // 2
    edges = mean_flow * 2.0 ** (np.arange(-half, half + 1, dtype=float) + 0.5)

    state = np.where(q <= 0, 0, np.digitize(q, edges) + 1)
    state = np.clip(state, 0, n_classes - 1)

    matrix = np.zeros((n_classes, n_periods))
    np.add.at(matrix, (state, time_index), 1.0)

    total = matrix.sum()
    if total == 0:
        return {'predictability': np.nan, 'constancy': np.nan,
                'contingency': np.nan, 'n_classes': n_classes}

    def entropy(counts):
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log(p)))

    h_time = entropy(matrix.sum(axis=0))       # uncertainty with respect to time
    h_state = entropy(matrix.sum(axis=1))      # uncertainty with respect to state
    h_joint = entropy(matrix.ravel())

    log_s = np.log(n_classes)

    constancy = 1.0 - h_state / log_s
    contingency = (h_time + h_state - h_joint) / log_s

    return {'predictability': float(constancy + contingency),
            'constancy': float(constancy),
            'contingency': float(contingency),
            'n_classes': int(n_classes)}


# %% timing
def half_flow_date(dates, q, water_year_labels, complete_years):
    """Day of the water year by which half the annual volume has passed.

    Timing shifts often appear long before annual totals change, so this is a
    more sensitive indicator of regime change than annual flow.
    """
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({'Date': dates, 'Q': np.asarray(q, dtype=float),
                          'WaterYear': water_year_labels})
    frame = frame[frame['WaterYear'].isin(complete_years)]

    rows = []
    for year, group in frame.groupby('WaterYear'):
        values = group['Q'].to_numpy()
        total = np.nansum(values)
        if not np.isfinite(total) or total <= 0:
            continue
        cumulative = np.nancumsum(values)
        index = int(np.searchsorted(cumulative, 0.5 * total))
        index = min(index, len(group) - 1)
        rows.append({'WaterYear': int(year),
                     'HalfFlowDate': group['Date'].iloc[index],
                     'DayOfWaterYear': int(index + 1),
                     'AnnualTotal': float(total)})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def seasonality(dates, q):
    """Flow-weighted mean day of year and the strength of the seasonal signal.

    Each day is a unit vector on a circle of one year and is weighted by flow.
    The resultant length runs from 0, meaning flow is spread evenly through the
    year, to 1, meaning all flow arrives on a single day. Monsoonal rivers sit
    high, spring-fed perennial rivers sit low.
    """
    dates = pd.DatetimeIndex(dates)
    q = np.asarray(q, dtype=float)

    usable = np.isfinite(q) & (q > 0)
    if usable.sum() == 0:
        return {'mean_day': np.nan, 'strength': np.nan}

    angle = 2.0 * np.pi * (dates[usable].dayofyear.to_numpy() - 1) / DAYS_IN_YEAR
    weight = q[usable]

    x = float(np.sum(weight * np.cos(angle)) / np.sum(weight))
    y = float(np.sum(weight * np.sin(angle)) / np.sum(weight))

    mean_angle = np.arctan2(y, x) % (2.0 * np.pi)

    return {'mean_day': float(mean_angle / (2.0 * np.pi) * DAYS_IN_YEAR) + 1.0,
            'strength': float(np.hypot(x, y))}


# %% flow duration curve shape
def fdc_indices(q):
    """Slope of the flow duration curve and the high to low flow ratio.

    The slope between the 33rd and 66th exceedance percentiles is a standard
    catchment signature in the regionalisation literature. A steep curve means a
    flashy catchment with little storage; a flat one means sustained flow.
    Undefined when the 66th percentile flow is zero, which is itself
    informative for an intermittent river.
    """
    values = _finite(q)
    if values.size < 10:
        return {'fdc_slope': np.nan, 'q5_q95_ratio': np.nan,
                'q33': np.nan, 'q66': np.nan, 'zero_flow_fraction': np.nan}

    def exceeded(percent):
        return float(np.percentile(values, 100.0 - percent))

    q33, q66 = exceeded(33.0), exceeded(66.0)
    q5, q95 = exceeded(5.0), exceeded(95.0)

    slope = ((np.log(q33) - np.log(q66)) / (0.66 - 0.33)
             if q33 > 0 and q66 > 0 else np.nan)

    return {'fdc_slope': float(slope) if np.isfinite(slope) else np.nan,
            'q5_q95_ratio': float(q5 / q95) if q95 > 0 else np.nan,
            'q33': q33, 'q66': q66,
            'zero_flow_fraction': float(np.mean(values <= 0))}


# %% flashiness and variability
def richards_baker_index(q):
    """Path length of the hydrograph divided by total flow.

    Higher means flashier. Sensitive to gap filling, because a smoothly
    interpolated gap has almost no path length.
    """
    values = _finite(q)
    if values.size < 2 or values.sum() <= 0:
        return np.nan
    return float(np.abs(np.diff(values)).sum() / values.sum())


def coefficient_of_variation(q):
    values = _finite(q)
    if values.size < 2:
        return np.nan
    mean = values.mean()
    return float(values.std(ddof=1) / mean) if mean > 0 else np.nan


# %% climate elasticity
def streamflow_elasticity(annual_precip_mm, annual_flow_mm):
    """Non-parametric elasticity of streamflow to precipitation.

    The proportional change in runoff for a proportional change in rainfall,
    after Sankarasubramanian et al. (2001). A value of 2 means a 10 per cent
    rainfall decline produces a 20 per cent runoff decline. This is the number a
    water planner actually wants, and it connects the tool directly to the
    climate projection literature.
    """
    p = np.asarray(annual_precip_mm, dtype=float)
    q = np.asarray(annual_flow_mm, dtype=float)

    usable = np.isfinite(p) & np.isfinite(q)
    p, q = p[usable], q[usable]

    if p.size < 5:
        return {'elasticity': np.nan, 'n_years': int(p.size)}

    p_bar, q_bar = p.mean(), q.mean()
    if p_bar <= 0 or q_bar <= 0:
        return {'elasticity': np.nan, 'n_years': int(p.size)}

    delta_p = p - p_bar
    valid = np.abs(delta_p) > 1e-9

    if valid.sum() < 5:
        return {'elasticity': np.nan, 'n_years': int(p.size)}

    ratios = ((q[valid] - q_bar) / delta_p[valid]) * (p_bar / q_bar)

    return {'elasticity': float(np.median(ratios)), 'n_years': int(p.size)}


# %% trend
def mann_kendall(values, alpha=0.05):
    """Mann-Kendall trend test with the tie correction, plus the Sen slope.

    Non-parametric, so it makes no distributional assumption, which suits
    annual hydrological series. The Sen slope is the median of all pairwise
    slopes and is the estimate to report alongside the p value.
    """
    x = np.asarray(values, dtype=float)
    keep = np.isfinite(x)
    x = x[keep]
    n = x.size

    if n < 8:
        return {'n': int(n), 'tau': np.nan, 'p_value': np.nan,
                'sen_slope': np.nan, 'trend': 'insufficient data'}

    signs = np.sign(x[None, :] - x[:, None])
    s = float(np.sum(np.triu(signs, k=1)))

    _, counts = np.unique(x, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if variance <= 0:
        return {'n': int(n), 'tau': np.nan, 'p_value': np.nan,
                'sen_slope': np.nan, 'trend': 'no variance'}

    if s > 0:
        z = (s - 1) / np.sqrt(variance)
    elif s < 0:
        z = (s + 1) / np.sqrt(variance)
    else:
        z = 0.0

    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    tau = float(s / (0.5 * n * (n - 1)))

    i, j = np.triu_indices(n, k=1)
    sen = float(np.median((x[j] - x[i]) / (j - i)))

    if p_value >= alpha:
        trend = 'no significant trend'
    else:
        trend = 'increasing' if s > 0 else 'decreasing'

    return {'n': int(n), 'tau': tau, 'p_value': p_value,
            'sen_slope': sen, 'trend': trend}


def trend_table(annual, year_column, value_columns, alpha=0.05):
    """Mann-Kendall and Sen slope for each annual series."""
    rows = []
    for column in value_columns:
        if column not in annual.columns:
            continue
        result = mann_kendall(annual[column].to_numpy(), alpha=alpha)
        rows.append({'Series': column, 'N': result['n'],
                     'KendallTau': result['tau'],
                     'PValue': result['p_value'],
                     'SenSlopePerYear': result['sen_slope'],
                     'Trend': result['trend']})
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# %% wet season onset
def anomalous_accumulation_onset(dates, values, water_year_labels, complete_years):
    """Wet season onset and retreat by the anomalous accumulation method.

    Accumulate the departure of each day from the annual mean daily rate. The
    curve falls through the dry season and rises through the wet, so its minimum
    is the onset and its maximum is the retreat. This is objective, needs no
    threshold, and works on rainfall or on flow, which is what makes the lag
    between the two computable.

    Reference: Liebmann and Marengo (2001).
    """
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({'Date': dates, 'Value': np.asarray(values, dtype=float),
                          'WaterYear': water_year_labels})
    frame = frame[frame['WaterYear'].isin(complete_years)]

    rows = []
    for year, group in frame.groupby('WaterYear'):
        v = group['Value'].to_numpy()
        if not np.isfinite(v).all() or np.nansum(v) <= 0:
            continue

        anomaly = np.cumsum(v - np.mean(v))
        onset_index = int(np.argmin(anomaly))
        retreat_index = int(np.argmax(anomaly))

        if retreat_index <= onset_index:
            continue

        rows.append({'WaterYear': int(year),
                     'OnsetDate': group['Date'].iloc[onset_index],
                     'OnsetDayOfWaterYear': onset_index + 1,
                     'RetreatDate': group['Date'].iloc[retreat_index],
                     'RetreatDayOfWaterYear': retreat_index + 1,
                     'SeasonLengthDays': retreat_index - onset_index})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def onset_lag(rain_onset, flow_onset):
    """Days between the rainfall wet season starting and the flow responding.

    This lag is a catchment storage signature. A karst or deeply weathered
    catchment absorbs the first rains and responds late; a shallow, sealed or
    already-wet catchment responds almost immediately.
    """
    if rain_onset.empty or flow_onset.empty:
        return pd.DataFrame()

    merged = rain_onset.merge(flow_onset, on='WaterYear', suffixes=('_Rain', '_Flow'))
    if merged.empty:
        return pd.DataFrame()

    merged['OnsetLagDays'] = (merged['OnsetDayOfWaterYear_Flow']
                              - merged['OnsetDayOfWaterYear_Rain'])
    merged['RetreatLagDays'] = (merged['RetreatDayOfWaterYear_Flow']
                                - merged['RetreatDayOfWaterYear_Rain'])

    return merged[['WaterYear', 'OnsetDate_Rain', 'OnsetDate_Flow', 'OnsetLagDays',
                   'RetreatDate_Rain', 'RetreatDate_Flow', 'RetreatLagDays']]


# %% assembly
def whole_record_indices(dates, q_mmd, rain_mmd=None):
    """Single-value signatures for the whole record, as a two column table."""
    colwell = colwell_indices(dates, q_mmd)
    season = seasonality(dates, q_mmd)
    fdc = fdc_indices(q_mmd)

    items = [
        ('Colwell predictability (P)', colwell['predictability']),
        ('Colwell constancy (C)', colwell['constancy']),
        ('Colwell contingency (M)', colwell['contingency']),
        ('Seasonality strength (0-1)', season['strength']),
        ('Flow-weighted mean day of year', season['mean_day']),
        ('Flow duration curve slope (33-66%)', fdc['fdc_slope']),
        ('Q5 to Q95 ratio', fdc['q5_q95_ratio']),
        ('Zero-flow fraction', fdc['zero_flow_fraction']),
        ('Richards-Baker flashiness index', richards_baker_index(q_mmd)),
        ('Coefficient of variation of daily flow', coefficient_of_variation(q_mmd)),
    ]

    if rain_mmd is not None:
        rain = _finite(rain_mmd)
        flow = _finite(q_mmd)
        if rain.size and flow.size and rain.mean() > 0:
            items.append(('Runoff coefficient (record mean)',
                          float(flow.mean() / rain.mean())))

    return pd.DataFrame(items, columns=['Index', 'Value'])


# %% annual regime indices
def annual_regime_indices(dates, q, water_year_labels, complete_years):
    """Flashiness and variability per water year.

    Reported annually rather than only for the whole record, because a trend in
    flashiness is a more direct signal of catchment change than a trend in
    annual volume, and because a single whole-record value hides the years in
    which the metric was carried by gap filled data.
    """
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({'Date': dates, 'Q': np.asarray(q, dtype=float),
                          'WaterYear': water_year_labels})
    frame = frame[frame['WaterYear'].isin(complete_years)]

    rows = []
    for year, group in frame.groupby('WaterYear'):
        values = group['Q'].to_numpy()
        fdc = fdc_indices(values)
        rows.append({'WaterYear': int(year),
                     'RichardsBakerIndex': richards_baker_index(values),
                     'CoefficientOfVariation': coefficient_of_variation(values),
                     'FDCSlope': fdc['fdc_slope'],
                     'ZeroFlowFraction': fdc['zero_flow_fraction'],
                     'MeanFlow_mmd': float(np.mean(values))})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def rolling_colwell(dates, q, water_year_labels, complete_years, window=15):
    """Colwell indices computed on a moving window of water years.

    A single whole-record value cannot show whether a regime is becoming more or
    less predictable. Running the decomposition over a moving window does, and
    separating constancy from contingency shows which of the two is moving: a
    river losing contingency is losing its seasonal signal, which is a different
    problem from one losing constancy.
    """
    years = sorted(int(y) for y in complete_years)
    if len(years) < window + 2:
        return pd.DataFrame()

    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({'Date': dates, 'Q': np.asarray(q, dtype=float),
                          'WaterYear': water_year_labels})

    rows = []
    for i in range(len(years) - window + 1):
        block = years[i:i + window]
        subset = frame[frame['WaterYear'].isin(block)]
        if subset.empty:
            continue

        result = colwell_indices(subset['Date'], subset['Q'].to_numpy())
        rows.append({'CentreWaterYear': int(block[len(block) // 2]),
                     'StartWaterYear': int(block[0]),
                     'EndWaterYear': int(block[-1]),
                     'Predictability': result['predictability'],
                     'Constancy': result['constancy'],
                     'Contingency': result['contingency']})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
