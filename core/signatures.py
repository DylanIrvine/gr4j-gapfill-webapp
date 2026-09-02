# core/signatures.py
# Water year, seasonal and monthly aggregation of a daily flow record, producing
# the products published by the Bureau of Meteorology Hydrologic Reference
# Stations, plus a few they do not publish.
#
# Conventions, stated explicitly because getting them wrong is silent
# ------------------------------------------------------------------
#   Water year label   the calendar year the water year STARTS in. With a
#                      September start, September 2024 to August 2025 is the
#                      2024 water year.
#   Partial periods    excluded. A water year, season or month is reported only
#                      when the record spans the whole of it. This follows the
#                      HRS convention and means the first and last periods of a
#                      record usually drop out.
#   Season label       the calendar year the season STARTS in, so Summer 2024 is
#                      December 2024 to February 2025. Seasons are the standard
#                      meteorological ones and are NOT redefined by the water
#                      year start month, since a season that straddled a water
#                      year boundary would otherwise be split.
#   Percentiles        exceedance convention. Q10 is the flow exceeded 10 per
#                      cent of the time, so it is the 90th percentile of the
#                      flow values. Q90 is the flow exceeded 90 per cent of the
#                      time. This matches the HRS definition.
#
# Provenance columns
# ------------------
# Every aggregated product carries PercentFilled, the share of days in that
# period that came from the model rather than the gauge. The HRS reports a
# single figure for the whole record, which cannot distinguish a water year that
# is 2 per cent modelled from one that is 80 per cent modelled. Cease-to-flow
# counts additionally separate observed from modelled days, because a model that
# cannot produce exactly zero flow will never register a modelled cease-to-flow.

# %%
import calendar

import numpy as np
import pandas as pd

# %% constants
MONTH_NAMES = list(calendar.month_name)[1:]

SEASON_OF_MONTH = {12: 'Summer', 1: 'Summer', 2: 'Summer',
                   3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                   6: 'Winter', 7: 'Winter', 8: 'Winter',
                   9: 'Spring', 10: 'Spring', 11: 'Spring'}

SEASON_START_MONTH = {'Summer': 12, 'Autumn': 3, 'Winter': 6, 'Spring': 9}
SEASON_ORDER = ('Summer', 'Autumn', 'Winter', 'Spring')

MOVING_AVERAGE_WINDOWS = (3, 5, 11)


# %% period labelling
def water_year(dates, start_month=1):
    """Water year label, being the calendar year the water year starts in."""
    d = pd.DatetimeIndex(dates)
    return np.where(d.month >= start_month, d.year, d.year - 1)


def season_labels(dates):
    """Season name and the calendar year the season starts in."""
    d = pd.DatetimeIndex(dates)
    season = np.array([SEASON_OF_MONTH[m] for m in d.month])
    year = np.where(d.month <= 2, d.year - 1, d.year)
    return season, year


def _water_year_span(year, start_month):
    start = pd.Timestamp(year=int(year), month=int(start_month), day=1)
    return start, start + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def _season_span(season, year):
    start = pd.Timestamp(year=int(year), month=SEASON_START_MONTH[season], day=1)
    return start, start + pd.DateOffset(months=3) - pd.Timedelta(days=1)


def _month_span(year, month):
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    return start, start + pd.DateOffset(months=1) - pd.Timedelta(days=1)


# %% frame assembly
def build_daily_frame(dates, q_mmd, filled_flag, area_km2, start_month=1,
                      baseflow_mmd=None, ctf_flag=None, model_components=None):
    """Daily frame with period labels and both unit systems.

    1 mm over 1 km2 is 1 ML, so ML/d is simply mm/d multiplied by the area.

    baseflow_mmd is the Lyne and Hollick digital-filter separation of the gap
    filled series; its columns carry an _LH suffix so the method is explicit.

    model_components, when given, is the runoff split produced internally by a
    process model (SIMHYD) from its calibrated parameter set: a dict with
    'surface', 'interflow' and 'baseflow' arrays in mm/d. These are a different
    quantity from the Lyne and Hollick baseflow (they come from the model, not
    from the gap filled hydrograph), so they get a _SIMHYD suffix and both sets
    of columns sit side by side in the one frame.
    """
    dates = pd.DatetimeIndex(dates)

    frame = pd.DataFrame({
        'Date': dates,
        'Q_mmd': np.asarray(q_mmd, dtype=float),
        'Q_MLd': np.asarray(q_mmd, dtype=float) * float(area_km2),
        'Filled': np.asarray(filled_flag).astype(int),
    })

    if baseflow_mmd is not None:
        frame['Qbase_LH_mmd'] = np.asarray(baseflow_mmd, dtype=float)
        frame['Qbase_LH_MLd'] = frame['Qbase_LH_mmd'] * float(area_km2)
        frame['Qquick_LH_MLd'] = frame['Q_MLd'] - frame['Qbase_LH_MLd']

    if model_components is not None:
        surface = np.asarray(model_components['surface'], dtype=float)
        interflow = np.asarray(model_components['interflow'], dtype=float)
        base = np.asarray(model_components['baseflow'], dtype=float)
        frame['Qsurface_SIMHYD_mmd'] = surface
        frame['Qsurface_SIMHYD_MLd'] = surface * float(area_km2)
        frame['Qinterflow_SIMHYD_mmd'] = interflow
        frame['Qinterflow_SIMHYD_MLd'] = interflow * float(area_km2)
        frame['Qbase_SIMHYD_mmd'] = base
        frame['Qbase_SIMHYD_MLd'] = base * float(area_km2)
        frame['Qtotal_SIMHYD_mmd'] = surface + interflow + base

    if ctf_flag is not None:
        frame['CeaseToFlow'] = np.asarray(ctf_flag).astype(int)

    frame['WaterYear'] = water_year(dates, start_month)
    season, season_year = season_labels(dates)
    frame['Season'] = season
    frame['SeasonYear'] = season_year
    frame['Year'] = dates.year
    frame['Month'] = dates.month

    return frame


def _complete_periods(frame, key_columns, span_function):
    """Keys whose full calendar span lies inside the record and is fully present."""
    first, last = frame['Date'].min(), frame['Date'].max()
    counts = frame.groupby(key_columns, observed=True).size()

    keep = []
    for key, n_days in counts.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        start, end = span_function(*key_tuple)
        expected = (end - start).days + 1
        if start >= first and end <= last and n_days == expected:
            keep.append(key)

    return set(keep)


# %% annual products
def annual_flow(frame, start_month=1):
    """Water year totals in GL, with the share of days that were gap filled."""
    complete = _complete_periods(frame, 'WaterYear',
                                 lambda y: _water_year_span(y, start_month))

    grouped = frame[frame['WaterYear'].isin(complete)].groupby('WaterYear')

    out = pd.DataFrame({
        'WaterYear': sorted(complete),
        'Flow_GL': grouped['Q_MLd'].sum().reindex(sorted(complete)).to_numpy() / 1000.0,
        'MeanFlow_MLd': grouped['Q_MLd'].mean().reindex(sorted(complete)).to_numpy(),
        'Days': grouped.size().reindex(sorted(complete)).to_numpy(),
        'PercentFilled': 100.0 * grouped['Filled'].mean().reindex(sorted(complete)).to_numpy(),
    })

    return out.reset_index(drop=True)


def annual_anomaly(annual, windows=MOVING_AVERAGE_WINDOWS):
    """Departure from the mean annual flow, with centred moving averages."""
    out = annual[['WaterYear', 'Flow_GL', 'PercentFilled']].copy()
    mean_flow = out['Flow_GL'].mean()
    out['MeanAnnualFlow_GL'] = mean_flow
    out['Anomaly_GL'] = out['Flow_GL'] - mean_flow

    for window in windows:
        out[f'Anomaly_MA{window}_GL'] = (out['Anomaly_GL']
                                         .rolling(window, center=True, min_periods=window)
                                         .mean())

    return out


def annual_percentiles(frame, start_month=1):
    """Q10, Q50 and Q90 per water year, on the exceedance convention."""
    complete = _complete_periods(frame, 'WaterYear',
                                 lambda y: _water_year_span(y, start_month))
    subset = frame[frame['WaterYear'].isin(complete)]

    rows = []
    for year, group in subset.groupby('WaterYear'):
        q = group['Q_MLd'].to_numpy()
        rows.append({'WaterYear': int(year),
                     'Q10_MLd': float(np.percentile(q, 90)),
                     'Q50_MLd': float(np.percentile(q, 50)),
                     'Q90_MLd': float(np.percentile(q, 10)),
                     'PercentFilled': 100.0 * float(group['Filled'].mean())})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def annual_maximum(frame, start_month=1):
    """Largest daily flow in each water year, and the day it occurred."""
    complete = _complete_periods(frame, 'WaterYear',
                                 lambda y: _water_year_span(y, start_month))
    subset = frame[frame['WaterYear'].isin(complete)]

    rows = []
    for year, group in subset.groupby('WaterYear'):
        peak = group.loc[group['Q_MLd'].idxmax()]
        rows.append({'WaterYear': int(year),
                     'MaxDailyFlow_MLd': float(peak['Q_MLd']),
                     'Date': peak['Date'],
                     'WasFilled': int(peak['Filled']),
                     'PercentFilled': 100.0 * float(group['Filled'].mean())})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def annual_baseflow(frame, start_month=1, base_col='Qbase_LH_MLd'):
    """Water year baseflow totals and the baseflow index.

    base_col selects the separation: 'Qbase_LH_MLd' for the Lyne and Hollick
    filter, 'Qbase_SIMHYD_MLd' for the SIMHYD model's own baseflow. The output
    schema is the same either way.
    """
    if base_col not in frame.columns:
        return pd.DataFrame()

    complete = _complete_periods(frame, 'WaterYear',
                                 lambda y: _water_year_span(y, start_month))
    subset = frame[frame['WaterYear'].isin(complete)]

    rows = []
    for year, group in subset.groupby('WaterYear'):
        total = float(group['Q_MLd'].sum())
        base = float(group[base_col].sum())
        rows.append({'WaterYear': int(year),
                     'Baseflow_GL': base / 1000.0,
                     'Quickflow_GL': (total - base) / 1000.0,
                     'TotalFlow_GL': total / 1000.0,
                     'BFI': base / total if total > 0 else np.nan,
                     'PercentFilled': 100.0 * float(group['Filled'].mean())})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


def annual_cease_to_flow(frame, start_month=1):
    """Cease-to-flow days per water year, split by observed and modelled.

    The split matters. GR6J's exponential store cannot reach exactly zero, so at
    a threshold of zero it will never produce a modelled cease-to-flow day, and
    any gap filled over a dry period will be counted as flowing.
    """
    if 'CeaseToFlow' not in frame.columns:
        return pd.DataFrame()

    complete = _complete_periods(frame, 'WaterYear',
                                 lambda y: _water_year_span(y, start_month))
    subset = frame[frame['WaterYear'].isin(complete)]

    rows = []
    for year, group in subset.groupby('WaterYear'):
        ctf = group['CeaseToFlow'].to_numpy().astype(bool)
        filled = group['Filled'].to_numpy().astype(bool)
        rows.append({'WaterYear': int(year),
                     'CeaseToFlowDays': int(ctf.sum()),
                     'ObservedCeaseToFlowDays': int((ctf & ~filled).sum()),
                     'ModelledCeaseToFlowDays': int((ctf & filled).sum()),
                     'PercentFilled': 100.0 * float(filled.mean())})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('WaterYear').reset_index(drop=True)


# %% seasonal and monthly products
def seasonal_flow(frame):
    """Seasonal totals in ML, with the anomaly against that season's own mean."""
    complete = _complete_periods(frame, ['SeasonYear', 'Season'],
                                 lambda y, s: _season_span(s, y))

    subset = frame[[(y, s) in complete
                    for y, s in zip(frame['SeasonYear'], frame['Season'])]]

    rows = []
    for (year, season), group in subset.groupby(['SeasonYear', 'Season']):
        rows.append({'SeasonYear': int(year), 'Season': season,
                     'Flow_ML': float(group['Q_MLd'].sum()),
                     'Days': int(len(group)),
                     'PercentFilled': 100.0 * float(group['Filled'].mean())})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out['SeasonMean_ML'] = out.groupby('Season')['Flow_ML'].transform('mean')
    out['Anomaly_ML'] = out['Flow_ML'] - out['SeasonMean_ML']

    out['Season'] = pd.Categorical(out['Season'], categories=SEASON_ORDER, ordered=True)
    return out.sort_values(['SeasonYear', 'Season']).reset_index(drop=True)


def monthly_flow(frame):
    """Monthly totals in ML, with the anomaly against that month's own mean."""
    complete = _complete_periods(frame, ['Year', 'Month'], _month_span)

    subset = frame[[(y, m) in complete
                    for y, m in zip(frame['Year'], frame['Month'])]]

    rows = []
    for (year, month), group in subset.groupby(['Year', 'Month']):
        rows.append({'Year': int(year), 'Month': int(month),
                     'MonthName': MONTH_NAMES[int(month) - 1],
                     'Flow_ML': float(group['Q_MLd'].sum()),
                     'Days': int(len(group)),
                     'PercentFilled': 100.0 * float(group['Filled'].mean())})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out['MonthMean_ML'] = out.groupby('Month')['Flow_ML'].transform('mean')
    out['Anomaly_ML'] = out['Flow_ML'] - out['MonthMean_ML']
    return out.sort_values(['Year', 'Month']).reset_index(drop=True)


def _distribution_table(values_by_group, group_name, order=None):
    rows = []
    for key, values in values_by_group:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({group_name: key, 'N': int(v.size),
                     'Minimum': float(v.min()),
                     'P25': float(np.percentile(v, 25)),
                     'Median': float(np.percentile(v, 50)),
                     'Mean': float(v.mean()),
                     'P75': float(np.percentile(v, 75)),
                     'Maximum': float(v.max())})

    out = pd.DataFrame(rows)
    if not out.empty and order is not None:
        out[group_name] = pd.Categorical(out[group_name], categories=order, ordered=True)
        out = out.sort_values(group_name)
    return out.reset_index(drop=True)


def seasonal_distribution(seasonal):
    """Five-number summary of seasonal flow, the data behind a seasonal boxplot."""
    if seasonal.empty:
        return pd.DataFrame()
    groups = [(s, g['Flow_ML'].to_numpy()) for s, g in seasonal.groupby('Season', observed=True)]
    return _distribution_table(groups, 'Season', order=SEASON_ORDER)


def monthly_distribution(monthly):
    """Five-number summary of monthly flow, the data behind a monthly boxplot."""
    if monthly.empty:
        return pd.DataFrame()
    groups = [(MONTH_NAMES[int(m) - 1], g['Flow_ML'].to_numpy())
              for m, g in monthly.groupby('Month')]
    return _distribution_table(groups, 'Month', order=MONTH_NAMES)


# %% flow duration curve
def flow_duration_curve(q_mmd, area_km2):
    """Exceedance probability against flow, both unit systems."""
    q = np.asarray(q_mmd, dtype=float)
    q = np.sort(q[np.isfinite(q)])[::-1]

    if q.size == 0:
        return pd.DataFrame()

    exceedance = np.arange(1, q.size + 1) / (q.size + 1) * 100.0

    return pd.DataFrame({'Exceedance_percent': exceedance,
                         'Flow_mmd': q,
                         'Flow_MLd': q * float(area_km2)})


# %% assembly
def build_all_products(frame, area_km2, start_month=1):
    """Every product as a dict of name to DataFrame, ready to write as CSV."""
    annual = annual_flow(frame, start_month)
    seasonal = seasonal_flow(frame)
    monthly = monthly_flow(frame)

    products = {
        'daily_flow': frame[['Date', 'Q_mmd', 'Q_MLd', 'Filled']],
        'flow_duration_curve': flow_duration_curve(frame['Q_mmd'], area_km2),
        'annual_flow': annual,
        'annual_anomaly': annual_anomaly(annual),
        'annual_percentiles': annual_percentiles(frame, start_month),
        'annual_maximum': annual_maximum(frame, start_month),
        'seasonal_flow': seasonal,
        'seasonal_distribution': seasonal_distribution(seasonal),
        'monthly_flow': monthly,
        'monthly_distribution': monthly_distribution(monthly),
    }

    if 'Qbase_LH_MLd' in frame.columns or 'Qbase_SIMHYD_MLd' in frame.columns:
        # one combined daily table, with the method flagged in each column name
        daily_cols = ['Date', 'Q_MLd']
        for col in ('Qbase_LH_MLd', 'Qquick_LH_MLd',
                    'Qsurface_SIMHYD_MLd', 'Qinterflow_SIMHYD_MLd', 'Qbase_SIMHYD_MLd'):
            if col in frame.columns:
                daily_cols.append(col)
        daily_cols.append('Filled')
        products['daily_baseflow'] = frame[daily_cols]

    if 'Qbase_LH_MLd' in frame.columns:
        products['annual_baseflow'] = annual_baseflow(frame, start_month,
                                                      base_col='Qbase_LH_MLd')
    if 'Qbase_SIMHYD_MLd' in frame.columns:
        products['annual_baseflow_simhyd'] = annual_baseflow(frame, start_month,
                                                             base_col='Qbase_SIMHYD_MLd')

    if 'CeaseToFlow' in frame.columns:
        products['annual_cease_to_flow'] = annual_cease_to_flow(frame, start_month)

    return {name: table for name, table in products.items() if not table.empty}
