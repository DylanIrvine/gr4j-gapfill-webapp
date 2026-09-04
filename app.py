# app.py
# HydroSTITCH: rainfall-runoff calibration, gap filling and analysis.
# Models: the GR family (GR4J, GR5J, GR6J) and SIMHYD.
# Dylan Irvine, Charles Darwin University
# Requires streamlit >= 1.30

import gc
import inspect
import math
import os
import zipfile
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.models import (simulate, simhyd_components, MODELS, MODEL_PARAMS,
                         MODEL_INFO, PARAM_NAMES, PARAM_BOUNDS, PARAM_LABELS,
                         PARAM_ROUNDING, MODEL_NOTES, STRICTLY_POSITIVE_PARAMS,
                         SIMHYD_OVERFLOW_CHOICES, NUMBA_AVAILABLE)
from core.metrics import (kge, nse, score, criterion_label, composite_label,
                          resolve_kge_bias, kge_bias_is_unstable,
                          METRICS, TRANSFORMS, TRANSFORM_LABELS, COMPOSITE_TRANSFORMS)
from core.units import cumecs_to_mmd, mmd_to_cumecs, mld_to_mmd, mmd_to_mld
from core.calibration import calibrate_gr
from core.gapfill import (gapfill_p50, gapfill_snapped, gapfill_gaussian_process,
                          gapfill_ar1, gapfill_enkf, identify_gaps, clip_negative)
from core.baseflow import (lyne_hollick, recession_alpha, cease_to_flow,
                           DEFAULT_ALPHA, DEFAULT_PASSES, DEFAULT_REFLECT,
                           RECESSION_MIN_LENGTH, RECESSION_SKIP_DAYS,
                           RECESSION_QUANTILE)
from core.signatures import (build_daily_frame, build_all_products, MONTH_NAMES,
                             water_year, _complete_periods, _water_year_span)
from core.indices import (cease_to_flow_spells, annual_cease_to_flow_spells,
                          colwell_indices, half_flow_date, seasonality, fdc_indices,
                          streamflow_elasticity, anomalous_accumulation_onset,
                          onset_lag, whole_record_indices, trend_table,
                          annual_regime_indices, rolling_colwell)
from core.evaluation import (efficiency_table, signature_report, worst_signatures)
from core.rainfall import (annual_rainfall, spi, cumulative_by_water_year,
                           annual_anomaly_series)
from core.plots import (cumulative_spaghetti, anomaly_bars, rainfall_runoff_cumulative)
from core.baseflow import recession_analysis
from core.usage import increment as _record_run
from core.dates import parse_dates, infer_dayfirst, DateParseError

# %% interface styling
# Streamlit's primary button is red in the default theme, and the workflow is a
# long single column in which every step looks alike. Making the one action that
# starts a multi-minute computation visually distinct is worth the two lines.
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #C0392B;
    border-color: #C0392B;
    color: white;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #A93226;
    border-color: #A93226;
    color: white;
}
div.stButton > button[kind="primary"]:disabled {
    background-color: #E6B0AA;
    border-color: #E6B0AA;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# %% plot settings
plt.style.use('default')
plt.rc('axes', linewidth=0.5)
# Arial first, with fallbacks so a machine without it degrades to the nearest
# metric-compatible face rather than to the matplotlib default. legend.frameon
# off applies to every legend, so individual calls need not repeat it.
plt.rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Arial', 'Helvetica', 'Liberation Sans',
                                 'Nimbus Sans', 'DejaVu Sans']})
plt.rcParams.update({'legend.frameon': False,
                     'legend.framealpha': 0.0,
                     'legend.borderpad': 0.3,
                     'axes.unicode_minus': False})
plt.rcParams.update({'font.size': 8, 'legend.labelspacing': 0.1, 'lines.linewidth': 1,
                     'xtick.direction': 'inout', 'ytick.direction': 'inout'})

# %% constants
EPS = 0.01
# Lower bound for any log-scaled flow (or flow-rate) axis, in mm/d. Without it a
# single near-zero modelled value drags the axis down past 1e-20 and the whole
# hydrograph collapses into the top of the panel. 1e-5 mm/d is ~3 mm/yr.
FLOW_FLOOR_MMD = 1e-5
C_OBS, C_SIM, C_CAL = 'black', 'royalblue', '#0DB14B'
C_PARAM = ['#FCB711', '#F37021', '#CC004C', '#6460AA', '#0DB14B', '#2BA9E0']
MAX_EXPORT_MODELS = 30
MAX_HISTORY = 20
C_BASE = '#2BA9E0'
FIGURE_DPI = 300
CREDIT = 'Produced with HydroSTITCH, Charles Darwin University'
LONG_FORCING_GAP = 5
CACHE_TTL = 3600

# Version stamp for the dict held in st.session_state['cal']. Streamlit reruns
# the script in place when new source is deployed, so stored results can outlive
# the code that wrote them. Increment whenever a key is added, renamed or
# removed, and stale results are discarded rather than raising a KeyError.
CAL_SCHEMA = 7

FLOW_UNITS = ['m3/s', 'ML/d', 'mm/d']
UNIT_SUFFIX = {'m3/s': 'm3s', 'ML/d': 'MLd', 'mm/d': 'mmd'}
FLOW_SERIES = ['Observed', 'Gapfilled', 'P05', 'P50', 'P95']

# Calibration / validation split. The hold-out period is withheld from the
# objective and used only to measure performance on days the model never saw
# during fitting. It is a single continuous run: the model still simulates over
# the whole record, only the objective and the reported metrics are windowed.
HOLDOUT_POSITIONS = ['Most recent years', 'First years after the warm-up']
# Days of calibration data left after warm-up and hold-out below which the
# calibration is blocked (it would mostly hit the objective's penalty floor) or
# merely warned about. These are guides, not physics; the point is that a
# hold-out on a short record weakens the delivered model, it does not only
# shrink the sample the score is measured on.
MIN_CAL_DAYS_BLOCK = 200
MIN_CAL_DAYS_WARN = 730
# The four criteria reported per period, mirroring the whole-record panel.
PERIOD_METRICS = [('KGE(Q)', 'KGE', 'none'), ('NSE(Q)', 'NSE', 'none'),
                  ('KGE(log Q)', 'KGE', 'log'), ('KGE(1/Q)', 'KGE', 'inverse')]

GAP_METHODS = ['Behavioural Median', 'Endpoint Snapped Residuals',
               'Gaussian Process Residuals', 'AR(1) Residuals',
               'Ensemble Kalman Smoother']


# %% module compatibility check
# app.py and the modules in core/ are updated together. If one is deployed
# without the other, the failure surfaces as a bare TypeError at the call site,
# and on Streamlit Cloud the message is redacted. This turns that into
# something actionable.

REQUIRED_ARGUMENTS = {
    'core/calibration.py': (calibrate_gr, ['model', 'transform_kind', 'composite_weight',
                                          'bounds', 'seed', 'refine_sample',
                                          'kge_bias', 'fit_mask', 'progress_callback',
                                          'simhyd_overflow_to_gw']),
    'core/models.py': (simulate, ['model', 'simhyd_overflow_to_gw']),
    'core/metrics.py': (score, ['metric', 'transform_kind']),
}

_stale = []
for _path, (_function, _expected) in REQUIRED_ARGUMENTS.items():
    _present = inspect.signature(_function).parameters
    _missing = [name for name in _expected if name not in _present]
    if _missing:
        _stale.append(f'{_path} is missing the argument(s) {", ".join(_missing)} '
                      f'in {_function.__name__}()')

if _stale:
    st.error('The files in core/ are out of step with app.py. '
             + ' '.join(_stale)
             + '. Update the core modules to the versions that go with this app.py '
               'and redeploy.')
    st.stop()


# %% unit conversion
# GR models work in mm/d throughout, so flow is converted on the way in and
# converted back on the way out. Both conversions use the same catchment area,
# so the round trip is exact to floating point.

def to_mmd(q, units, area_km2):
    if units == 'm3/s':
        return cumecs_to_mmd(q, area_km2)
    if units == 'ML/d':
        return mld_to_mmd(q, area_km2)
    return np.asarray(q, dtype=float)


def from_mmd(q_mmd, units, area_km2):
    if units == 'm3/s':
        return mmd_to_cumecs(q_mmd, area_km2)
    if units == 'ML/d':
        return mmd_to_mld(q_mmd, area_km2)
    return np.asarray(q_mmd, dtype=float)


def unit_factor(units, area_km2):
    """How many mm/d one unit of the input series represents on this catchment."""
    return float(to_mmd(np.array([1.0]), units, area_km2)[0])


# %% figure capture
# Rebuilt on every script run. Populated by show().
FIGURES = {}


# %% plotting helpers
def section_break():
    st.markdown('---')


def new_fig(w_cm, h_cm, rect):
    fig = plt.figure(figsize=(w_cm / 2.54, h_cm / 2.54))
    return fig, fig.add_axes(rect)


def show(fig, name=None):
    """Render a figure, keep a PNG copy for the download package, then close it.

    Streamlit gives dataframes a download button but not figures, so the PNG
    bytes are captured here as each figure is drawn and bundled into the results
    zip in section 6. FIGURES is a plain module-level dict, which Streamlit
    resets on every script run, so it always holds the figures currently on
    screen rather than accumulating stale ones.
    """
    if name is not None:
        png = BytesIO()
        fig.savefig(png, format='png', dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        FIGURES[name] = png.getvalue()

    st.pyplot(fig)
    plt.close(fig)


def fdc(q):
    """Flow duration curve. Returns exceedance (%) and flows sorted high to low."""
    q = np.sort(q[np.isfinite(q)])[::-1]
    return np.arange(1, len(q) + 1) / (len(q) + 1) * 100, q


def plot_hydrograph(dates, q_obs, series, log_y=False):
    fig, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
    ax.plot(dates, q_obs, color=C_OBS, alpha=0.6, linewidth=1, label='Observed')
    for values, colour, label, lw in series:
        ax.plot(dates, values, color=colour, alpha=0.8, linewidth=lw, label=label)

    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (mm/d)')

    # span exactly the record, and anchor a linear axis at zero so the
    # hydrograph is not floated above the baseline by a stray small value
    ax.set_xlim(pd.Timestamp(min(dates)), pd.Timestamp(max(dates)))
    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(bottom=FLOW_FLOOR_MMD)
    else:
        ax.set_ylim(bottom=0)

    ax.legend()
    return fig, ax


def plot_log_residuals(dates, q_obs, q_mod, colour, holdout_mask=None):
    fig, ax = new_fig(17, 6, [0.10, 0.18, 0.85, 0.72])
    ax.plot(dates, np.log(q_obs + EPS) - np.log(q_mod + EPS), color=colour, linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    shade_periods(ax, dates, holdout_mask)
    if holdout_mask is not None and np.any(holdout_mask):
        ax.legend(loc='upper right', fontsize=7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Residual')
    return fig


def plot_scatter(x, y, colour, xlabel, ylabel):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fig, ax = new_fig(8, 8, [0.20, 0.18, 0.72, 0.72])
    ax.scatter(x[m], y[m], marker='o', s=5, lw=0, alpha=0.4, color=colour)
    lo = min(np.nanmin(x[m]), np.nanmin(y[m]))
    hi = max(np.nanmax(x[m]), np.nanmax(y[m]))
    ax.plot([lo, hi], [lo, hi], 'k--')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_parameter_pairs(behavioural_df, names, best_params):
    """Lower-triangle scatter matrix of the behavioural set, coloured by score.

    A histogram is a marginal projection. If the behavioural members lie along a
    curved ridge in parameter space, projecting that ridge onto one axis piles
    up density where the ridge runs parallel to the axis and thins it elsewhere,
    producing apparent modes that correspond to no distinct solution. This plot
    shows the joint structure, so two genuinely separate optima can be told
    apart from one bent ridge.
    """
    n = len(names)
    if n < 2 or len(behavioural_df) < 2:
        return None

    size = min(3.8 * (n - 1), 18.0)
    fig = plt.figure(figsize=(size / 2.54, size / 2.54))
    scores = behavioural_df['Score'].to_numpy()
    points = None

    for i in range(1, n):
        for j in range(i):
            ax = fig.add_subplot(n - 1, n - 1, (i - 1) * (n - 1) + j + 1)
            points = ax.scatter(behavioural_df[names[j]], behavioural_df[names[i]],
                                c=scores, cmap='viridis', s=12, lw=0, alpha=0.85)
            ax.plot(best_params[names[j]], best_params[names[i]], 'x',
                    color='black', markersize=8, markeredgewidth=1.5)
            ax.tick_params(labelsize=6)

            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=8)
                ax.tick_params(axis='x', labelrotation=45)
            else:
                ax.set_xticklabels([])

    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.97, top=0.97,
                        hspace=0.12, wspace=0.12)

    if n >= 3 and points is not None:
        cax = fig.add_axes([0.75, 0.55, 0.02, 0.40])
        bar = fig.colorbar(points, cax=cax)
        bar.set_label('Score', fontsize=8)
        bar.ax.tick_params(labelsize=6)

    return fig


def longest_gap(series):
    return max([g['length_days'] for g in identify_gaps(series)], default=0)


# %% calibration / validation split helpers
def holdout_mask_from_dates(dates, warmup_days, holdout_years, position):
    """Boolean array, True where a day falls in the held-out validation period.

    The hold-out is defined in calendar time from the dates, not by index, so it
    means the same thing regardless of gaps in the record. 'Most recent years'
    takes the last holdout_years of the record; 'First years after the warm-up'
    takes that span starting from the first day past the warm-up. Fractional
    years are allowed and interpreted as 365.25-day units.
    """
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    n = len(di)
    mask = np.zeros(n, dtype=bool)

    if not holdout_years or holdout_years <= 0 or n == 0:
        return mask

    span = pd.Timedelta(days=float(holdout_years) * 365.25)

    if position == HOLDOUT_POSITIONS[0]:            # most recent years
        cutoff = di.max() - span
        mask = np.asarray(di > cutoff, dtype=bool)
    else:                                            # first years after warm-up
        if warmup_days >= n:
            return mask
        start = di[int(warmup_days)]
        mask = np.asarray((di >= start) & (di < start + span), dtype=bool)

    return mask


def period_labels(n, warmup_days, holdout_mask):
    """Per-day label: Warm-up, Validation, or Calibration.

    Warm-up wins where it overlaps the hold-out, since a warm-up day is never
    scored under either heading. Only the recent-years hold-out on a very short
    record can produce such an overlap.
    """
    labels = np.full(n, 'Calibration', dtype=object)
    if holdout_mask is not None:
        labels[np.asarray(holdout_mask, dtype=bool)] = 'Validation'
    if warmup_days > 0:
        labels[:int(warmup_days)] = 'Warm-up'
    return labels


def scoring_masks(q_obs, warmup_days, holdout_mask):
    """Return (calibration, validation) boolean masks for reporting metrics.

    Both exclude the warm-up and any non-finite observation, so a period score
    is computed only on days that were genuinely available. The validation mask
    is the held-out days; the calibration mask is everything else that was
    scored during fitting.
    """
    n = len(q_obs)
    warm = np.arange(n) < int(warmup_days)
    finite = np.isfinite(q_obs)
    if holdout_mask is None:
        hold = np.zeros(n, dtype=bool)
    else:
        hold = np.asarray(holdout_mask, dtype=bool)
    val = hold & ~warm & finite
    cal = ~hold & ~warm & finite
    return cal, val


def period_efficiency(q_obs, q_sim, mask, epsilon):
    """The four reported criteria over the days in mask, at a fixed offset.

    epsilon is passed through so the log and inverse criteria use the same
    transform offset for every period, which is what makes a calibration value
    and a validation value directly comparable.
    """
    if mask is None or not np.any(mask):
        return {label: np.nan for label, _, _ in PERIOD_METRICS}
    o, s = q_obs[mask], q_sim[mask]
    return {label: score(o, s, metric, transform, epsilon=epsilon)
            for label, metric, transform in PERIOD_METRICS}


def holdout_representativeness(dates, rain, q_obs, holdout_mask, warmup_days,
                              min_year_days=360, min_years=3):
    """Place the held-out window against the record, to guard against reading a
    validation number from a period that was unusually wet or dry.

    Rainfall uses the full forcing, which is complete by this point; observed
    flow uses observed days only. The headline is a percentile of the hold-out's
    annual-equivalent rainfall (its mean daily rainfall scaled to a year) among
    the record's complete calendar years, which answers 'how wet a year was this
    like' without depending on the water-year configuration set later. When the
    record spans too few complete years for that to mean anything, it falls back
    to the ratio of the hold-out means to the calibration-period means, which
    needs no distribution. A value far from the middle (or from 100 per cent)
    means the split is testing transfer to conditions unlike the calibration
    data rather than a like-for-like hold-out.
    """
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    hold = np.asarray(holdout_mask, dtype=bool)
    n = len(di)

    out = {'ok': bool(hold.any())}
    if not out['ok']:
        return out

    warm = np.arange(n) < int(warmup_days)
    cal_days = ~hold & ~warm
    rain = np.asarray(rain, dtype=float)
    q = np.asarray(q_obs, dtype=float)

    ho_rain = float(np.nanmean(rain[hold]))
    cal_rain = float(np.nanmean(rain[cal_days])) if cal_days.any() else np.nan
    out['rain_ratio'] = (ho_rain / cal_rain
                         if np.isfinite(cal_rain) and cal_rain > 0 else np.nan)

    ho_qmask, cal_qmask = hold & np.isfinite(q), cal_days & np.isfinite(q)
    ho_flow = float(np.nanmean(q[ho_qmask])) if ho_qmask.any() else np.nan
    cal_flow = float(np.nanmean(q[cal_qmask])) if cal_qmask.any() else np.nan
    out['flow_ratio'] = (ho_flow / cal_flow
                         if np.isfinite(cal_flow) and cal_flow > 0 and np.isfinite(ho_flow)
                         else np.nan)

    # percentile of annual-equivalent rainfall among complete calendar years
    years = di.year.to_numpy()
    unique_years = np.unique(years)
    rain_totals = [float(np.nansum(rain[years == y])) for y in unique_years
                   if (years == y).sum() >= min_year_days]
    out['n_complete_years'] = len(rain_totals)

    if len(rain_totals) >= min_years:
        totals = np.asarray(rain_totals)
        out['rain_percentile'] = 100.0 * float(np.mean(totals < ho_rain * 365.25))

        flow_totals = [float(np.nansum(q[(years == y) & np.isfinite(q)])) for y in unique_years
                       if ((years == y) & np.isfinite(q)).sum() >= min_year_days]
        if len(flow_totals) >= min_years and np.isfinite(ho_flow):
            ftot = np.asarray(flow_totals)
            out['flow_percentile'] = 100.0 * float(np.mean(ftot < ho_flow * 365.25))

    return out


def shade_periods(ax, dates, mask, colour='0.85', label='Validation (unseen)'):
    """Shade the contiguous runs where mask is True, behind everything else.

    A single legend entry is emitted for the first run only, so the legend is
    not cluttered by one entry per block. Does nothing when the mask is empty,
    so plots with no hold-out are unchanged.
    """
    if mask is None:
        return
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return

    di = pd.DatetimeIndex(pd.to_datetime(dates))
    edges = np.diff(m.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m)]

    first = True
    for s, e in zip(starts, ends):
        ax.axvspan(di[s], di[e - 1], color=colour, alpha=0.6, linewidth=0,
                   zorder=0, label=label if first else None)
        first = False


# %% cached computation
@st.cache_data(show_spinner=False, max_entries=8, ttl=CACHE_TTL)
def run_model(rain, pet, model, param_values, simhyd_overflow_to_gw=False):
    params = dict(zip(PARAM_NAMES[model], param_values))
    return simulate(rain, pet, params, model=model,
                    simhyd_overflow_to_gw=simhyd_overflow_to_gw)


@st.cache_data(show_spinner='Gap filling...', max_entries=3, ttl=CACHE_TTL)
def run_gapfill(q_obs, q50, method, rain=None, pet=None, best_params=None,
                model=None, simhyd_overflow_to_gw=False):
    """Fill the gaps by the chosen method.

    The four residual methods need only q_obs and the behavioural median. The
    Ensemble Kalman Smoother re-runs the calibrated model, so it also needs the
    forcing, the best parameter set and the model name.
    """
    if method == 'Behavioural Median':
        return gapfill_p50(q_obs, q50)
    if method == 'Endpoint Snapped Residuals':
        return gapfill_snapped(q_obs, q50)
    if method == 'AR(1) Residuals':
        return gapfill_ar1(q_obs, q50)
    if method == 'Ensemble Kalman Smoother':
        if (rain is None or pet is None or best_params is None
                or len(rain) != len(q_obs) or len(pet) != len(q_obs)):
            # forcing not aligned with the calibration record; use the GP instead
            return gapfill_gaussian_process(q_obs, q50)
        return gapfill_enkf(q_obs, rain, pet, best_params, model=model,
                            simhyd_overflow_to_gw=simhyd_overflow_to_gw)
    return gapfill_gaussian_process(q_obs, q50)


@st.cache_data(show_spinner='Separating baseflow...', max_entries=3, ttl=CACHE_TTL)
def run_baseflow(q, alpha, passes, n_reflect):
    return lyne_hollick(q, alpha=alpha, passes=passes, n_reflect=n_reflect)


@st.cache_data(show_spinner=False, max_entries=4, ttl=CACHE_TTL)
def run_simhyd_components(rain, pet, param_values, overflow_to_gw=False):
    """SIMHYD's internal runoff split for the calibrated parameter set.

    This is the model's own baseflow, computed by re-running SIMHYD over the
    whole record; it is not a separation of the gap filled series. overflow_to_gw
    must match whatever the calibration used.
    """
    params = dict(zip(PARAM_NAMES['SIMHYD'], param_values))
    return simhyd_components(rain, pet, params, overflow_to_gw=overflow_to_gw)


@st.cache_data(show_spinner='Evaluating against signatures...', max_entries=3, ttl=CACHE_TTL)
def run_signature_report(dates, q_obs, q_median, q_best, alpha, passes, n_reflect,
                         ctf_threshold, warmup_days):
    simulations = {'Behavioural median': q_median, 'Best model': q_best}
    return (signature_report(dates, q_obs, simulations, alpha=alpha, passes=passes,
                             n_reflect=n_reflect, ctf_threshold=ctf_threshold,
                             warmup_days=warmup_days),
            efficiency_table(q_obs, simulations, warmup_days=warmup_days))


@st.cache_data(show_spinner=False, max_entries=3, ttl=CACHE_TTL)
def run_recession_analysis(q, min_length, skip_days):
    return recession_analysis(q, min_length=min_length, skip_days=skip_days)


@st.cache_data(show_spinner=False, max_entries=3, ttl=CACHE_TTL)
def run_recession_alpha(q, min_length, skip_days, quantile):
    return recession_alpha(q, min_length=min_length, skip_days=skip_days,
                           quantile=quantile)


def _write_workbook_rowwise(sheets):
    """Write the sheets one row at a time using xlsxwriter constant memory mode.

    Row order is not optional here. In constant_memory mode xlsxwriter holds
    exactly one row and flushes it the moment a different row is written.
    pandas.to_excel writes a DataFrame COLUMN BY COLUMN, so under that mode
    every row is flushed while the first column is being written and all later
    columns are silently discarded. The result opens cleanly in Excel with only
    column A populated, which is how the bug went unnoticed. Writing row by row
    is what makes the mode safe, and it also keeps peak memory about ten times
    lower than the pandas path and runs roughly twice as fast.

    NaN becomes None so the cell is left blank rather than raising. Dates use
    the workbook default format.
    """
    import xlsxwriter

    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {'constant_memory': True,
                                            'default_date_format': 'yyyy-mm-dd'})

    for name, df in sheets:
        worksheet = workbook.add_worksheet(name)
        worksheet.write_row(0, 0, [str(column) for column in df.columns])

        for r, row in enumerate(df.itertuples(index=False, name=None), start=1):
            worksheet.write_row(r, 0, [None if isinstance(v, float) and v != v else v
                                       for v in row])

    workbook.close()
    return buffer.getvalue()


@st.cache_data(show_spinner='Building workbook...', max_entries=2, ttl=CACHE_TTL)
def build_workbook(output_df, behavioural_df, ensemble_df, metadata_df):
    """Return workbook bytes. Bytes rather than a BytesIO, because a buffer held
    across reruns can be left at EOF and download as an empty file."""
    sheets = [('GapFilled', output_df),
              ('BehaviouralModels', behavioural_df),
              ('EnsembleHydrographs', ensemble_df),
              ('Metadata', metadata_df)]

    try:
        return _write_workbook_rowwise(sheets)
    except ImportError:
        # openpyxl is correct but instantiates a Python object per cell, which
        # on a multi-decade record runs to hundreds of thousands of objects
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for name, df in sheets:
                df.to_excel(writer, sheet_name=name, index=False)
        return buffer.getvalue()


def build_results_zip(workbook_bytes, workbook_name, figures, readme_text, products=None):
    """Bundle the workbook, every figure on screen, the CSV products and a note."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(workbook_name, workbook_bytes)
        archive.writestr('README.txt', readme_text)

        for name, png in figures.items():
            archive.writestr(f'figures/{name}.png', png)

        for name, table in (products or {}).items():
            archive.writestr(f'csv/{name}.csv', table.to_csv(index=False))

    return buffer.getvalue()


# %% export table builders
def build_output_df(dates, series_mmd, units, area_km2, include_mmd, include_native,
                    warmup_days=0, holdout_mask=None):
    data = {'Date': dates}

    if include_mmd:
        for name in FLOW_SERIES:
            data[f'{name}_mmd'] = series_mmd[name]

    if include_native:
        suffix = UNIT_SUFFIX[units]
        for name in FLOW_SERIES:
            data[f'{name}_{suffix}'] = from_mmd(series_mmd[name], units, area_km2)

    data['FilledFlag'] = np.isnan(series_mmd['Observed']).astype(int)
    data['Period'] = period_labels(len(np.asarray(dates)), warmup_days, holdout_mask)
    return pd.DataFrame(data)


def build_ensemble_df(dates, ensemble_mmd, units, area_km2, native):
    values = from_mmd(ensemble_mmd, units, area_km2) if native else ensemble_mmd
    suffix = UNIT_SUFFIX[units] if native else 'mmd'

    ensemble_df = pd.DataFrame(values.T,
                               columns=[f'Model_{i + 1:03d}_{suffix}'
                                        for i in range(len(values))])
    ensemble_df.insert(0, 'Date', np.asarray(dates))
    return ensemble_df


def build_metadata_df(cal, gap_method, n_missing, n_clipped, ensemble_units, sheet_units):
    units = cal['flow_units']
    area = cal['area_km2']
    best = cal['best_params']
    model = cal['model']

    items = [
        ('Hydrological model', model),
        ('Calibration criterion', cal['criterion']),
        ('Random seed', str(cal['seed'])),
        ('Behavioural set source', cal['behavioural_source']),
        ('Model implementation', 'numba compiled' if NUMBA_AVAILABLE else 'pure Python'),
        ('Local sample size', str(cal['n_sampled'])),
        ('Best criterion value', f"{cal['best_score']:.4f}"),
        ('Transform offset epsilon (mm/d)', f"{cal['epsilon']:.6g}"),
        ('Warm-up days excluded', str(cal['warmup_days'])),
        ('Behavioural models retained', str(cal['n_behavioural'])),
        ('Input flow units', units),
        ('Catchment area (km2)', f'{area:g}'),
        ('1 input unit expressed in mm/d', f'{unit_factor(units, area):.6g}'),
        ('GapFilled sheet units', sheet_units),
        ('EnsembleHydrographs sheet units', ensemble_units),
        ('Rainfall and PET units', 'mm/d'),
        ('Gap filling method', gap_method),
        ('Days gap filled', str(n_missing)),
        ('Filled values clipped at zero', str(n_clipped)),
        ('Forcing interpolated', str(cal['forcing_interpolated'])),
    ]

    # Calibration / validation split. Reported here so the workbook records
    # whether the metrics are in-sample or a genuine hold-out test, and which
    # days were withheld.
    if cal.get('holdout_years', 0):
        items.append(('Validation hold-out (years)', f"{cal['holdout_years']:g}"))
        items.append(('Validation hold-out position', str(cal.get('holdout_position', ''))))
        val_range = cal.get('val_date_range')
        if val_range:
            items.append(('Validation period', f'{val_range[0]} to {val_range[1]}'))
        items.append(('Validation days scored', str(cal.get('n_val_scored', 0))))
        items.append(('Calibration days scored', str(cal.get('n_cal_scored', 0))))
        items.append(('Delivered model refit on all data', 'no (single run, hold-out excluded)'))

        rep = cal.get('representativeness') or {}
        if rep.get('ok'):
            if 'rain_percentile' in rep:
                value = (f"rainfall {rep['rain_percentile']:.0f}th percentile of "
                         f"{rep['n_complete_years']} complete years")
                if 'flow_percentile' in rep:
                    value += f", flow {rep['flow_percentile']:.0f}th percentile"
                items.append(('Hold-out representativeness', value))
            elif np.isfinite(rep.get('rain_ratio', np.nan)):
                value = f"rainfall {100 * rep['rain_ratio']:.0f}% of calibration-period mean"
                if np.isfinite(rep.get('flow_ratio', np.nan)):
                    value += f", flow {100 * rep['flow_ratio']:.0f}%"
                items.append(('Hold-out representativeness', value + ' (record too short for a '
                                                                    'percentile)'))

        scores = cal.get('period_scores')
        if scores:
            for label, _, _ in PERIOD_METRICS:
                items.append((f'{label} calibration',
                              f"{scores['calibration'][label]:.4f}"))
                items.append((f'{label} validation (unseen)',
                              f"{scores['validation'][label]:.4f}"))
    else:
        items.append(('Validation hold-out', 'none (calibrated on all observed data)'))

    bounds = cal['bounds']
    items.append(('Parameter bounds', 'manually set' if cal['bounds_customised'] else 'defaults'))

    for name in PARAM_NAMES[model]:
        lo, hi = bounds[name]
        items.append((f'{name} {PARAM_LABELS[name]}',
                      f'{best[name]:.4f} (bounds {lo:g} to {hi:g})'))

    items += [
        ('P05, P50, P95', 'Percentiles across the behavioural ensemble, per day'),
        ('FilledFlag', '1 where the observed record was missing and has been filled'),
        ('Period', 'Warm-up, Calibration or Validation membership of each day'),
        ('Model implementation', 'Transcribed from airGR Fortran and verified against it'),
    ]

    return pd.DataFrame(items, columns=['Item', 'Value'])


# %% header
head_text, head_logo = st.columns([3, 1], vertical_alignment='center')

with head_text:
    st.title('HYDROSTITCH')
    st.subheader("HYDROlogical Signatures, Time-Series Infilling and Tools for Catchment Hydrology")
    st.write('Dylan Irvine, Charles Darwin University.\n')

with head_logo:
    st.image('docs/HS_Logo.png', width='stretch')


# %% run counter
# Counts each completed calibration (see the Calibrate block below), not page
# loads. Preferred backend is Upstash Redis (an atomic INCR that survives
# redeploys); it needs, in st.secrets under [redis] or as environment variables:
#     rest_url   / UPSTASH_REDIS_REST_URL
#     rest_token / UPSTASH_REDIS_REST_TOKEN
#     key        / HYDROSTITCH_RUN_COUNT_KEY   (optional, default "hydrostitch:runs")
# One Upstash database can hold many counters - just give each app its own key.
# Without Redis it falls back to a JSON file, then to an in-memory tally per
# running container. See core/usage.py.
def _run_count_secret(name, env):
    try:
        section = st.secrets['redis'] if 'redis' in st.secrets else {}
        if name in section:
            return str(section[name])
    except Exception:
        pass
    return os.environ.get(env)


def _run_count_setting(key, env, default=0):
    try:
        if key in st.secrets:
            return int(st.secrets[key])
    except Exception:
        pass
    try:
        return int(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default


@st.cache_resource
def _in_memory_run_count():
    return {'n': 0}


def count_completed_run():
    """Record one completed run and return the new total (never raises)."""
    try:
        rc_path = _run_count_setting('run_count_path', 'HYDROSTITCH_RUN_COUNT_PATH', None) or None
        rc_start = _run_count_setting('run_count_start', 'HYDROSTITCH_RUN_COUNT_START', 0)
        total = _record_run(
            path=rc_path, start=rc_start,
            redis_url=_run_count_secret('rest_url', 'UPSTASH_REDIS_REST_URL'),
            redis_token=_run_count_secret('rest_token', 'UPSTASH_REDIS_REST_TOKEN'),
            redis_key=_run_count_secret('key', 'HYDROSTITCH_RUN_COUNT_KEY') or 'hydrostitch:runs',
        )
    except Exception:
        total = None
    if total is None:                    # no persistent backend: this container only
        mem = _in_memory_run_count()
        mem['n'] += 1
        total = mem['n']
    st.session_state['run_count'] = total
    return total


if st.session_state.get('run_count'):
    st.caption(f"HydroSTITCH runs completed: {st.session_state['run_count']:,}")

st.write(
    'HydroSTITCH runs several lumped parameter conceptual rainfall-runoff models, calibrating the '
    'unknown parameters, to best reproduce observed flow data. The resulting models can then be used '
    'to gap fill hydrographs. Optionally, the continuous data can then be used to perform '
    'a range of hydrological analyses. The models are set up to simulate daily streamflow using only '
    'catchment-averaged daily precipitation and potential evapotranspiration data. This tool '
    'applies the GR4J, GR5J and GR6J, and SIMHYD models with no coding required. Upload your file, '
    'follow the workflow, and you will have calibrated models and gap-filled hydrographs.\n\n'
    'Notably, numerous metrics are provided to ensure that you do not obtain a model with a '
    'good fit, but with highly inappropriate model parameters.\n\n'
)
with st.expander('**Selected References**'):
  st.write(
    'GR models:\n\n'
    'Perrin, C., Michel, C., and Andréassian, V. (2003). Improvement of a parsimonious model for '
    'streamflow simulation. Journal of Hydrology 279(1), 275-289.\n\n'
    'Le Moine, N. (2008). Le bassin versant de surface vu par le souterrain: une voie d\'amélioration'
    'des performances et du réalisme des modèles pluie-débit? PhD thesis (in French), UPMC, Cemagref '
    'Antony, Paris, France.\n\n'
    'Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T., and Andréassian, V. (2011). A '
    'downward structural sensitivity analysis of hydrological models to improve low-flow '
    'simulation. Journal of Hydrology 411(1-2), 66-76.\n\n'

    'SIMHYD model:\n\n'
    'Chiew, F.H.S., Peel, M.C. & Western, A.W. (2002). Application and testing of the simple rainfall-runoff '
    ' model SIMHYD. Cooperative Research Centre for Catchment Hydrology, University of Melbourne.\n\n'
    'Chiew, F.H.S., Teng, J., Vaze, J., Post, D.A., Perraud, J.M., Kirono, D.G.C., and Viney, N.R. '
    '(2009). Estimating climate change impact on runoff across southeast Australia: Method, results '
    'and implications of the modeling method. Water Resources Research 45, W10414.\n\n'
    'Other software packages:\n\n'
    'Andrews, F.T., Croke, B.W., Jakeman, A.J. (2011) An open software environment for hydrological '
    'model assessment and development. Environmental Modelling & Software 26(10) 1171-1185.\n\n'
    'Coron, L., Thirel, G., Delaigue, O., Perrin, C., and Andréassian, V. (2017). The suite of lumped '
    'GR hydrological models in an R package. Environmental Modelling and Software 94, 166-171.\n\n'
    )

# %% 1. upload
st.subheader('1. Upload Data')
st.write('Upload a csv containing date, rainfall, PET, and streamflow. Select the date format '
         'below after uploading. Rain and PET must be in mm/d, but flow can be m3/s, ML/d, or mm/d.')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is None:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f'Could not read the csv: {exc}')
    st.stop()

section_break()
st.subheader('2. Data Preview and catchment information')
st.dataframe(df.head())

columns = df.columns.tolist()
date_col = st.selectbox('Date Column', columns)
rain_col = st.selectbox('Rain Column', columns)
pet_col = st.selectbox('PET Column', columns)
flow_col = st.selectbox('Flow Column', columns)

DATE_FORMATS = {
    'Auto (detect day / month)': None,
    'Day first  d/m/yyyy or dd/mm/yyyy': '%d/%m/%Y',
    'Month first  m/d/yyyy or mm/dd/yyyy': '%m/%d/%Y',
    'ISO  yyyy-mm-dd': '%Y-%m-%d',
    'Day first, 2-digit year  dd/mm/yy': '%d/%m/%y',
}
date_format_label = st.selectbox('Date Format', list(DATE_FORMATS), index=0)
st.caption('Auto reads each row on its own and works out day-first from the data. '
           'One- and two-digit days and months, ISO rows, month names and trailing '
           'times are all handled; a column that parses out of chronological order '
           'is reported rather than silently scrambled.')

section_break()
st.subheader('Catchment Information')
area_km2 = st.number_input('Catchment Area (km²)', min_value=0.001, value=1000.0, step=1.0)
flow_units = st.selectbox('Flow Units', FLOW_UNITS)

_date_format = DATE_FORMATS[date_format_label]
try:
    if _date_format is None:
        _hint = infer_dayfirst(df[date_col])
        _dates_idx = parse_dates(df[date_col],
                                 dayfirst=(True if _hint is None else _hint),
                                 coerce=True)
        if _hint is None:
            st.info('Could not tell day-first from month-first from the data; assumed '
                    'day-first (Australian). If the parsed dates below look wrong, pick '
                    'an explicit format.')
    else:
        _dates_idx = parse_dates(df[date_col], date_format=_date_format, coerce=True)
except DateParseError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f'Could not parse the date column: {exc}')
    st.stop()

_bad_dates = np.asarray(_dates_idx.isna())
if _bad_dates.any():
    st.warning(f'{int(_bad_dates.sum())} row(s) had an unreadable date and were excluded.')
    df = df[~_bad_dates].reset_index(drop=True)
    _dates_idx = _dates_idx[~_bad_dates]

# a plain datetime64 Series (not a DatetimeIndex): that is the type the rest of
# the app and the @st.cache_data functions were written against - Streamlit can
# hash a Series but not a DatetimeIndex (it iterates it into Timestamps).
dates = pd.Series(pd.DatetimeIndex(_dates_idx)).reset_index(drop=True)

try:
    rain = np.asarray(df[rain_col], dtype=float)
    pet = np.asarray(df[pet_col], dtype=float)
    flow = np.asarray(df[flow_col], dtype=float)
except Exception as exc:
    st.error(f'Could not read the rain, PET or flow column: {exc}')
    st.stop()

q_obs_mmd = to_mmd(flow, flow_units, area_km2)

section_break()
st.subheader('Data Summary')
st.write(f'Record Length: {len(df)} days')
st.write(f'Flow Missing Values: {int(np.isnan(q_obs_mmd).sum())}')
st.write(f'Rain Missing Values: {int(np.isnan(rain).sum())}')
st.write(f'PET Missing Values: {int(np.isnan(pet).sum())}')
st.write(f'Record starts: {dates.min()}')
st.write(f'Record ends: {dates.max()}')

n_observed = int(np.isfinite(q_obs_mmd).sum())
n_zero_flow = int((np.isfinite(q_obs_mmd) & (q_obs_mmd <= 0)).sum())
zero_fraction = n_zero_flow / n_observed if n_observed > 0 else 0.0
st.write(f'Zero-flow days: {n_zero_flow} of {n_observed} observed '
         f'({100 * zero_fraction:.1f} per cent)')

# %% units and scaling check
# The catchment area is the only thing linking discharge to depth. Get it wrong
# and every metric, parameter and output is rescaled without any error being
# raised, so the conversion is stated explicitly and sense checked here.
section_break()
st.subheader('Flow Units and Scaling')

factor = unit_factor(flow_units, area_km2)

col1, col2 = st.columns(2)
col1.metric('Input flow units', flow_units)
col2.metric('Model units', 'mm/d')

if flow_units == 'mm/d':
    st.write('Flow is already a depth, so no conversion is applied. The catchment area is still '
             'used to report volumes.')
else:
    st.write(f'Over a {area_km2:g} km² catchment, 1 {flow_units} is equivalent to '
             f'{factor:.6g} mm/d. Flow is converted to mm/d for modelling and converted back for '
             'export.')

sense_mask = np.isfinite(q_obs_mmd) & np.isfinite(rain)

if sense_mask.sum() > 0:
    mean_q_mmd = float(np.mean(q_obs_mmd[sense_mask]))
    mean_p_mmd = float(np.mean(rain[sense_mask]))
    mean_q_native = float(from_mmd(np.array([mean_q_mmd]), flow_units, area_km2)[0])

    st.write(f'Mean observed flow: {mean_q_native:.4g} {flow_units}, equal to {mean_q_mmd:.4g} mm/d '
             f'or {mean_q_mmd * area_km2:.4g} ML/d.')
    st.write(f'Mean rainfall over the same days: {mean_p_mmd:.4g} mm/d.')

    if mean_p_mmd > 0:
        runoff_coefficient = mean_q_mmd / mean_p_mmd
        st.write(f'Implied runoff coefficient: {runoff_coefficient:.3f}')

        if runoff_coefficient > 1.0:
            st.error('The runoff coefficient exceeds 1, meaning more water is leaving the '
                     'catchment than falls on it as rain. Check the catchment area and the flow '
                     'units, since one of them is almost certainly wrong. Regulated or '
                     'groundwater-fed systems are the only legitimate exception.')
        elif runoff_coefficient < 0.001:
            st.warning('The runoff coefficient is below 0.001. This can be genuine for an '
                       'ephemeral dryland catchment, but it is also what a wrong catchment area or '
                       'a units mismatch looks like. Worth confirming before calibrating.')

# %% forcing completeness
n_rain_missing = int(np.isnan(rain).sum())
n_pet_missing = int(np.isnan(pet).sum())
forcing_interpolated = False

if n_rain_missing > 0 or n_pet_missing > 0:

    section_break()
    st.warning(f'The forcing record is incomplete: {n_rain_missing} missing rainfall values and '
               f'{n_pet_missing} missing PET values. The model carries state forward, so the '
               'simulation will be NaN from the first missing value onwards unless these are filled.')

    forcing_interpolated = st.checkbox('Linearly interpolate missing rainfall and PET', value=False)

    if not forcing_interpolated:
        st.info('Tick the box above to interpolate, or upload a complete forcing record.')
        st.stop()

    worst_gap = max(longest_gap(rain), longest_gap(pet))

    rain = pd.Series(rain).interpolate(method='linear', limit_direction='both').to_numpy()
    pet = pd.Series(pet).interpolate(method='linear', limit_direction='both').to_numpy()

    if np.isnan(rain).any() or np.isnan(pet).any():
        st.error('Interpolation failed. The rainfall or PET column contains no valid values.')
        st.stop()

    st.write(f'Interpolated {n_rain_missing} rainfall and {n_pet_missing} PET values. '
             f'Longest interpolated run: {worst_gap} days.')

    if worst_gap > LONG_FORCING_GAP:
        st.warning(f'At least one forcing gap is {worst_gap} days long. Linear interpolation of '
                   'daily rainfall is not physically meaningful over runs of this length, because '
                   'it smears or erases individual rain events. Patching from a nearby gauge or a '
                   'gridded product such as SILO would be preferable.')

if n_observed < 730:
    st.warning('Less than two years of observed flow available.')

# %% 2. model selection
section_break()
st.subheader('3. Model Selection')

model = st.selectbox('Hydrological Model', MODELS, index=0)
model_info = MODEL_INFO[model]
st.caption(model_info.notes)
with st.expander('Model structures and what each parameter does'):
    diagram_image, diagram_caption, diagram_notes = model_info.diagram
    st.image(diagram_image, caption=diagram_caption, width='stretch')
    st.markdown(diagram_notes)

    st.dataframe(
        pd.DataFrame([
            {'Parameter': name,
             # spec.label carries the units in parentheses, so strip them here
             'Meaning': spec.label.rsplit(' (', 1)[0],
             'Units': spec.units,
             'Full range': f'{spec.bounds[0]:g} to {spec.bounds[1]:g}',
             'Typical': spec.typical}
            for name, spec in MODEL_PARAMS[model].items()
        ]),
        hide_index=True, use_container_width=True)

    st.caption('Full range is what the optimiser is permitted to explore. Typical is what '
               'these parameters usually take on Australian catchments. A calibrated value '
               'far outside the typical range is not necessarily wrong, but it is usually '
               'compensating for something: a catchment area that is off, a forcing problem, '
               'or a structure that cannot represent the catchment. Check before interpreting it.')


if not model_info.can_produce_zero_flow and zero_fraction > 0.05:
    st.warning(f'{100 * zero_fraction:.0f} per cent of observed days are zero flow. The {model} '
               'exponential store asymptotes towards zero but never reaches it, so it will '
               'produce a persistent low trickle where the river is actually dry. A structure '
               'that can reach zero flow (GR4J, GR5J or SIMHYD) is likely a better fit here.')

if model_info.has_exchange_threshold:
    st.caption('The exchange term X2*(R/X3 - X5) is applied twice in GR5J and three times in '
               'GR6J, so large values of X2 combined with X5 can move a great deal of water into '
               'or out of the catchment. Check the calibrated water balance rather than trusting '
               'the efficiency score alone.')

simhyd_overflow_to_gw = False
if model == 'SIMHYD':
    _overflow_label = st.selectbox(
        'Soil-store overflow', list(SIMHYD_OVERFLOW_CHOICES), index=0,
        help='When the soil moisture store fills past its capacity SMSC, the excess is '
             'handled one of two ways. hydromad, the reference implementation, discards it. '
             'Chiew et al. (2009) Figure 2 routes it into the groundwater store, so it '
             'reappears later as baseflow. The two agree exactly whenever the soil store '
             'never fills; they diverge on wet catchments with a small SMSC. Keep this the '
             'same between calibration and analysis.')
    simhyd_overflow_to_gw = SIMHYD_OVERFLOW_CHOICES[_overflow_label]
    if simhyd_overflow_to_gw:
        st.caption('Using the Chiew et al. (2009) overflow path. This departs from hydromad, '
                   'so a numeric comparison against a hydromad run will not match on days the '
                   'soil store overflows.')

param_names = PARAM_NAMES[model]

data_key = (uploaded_file.name, getattr(uploaded_file, 'size', len(df)), date_col, rain_col,
            pet_col, flow_col, flow_units, float(area_km2), forcing_interpolated, model,
            simhyd_overflow_to_gw)

# %% 3. manual simulation
section_break()
st.subheader('4. Manual Simulation')
st.write(f'Adjust the {model} parameters by hand and assess model behaviour before running '
         'automatic calibration. All plots are in mm/d, the units the model works in. Exports can '
         'be written in mm/d, the input units, or both.')

manual_values = []
for name in param_names:
    spec = MODEL_PARAMS[model][name]
    lo, hi = spec.bounds
    manual_values.append(st.number_input(f'{name} {spec.label}',
                                         min_value=float(lo), max_value=float(hi),
                                         value=float(np.clip(spec.default, lo, hi)),
                                         key=f'manual_{model}_{name}'))

q_sim_manual = run_model(rain, pet, model, tuple(manual_values),
                         simhyd_overflow_to_gw=simhyd_overflow_to_gw)

section_break()
st.subheader('Observed vs Simulated (mm/d) (exploration parameters)')

col1, col2, col3 = st.columns(3)
col1.metric('KGE', f'{kge(q_obs_mmd, q_sim_manual):.3f}')
col2.metric('NSE', f'{nse(q_obs_mmd, q_sim_manual):.3f}')
col3.metric('KGE(1/Q)', f'{score(q_obs_mmd, q_sim_manual, "KGE", "inverse"):.3f}')

fig, _ = plot_hydrograph(dates, q_obs_mmd, [(q_sim_manual, C_SIM, model, 2)])
show(fig, 'exploration_hydrograph')

st.subheader('Residuals (exploration parameters)')
show(plot_log_residuals(dates, q_obs_mmd, q_sim_manual, C_SIM), 'exploration_residuals')

st.subheader('Observed vs Simulated Scatter (exploration parameters)')
show(plot_scatter(q_obs_mmd, q_sim_manual, C_SIM, 'Observed (mm/d)', 'Simulated (mm/d)'),
     'exploration_scatter')

# %% 4. calibration
section_break()
st.subheader('5. Model Calibration')
st.write('The objective function has two parts: the efficiency criterion, and the transformation '
         'applied to both flow series before the criterion is computed. Squared-error criteria on '
         'untransformed flow are dominated by peaks, so a parameter that only affects recessions '
         'will barely register. If low flows are what you care about, calibrate on a transformed '
         'series.')

if NUMBA_AVAILABLE:
    st.success('Model compiled with numba. Calibration takes seconds, so larger budgets and '
               'local sampling of the behavioural set are cheap.')
else:
    st.warning('numba is not installed, so the model is running in pure Python. Results are '
               'identical but roughly 40 times slower: a GR6J calibration that would take '
               'seconds takes minutes. Add "numba" to requirements.txt to enable it. Local '
               'sampling of the behavioural set defaults to off while it is absent, because '
               'every sampled point is a full model run.')

col1, col2 = st.columns(2)
metric = col1.selectbox('Efficiency Criterion', METRICS)
criterion_type = col2.selectbox('Criterion Type', ['Single transformation', 'Composite'])

composite_weight = None

if criterion_type == 'Single transformation':
    transform_kind = st.selectbox('Flow Transformation', TRANSFORMS,
                                  format_func=lambda k: TRANSFORM_LABELS[k])
    criterion = criterion_label(metric, transform_kind)
    st.caption(f'Calibration will maximise {criterion}. Values are not comparable across '
               'transformations, so do not read a higher number under one transform as a better '
               'model than a lower number under another.')

    if metric == 'KGE' and transform_kind == 'log':
        st.info('KGE on log-transformed flow uses a standardised bias component here, '
                'and is written KGE* to mark it. The usual bias term is a ratio of means, '
                'which is well defined for discharge but not for ln(Q + eps), a quantity '
                'that passes through zero. With the standard formula an identical five per '
                'cent overestimate can score anywhere from 0.37 to 0.99 depending only on '
                'the units the flow is expressed in. Values marked KGE* are therefore not '
                'directly comparable with a KGE(log Q) computed elsewhere.')
else:
    transform_kind = 'none'
    composite_weight = st.slider(
        f'Weight on {criterion_label(metric, COMPOSITE_TRANSFORMS[0])}',
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help='The remainder goes to the logarithmic component. 0.5 is a reasonable default; '
             'sweeping the weight traces the trade-off between high-flow and low-flow fit.')
    criterion = composite_label(metric, composite_weight)
    st.caption(f'Calibration will maximise {criterion}. This asks the optimiser to perform at '
               'both ends of the hydrograph rather than trading one for the other. It is a '
               'pragmatic calibration target, not a likelihood, so describe it that way in any '
               'write-up. Sweeping the weight from 0 to 1 and plotting the resulting flow '
               'duration curves traces the trade-off explicitly.')

if (model_info.low_flow_criterion_note and criterion_type == 'Single transformation'
        and transform_kind == 'none'):
    st.warning(model_info.low_flow_criterion_note)

warmup_days = int(st.number_input('Warm-up Days', value=730, min_value=0))

if model_info.min_warmup_days and warmup_days < model_info.min_warmup_days:
    st.warning(model_info.warmup_note.format(warmup_days=warmup_days,
                                             min_days=model_info.min_warmup_days))

# %% calibration / validation split
st.markdown('**Validation hold-out**')
st.write('Optionally withhold a block of years from calibration and use it only to measure how '
         'the model performs on days it never saw during fitting. This is a single run: the model '
         'is calibrated on the remaining days and that same model is what gap fills and is '
         'downloaded, so the validation score describes the model you actually get. Set the '
         'hold-out to 0 to calibrate on all data.')

col_h1, col_h2 = st.columns(2)
holdout_years = float(col_h1.number_input('Hold-out length (years)', value=0.0, min_value=0.0,
                                          step=1.0,
                                          help='Years of observed flow withheld from '
                                               'calibration. 0 uses all data.'))
holdout_position = col_h2.selectbox('Hold-out position', HOLDOUT_POSITIONS, index=0,
                                    disabled=holdout_years <= 0,
                                    help='Whether the withheld years are taken from the end of '
                                         'the record or from just after the warm-up.')

holdout_mask = holdout_mask_from_dates(dates, warmup_days, holdout_years, holdout_position)
fit_mask = None if holdout_years <= 0 else ~holdout_mask
holdout_ok = True

if holdout_years > 0:
    cal_score_mask, val_score_mask = scoring_masks(q_obs_mmd, warmup_days, holdout_mask)
    n_cal_left = int(cal_score_mask.sum())
    n_val = int(val_score_mask.sum())
    di_all = pd.DatetimeIndex(pd.to_datetime(dates))

    st.caption('The warm-up period above already removes the first '
               f'{warmup_days} days from fitting. A hold-out removes more on top of that. On a '
               'short record a long hold-out is unwise: because this is a single run with no '
               'refit, the days you withhold are gone from the delivered model for good, so it is '
               'fit on less data and is weaker for it, not merely measured on a smaller sample. '
               'Keep the hold-out short relative to the record.')

    if n_val > 0:
        val_dates = di_all[val_score_mask]
        st.write(f'Validation period: {val_dates.min():%Y-%m-%d} to {val_dates.max():%Y-%m-%d}, '
                 f'{n_val} observed days (about {n_val / 365.25:.1f} years). '
                 f'Calibration data left after warm-up and hold-out: {n_cal_left} days '
                 f'(about {n_cal_left / 365.25:.1f} years).')
    else:
        st.warning('The hold-out as configured contains no observed flow, so there is nothing to '
                   'validate against. Change its length or position.')

    if n_cal_left < MIN_CAL_DAYS_BLOCK:
        st.error(f'Only {n_cal_left} days of observed flow remain for calibration after the '
                 f'warm-up and hold-out, below the {MIN_CAL_DAYS_BLOCK} needed for a meaningful '
                 'fit. Shorten the hold-out or the warm-up.')
        holdout_ok = False
    elif n_cal_left < MIN_CAL_DAYS_WARN:
        st.warning(f'Only about {n_cal_left / 365.25:.1f} years of observed flow remain for '
                   'calibration after the warm-up and hold-out. The fit and every output derived '
                   'from it rest on a short record; treat the results with corresponding caution.')

st.write('Models within this distance of the best objective score are retained as behavioural '
         'models, up to 200 configurations.')
behavioural_delta = st.number_input('Behavioural Model Delta', value=0.05, min_value=0.001,
                                    max_value=0.50, step=0.01)

with st.expander('Advanced Calibration Settings'):
    maxiter = int(st.number_input('Maximum Iterations', value=25, min_value=1))
    popsize = int(st.number_input('Population Size', value=12, min_value=1))
    seed = int(st.number_input('Random Seed', value=1, min_value=0, step=1))

    st.markdown('**Behavioural set**')
    refine = st.checkbox('Refine by local sampling', value=NUMBA_AVAILABLE,
                         help='Recommended when the model is compiled. Each sampled point is a '
                              'full model run, so without numba this is expensive.')
    refine_sample = int(st.number_input('Local sample size',
                                        value=3000 if NUMBA_AVAILABLE else 500,
                                        min_value=0, step=500, disabled=not refine))
    refine_scale = st.slider('Sampling margin', min_value=0.0, max_value=0.5, value=0.10,
                             step=0.05, disabled=not refine)
    st.caption('Differential evolution is good at locating the behavioural region and bad at '
               'sampling it, because the density in its trajectory reflects the path the '
               'population happened to take rather than where good models are. With this on, '
               'the search locates the region and a Latin hypercube then samples it, so the '
               'behavioural spread is an actual sample of an actual region. Yield is typically '
               'one to four per cent, because the region is a thin ridge and its bounding box is '
               'mostly empty, so a few thousand points are needed. With the compiled model this '
               'costs seconds. The margin widens the box beyond what the search found.')
    st.caption('The seed fixes the differential evolution random state, so a repeated run with '
               'identical settings reproduces exactly. Changing it and re-running is the check '
               'for whether an apparent optimum is real: if two seeds land on materially '
               'different parameters at similar scores, the search has not converged or the '
               'catchment supports more than one solution. Each run is compared against the '
               'others in the run history below rather than pooled, since pooling trajectories '
               'from different seeds gives a set that is neither a sample nor a trajectory.')
    if NUMBA_AVAILABLE:
        st.caption('The model is compiled with numba, roughly 40 times faster than the pure '
                   'Python fallback, so larger budgets are cheap. A GR6J calibration at 50 '
                   'generations takes seconds rather than minutes.')
    else:
        st.caption('numba is not installed, so the model runs in pure Python. Results are '
                   'identical but roughly 40 times slower. Installing numba makes larger '
                   'budgets and local sampling practical. It is deliberately not a hard '
                   'requirement, because it pins a supported numpy range.')

    st.caption(f'Differential evolution sizes the population as popsize times the number of '
               f'parameters, so {model} runs {popsize * len(param_names)} members per generation '
               f'against {popsize * 4} for a four-parameter model. Expect GR6J to take around half '
               'again as long as GR4J for the same settings.')

    st.markdown('**Parameter Bounds**')
    st.caption('The defaults follow the ranges in general use for the GR models. Widening a bound '
               'gives the optimiser more room to compensate for a data problem, and narrowing one '
               'suppresses the bound-hit warning rather than fixing whatever caused it. Treat a '
               'constrained run as a diagnostic to report alongside the unconstrained one, not as '
               'a replacement for it.')

    custom_bounds = st.checkbox('Set parameter bounds manually', value=False,
                                key=f'custom_bounds_{model}')

    param_bounds = {name: PARAM_BOUNDS[name] for name in param_names}
    bounds_valid = True

    if custom_bounds:
        for name in param_names:
            default_lo, default_hi = PARAM_BOUNDS[name]
            st.write(f'{name} {PARAM_LABELS[name]}')
            col_lo, col_hi = st.columns(2)

            lo = col_lo.number_input('Lower', value=float(default_lo), format='%.4f',
                                     key=f'bound_{model}_{name}_lo', label_visibility='collapsed')
            hi = col_hi.number_input('Upper', value=float(default_hi), format='%.4f',
                                     key=f'bound_{model}_{name}_hi', label_visibility='collapsed')

            param_bounds[name] = (lo, hi)

            if lo >= hi:
                st.error(f'{name}: the lower bound must be below the upper bound.')
                bounds_valid = False
            elif name in STRICTLY_POSITIVE_PARAMS and lo <= 0:
                st.error(f'{name}: the lower bound must be greater than zero, since the model '
                         f'divides by {name}.')
                bounds_valid = False

        widened = [name for name in param_names
                   if param_bounds[name][0] < PARAM_BOUNDS[name][0]
                   or param_bounds[name][1] > PARAM_BOUNDS[name][1]]
        narrowed = [name for name in param_names
                    if param_bounds[name][0] > PARAM_BOUNDS[name][0]
                    or param_bounds[name][1] < PARAM_BOUNDS[name][1]]

        if widened:
            st.warning(f'Widened beyond the defaults: {", ".join(widened)}. A parameter that then '
                       'settles far outside the usual range is more often compensating for '
                       'something wrong in the forcing, the catchment area, or the model '
                       'structure than describing the catchment.')
        if narrowed:
            st.info(f'Narrowed relative to the defaults: {", ".join(narrowed)}. If the objective '
                    'score barely changes, the unconstrained value was not doing much work. If it '
                    'collapses, the model needs that value to fit, which is itself the finding.')

# rough cost estimate, so a long run is not a surprise
_evals = popsize * len(param_names) * (maxiter + 1) + (refine_sample if refine else 0)
_per_1000_days = 0.00015 if NUMBA_AVAILABLE else 0.0056     # seconds, measured
_estimate = _evals * _per_1000_days * len(rain) / 1000.0

st.caption(f'About {_evals:,} model evaluations, roughly '
           + (f'{_estimate:.0f} seconds.' if _estimate < 90
              else f'{_estimate / 60:.1f} minutes.')
           + ('' if NUMBA_AVAILABLE else ' Installing numba would cut this to a few seconds.'))

if not bounds_valid:
    st.error('Fix the parameter bounds above before calibrating.')

if st.button('Calibrate', type='primary', disabled=not (bounds_valid and holdout_ok),
             use_container_width=True):

    cal_bar = st.progress(0.0, text=f'Calibrating {model}...')
    generation = {'n': 0}

    def report_progress(params, convergence):
        generation['n'] += 1
        cal_bar.progress(min(generation['n'] / max(maxiter, 1), 1.0),
                         text=f"Calibrating {model}, generation {generation['n']} of {maxiter}")

    cal_results = calibrate_gr(precip=rain, pet=pet, q_obs=q_obs_mmd, model=model,
                               warmup_days=warmup_days, metric=metric,
                               transform_kind=transform_kind,
                               composite_weight=composite_weight,
                               maxiter=maxiter, popsize=popsize,
                               behavioural_delta=behavioural_delta, bounds=param_bounds,
                               seed=seed,
                               refine_sample=refine_sample if refine else 0,
                               refine_scale=refine_scale,
                               fit_mask=fit_mask,
                               simhyd_overflow_to_gw=simhyd_overflow_to_gw,
                               progress_callback=report_progress)

    cal_bar.empty()
    behavioural_df = cal_results['behavioural_df']

    if len(behavioural_df) == 0:
        st.error('No behavioural models met the acceptance criterion. Increase the behavioural '
                 'delta, or increase the maximum iterations.')
        st.stop()

    best_params = cal_results['best_params']

    ens_bar = st.progress(0.0, text='Simulating the behavioural ensemble...')
    ensemble = np.empty((len(behavioural_df), len(rain)))

    for i, (_, row) in enumerate(behavioural_df.iterrows()):
        ensemble[i] = simulate(rain, pet, {p: row[p] for p in param_names}, model=model,
                               simhyd_overflow_to_gw=simhyd_overflow_to_gw)
        ens_bar.progress((i + 1) / len(behavioural_df))

    ens_bar.empty()

    # Only summary arrays and the ensemble members that are actually exported are
    # retained. Holding the full ensemble, then rebuilding the flow duration curve
    # array from it on every rerun, is what pushes a multi-decade record past the
    # container memory limit and gets the app restarted.
    obs_mask = np.isfinite(q_obs_mmd)
    fdc_ensemble = np.array([fdc(sim[obs_mask])[1] for sim in ensemble])

    st.session_state['cal'] = {
        'schema': CAL_SCHEMA,
        'data_key': data_key,
        'model': model,
        'criterion': criterion,
        'metric': metric,
        'transform_kind': transform_kind,
        'composite_weight': composite_weight,
        'seed': int(seed),
        'behavioural_source': cal_results['behavioural_source'],
        'n_sampled': cal_results['n_sampled'],
        'epsilon': cal_results['epsilon'],
        'bounds': cal_results['bounds'],
        'bounds_customised': bool(custom_bounds),
        'warmup_days': warmup_days,
        'holdout_years': float(holdout_years),
        'holdout_position': holdout_position if holdout_years > 0 else None,
        'holdout_mask': holdout_mask if holdout_years > 0 else None,
        'representativeness': (holdout_representativeness(dates, rain, q_obs_mmd,
                                                         holdout_mask, warmup_days)
                               if holdout_years > 0 else None),
        'flow_units': flow_units,
        'area_km2': float(area_km2),
        'forcing_interpolated': forcing_interpolated,
        'simhyd_overflow_to_gw': bool(simhyd_overflow_to_gw),
        'dates': dates,
        'q_obs': q_obs_mmd,
        'behavioural_df': behavioural_df,
        'n_behavioural': len(behavioural_df),
        'best_params': best_params,
        'best_score': cal_results['best_score'],
        'q_cal': simulate(rain, pet, best_params, model=model,
                          simhyd_overflow_to_gw=simhyd_overflow_to_gw),
        'q05': np.nanpercentile(ensemble, 5, axis=0),
        'q50': np.nanpercentile(ensemble, 50, axis=0),
        'q95': np.nanpercentile(ensemble, 95, axis=0),
        'fdc05': np.nanpercentile(fdc_ensemble, 5, axis=0),
        'fdc50': np.nanpercentile(fdc_ensemble, 50, axis=0),
        'fdc95': np.nanpercentile(fdc_ensemble, 95, axis=0),
        'ensemble_export': ensemble[:min(MAX_EXPORT_MODELS, len(ensemble))].copy(),
    }

    # A compact record of every run in this session, so seed stability can be
    # read off a table rather than compared against screenshots. Only scalars
    # are kept, so the cost is negligible.
    history = st.session_state.setdefault('history', [])
    record = {'Run': len(history) + 1, 'Model': model, 'Seed': int(seed),
              'Criterion': criterion, 'Score': round(cal_results['best_score'], 4),
              'Behavioural': len(behavioural_df)}
    record.update({name: round(best_params[name], PARAM_ROUNDING[name])
                   for name in param_names})
    history.append(record)
    st.session_state['history'] = history[-MAX_HISTORY:]

    # count this completed run; harmless if no counter backend is configured
    _run_total = count_completed_run()
    if _run_total is not None:
        st.success(f'Calibration complete. This was run #{_run_total:,} of HydroSTITCH.')

    del ensemble, fdc_ensemble
    gc.collect()

# %% 4b. calibration results, rendered from session_state on every rerun
cal = st.session_state.get('cal')

if cal is not None and cal.get('schema') != CAL_SCHEMA:
    del st.session_state['cal']
    cal = None
    st.warning('Stored results were written by an earlier version of this app and have been '
               'discarded. Please re-run the calibration.')

if cal is None:
    st.info('Run a calibration to produce the behavioural ensemble, the gap filled series, and the '
            'downloadable workbook. Note that results are held for the current browser session '
            'only, so refreshing the page clears them.')
    st.stop()

if cal['data_key'] != data_key:
    st.warning('The data, model, column selections, or catchment settings have changed since this '
               'calibration was run. The results and download below still refer to the earlier '
               'configuration. Re-run the calibration to update them.')
    if st.button('Clear calibration results'):
        del st.session_state['cal']
        st.rerun()

cal_model = cal['model']
cal_params = PARAM_NAMES[cal_model]
cal_bounds = cal['bounds']
cal_dates = cal['dates']
q_obs = cal['q_obs']
cal_units = cal['flow_units']
cal_area = cal['area_km2']
behavioural_df = cal['behavioural_df']
best_params = cal['best_params']
q_cal = cal['q_cal']
q05, q50, q95 = cal['q05'], cal['q50'], cal['q95']

history = st.session_state.get('history', [])
if len(history) > 1:
    st.subheader('Run History (this session)')
    st.write('One row per calibration run in this browser session. Change the seed in Advanced '
             'Calibration Settings and re-run to test whether an optimum is stable: parameters '
             'that move materially between seeds at similar scores are not identified by the '
             'data.')
    st.dataframe(pd.DataFrame(history), hide_index=True)
    if st.button('Clear run history'):
        del st.session_state['history']
        st.rerun()
    section_break()

st.write(f"Model: {cal_model}. Behavioural models retained: {cal['n_behavioural']}, "
         f"from the {cal['behavioural_source']}.")

if cal['behavioural_source'].startswith('differential') and cal['n_sampled'] > 0:
    st.warning(f'Local sampling drew {cal["n_sampled"]} points but fewer than ten were '
               'behavioural, so the set fell back to the search trajectory. Increase the sample '
               'size or the margin in Advanced Calibration Settings, or accept that the spread '
               'below is a search trajectory rather than a sample and describe it that way.')
st.write(f"Best {cal['criterion']}: {cal['best_score']:.3f}")
st.dataframe(behavioural_df.head(20))

st.subheader('Behavioural Parameter Summary')
st.dataframe(behavioural_df[list(cal_params) + ['Score']].describe())

st.subheader('Calibration Results')
st.json(best_params)

# %% convergence diagnostics
# Ported from the package, where a sweep produces more calibrations than can be
# inspected. The same checks are useful here for the opposite reason: a single
# calibration is easy to over-read, and these name the specific ways in which a
# good-looking score can rest on a search that did not converge.

DEGENERATE_FRACTION = 1e-4      # spread below this fraction of the range
SCORE_CORRELATION_LIMIT = 0.4   # score still varying with a parameter

diagnostic_messages = []

for name in cal_params:
    low, high = cal_bounds[name]
    span = high - low
    values = behavioural_df[name].to_numpy(dtype=float)

    if len(values) > 2 and np.std(values) < DEGENERATE_FRACTION * span:
        diagnostic_messages.append(
            f'{name} shows no spread across the retained set, so the set records a '
            'collapse of the search rather than a region of parameter space.')

if len(behavioural_df) > 2 and 'Score' in behavioural_df:
    scores = behavioural_df['Score'].to_numpy(dtype=float)
    correlations = {}
    if np.std(scores) > 0:
        for name in cal_params:
            values = behavioural_df[name].to_numpy(dtype=float)
            if np.std(values) > 0:
                correlations[name] = float(np.corrcoef(values, scores)[0, 1])

    if correlations:
        worst = max(correlations, key=lambda k: abs(correlations[k]))
        if abs(correlations[worst]) > SCORE_CORRELATION_LIMIT:
            diagnostic_messages.append(
                f'The score still varies systematically with {worst} across the retained '
                f'set (r = {correlations[worst]:+.2f}). In a converged set the score is '
                'near flat with respect to every parameter, so this usually means the '
                'search was still descending a ridge when it stopped. Raise the maximum '
                'iterations and re-run.')

if diagnostic_messages:
    for message in diagnostic_messages:
        st.warning(message)
else:
    st.success('Convergence diagnostics passed: no parameter is degenerate and the score '
               'is flat with respect to the retained parameter sets.')

for name in cal_params:
    lo, hi = cal_bounds[name]
    tol = 0.001 * (hi - lo)
    if best_params[name] <= lo + tol:
        st.warning(f'{name} reached its lower bound of {lo:g}. The optimiser wanted to go further '
                   'and could not, so this value is set by the constraint rather than by the data.')
    if best_params[name] >= hi - tol:
        st.warning(f'{name} reached its upper bound of {hi:g}. The optimiser wanted to go further '
                   'and could not, so this value is set by the constraint rather than by the data.')

if cal['bounds_customised']:
    st.info('This calibration used manually set parameter bounds. They are recorded on the '
            'Metadata sheet of the workbook.')

# criteria across transformations, so the trade-off is visible
st.subheader('Best Model Performance')

cal_score_mask, val_score_mask = scoring_masks(q_obs, cal['warmup_days'], cal.get('holdout_mask'))

if cal.get('holdout_years', 0) and val_score_mask.any():
    # Per-period efficiency. Both columns use the calibration-derived offset held
    # in cal['epsilon'], so the log and inverse criteria are on the same footing
    # across periods and the two columns can be read against each other.
    cal_scores = period_efficiency(q_obs, q_cal, cal_score_mask, cal['epsilon'])
    val_scores = period_efficiency(q_obs, q_cal, val_score_mask, cal['epsilon'])

    perf = pd.DataFrame({
        'Metric': [label for label, _, _ in PERIOD_METRICS],
        'Calibration (seen)': [cal_scores[label] for label, _, _ in PERIOD_METRICS],
        'Validation (unseen)': [val_scores[label] for label, _, _ in PERIOD_METRICS],
    })
    perf['Change'] = perf['Validation (unseen)'] - perf['Calibration (seen)']
    st.dataframe(perf.round(3), hide_index=True, use_container_width=True)

    # kept on the cal dict so the export section and readme report the same
    # numbers without recomputing them
    cal['period_scores'] = {'calibration': cal_scores, 'validation': val_scores}
    cal['n_cal_scored'] = int(cal_score_mask.sum())
    cal['n_val_scored'] = int(val_score_mask.sum())
    _val_dates = pd.DatetimeIndex(pd.to_datetime(cal_dates))[val_score_mask]
    cal['val_date_range'] = (f'{_val_dates.min():%Y-%m-%d}', f'{_val_dates.max():%Y-%m-%d}')

    st.caption('The validation column is computed on the held-out days, which the delivered model '
               'never saw during fitting, so it is the honest estimate of how well the gap filling '
               'performs on unseen data. The delivered model is this same model, not a separate '
               'full-record fit, so these numbers describe what you download. A large fall from '
               'the calibration column to the validation column is the signature of overfitting '
               'or parameters that do not transfer across the conditions in the two periods. Both '
               'columns exclude the warm-up and use a shared transform offset. The validation '
               'block spans a sustained stretch with no nearby observed flow, which is harder than '
               'the shorter interior gaps that make up most fills, so read it as a conservative '
               'floor rather than an exact interior-gap skill.')

    rep = cal.get('representativeness')
    if rep and rep.get('ok'):
        if 'rain_percentile' in rep:
            line = ('Hold-out representativeness: its rainfall sits around the '
                    f"{rep['rain_percentile']:.0f}th percentile of the "
                    f"{rep['n_complete_years']} complete calendar years in the record")
            if 'flow_percentile' in rep:
                line += f", and its observed flow around the {rep['flow_percentile']:.0f}th"
            line += ('. A hold-out far from the middle is testing the model under conditions '
                     'unlike the calibration data: a harder and more informative test, but not a '
                     'like-for-like one, so weigh the drop from calibration to validation with '
                     'that in mind.')
            st.caption(line)
        elif np.isfinite(rep.get('rain_ratio', np.nan)):
            line = ('Hold-out representativeness: the record has only '
                    f"{rep['n_complete_years']} complete calendar years, too few for a "
                    'percentile, so this is a ratio instead. The hold-out averages '
                    f"{100 * rep['rain_ratio']:.0f} per cent of the calibration period's mean "
                    'daily rainfall')
            if np.isfinite(rep.get('flow_ratio', np.nan)):
                line += f", and {100 * rep['flow_ratio']:.0f} per cent of its mean observed flow"
            line += ('. Well away from 100 per cent means the two periods sample different '
                     'conditions, so the validation number is a transfer test rather than a '
                     'like-for-like one.')
            st.caption(line)
else:
    cols = st.columns(4)
    cols[0].metric('KGE(Q)', f'{score(q_obs, q_cal, "KGE", "none"):.3f}')
    cols[1].metric('NSE(Q)', f'{score(q_obs, q_cal, "NSE", "none"):.3f}')
    cols[2].metric('KGE(log Q)', f'{score(q_obs, q_cal, "KGE", "log"):.3f}')
    cols[3].metric('KGE(1/Q)', f'{score(q_obs, q_cal, "KGE", "inverse"):.3f}')
    st.caption('Reported over the whole record including warm-up, so these differ slightly from '
               'the calibration score, which excludes the warm-up period. No validation hold-out '
               'was set, so every observed day informed the fit and none of these numbers is an '
               'out-of-sample test.')

# hydrograph
st.subheader('Behavioural Ensemble Hydrograph')
fig_cal, ax = plot_hydrograph(cal_dates, q_obs, [(q50, C_CAL, 'Behavioural median', 1.5)])
ax.fill_between(cal_dates, q05, q95, color=C_CAL, alpha=0.25, label='5-95% behavioural range')
shade_periods(ax, cal_dates, cal.get('holdout_mask'))
ax.legend()
show(fig_cal, 'ensemble_hydrograph')

st.subheader('Behavioural Median Residuals')
show(plot_log_residuals(cal_dates, q_obs, q50, C_CAL, holdout_mask=cal.get('holdout_mask')),
     'behavioural_median_residuals')

# flow duration curve, from percentiles computed once at calibration time
ex_obs, q_obs_fdc = fdc(q_obs)

st.subheader('Flow Duration Curve')
fig_fdc, ax = new_fig(10, 8, [0.15, 0.15, 0.75, 0.75])
ax.fill_between(ex_obs, cal['fdc05'], cal['fdc95'], color=C_CAL, alpha=0.5,
                label='5-95% behavioural range', zorder=1)
ax.plot(ex_obs, cal['fdc50'], color=C_CAL, linewidth=2, label='Behavioural median', zorder=2)
ax.plot(ex_obs, q_obs_fdc, color=C_OBS, linewidth=2, label='Observed', zorder=3)
ax.set_yscale('log')
ax.set_ylim(bottom=FLOW_FLOOR_MMD)
ax.set_xlabel('Exceedance (%)')
ax.set_ylabel('Flow (mm/d)')
ax.legend()
show(fig_fdc, 'flow_duration_curve')

st.subheader('Best Model Scatter Plot')
show(plot_scatter(q_obs, q_cal, C_CAL, 'Observed (mm/d)', f'Calibrated {cal_model} (mm/d)'),
     'best_model_scatter')

# parameter distributions, grid sized to the number of parameters
st.subheader('Behavioural Parameter Distributions')
n_rows = math.ceil(len(cal_params) / 2)
fig_hist = plt.figure(figsize=(17 / 2.54, 6 * n_rows / 2.54))

for i, name in enumerate(cal_params):
    ax = fig_hist.add_subplot(n_rows, 2, i + 1)
    ax.hist(behavioural_df[name], bins=20, color=C_PARAM[i % len(C_PARAM)],
            edgecolor='black', linewidth=0.5)
    ax.axvline(best_params[name], color='black', linestyle='--', linewidth=1.5,
               label=f'{best_params[name]:.2f}')
    ax.set_title(name)
    ax.set_xlabel(PARAM_LABELS[name])
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)

fig_hist.subplots_adjust(hspace=0.55, wspace=0.20)
show(fig_hist, 'parameter_distributions')

with st.expander('Advanced Parameter Diagnostics'):
    st.subheader('Behavioural Parameter Correlation Matrix')
    st.dataframe(behavioural_df[list(cal_params) + ['Score']].corr())

    st.subheader('Behavioural Parameter Pairs')
    st.write('Each panel is one pair of parameters, coloured by objective score, with the best '
             'set marked by a cross. Read this before interpreting the histograms above. A '
             'histogram is a projection onto one axis, so a curved ridge in parameter space can '
             'produce apparent modes that are not separate solutions. If two clusters here are '
             'joined by a continuous arc of similar colour, there is one solution and the '
             'histogram is misleading. If they are genuinely separate, with a gap between them, '
             'the catchment supports more than one configuration and no amount of extra search '
             'effort will resolve it. Strong diagonal or curved structure in a panel means those '
             'two parameters compensate for each other and neither is individually identifiable.')

    fig_pairs = plot_parameter_pairs(behavioural_df, cal_params, best_params)
    if fig_pairs is not None:
        show(fig_pairs, 'parameter_pairs')
    st.caption('The behavioural set is drawn from the differential evolution trajectory rather '
               'than a random sample of parameter space, so the spread reflects local sensitivity '
               'around the optimum rather than a formal predictive uncertainty. Adding parameters '
               'widens the region of near-equivalent performance, so treat the spreads from the '
               'more highly parameterised models (GR6J, SIMHYD) with more caution than GR4J ones.')

# %% 5. gap filling
section_break()
st.subheader('6. Gap Filling')

gap_method = st.selectbox('Gap Filling Method', GAP_METHODS)
if gap_method == 'Ensemble Kalman Smoother':
    st.caption('Takes the deterministic calibrated run as the background, then '
               'updates each gap with the observations either side of it through '
               'the covariance of a 60-member ensemble with perturbed rainfall '
               'and PET. Slower than the residual methods, and it is the only one '
               'that uses the forcing through the gap.')

q_gapfilled = run_gapfill(q_obs, q50, gap_method, rain=rain, pet=pet,
                          best_params=cal['best_params'], model=cal_model,
                          simhyd_overflow_to_gw=cal.get('simhyd_overflow_to_gw', False))
q_gapfilled, n_clipped = clip_negative(q_gapfilled, q_obs)

n_missing = int(np.isnan(q_obs).sum())
gap_lengths = [g['length_days'] for g in identify_gaps(q_obs)]

st.write(f'Days gap filled: {n_missing} of {len(q_obs)}, across {len(gap_lengths)} gaps. '
         f'Longest gap: {max(gap_lengths, default=0)} days.')

if n_missing > 0:
    filled_mask = np.isnan(q_obs)
    filled_volume_ml = float(np.nansum(q_gapfilled[filled_mask])) * cal_area
    st.write(f'Volume attributed to filled days: {filled_volume_ml:,.0f} ML '
             f'({filled_volume_ml / 1000:,.1f} GL).')

if n_clipped > 0:
    st.warning(f'{n_clipped} gap filled values were negative and have been clipped to zero. '
               'Residual-based filling can push recession flows below zero where the model carries '
               'a persistent positive bias. Observed values are never clipped.')

st.subheader('Gap Filled Hydrograph')
fig_gap, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
ax.plot(cal_dates, q_gapfilled, color=C_CAL, linewidth=1, label='Gap filled')
ax.plot(cal_dates, q_obs, color=C_OBS, linewidth=1, label='Observed')
shade_periods(ax, cal_dates, cal.get('holdout_mask'))
ax.set_xlabel('Date')
ax.set_ylabel('Flow (mm/d)')
ax.set_xlim(pd.Timestamp(cal_dates.min()), pd.Timestamp(cal_dates.max()))
ax.set_ylim(bottom=0)
ax.legend()
show(fig_gap, 'gapfilled_hydrograph')

# %% 6. optional analysis
# Everything below is opt-in. The common path is upload, calibrate, gap fill,
# download; a user who wants a plausible hydrograph should not have to scroll
# past baseflow separation and Colwell decomposition to reach the download
# button. Gating also cuts rerun cost, because Streamlit re-executes every
# widget above the one being changed.
section_break()
st.subheader('7. Further Analysis and outputs (optional)')

st.write('The gap filled series above is the main output and the download below is ready now. '
         'The analyses here are optional and add to the download package when enabled.')

col_opt1, col_opt2 = st.columns(2)

show_evaluation = col_opt1.checkbox(
    'Evaluate the model against hydrological signatures', value=False,
    help='How well the calibrated model reproduces the flow duration curve, baseflow index, '
         'flashiness, seasonality and regime, rather than a single efficiency score. Fast.')

show_analysis = col_opt2.checkbox(
    'Baseflow separation, water year products and long-term analysis', value=False,
    help='Baseflow separation, recession analysis, water year and seasonal products, '
         'signature indices, trends and the long-term rainfall and flow figures. Slower, '
         'and adds around twenty CSV products to the download.')

# minimal product set, so the download works whether or not the analyses are run
products = {'daily_flow': pd.DataFrame(
    {'Date': cal_dates,
     'Q_mmd': q_gapfilled,
     'Q_MLd': q_gapfilled * cal_area,
     'Filled': np.isnan(q_obs).astype(int),
     'Period': period_labels(len(cal_dates), cal['warmup_days'], cal.get('holdout_mask'))})}

if show_evaluation:
    section_break()
    st.subheader('Model Evaluation Against the Observed Record')

    st.write('A single efficiency score answers one narrow question, weighted one particular way. '
             'It cannot say whether the model reproduces the shape of the flow duration curve, the '
             'proportion of flow arriving as baseflow, the flashiness of the hydrograph or the '
             'number of days the river stops flowing. Those are the properties that get used '
             'downstream, so they are what the model should be judged on. Everything below is '
             'computed on observed days only, so gap filled values never enter the evaluation.')

    sig_report, eff_table = run_signature_report(
        cal_dates, q_obs, q50, q_cal, DEFAULT_ALPHA, DEFAULT_PASSES, DEFAULT_REFLECT,
        0.0, cal['warmup_days'])

    if eff_table is not None and not eff_table.empty:
        st.markdown('**Efficiency criteria under every transformation**')
        st.dataframe(eff_table.set_index('Model').T, use_container_width=True)
        st.caption('KGE is shown with its three components. A composite of 0.85 built from a '
                   'correlation of 0.95 and a variability ratio of 1.3 describes a very different '
                   'failure from one built from a correlation of 0.87 and a variability ratio of '
                   '1.0, and the composite alone cannot tell them apart. Alpha above 1 means the '
                   'simulation is too variable; beta above 1 means it carries a positive volume bias.')
        products['efficiency_criteria'] = eff_table

    if sig_report is not None and not sig_report.empty:
        st.markdown('**Hydrological signatures, observed against modelled**')
        st.dataframe(sig_report, hide_index=True, use_container_width=True)
        products['signature_evaluation'] = sig_report

        poor = worst_signatures(sig_report, threshold=25.0)
        if poor:
            st.warning('Signatures reproduced worst, each out by more than 25 per cent in at least '
                       'one of the two simulations: ' + '; '.join(poor[:6])
                       + ('.' if len(poor) <= 6 else ', and others.')
                       + ' A model can score well on an efficiency criterion and still get these '
                         'wrong, which matters if any of them is the quantity the work depends on.')
        else:
            st.success('Every signature is reproduced to within 25 per cent by both the behavioural '
                       'median and the best model.')

        st.caption('The behavioural median is the day-by-day median of the retained ensemble and is '
                   'not itself a model run, so it can reproduce signatures that no single member '
                   'reproduces, and can smooth away flashiness that every member has. The best model '
                   'is one parameter set and is internally consistent. Where the two differ '
                   'markedly on a signature, prefer the best model for anything requiring physical '
                   'consistency and the median for anything requiring a central estimate.')



if show_analysis:
    section_break()
    st.subheader('Hydrological Analysis')
    st.write('Baseflow separation and the water year, seasonal and monthly summaries, computed on the '
             'gap filled series. Every product is written to the download package as a CSV.')

    col_wy, col_ctf = st.columns(2)

    wy_start_month = col_wy.selectbox(
        'Water Year Starts In', list(range(1, 13)), index=8,
        format_func=lambda m: MONTH_NAMES[m - 1],
        help='A water year is labelled by the calendar year it starts in, so with a September '
             'start, September 2024 to August 2025 is the 2024 water year. September suits the '
             'wet-dry tropics, since it sits near the dry season minimum and keeps a wet season '
             'inside one water year.')

    ctf_threshold = col_ctf.number_input(
        'Cease-to-Flow Threshold (mm/d)', value=0.0, min_value=0.0, format='%.5f',
        help='Days at or below this are counted as cease-to-flow. Zero is the strict definition. '
             'A small positive value is often more useful, and is necessary for GR6J, whose '
             'exponential store cannot reach exactly zero, so modelled dry periods would otherwise '
             'never register. GR4J, GR5J and SIMHYD can all produce exactly zero flow, so a zero '
             'threshold is meaningful for them.')

    st.markdown('**Baseflow Separation (Lyne and Hollick)**')

    alpha_mode = st.radio('Filter coefficient', ['Derive from recession', 'Set manually'],
                          horizontal=True)

    with st.expander('Advanced Baseflow Settings'):
        bf_passes = int(st.number_input('Filter Passes', value=DEFAULT_PASSES, min_value=3, step=2))
        bf_reflect = int(st.number_input('Reflection Length (days)', value=DEFAULT_REFLECT,
                                         min_value=5))
        st.caption('Passes must be odd. The reflection pads both ends of each block so the recursive '
                   'filter has run-in, and is trimmed off afterwards.')

        st.markdown('**Recession extraction**')
        rec_min_length = int(st.number_input('Minimum Recession Length (days)',
                                             value=RECESSION_MIN_LENGTH, min_value=2))
        rec_skip = int(st.number_input('Days Skipped After Each Peak',
                                       value=RECESSION_SKIP_DAYS, min_value=0))
        rec_quantile = st.slider('Ratio Quantile', min_value=0.05, max_value=0.95,
                                 value=RECESSION_QUANTILE, step=0.05)
        st.caption('Consecutive falling days form a recession segment. The first few days after a '
                   'peak are still dominated by quickflow, so they are dropped. Alpha is then the '
                   'chosen quantile of the daily ratio Q(t)/Q(t-1). A real hydrograph mixes a fast '
                   'and a slow recession, so the low quantiles sample the fast component and the '
                   'high quantiles the slow one. A wide interquartile range below means the estimate '
                   'is not uniquely defined for this catchment.')

    recession = run_recession_alpha(q_gapfilled, rec_min_length, rec_skip, rec_quantile)

    if alpha_mode == 'Derive from recession':
        if np.isfinite(recession['alpha']):
            bf_alpha = float(recession['alpha'])
            st.write(f'Derived alpha: **{bf_alpha:.4f}** from {recession["n_segments"]} recession '
                     f'segments and {recession["n_ratios"]} days. Ratio quartiles '
                     f'{recession["q25"]:.4f} / {recession["q50"]:.4f} / {recession["q75"]:.4f}.')
            spread = recession['q75'] - recession['q25']
            if spread > 0.05:
                st.warning(f'The interquartile range of the daily recession ratio is {spread:.3f}, '
                           'which is wide. The hydrograph is mixing fast and slow recessions, so the '
                           'derived alpha depends materially on the quantile chosen. Report the '
                           'quartiles alongside the value, and consider a manual alpha for '
                           'comparability with other studies.')
        else:
            bf_alpha = DEFAULT_ALPHA
            st.warning(f'Too few usable recession days were found, so the conventional '
                       f'{DEFAULT_ALPHA} is used instead.')
    else:
        bf_alpha = st.slider('Alpha', min_value=0.900, max_value=0.995, value=DEFAULT_ALPHA,
                             step=0.005)
        if np.isfinite(recession['alpha']):
            st.caption(f'For comparison, the recession-derived value for this series is '
                       f'{recession["alpha"]:.4f}. The conventional {DEFAULT_ALPHA} comes from '
                       'Nathan and McMahon (1990), who chose it because the separations resembled '
                       'manual ones on Australian catchments, not from recession theory.')

    separation = run_baseflow(q_gapfilled, bf_alpha, bf_passes, bf_reflect)
    q_baseflow = separation['baseflow']

    # SIMHYD carries its own baseflow. When it is the calibrated model, compute
    # the model's runoff split from the best parameter set and show it as a
    # second, separate separation. It is a different quantity from the filter:
    # the filter separates the gap filled hydrograph, this is the model's own
    # accounting over a clean re-run.
    simhyd_split = None
    simhyd_bfi = np.nan
    if MODEL_INFO[cal_model].provides_components:
        best_values = tuple(cal['best_params'][p] for p in cal_params)
        simhyd_split = run_simhyd_components(rain, pet, best_values,
                                            overflow_to_gw=cal.get('simhyd_overflow_to_gw', False))
        _mtot = float(np.nansum(simhyd_split['total']))
        simhyd_bfi = (float(np.nansum(simhyd_split['baseflow'])) / _mtot
                      if _mtot > 0 else np.nan)

    metric_cols = st.columns(4 if simhyd_split is not None else 3)
    metric_cols[0].metric('BFI (Lyne–Hollick)', f'{separation["bfi"]:.3f}')
    metric_cols[1].metric('Alpha', f'{bf_alpha:.4f}')
    metric_cols[2].metric('Passes', f'{bf_passes}')
    if simhyd_split is not None:
        metric_cols[3].metric('BFI (SIMHYD model)', f'{simhyd_bfi:.3f}')

    st.subheader('Baseflow Separation — Lyne–Hollick digital filter')
    st.caption('Applied to the gap filled series (observed flow, with the behavioural median '
               'spliced into the gaps).')
    fig_bf, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
    ax.plot(cal_dates, q_gapfilled, color=C_OBS, linewidth=0.8, label='Total flow')
    ax.plot(cal_dates, q_baseflow, color=C_BASE, linewidth=1.2, label='Baseflow')
    ax.fill_between(cal_dates, FLOW_FLOOR_MMD, np.clip(q_baseflow, FLOW_FLOOR_MMD, None),
                    color=C_BASE, alpha=0.3)
    shade_periods(ax, cal_dates, cal.get('holdout_mask'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (mm/d)')
    ax.set_xlim(pd.Timestamp(cal_dates.min()), pd.Timestamp(cal_dates.max()))
    ax.set_yscale('log')
    ax.set_ylim(bottom=FLOW_FLOOR_MMD)
    ax.legend()
    show(fig_bf, 'baseflow_separation_lyne_hollick')

    if simhyd_split is not None:
        st.subheader('Baseflow Separation — SIMHYD model components')
        st.caption('The runoff paths SIMHYD itself produces from the calibrated best parameter '
                   'set, re-run over the whole record. This is the model\'s baseflow, not a '
                   'filter of the observed hydrograph, so it will differ from the panel above.')

        # SIMHYD's internal split is not identifiable from a single streamflow
        # series. When SUB collapses toward zero and CRAK toward one the model
        # routes almost everything through the groundwater store, so "baseflow"
        # becomes a relabelling of total runoff. Flag that rather than present
        # the split as quantitative.
        _sh_inter_frac = float(np.nansum(simhyd_split['interflow'])
                               / max(np.nansum(simhyd_split['total']), 1e-9))
        if simhyd_bfi > 0.9 or _sh_inter_frac < 0.02:
            st.warning('SIMHYD is routing almost all runoff through its groundwater store on '
                       f'this catchment (model BFI {simhyd_bfi:.2f}, interflow '
                       f'{100 * _sh_inter_frac:.0f}% of runoff). The internal baseflow / '
                       'interflow / infiltration-excess split is not constrained by a single '
                       'streamflow series and is close to degenerate here. Treat the SIMHYD '
                       'component columns and this figure as structural, not quantitative; the '
                       'Lyne-Hollick separation above is the defensible baseflow product.')
        q_model_total = simhyd_split['total']
        q_model_base = simhyd_split['baseflow']
        q_model_inter = simhyd_split['interflow']
        q_model_surface = simhyd_split['surface']
        fig_sh, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
        ax.stackplot(cal_dates, q_model_base, q_model_inter, q_model_surface,
                     labels=['Baseflow', 'Interflow', 'Infiltration-excess'],
                     colors=[C_BASE, '#6460AA', '#F37021'], alpha=0.85)
        ax.plot(cal_dates, q_model_total, color=C_OBS, linewidth=0.6, label='Model total')
        shade_periods(ax, cal_dates, cal.get('holdout_mask'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow (mm/d)')
        ax.set_xlim(pd.Timestamp(cal_dates.min()), pd.Timestamp(cal_dates.max()))
        ax.legend(loc='upper right', ncol=2)
        show(fig_sh, 'baseflow_separation_simhyd_components')

    # %% products
    ctf_days = cease_to_flow(q_gapfilled, threshold=ctf_threshold)

    daily_frame = build_daily_frame(cal_dates, q_gapfilled, np.isnan(q_obs).astype(int),
                                    cal_area, start_month=wy_start_month,
                                    baseflow_mmd=q_baseflow, ctf_flag=ctf_days,
                                    model_components=simhyd_split)

    products = build_all_products(daily_frame, cal_area, start_month=wy_start_month)

    annual = products.get('annual_flow')

    if annual is None or annual.empty:
        st.warning('The record does not span a single complete water year, so no annual products '
                   'could be produced. Partial periods are excluded at both ends of the record.')
    else:
        st.subheader('Annual Flow by Water Year')
        st.write(f'{len(annual)} complete water years, {int(annual["WaterYear"].min())} to '
                 f'{int(annual["WaterYear"].max())}. Partial water years at each end of the record '
                 'are excluded, following the Hydrologic Reference Stations convention.')

        anomaly = products['annual_anomaly']

        fig_annual, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
        colours = [C_CAL if pf < 20 else '#F37021' for pf in annual['PercentFilled']]
        ax.bar(annual['WaterYear'], annual['Flow_GL'], color=colours, edgecolor='black',
               linewidth=0.4, label='Annual flow')
        ax.axhline(annual['Flow_GL'].mean(), color='black', linestyle='--', linewidth=1,
                   label=f'Mean {annual["Flow_GL"].mean():.1f} GL')
        ax.plot(anomaly['WaterYear'], anomaly['Anomaly_MA5_GL'] + annual['Flow_GL'].mean(),
                color=C_OBS, linewidth=1.5, label='5 year moving average')
        ax.set_xlabel(f'Water year (starting {MONTH_NAMES[wy_start_month - 1]})')
        ax.set_ylabel('Flow (GL/yr)')
        ax.legend(fontsize=7)
        show(fig_annual, 'annual_flow')
        st.caption('Bars are shaded orange where more than 20 per cent of the water year was gap '
                   'filled, so a year carried largely by the model is visible rather than implied.')

        def _annual_baseflow_chart(ab, title, slug):
            st.subheader(title)
            fig_ab, ax = new_fig(17, 7, [0.10, 0.16, 0.78, 0.74])
            ax.bar(ab['WaterYear'], ab['TotalFlow_GL'], color='#DDDDDD', edgecolor='black',
                   linewidth=0.4, label='Total flow')
            ax.bar(ab['WaterYear'], ab['Baseflow_GL'], color=C_BASE, edgecolor='black',
                   linewidth=0.4, label='Baseflow')
            ax.set_xlabel(f'Water year (starting {MONTH_NAMES[wy_start_month - 1]})')
            ax.set_ylabel('Flow (GL/yr)')
            ax.legend(fontsize=7, loc='upper left')
            ax_bfi = ax.twinx()
            ax_bfi.plot(ab['WaterYear'], ab['BFI'], color='black', marker='o', markersize=3,
                        linewidth=1)
            ax_bfi.set_ylabel('BFI (-)')
            ax_bfi.set_ylim(0, 1)
            show(fig_ab, slug)

        if 'annual_baseflow' in products:
            label = ('Annual Baseflow (Lyne–Hollick)' if 'annual_baseflow_simhyd' in products
                     else 'Annual Baseflow')
            _annual_baseflow_chart(products['annual_baseflow'], label, 'annual_baseflow')

        if 'annual_baseflow_simhyd' in products:
            _annual_baseflow_chart(products['annual_baseflow_simhyd'],
                                   'Annual Baseflow (SIMHYD model)', 'annual_baseflow_simhyd')

        if 'annual_cease_to_flow' in products:
            ctf_table = products['annual_cease_to_flow']
            total_ctf = int(ctf_table['CeaseToFlowDays'].sum())
            modelled_ctf = int(ctf_table['ModelledCeaseToFlowDays'].sum())
            st.write(f'Cease-to-flow days at or below {ctf_threshold:g} mm/d: {total_ctf} across the '
                     f'complete water years, of which {modelled_ctf} fall on gap filled days.')
            if not MODEL_INFO[cal_model].can_produce_zero_flow and ctf_threshold == 0.0:
                st.warning(f'{cal_model} cannot produce exactly zero flow, so at a threshold of zero '
                           'no gap filled day will ever be counted as cease-to-flow. Any dry period '
                           'that was filled is being recorded as flowing. Set a small positive '
                           'threshold, or read the observed and modelled columns separately.')

        with st.expander('Annual Summary Table'):
            st.dataframe(annual, hide_index=True)


    st.subheader('Signature Indices and Long-Term Analysis')

    complete_years = _complete_periods(daily_frame, 'WaterYear',
                                       lambda y: _water_year_span(y, wy_start_month))
    wy_labels = daily_frame['WaterYear'].to_numpy()

    col_ref1, col_ref2 = st.columns(2)

    if complete_years:
        year_low, year_high = int(min(complete_years)), int(max(complete_years))
    else:
        year_low = year_high = int(pd.DatetimeIndex(cal_dates).year.min())

    reference_start = col_ref1.number_input('Reference Period Start (water year)',
                                            value=year_low, min_value=year_low,
                                            max_value=year_high, step=1)
    reference_end = col_ref2.number_input('Reference Period End (water year)',
                                          value=year_high, min_value=year_low,
                                          max_value=year_high, step=1)

    if reference_start > reference_end:
        st.error('The reference period start must not be after the end.')
        reference_start, reference_end = year_low, year_high

    st.caption('Anomalies are computed against the mean of this period rather than the whole '
               'record, because a mean taken over a record containing a trend is not a stable '
               'baseline, and because comparability with published anomalies requires matching '
               'their reference period.')

    # --- whole record indices ---
    record_indices = whole_record_indices(cal_dates, q_gapfilled, rain)
    products['record_indices'] = record_indices

    st.markdown('**Whole-record signatures**')
    idx_cols = st.columns(4)
    lookup = dict(zip(record_indices['Index'], record_indices['Value']))
    idx_cols[0].metric('Colwell predictability', f'{lookup.get("Colwell predictability (P)", float("nan")):.3f}')
    idx_cols[1].metric('Constancy / contingency',
                       f'{lookup.get("Colwell constancy (C)", float("nan")):.2f} / '
                       f'{lookup.get("Colwell contingency (M)", float("nan")):.2f}')
    idx_cols[2].metric('Seasonality strength', f'{lookup.get("Seasonality strength (0-1)", float("nan")):.3f}')
    idx_cols[3].metric('Flashiness (RBI)', f'{lookup.get("Richards-Baker flashiness index", float("nan")):.3f}')

    with st.expander('What these mean'):
        st.write('**Colwell predictability** is how well the flow can be anticipated from the date '
                 'alone, and it splits into **constancy**, meaning the flow is always much the same, '
                 'and **contingency**, meaning the flow is reliably different at different times of '
                 'year. Predictability is the sum of the two. Northern Australian rivers are the '
                 'archetype of low constancy and high contingency: wildly variable, but variable on '
                 'a schedule. A river with high constancy instead is spring fed.')
        st.write('**Seasonality strength** runs from 0, flow spread evenly through the year, to 1, '
                 'all flow arriving on one day. **Flashiness** is the path length of the hydrograph '
                 'divided by total flow, so a smoothly interpolated gap lowers it, which is one '
                 'reason to read it against the percentage filled.')
        st.dataframe(record_indices, hide_index=True)

    # --- recession analysis ---
    st.markdown('**Recession analysis**')
    recession_fit = run_recession_analysis(q_gapfilled, rec_min_length, rec_skip)

    if np.isfinite(recession_fit['b']):
        rec_cols = st.columns(3)
        rec_cols[0].metric('Exponent b', f'{recession_fit["b"]:.3f}')
        rec_cols[1].metric('Coefficient a', f'{recession_fit["a"]:.4f}')
        rec_cols[2].metric('Envelope fit R2', f'{recession_fit["r_squared"]:.3f}')

        fig_rec, ax = new_fig(10, 9, [0.16, 0.14, 0.80, 0.80])
        ax.scatter(recession_fit['q_mid'], recession_fit['dqdt'], s=4, lw=0, alpha=0.18,
                   color=C_CAL)
        ax.plot(recession_fit['envelope_q'], recession_fit['envelope_dqdt'], 'o',
                color='black', markersize=4, label='Lower envelope')
        fitted = recession_fit['a'] * recession_fit['envelope_q'] ** recession_fit['b']
        ax.plot(recession_fit['envelope_q'], fitted, '-', color='#D0245C',
                linewidth=1.6,
                label=f'-dQ/dt = {recession_fit["a"]:.3g} Q$^{{{recession_fit["b"]:.2f}}}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=FLOW_FLOOR_MMD)
        ax.set_ylim(bottom=FLOW_FLOOR_MMD)
        ax.set_xlabel('Q (mm/d)')
        ax.set_ylabel('-dQ/dt (mm/d per day)')
        ax.legend(fontsize=7)
        show(fig_rec, 'recession_analysis')

        if recession_fit['b'] < 0.75:
            st.info(f'b = {recession_fit["b"]:.2f}, well below 1. Recession slows faster than a '
                    'linear reservoir would, which usually indicates more than one drainage '
                    'process or a store that is still being recharged during the falling limb.')
        elif recession_fit['b'] > 1.75:
            st.info(f'b = {recession_fit["b"]:.2f}. Values near 1.5 correspond to the Boussinesq '
                    'late-time solution for a horizontal unconfined aquifer and near 3 to the '
                    'early-time solution, so a high b points to shallow aquifer drainage rather '
                    'than a simple linear store.')
        else:
            st.info(f'b = {recession_fit["b"]:.2f}, close to 1, which is a linear reservoir. That '
                    'is the assumption behind an exponential recession and behind using a single '
                    'filter coefficient, so a constant alpha is defensible for this catchment.')
    else:
        st.info('Too few usable recession segments were found for a Brutsaert and Nieber fit.')

    # --- annual signature series ---
    if complete_years:
        ctf_flags = daily_frame['CeaseToFlow'].to_numpy() if 'CeaseToFlow' in daily_frame else None

        spells = cease_to_flow_spells(cal_dates, ctf_flags, np.isnan(q_obs))
        if not spells.empty:
            products['cease_to_flow_spells'] = spells
            annual_spells = annual_cease_to_flow_spells(
                spells, lambda dt: water_year([dt], wy_start_month)[0], complete_years)
            if not annual_spells.empty:
                products['annual_cease_to_flow_spells'] = annual_spells
                st.markdown('**Cease-to-flow spells**')
                st.write(f'{len(spells)} no-flow spells in the record. Longest '
                         f'{int(spells["LengthDays"].max())} days. A water year with its no-flow '
                         'days in one block is a very different river from one with the same total '
                         'spread across many short events, which is why the spell length matters '
                         'more than the annual count for refuge persistence and fish passage.')
                st.dataframe(annual_spells.head(10), hide_index=True)

        timing = half_flow_date(cal_dates, q_gapfilled, wy_labels, complete_years)
        if not timing.empty:
            products['annual_half_flow_date'] = timing

        rain_onset = anomalous_accumulation_onset(cal_dates, rain, wy_labels, complete_years)
        flow_onset = anomalous_accumulation_onset(cal_dates, q_gapfilled, wy_labels, complete_years)
        lag = onset_lag(rain_onset, flow_onset)

        if not rain_onset.empty:
            products['annual_wet_season_rainfall'] = rain_onset
        if not flow_onset.empty:
            products['annual_wet_season_flow'] = flow_onset
        onset_coverage = (len(rain_onset) / len(complete_years)) if complete_years else 0.0
        if onset_coverage < 0.7:
            st.warning(f'The wet season onset method resolved only {len(rain_onset)} of '
                       f'{len(complete_years)} water years. That almost always means the water year '
                       'start month is poorly chosen for this catchment: the method needs the wet '
                       'season to sit inside the water year rather than straddle its boundary. Try '
                       'setting the start month to a month in the middle of the dry season.')

        if not lag.empty:
            products['annual_onset_lag'] = lag
            st.markdown('**Wet season onset**')
            st.write(f'Mean lag between the rainfall wet season beginning and the flow responding: '
                     f'**{lag["OnsetLagDays"].mean():.0f} days** (range {int(lag["OnsetLagDays"].min())} '
                     f'to {int(lag["OnsetLagDays"].max())}). Onset is found by the anomalous '
                     'accumulation method, which needs no threshold. This lag is a catchment storage '
                     'signature: a deeply weathered or karstic catchment absorbs the first rains and '
                     'responds late, a shallow or already wet one responds almost at once.')

        rain_annual = annual_rainfall(cal_dates, rain, wy_labels, complete_years)
        if not rain_annual.empty:
            products['annual_rainfall'] = rain_annual

            merged = annual.merge(rain_annual, on='WaterYear', how='inner')
            if len(merged) >= 5:
                flow_mm = merged['Flow_GL'].to_numpy() * 1000.0 / cal_area
                elasticity = streamflow_elasticity(merged['Rainfall_mm'].to_numpy(), flow_mm)
                merged['Runoff_mm'] = flow_mm
                merged['RunoffCoefficient'] = flow_mm / merged['Rainfall_mm'].to_numpy()
                products['annual_rainfall_runoff'] = merged[
                    ['WaterYear', 'Rainfall_mm', 'Runoff_mm', 'RunoffCoefficient']]

                if np.isfinite(elasticity['elasticity']):
                    st.markdown('**Streamflow elasticity to rainfall**')
                    st.metric('Elasticity', f'{elasticity["elasticity"]:.2f}',
                              help='Proportional change in runoff per proportional change in '
                                   'rainfall. A value of 2 means a 10 per cent rainfall decline '
                                   'produces a 20 per cent runoff decline.')
                    st.caption(f'Estimated non-parametrically over {elasticity["n_years"]} complete '
                               'water years, after Sankarasubramanian et al. (2001). This is the '
                               'number to carry into a climate projection.')

        # --- regime indices per water year ---
        regime = annual_regime_indices(cal_dates, q_gapfilled, wy_labels, complete_years)
        if not regime.empty:
            products['annual_regime_indices'] = regime

            st.markdown('**Flashiness and regime by water year**')
            fig_regime, ax = new_fig(17, 7, [0.10, 0.16, 0.78, 0.76])
            ax.bar(regime['WaterYear'], regime['RichardsBakerIndex'], color=C_CAL,
                   edgecolor='black', linewidth=0.4, label='Richards-Baker index')
            ax.set_xlabel(f'Water year (starting {MONTH_NAMES[wy_start_month - 1]})')
            ax.set_ylabel('Richards-Baker index (-)')
            ax_cv = ax.twinx()
            ax_cv.plot(regime['WaterYear'], regime['CoefficientOfVariation'], color='black',
                       marker='o', markersize=3, linewidth=1, label='Coefficient of variation')
            ax_cv.set_ylabel('Coefficient of variation (-)')
            ax.legend(loc='upper left', fontsize=7)
            ax_cv.legend(loc='upper right', fontsize=7)
            show(fig_regime, 'annual_flashiness')
            st.caption('Flashiness is the path length of the hydrograph divided by total flow, so a '
                       'smoothly interpolated gap lowers it. Read this against the percentage filled '
                       'column in the annual table before interpreting any trend.')

        rolling = rolling_colwell(cal_dates, q_gapfilled, wy_labels, complete_years, window=15)
        if not rolling.empty:
            products['rolling_colwell'] = rolling

            st.markdown('**Regime predictability through time**')
            fig_col, ax = new_fig(17, 7, [0.10, 0.16, 0.86, 0.76])
            ax.plot(rolling['CentreWaterYear'], rolling['Predictability'], color='black',
                    linewidth=1.8, label='Predictability (P)')
            ax.plot(rolling['CentreWaterYear'], rolling['Constancy'], color=C_BASE,
                    linewidth=1.4, label='Constancy (C)')
            ax.plot(rolling['CentreWaterYear'], rolling['Contingency'], color='#F37021',
                    linewidth=1.4, label='Contingency (M)')
            ax.set_xlabel('Centre of 15 year window (water year)')
            ax.set_ylabel('Colwell index (-)')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            show(fig_col, 'rolling_colwell')
            st.caption('Predictability is the sum of constancy and contingency. A river losing '
                       'contingency is losing its seasonal signal, which is a different problem from '
                       'one losing constancy, and a single whole-record value cannot distinguish '
                       'either from no change at all.')

        # --- trends ---
        trend_inputs = annual.merge(rain_annual[['WaterYear', 'Rainfall_mm']],
                                    on='WaterYear', how='left') if not rain_annual.empty else annual
        if 'annual_baseflow' in products:
            trend_inputs = trend_inputs.merge(
                products['annual_baseflow'][['WaterYear', 'Baseflow_GL', 'BFI']],
                on='WaterYear', how='left')
        if not timing.empty:
            trend_inputs = trend_inputs.merge(timing[['WaterYear', 'DayOfWaterYear']],
                                              on='WaterYear', how='left')

        if not regime.empty:
            trend_inputs = trend_inputs.merge(
                regime[['WaterYear', 'RichardsBakerIndex', 'CoefficientOfVariation']],
                on='WaterYear', how='left')

        trends = trend_table(trend_inputs, 'WaterYear',
                             ['Flow_GL', 'Rainfall_mm', 'Baseflow_GL', 'BFI', 'DayOfWaterYear',
                              'RichardsBakerIndex', 'CoefficientOfVariation'])
        if not trends.empty:
            products['trends'] = trends
            st.markdown('**Trends (Mann-Kendall with Sen slope)**')
            st.dataframe(trends, hide_index=True)
            st.caption('Non-parametric, so no distributional assumption is made. The Sen slope is '
                       'the estimate to report alongside the p value. DayOfWaterYear is the timing '
                       'of the half-flow date, which often moves before annual totals do.')

        # --- SPI ---
        spi_table = spi(cal_dates, rain)
        if not spi_table.empty:
            products['spi'] = spi_table

        # --- showcase plots ---
        st.markdown('**Long-term rainfall and flow**')

        rain_wide = cumulative_by_water_year(cal_dates, rain, wy_labels, complete_years)
        flow_wide = cumulative_by_water_year(cal_dates, q_gapfilled, wy_labels, complete_years)

        fig_rain_cum = cumulative_spaghetti(
            rain_wide, start_month=wy_start_month, ylabel='Cumulative rainfall (mm)',
            title='Catchment rainfall by water year',
            subtitle=f'Water year starts in {MONTH_NAMES[wy_start_month - 1]}', credit=CREDIT)
        if fig_rain_cum is not None:
            show(fig_rain_cum, 'cumulative_rainfall')

        fig_flow_cum = cumulative_spaghetti(
            flow_wide, start_month=wy_start_month, ylabel='Cumulative runoff (mm)',
            title='Catchment runoff by water year',
            subtitle=f'Water year starts in {MONTH_NAMES[wy_start_month - 1]}', credit=CREDIT)
        if fig_flow_cum is not None:
            show(fig_flow_cum, 'cumulative_runoff')

        st.caption('Blue is the wettest water year on record, red the driest, black the most recent '
                   'complete year, and grey every other year.')

        if not rain_annual.empty:
            rain_anom = annual_anomaly_series(rain_annual, 'WaterYear', 'Rainfall_mm',
                                              int(reference_start), int(reference_end),
                                              moving_windows=(10,))
            products['annual_rainfall_anomaly'] = rain_anom
            fig_rain_anom = anomaly_bars(
                rain_anom, 'WaterYear', 'Anomaly', int(reference_start), int(reference_end),
                moving_column='Anomaly_MA10', ylabel='Rainfall anomaly (mm)',
                title='Water year rainfall anomaly',
                subtitle=f'Reference period {int(reference_start)} to {int(reference_end)}',
                credit=CREDIT)
            if fig_rain_anom is not None:
                show(fig_rain_anom, 'rainfall_anomaly')

        flow_anom = annual_anomaly_series(annual, 'WaterYear', 'Flow_GL',
                                          int(reference_start), int(reference_end),
                                          moving_windows=(10,))
        products['annual_flow_anomaly'] = flow_anom
        fig_flow_anom = anomaly_bars(
            flow_anom, 'WaterYear', 'Anomaly', int(reference_start), int(reference_end),
            moving_column='Anomaly_MA10', ylabel='Flow anomaly (GL)',
            title='Water year flow anomaly',
            subtitle=f'Reference period {int(reference_start)} to {int(reference_end)}',
            credit=CREDIT)
        if fig_flow_anom is not None:
            show(fig_flow_anom, 'flow_anomaly')

        # --- paired rainfall and runoff for the most recent complete year ---
        most_recent = int(max(complete_years))
        fig_paired = rainfall_runoff_cumulative(rain_wide, flow_wide, most_recent,
                                                start_month=wy_start_month, credit=CREDIT)
        if fig_paired is not None:
            st.markdown('**Rainfall and runoff together**')
            show(fig_paired, 'rainfall_runoff_paired')
            st.caption('The vertical gap between the two curves is water that has fallen on the '
                       'catchment and has not yet left it, so the shape of that gap through the year '
                       'is the catchment storage behaviour drawn directly. Widening through the wet '
                       'season is filling; a slow closing through the dry season is release.')

    st.write(f'**{len(products)} CSV products** are included in the download package: '
             + ', '.join(sorted(products)) + '.')



# %% 7. export
section_break()
st.subheader('Download Results')

native_label = f'{cal_units} only (as uploaded)'
both_label = f'Both mm/d and {cal_units}'

if cal_units == 'mm/d':
    st.write('The flow column was uploaded in mm/d, so the export is in mm/d and no conversion is '
             'applied.')
    export_choice = 'mm/d only'
else:
    st.write(f'The model works in mm/d. Flow was uploaded in {cal_units} and can be written back '
             f'in either unit system, converted using the {cal_area:g} km² catchment area recorded '
             'at calibration time.')
    export_choice = st.radio('Export units', [both_label, 'mm/d only', native_label], index=0)

include_mmd = export_choice in ('mm/d only', both_label)
include_native = export_choice in (native_label, both_label) and cal_units != 'mm/d'

ensemble_native = include_native and not include_mmd
ensemble_units = cal_units if ensemble_native else 'mm/d'

if include_mmd and include_native:
    sheet_units = f'mm/d and {cal_units}, indicated by the column suffix'
    file_tag = f'mmd_and_{UNIT_SUFFIX[cal_units]}'
elif include_native:
    sheet_units = cal_units
    file_tag = UNIT_SUFFIX[cal_units]
else:
    sheet_units = 'mm/d'
    file_tag = 'mmd'

series_mmd = {'Observed': q_obs, 'Gapfilled': q_gapfilled,
              'P05': q05, 'P50': q50, 'P95': q95}

output_df = build_output_df(cal_dates, series_mmd, cal_units, cal_area,
                            include_mmd, include_native,
                            warmup_days=cal['warmup_days'],
                            holdout_mask=cal.get('holdout_mask'))
ensemble_df = build_ensemble_df(cal_dates, cal['ensemble_export'], cal_units, cal_area,
                                ensemble_native)
metadata_df = build_metadata_df(cal, gap_method, n_missing, n_clipped,
                                ensemble_units, sheet_units)

st.write('Columns in the workbook:')
st.dataframe(pd.DataFrame({'Column': output_df.columns,
                           'Units': ['date' if c == 'Date'
                                     else 'flag' if c == 'FilledFlag'
                                     else 'label' if c == 'Period'
                                     else cal_units if (cal_units != 'mm/d'
                                                        and c.endswith(UNIT_SUFFIX[cal_units]))
                                     else 'mm/d'
                                     for c in output_df.columns]}),
             hide_index=True)

workbook_bytes = build_workbook(output_df, behavioural_df, ensemble_df, metadata_df)

workbook_name = f'gr_gapfill_{cal_model.lower()}_{file_tag}.xlsx'

# Calibration / validation split summary for the readme. Built here so the
# literal below can splice it in with unpacking whether or not a hold-out ran.
if cal.get('holdout_years', 0):
    _scores = cal.get('period_scores') or {}
    _vr = cal.get('val_date_range')
    holdout_readme_lines = [
        '',
        'Calibration / validation split',
        f'  Hold-out: {cal["holdout_years"]:g} years, {cal.get("holdout_position", "")}',
    ]
    if _vr:
        holdout_readme_lines.append(
            f'  Validation period: {_vr[0]} to {_vr[1]} '
            f'({cal.get("n_val_scored", 0)} scored days)')
    holdout_readme_lines += [
        f'  Calibration days scored: {cal.get("n_cal_scored", 0)}',
        '  The delivered model was calibrated on the other days only and was NOT refit on',
        '  the validation period, so the validation metrics below describe the model as',
        '  delivered. They are the estimate of gap-fill skill on unseen data. A large fall',
        '  from calibration to validation points to overfitting or parameters that do not',
        '  transfer between the conditions in the two periods.',
    ]
    _rep = cal.get('representativeness') or {}
    if _rep.get('ok'):
        if 'rain_percentile' in _rep:
            _line = (f'  Representativeness: hold-out rainfall around the '
                     f'{_rep["rain_percentile"]:.0f}th percentile of '
                     f'{_rep["n_complete_years"]} complete calendar years')
            if 'flow_percentile' in _rep:
                _line += f', observed flow around the {_rep["flow_percentile"]:.0f}th'
            holdout_readme_lines.append(_line + '.')
        elif np.isfinite(_rep.get('rain_ratio', np.nan)):
            _line = (f'  Representativeness: hold-out rainfall {100 * _rep["rain_ratio"]:.0f}% '
                     'of the calibration-period mean')
            if np.isfinite(_rep.get('flow_ratio', np.nan)):
                _line += f', flow {100 * _rep["flow_ratio"]:.0f}%'
            holdout_readme_lines.append(_line + ' (record too short for a percentile).')
    if _scores:
        holdout_readme_lines.append('  Per-period efficiency (shared transform offset):')
        for _label, _, _ in PERIOD_METRICS:
            holdout_readme_lines.append(
                f'    {_label:<11} cal {_scores["calibration"][_label]:.3f}   '
                f'val {_scores["validation"][_label]:.3f}')
else:
    holdout_readme_lines = ['', 'Validation hold-out: none (calibrated on all observed data)']

readme_lines = [
    'HydroSTITCH results package',
    '',
    f'Generated: {datetime.now():%Y-%m-%d %H:%M}',
    f'Model: {cal_model}',
    *([f'SIMHYD soil-store overflow: '
       + ('Chiew et al. 2009 (recharges groundwater)' if cal.get('simhyd_overflow_to_gw')
          else 'hydromad (discarded)')]
      if cal_model == 'SIMHYD' else []),
    f'Calibration criterion: {cal["criterion"]}',
    f'Best criterion value: {cal["best_score"]:.4f}',
    f'Random seed: {cal["seed"]}',
    f'Warm-up days excluded: {cal["warmup_days"]}',
    f'Behavioural models retained: {cal["n_behavioural"]}',
    f'Gap filling method: {gap_method}',
    f'Catchment area: {cal_area:g} km2',
    f'Input flow units: {cal_units}',
    f'Workbook units: {sheet_units}',
    *holdout_readme_lines,
    '',
    'Contents',
    f'  {workbook_name}',
    '      four sheets: GapFilled, BehaviouralModels, EnsembleHydrographs, Metadata',
    '  figures/',
    '      PNG at 200 dpi of every figure shown in the app',
    '  csv/',
    '      water year, seasonal and monthly products from the gap filled series',
    '',
    'Conventions used in the csv products',
    '  Water year label is the calendar year the water year starts in.',
    '  Partial water years, seasons and months at each end of the record are excluded.',
    '  Seasons are the standard meteorological ones, labelled by the year they start in,',
    '    so Summer 2024 is December 2024 to February 2025.',
    '  Percentiles use the exceedance convention: Q10 is the flow exceeded 10 per cent',
    '    of the time, which is the 90th percentile of the flow values.',
    '  PercentFilled is the share of days in that period that came from the model',
    '    rather than the gauge. Read every aggregate against it.',
    '  Period marks each day as Warm-up, Calibration or Validation. The GapFilled sheet',
    '    and daily_flow.csv both carry it, so seen and unseen days can be told apart.',
    '',
    'CSV products included:',
]
if show_analysis:
    readme_lines += [
        f'Water year starts in: {MONTH_NAMES[wy_start_month - 1]} '
        f'(labelled by the year it starts in)',
        f'Baseflow filter: Lyne and Hollick, alpha {bf_alpha:.4f}, {bf_passes} passes',
        f'Alpha source: {alpha_mode.lower()}',
        f'Baseflow index over the record (Lyne-Hollick): {separation["bfi"]:.4f}',
        f'Cease-to-flow threshold: {ctf_threshold:g} mm/d',
        '',
    ]
    if simhyd_split is not None:
        readme_lines += [
            'Two baseflow separations are provided because SIMHYD produces its own:',
            '  * columns/files marked _LH are the Lyne and Hollick digital filter applied',
            '    to the gap filled hydrograph;',
            '  * columns/files marked _SIMHYD are the runoff paths SIMHYD itself generates',
            '    from the calibrated parameter set, re-run over the whole record. Baseflow',
            f'    index over the record (SIMHYD model): {simhyd_bfi:.4f}',
            '  The two are different quantities and will not agree exactly.',
            'Note: the daily baseflow columns were renamed in this version, e.g. Qbase_MLd',
            'is now Qbase_LH_MLd.',
            '',
        ]

readme_lines += [f'  csv/{name}.csv' for name in sorted(products)]
readme_lines += [
    '',
    'Figures included:',
]
readme_lines += [f'  figures/{name}.png' for name in sorted(FIGURES)]
readme_lines += [
    '',
    'Note on the behavioural ensemble: it is drawn from the differential evolution',
    'trajectory, not from a random sample of parameter space, so the 5 to 95 per cent',
    'range is a local sensitivity band around the optimum and not a calibrated',
    'predictive uncertainty.',
]
readme_text = '\n'.join(readme_lines)

col_xlsx, col_zip = st.columns(2)

col_xlsx.download_button(
    label='Workbook only (.xlsx)',
    data=workbook_bytes,
    file_name=workbook_name,
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    key='download_workbook',
    use_container_width=True,
)

col_zip.download_button(
    label=f'Everything, including {len(FIGURES)} figures (.zip)',
    data=build_results_zip(workbook_bytes, workbook_name, FIGURES, readme_text, products),
    file_name=f'gr_gapfill_{cal_model.lower()}_{file_tag}.zip',
    mime='application/zip',
    key='download_package',
    use_container_width=True,
)

st.caption(f'The zip additionally holds {len(products)} CSV products and {len(FIGURES)} '
           'figures. Workbook contents: four sheets, GapFilled in ' + sheet_units + ', '
           'BehaviouralModels holding the '
           f'{cal["n_behavioural"]} retained {cal_model} parameter sets, EnsembleHydrographs '
           f'holding {len(cal["ensemble_export"])} members in {ensemble_units}, and Metadata '
           'recording the model, criterion, units, catchment area and calibration settings.')
