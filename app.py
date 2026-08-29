# app.py
# GR gap filling web app (GR4J, GR5J, GR6J)
# Dylan Irvine, Charles Darwin University
# Requires streamlit >= 1.30

import gc
import inspect
import math
import zipfile
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.models import (simulate, MODELS, PARAM_NAMES, PARAM_BOUNDS,
                         PARAM_LABELS, PARAM_ROUNDING, MODEL_NOTES, NUMBA_AVAILABLE)
from core.metrics import (kge, nse, score, criterion_label, composite_label,
                          resolve_kge_bias, kge_bias_is_unstable,
                          METRICS, TRANSFORMS, TRANSFORM_LABELS, COMPOSITE_TRANSFORMS)
from core.units import cumecs_to_mmd, mmd_to_cumecs, mld_to_mmd, mmd_to_mld
from core.calibration import calibrate_gr
from core.gapfill import (gapfill_p50, gapfill_snapped, gapfill_gaussian_process,
                          identify_gaps, clip_negative)
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
C_OBS, C_SIM, C_CAL = 'black', 'royalblue', '#0DB14B'
C_PARAM = ['#FCB711', '#F37021', '#CC004C', '#6460AA', '#0DB14B', '#2BA9E0']
MAX_EXPORT_MODELS = 30
MAX_HISTORY = 20
C_BASE = '#2BA9E0'
FIGURE_DPI = 300
CREDIT = 'Produced with the GR gap filling tool, Charles Darwin University'
LONG_FORCING_GAP = 5
CACHE_TTL = 3600
GR6J_MIN_WARMUP = 1095

# Version stamp for the dict held in st.session_state['cal']. Streamlit reruns
# the script in place when new source is deployed, so stored results can outlive
# the code that wrote them. Increment whenever a key is added, renamed or
# removed, and stale results are discarded rather than raising a KeyError.
CAL_SCHEMA = 5

FLOW_UNITS = ['m3/s', 'ML/d', 'mm/d']
UNIT_SUFFIX = {'m3/s': 'm3s', 'ML/d': 'MLd', 'mm/d': 'mmd'}
FLOW_SERIES = ['Observed', 'Gapfilled', 'P05', 'P50', 'P95']

PARAM_DEFAULTS = {'X1': 500.0, 'X2': 0.0, 'X3': 100.0, 'X4': 2.0, 'X5': 0.0, 'X6': 10.0}

GAP_METHODS = ['Behavioural Median', 'Endpoint Snapped Residuals',
               'Gaussian Process Residuals']


# %% module compatibility check
# app.py and the modules in core/ are updated together. If one is deployed
# without the other, the failure surfaces as a bare TypeError at the call site,
# and on Streamlit Cloud the message is redacted. This turns that into
# something actionable.

REQUIRED_ARGUMENTS = {
    'core/calibration.py': (calibrate_gr, ['model', 'transform_kind', 'composite_weight',
                                          'bounds', 'seed', 'refine_sample',
                                          'kge_bias', 'progress_callback']),
    'core/models.py': (simulate, ['model']),
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
    else:
        ax.set_ylim(bottom=0)

    ax.legend()
    return fig, ax


def plot_log_residuals(dates, q_obs, q_mod, colour):
    fig, ax = new_fig(17, 6, [0.10, 0.18, 0.85, 0.72])
    ax.plot(dates, np.log(q_obs + EPS) - np.log(q_mod + EPS), color=colour, linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
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


# %% cached computation
@st.cache_data(show_spinner=False, max_entries=8, ttl=CACHE_TTL)
def run_model(rain, pet, model, param_values):
    params = dict(zip(PARAM_NAMES[model], param_values))
    return simulate(rain, pet, params, model=model)


@st.cache_data(show_spinner='Gap filling...', max_entries=3, ttl=CACHE_TTL)
def run_gapfill(q_obs, q50, method):
    if method == 'Behavioural Median':
        return gapfill_p50(q_obs, q50)
    if method == 'Endpoint Snapped Residuals':
        return gapfill_snapped(q_obs, q50)
    return gapfill_gaussian_process(q_obs, q50)


@st.cache_data(show_spinner='Separating baseflow...', max_entries=3, ttl=CACHE_TTL)
def run_baseflow(q, alpha, passes, n_reflect):
    return lyne_hollick(q, alpha=alpha, passes=passes, n_reflect=n_reflect)


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
def build_output_df(dates, series_mmd, units, area_km2, include_mmd, include_native):
    data = {'Date': dates}

    if include_mmd:
        for name in FLOW_SERIES:
            data[f'{name}_mmd'] = series_mmd[name]

    if include_native:
        suffix = UNIT_SUFFIX[units]
        for name in FLOW_SERIES:
            data[f'{name}_{suffix}'] = from_mmd(series_mmd[name], units, area_km2)

    data['FilledFlag'] = np.isnan(series_mmd['Observed']).astype(int)
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

    bounds = cal['bounds']
    items.append(('Parameter bounds', 'manually set' if cal['bounds_customised'] else 'defaults'))

    for name in PARAM_NAMES[model]:
        lo, hi = bounds[name]
        items.append((f'{name} {PARAM_LABELS[name]}',
                      f'{best[name]:.4f} (bounds {lo:g} to {hi:g})'))

    items += [
        ('P05, P50, P95', 'Percentiles across the behavioural ensemble, per day'),
        ('FilledFlag', '1 where the observed record was missing and has been filled'),
        ('Model implementation', 'Transcribed from airGR Fortran and verified against it'),
    ]

    return pd.DataFrame(items, columns=['Item', 'Value'])


# %% header
# %% header
head_text, head_logo = st.columns([4, 1], vertical_alignment='center')

with head_text:
    st.title('HYDROSTITCH')
    st.subheader("**HYDRO**logical **S**ignatures, **T**ime-**S**eries **I**nfilling and **T**ools for **C**atchment **H**ydrology")
    st.write('Dylan Irvine, Charles Darwin University.\n')

with head_logo:
    st.image('docs/HS_Logo.png', width=200)

st.write(
    'The GR models (Modèle du Génie Rural à N paramètres Journalier) are simple, lumped '
    'conceptual rainfall-runoff models. They simulate daily streamflow using only '
    'catchment-averaged daily precipitation and potential evapotranspiration data. This tool '
    'applies the GR4J, GR5J and GR6J models with no coding required. Upload your file, follow the '
    'workflow, and you will have calibrated models and gap-filled hydrographs.\n'
    'Notably, numerous metrics are provided to ensure that you do not obtain a model with a '
    'good fit, but with highly inappropriate model parameters.\n\n'
)
with st.expander('**Selected References**'):
  st.write(
    'Perrin, C., Michel, C., and Andréassian, V. (2003). Improvement of a parsimonious model for '
    'streamflow simulation. Journal of Hydrology 279(1), 275-289.\n\n'
    'Le Moine, N. (2008). Le bassin versant de surface vu par le souterrain: une voie d\'amélioration' 
    'des performances et du réalisme des modèles pluie-débit? PhD thesis (in French), UPMC, Cemagref '
    'Antony, Paris, France.\n\n'
    'Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T., and Andréassian, V. (2011). A '
    'downward structural sensitivity analysis of hydrological models to improve low-flow '
    'simulation. Journal of Hydrology 411(1-2), 66-76.\n\n'
    'Coron, L., Thirel, G., Delaigue, O., Perrin, C., and Andréassian, V. (2017). The suite of lumped '
    'GR hydrological models in an R package. Environmental Modelling and Software 94, 166-171.'
    )

# %% 1. upload
st.subheader('1. Upload Data')
st.write('Upload a csv containing date, rainfall, PET, and streamflow. Dates must be in dd/mm/yyyy '
         'format. Rain and PET must be in mm/d, but flow can be m3/s, ML/d, or mm/d.')

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

section_break()
st.subheader('Catchment Information')
area_km2 = st.number_input('Catchment Area (km²)', min_value=0.001, value=1000.0, step=1.0)
flow_units = st.selectbox('Flow Units', FLOW_UNITS)

try:
    dates = pd.to_datetime(df[date_col], dayfirst=True)
    rain = np.asarray(df[rain_col], dtype=float)
    pet = np.asarray(df[pet_col], dtype=float)
    flow = np.asarray(df[flow_col], dtype=float)
except Exception as exc:
    st.error(f'Could not parse the selected columns: {exc}')
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
st.caption(MODEL_NOTES[model])
with st.expander('Model structures and what each parameter does'):
    st.image('docs/gr_structures.png',
             caption='Production module shared by all three models, and the routing '
                     'differences between GR4J, GR5J and GR6J. Structure after '
                     'Perrin et al. (2003), Le Moine (2008) and Pushpalatha et al. (2011).',
             use_container_width=True)

    st.markdown("""
**Three differences that are easy to misread.**

- **Routing split.** GR4J and GR6J split effective rainfall 90/10 *before*
  convolution and use two unit hydrographs, of base X4 and 2·X4. GR5J routes all
  of Pr through a single hydrograph of base 2·X4 and splits *after*. GR5J is not
  GR4J with a modified exchange term.
- **Exchange applications.** F is applied twice in GR4J and GR5J, three times in
  GR6J. The same X2 moves substantially more water in GR6J.
- **Exponential store.** R2 is not bounded below at zero. That is what sustains a
  slow recession indefinitely, and it is why GR6J cannot produce exactly zero flow.
""")

    UNITS = {'X1': 'mm', 'X2': 'mm/d', 'X3': 'mm', 'X4': 'd', 'X5': '-', 'X6': 'mm'}
    TYPICAL = {'X1': '100 to 800', 'X2': '-5 to 3', 'X3': '20 to 500',
               'X4': '1 to 10', 'X5': '-1 to 1', 'X6': '1 to 60'}

    st.dataframe(
        pd.DataFrame([
            {'Parameter': name,
             # PARAM_LABELS already carries the units, so strip them here
             'Meaning': PARAM_LABELS[name].rsplit(' (', 1)[0],
             'Units': UNITS[name],
             'Full range': f'{PARAM_BOUNDS[name][0]:g} to {PARAM_BOUNDS[name][1]:g}',
             'Typical': TYPICAL[name]}
            for name in PARAM_NAMES[model]
        ]),
        hide_index=True, use_container_width=True)

    st.caption('Full range is what the optimiser is permitted to explore. Typical is what '
               'these parameters usually take on Australian catchments. A calibrated value '
               'far outside the typical range is not necessarily wrong, but it is usually '
               'compensating for something: a catchment area that is off, a forcing problem, '
               'or a structure that cannot represent the catchment. Check before interpreting it.')


if model == 'GR6J' and zero_fraction > 0.05:
    st.warning(f'{100 * zero_fraction:.0f} per cent of observed days are zero flow. The GR6J '
               'exponential store asymptotes towards zero but never reaches it, so it will '
               'produce a persistent low trickle where the river is actually dry. GR4J or GR5J '
               'is likely the better structure for this catchment.')

if model in ('GR5J', 'GR6J'):
    st.caption('The exchange term X2*(R/X3 - X5) is applied twice in GR5J and three times in '
               'GR6J, so large values of X2 combined with X5 can move a great deal of water into '
               'or out of the catchment. Check the calibrated water balance rather than trusting '
               'the efficiency score alone.')

param_names = PARAM_NAMES[model]

data_key = (uploaded_file.name, getattr(uploaded_file, 'size', len(df)), date_col, rain_col,
            pet_col, flow_col, flow_units, float(area_km2), forcing_interpolated, model)

# %% 3. manual simulation
section_break()
st.subheader('4. Manual Simulation')
st.write(f'Adjust the {model} parameters by hand and assess model behaviour before running '
         'automatic calibration. All plots are in mm/d, the units the model works in. Exports can '
         'be written in mm/d, the input units, or both.')

manual_values = []
for name in param_names:
    lo, hi = PARAM_BOUNDS[name]
    manual_values.append(st.number_input(f'{name} {PARAM_LABELS[name]}',
                                         min_value=float(lo), max_value=float(hi),
                                         value=float(np.clip(PARAM_DEFAULTS[name], lo, hi)),
                                         key=f'manual_{model}_{name}'))

q_sim_manual = run_model(rain, pet, model, tuple(manual_values))

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

if model == 'GR6J' and criterion_type == 'Single transformation' and transform_kind == 'none':
    st.warning('GR6J adds X6 specifically to control low flows, but an untransformed criterion is '
               'almost entirely determined by peak flows, so X6 will be poorly constrained. '
               'Consider the logarithmic or inverse transformation.')

warmup_days = int(st.number_input('Warm-up Days', value=730, min_value=0))

if model == 'GR6J' and warmup_days < GR6J_MIN_WARMUP:
    st.warning(f'The GR6J exponential store equilibrates slowly and is initialised at zero. With '
               f'only {warmup_days} warm-up days the calibration may be fitting the spin-up rather '
               f'than the catchment. At least {GR6J_MIN_WARMUP} days is safer.')

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
            elif name == 'X6' and lo <= 0:
                st.error('X6: the lower bound must be greater than zero, since X6 divides the '
                         'exponential store level.')
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

if st.button('Calibrate', type='primary', disabled=not bounds_valid,
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
        ensemble[i] = simulate(rain, pet, {p: row[p] for p in param_names}, model=model)
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
        'flow_units': flow_units,
        'area_km2': float(area_km2),
        'forcing_interpolated': forcing_interpolated,
        'dates': dates,
        'q_obs': q_obs_mmd,
        'behavioural_df': behavioural_df,
        'n_behavioural': len(behavioural_df),
        'best_params': best_params,
        'best_score': cal_results['best_score'],
        'q_cal': simulate(rain, pet, best_params, model=model),
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
cols = st.columns(4)
cols[0].metric('KGE(Q)', f'{score(q_obs, q_cal, "KGE", "none"):.3f}')
cols[1].metric('NSE(Q)', f'{score(q_obs, q_cal, "NSE", "none"):.3f}')
cols[2].metric('KGE(log Q)', f'{score(q_obs, q_cal, "KGE", "log"):.3f}')
cols[3].metric('KGE(1/Q)', f'{score(q_obs, q_cal, "KGE", "inverse"):.3f}')
st.caption('Reported over the whole record including warm-up, so these differ slightly from the '
           'calibration score, which excludes the warm-up period.')

# hydrograph
st.subheader('Behavioural Ensemble Hydrograph')
fig_cal, ax = plot_hydrograph(cal_dates, q_obs, [(q50, C_CAL, 'Behavioural median', 1.5)])
ax.fill_between(cal_dates, q05, q95, color=C_CAL, alpha=0.25, label='5-95% behavioural range')
ax.legend()
show(fig_cal, 'ensemble_hydrograph')

st.subheader('Behavioural Median Residuals')
show(plot_log_residuals(cal_dates, q_obs, q50, C_CAL), 'behavioural_median_residuals')

# flow duration curve, from percentiles computed once at calibration time
ex_obs, q_obs_fdc = fdc(q_obs)

st.subheader('Flow Duration Curve')
fig_fdc, ax = new_fig(10, 8, [0.15, 0.15, 0.75, 0.75])
ax.fill_between(ex_obs, cal['fdc05'], cal['fdc95'], color=C_CAL, alpha=0.5,
                label='5-95% behavioural range', zorder=1)
ax.plot(ex_obs, cal['fdc50'], color=C_CAL, linewidth=2, label='Behavioural median', zorder=2)
ax.plot(ex_obs, q_obs_fdc, color=C_OBS, linewidth=2, label='Observed', zorder=3)
ax.set_yscale('log')
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
               'widens the region of near-equivalent performance, so treat GR6J spreads with more '
               'caution than GR4J ones.')

# %% 5. gap filling
section_break()
st.subheader('6. Gap Filling')

gap_method = st.selectbox('Gap Filling Method', GAP_METHODS)

q_gapfilled = run_gapfill(q_obs, q50, gap_method)
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
products = {'daily_flow': pd.DataFrame({'Date': cal_dates,
                                        'Q_mmd': q_gapfilled,
                                        'Q_MLd': q_gapfilled * cal_area,
                                        'Filled': np.isnan(q_obs).astype(int)})}

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
             'never register.')

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

    col_a, col_b, col_c = st.columns(3)
    col_a.metric('Baseflow Index', f'{separation["bfi"]:.3f}')
    col_b.metric('Alpha', f'{bf_alpha:.4f}')
    col_c.metric('Passes', f'{bf_passes}')

    st.subheader('Baseflow Separation')
    fig_bf, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
    ax.plot(cal_dates, q_gapfilled, color=C_OBS, linewidth=0.8, label='Total flow')
    ax.plot(cal_dates, q_baseflow, color=C_BASE, linewidth=1.2, label='Baseflow')
    ax.fill_between(cal_dates, 0, q_baseflow, color=C_BASE, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (mm/d)')
    ax.set_xlim(pd.Timestamp(cal_dates.min()), pd.Timestamp(cal_dates.max()))
    ax.set_yscale('log')
    ax.legend()
    show(fig_bf, 'baseflow_separation')

    # %% products
    ctf_days = cease_to_flow(q_gapfilled, threshold=ctf_threshold)

    daily_frame = build_daily_frame(cal_dates, q_gapfilled, np.isnan(q_obs).astype(int),
                                    cal_area, start_month=wy_start_month,
                                    baseflow_mmd=q_baseflow, ctf_flag=ctf_days)

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

        if 'annual_baseflow' in products:
            st.subheader('Annual Baseflow')
            ab = products['annual_baseflow']
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
            show(fig_ab, 'annual_baseflow')

        if 'annual_cease_to_flow' in products:
            ctf_table = products['annual_cease_to_flow']
            total_ctf = int(ctf_table['CeaseToFlowDays'].sum())
            modelled_ctf = int(ctf_table['ModelledCeaseToFlowDays'].sum())
            st.write(f'Cease-to-flow days at or below {ctf_threshold:g} mm/d: {total_ctf} across the '
                     f'complete water years, of which {modelled_ctf} fall on gap filled days.')
            if cal_model == 'GR6J' and ctf_threshold == 0.0:
                st.warning('GR6J cannot produce exactly zero flow, so at a threshold of zero no gap '
                           'filled day will ever be counted as cease-to-flow. Any dry period that '
                           'was filled is being recorded as flowing. Set a small positive threshold, '
                           'or read the observed and modelled columns separately.')

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
                            include_mmd, include_native)
ensemble_df = build_ensemble_df(cal_dates, cal['ensemble_export'], cal_units, cal_area,
                                ensemble_native)
metadata_df = build_metadata_df(cal, gap_method, n_missing, n_clipped,
                                ensemble_units, sheet_units)

st.write('Columns in the workbook:')
st.dataframe(pd.DataFrame({'Column': output_df.columns,
                           'Units': ['date' if c == 'Date'
                                     else 'flag' if c == 'FilledFlag'
                                     else cal_units if (cal_units != 'mm/d'
                                                        and c.endswith(UNIT_SUFFIX[cal_units]))
                                     else 'mm/d'
                                     for c in output_df.columns]}),
             hide_index=True)

workbook_bytes = build_workbook(output_df, behavioural_df, ensemble_df, metadata_df)

workbook_name = f'gr_gapfill_{cal_model.lower()}_{file_tag}.xlsx'

readme_lines = [
    'GR gap filling results package',
    '',
    f'Generated: {datetime.now():%Y-%m-%d %H:%M}',
    f'Model: {cal_model}',
    f'Calibration criterion: {cal["criterion"]}',
    f'Best criterion value: {cal["best_score"]:.4f}',
    f'Random seed: {cal["seed"]}',
    f'Warm-up days excluded: {cal["warmup_days"]}',
    f'Behavioural models retained: {cal["n_behavioural"]}',
    f'Gap filling method: {gap_method}',
    f'Catchment area: {cal_area:g} km2',
    f'Input flow units: {cal_units}',
    f'Workbook units: {sheet_units}',
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
    '',
    'CSV products included:',
]
if show_analysis:
    readme_lines += [
        f'Water year starts in: {MONTH_NAMES[wy_start_month - 1]} '
        f'(labelled by the year it starts in)',
        f'Baseflow filter: Lyne and Hollick, alpha {bf_alpha:.4f}, {bf_passes} passes',
        f'Alpha source: {alpha_mode.lower()}',
        f'Baseflow index over the record: {separation["bfi"]:.4f}',
        f'Cease-to-flow threshold: {ctf_threshold:g} mm/d',
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
