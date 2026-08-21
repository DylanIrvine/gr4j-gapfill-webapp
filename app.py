# app.py
# GR4J gap filling web app
# Dylan Irvine, Charles Darwin University
# Requires streamlit >= 1.30

import gc
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.metrics import kge, nse
from core.gr4j import simulate
from core.units import cumecs_to_mmd, mmd_to_cumecs, mld_to_mmd, mmd_to_mld
from core.calibration import calibrate_gr4j, PARAM_BOUNDS
from core.gapfill import (gapfill_p50, gapfill_snapped, gapfill_gaussian_process,
                          identify_gaps, clip_negative)

# %% plot settings
plt.style.use('default')
plt.rc('axes', linewidth=0.5)
plt.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
plt.rcParams.update({'font.size': 8, 'legend.labelspacing': 0.1, 'lines.linewidth': 1,
                     'xtick.direction': 'out', 'ytick.direction': 'out'})

# %% constants
EPS = 0.01
C_OBS, C_SIM, C_CAL = 'black', 'royalblue', '#0DB14B'
C_PARAM = ['#FCB711', '#F37021', '#CC004C', '#6460AA']
MAX_EXPORT_MODELS = 30
LONG_FORCING_GAP = 5
CACHE_TTL = 3600

FLOW_UNITS = ['m3/s', 'ML/d', 'mm/d']

# Suffix appended to exported column names. Filesystem and Excel safe, so no
# slashes, and unambiguous about which unit system a column is in.
UNIT_SUFFIX = {'m3/s': 'm3s', 'ML/d': 'MLd', 'mm/d': 'mmd'}

FLOW_SERIES = ['Observed', 'Gapfilled', 'P05', 'P50', 'P95']

PARAM_LABELS = {'X1': 'Production Store Capacity (mm)', 'X2': 'Groundwater Exchange (mm/d)',
                'X3': 'Routing Store Capacity (mm)', 'X4': 'Time Base (days)'}

GAP_METHODS = ['Behavioural Median', 'Endpoint Snapped Residuals', 'Gaussian Process Residuals']


# %% unit conversion
# GR4J works in mm/d throughout, so flow is converted on the way in and
# converted back on the way out. Both conversions use the same catchment area,
# so the round trip is exact to floating point.

def to_mmd(q, units, area_km2):
    """Convert a flow series from the stated input units to mm/d."""
    if units == 'm3/s':
        return cumecs_to_mmd(q, area_km2)
    if units == 'ML/d':
        return mld_to_mmd(q, area_km2)
    return np.asarray(q, dtype=float)


def from_mmd(q_mmd, units, area_km2):
    """Convert a flow series from mm/d back to the stated input units."""
    if units == 'm3/s':
        return mmd_to_cumecs(q_mmd, area_km2)
    if units == 'ML/d':
        return mmd_to_mld(q_mmd, area_km2)
    return np.asarray(q_mmd, dtype=float)


def unit_factor(units, area_km2):
    """How many mm/d one unit of the input series represents on this catchment."""
    return float(to_mmd(np.array([1.0]), units, area_km2)[0])


# %% plotting helpers
def section_break():
    st.markdown('---')


def new_fig(w_cm, h_cm, rect):
    fig = plt.figure(figsize=(w_cm / 2.54, h_cm / 2.54))
    return fig, fig.add_axes(rect)


def show(fig):
    """Render a figure then close it, so repeated reruns do not leak memory."""
    st.pyplot(fig)
    plt.close(fig)


def fdc(q):
    """Flow duration curve. Returns exceedance (%) and flows sorted high to low."""
    q = np.sort(q[np.isfinite(q)])[::-1]
    return np.arange(1, len(q) + 1) / (len(q) + 1) * 100, q


def plot_hydrograph(dates, q_obs, series):
    """series is a list of (values, colour, label, linewidth) tuples."""
    fig, ax = new_fig(17, 8, [0.10, 0.15, 0.85, 0.75])
    ax.plot(dates, q_obs, color=C_OBS, alpha=0.6, linewidth=1, label='Observed')
    for values, colour, label, lw in series:
        ax.plot(dates, values, color=colour, alpha=0.8, linewidth=lw, label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (mm/d)')
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


def longest_gap(series):
    return max([g['length_days'] for g in identify_gaps(series)], default=0)


# %% cached computation
# Streamlit reruns the whole script on every widget interaction, including a
# click on the download button. Anything expensive that is neither cached nor
# held in session_state is recomputed on every one of those reruns.
#
# Caches are shared across sessions on the same container and are a common cause
# of gradual memory growth on Community Cloud, so entry counts are deliberately
# small and everything expires after an hour.

@st.cache_data(show_spinner=False, max_entries=8, ttl=CACHE_TTL)
def run_gr4j(rain, pet, x1, x2, x3, x4):
    return simulate(rain, pet, {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4})


@st.cache_data(show_spinner='Gap filling...', max_entries=3, ttl=CACHE_TTL)
def run_gapfill(q_obs, q50, method):
    if method == 'Behavioural Median':
        return gapfill_p50(q_obs, q50)
    if method == 'Endpoint Snapped Residuals':
        return gapfill_snapped(q_obs, q50)
    return gapfill_gaussian_process(q_obs, q50)


def _excel_writer(buffer):
    """Prefer xlsxwriter in constant memory mode.

    openpyxl instantiates a Python object per cell before saving, which for a
    multi-decade daily record with 30 ensemble columns runs to hundreds of
    thousands of objects. xlsxwriter with constant_memory streams each row to a
    temporary file instead and keeps memory flat.
    """
    try:
        import xlsxwriter  # noqa: F401
        return pd.ExcelWriter(buffer, engine='xlsxwriter',
                              engine_kwargs={'options': {'constant_memory': True}})
    except ImportError:
        return pd.ExcelWriter(buffer, engine='openpyxl')


@st.cache_data(show_spinner='Building workbook...', max_entries=2, ttl=CACHE_TTL)
def build_workbook(output_df, behavioural_df, ensemble_df, metadata_df):
    """Return workbook bytes. Bytes rather than a BytesIO, because a buffer held
    across reruns can be left at EOF and download as an empty file."""
    buffer = BytesIO()
    with _excel_writer(buffer) as writer:
        output_df.to_excel(writer, sheet_name='GapFilled', index=False)
        behavioural_df.to_excel(writer, sheet_name='BehaviouralModels', index=False)
        ensemble_df.to_excel(writer, sheet_name='EnsembleHydrographs', index=False)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    return buffer.getvalue()


# %% export table builders
def build_output_df(dates, series_mmd, units, area_km2, include_mmd, include_native):
    """Assemble the gap filled sheet, with flow columns in one or both unit systems.

    series_mmd is a dict of name to mm/d array, keyed by FLOW_SERIES.
    """
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
    """Ensemble hydrographs in a single unit system, to keep the sheet narrow."""
    values = from_mmd(ensemble_mmd, units, area_km2) if native else ensemble_mmd
    suffix = UNIT_SUFFIX[units] if native else 'mmd'

    ensemble_df = pd.DataFrame(values.T,
                               columns=[f'Model_{i + 1:03d}_{suffix}'
                                        for i in range(len(values))])
    ensemble_df.insert(0, 'Date', np.asarray(dates))
    return ensemble_df


def build_metadata_df(cal, gap_method, n_missing, n_clipped, ensemble_units, sheet_units):
    """Document what is in the workbook, and in which units.

    Without this, a downstream user opening the file has no way to tell whether
    a column is a depth or a discharge, or which catchment area was used to
    convert between them.
    """
    units = cal['flow_units']
    area = cal['area_km2']
    best = cal['best_params']

    items = [
        ('Input flow units', units),
        ('Catchment area (km2)', f'{area:g}'),
        ('1 input unit expressed in mm/d', f'{unit_factor(units, area):.6g}'),
        ('GapFilled sheet units', sheet_units),
        ('EnsembleHydrographs sheet units', ensemble_units),
        ('Rainfall and PET units', 'mm/d'),
        ('Objective function', cal['objective']),
        ('Best objective score', f"{cal['best_score']:.4f}"),
        ('Behavioural models retained', str(cal['n_behavioural'])),
        ('Gap filling method', gap_method),
        ('Days gap filled', str(n_missing)),
        ('Filled values clipped at zero', str(n_clipped)),
        ('Forcing interpolated', str(cal['forcing_interpolated'])),
        ('X1 production store capacity (mm)', f"{best['X1']:.3f}"),
        ('X2 groundwater exchange (mm/d)', f"{best['X2']:.3f}"),
        ('X3 routing store capacity (mm)', f"{best['X3']:.3f}"),
        ('X4 time base (days)', f"{best['X4']:.3f}"),
        ('P05, P50, P95', 'Percentiles across the behavioural ensemble, per day'),
        ('FilledFlag', '1 where the observed record was missing and has been filled'),
    ]

    return pd.DataFrame(items, columns=['Item', 'Value'])


# %% header
st.title('GR4J Gap Filling Tool')
st.write('Dylan Irvine, Charles Darwin University.\n')
st.write(
    'The GR4J model (Modèle du Génie Rural à 4 paramètres Journalier) is a simple, lumped '
    'conceptual rainfall-runoff model. It simulates daily streamflow using only catchment-averaged '
    'daily precipitation and potential evapotranspiration data. The model was originally published '
    'by Perrin et al. (2003).\n\n The gr4j-gapfill-webapp provides users with an online version of '
    'the tool that can be applied with no coding required. Simply upload your file, follow the '
    'workflow, and you will have calibrated models and/or gap-filled hydrographs.\n\n'
    'Original reference \n\nPerrin, C., Michel, C., and Andréassian, V. "Improvement of a '
    'parsimonious model for streamflow simulation." Journal of Hydrology 279, no. 1 (2003): 275-289.'
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
st.subheader('Data Preview')
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

obs_mask_all = np.isfinite(q_obs_mmd) & np.isfinite(rain)

if obs_mask_all.sum() > 0:
    mean_q_mmd = float(np.mean(q_obs_mmd[obs_mask_all]))
    mean_p_mmd = float(np.mean(rain[obs_mask_all]))
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
# GR4J carries state forward day by day, so a single missing rainfall or PET
# value turns every subsequent day into NaN.
n_rain_missing = int(np.isnan(rain).sum())
n_pet_missing = int(np.isnan(pet).sum())
forcing_interpolated = False

if n_rain_missing > 0 or n_pet_missing > 0:

    section_break()
    st.warning(f'The forcing record is incomplete: {n_rain_missing} missing rainfall values and '
               f'{n_pet_missing} missing PET values. GR4J carries state forward, so the simulation '
               'will be NaN from the first missing value onwards unless these are filled.')

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

if np.isfinite(q_obs_mmd).sum() < 730:
    st.warning('Less than two years of observed flow available.')

# Identifies the current data configuration, so stored calibration results can
# be checked for staleness after the user changes a column, the area or the units.
data_key = (uploaded_file.name, getattr(uploaded_file, 'size', len(df)), date_col, rain_col,
            pet_col, flow_col, flow_units, float(area_km2), forcing_interpolated)

# %% 2. manual simulation
section_break()
st.subheader('2. Manual GR4J Simulation')
st.write('Use this section to manually adjust the GR4J parameters and assess model behaviour using '
         'the hydrograph, residuals, scatter plot, KGE, and NSE before running automatic '
         'calibration. All plots in this section and below are in mm/d, the units the model works '
         'in. Exports can be written in mm/d, the input units, or both.')

x1 = st.number_input('X1 Production Store Capacity (mm)', min_value=1.0, max_value=3000.0, value=500.0)
x2 = st.number_input('X2 Groundwater Exchange (mm/d)', min_value=-25.0, max_value=5.0, value=0.0)
x3 = st.number_input('X3 Routing Store Capacity (mm)', min_value=1.0, max_value=1000.0, value=100.0)
x4 = st.number_input('X4 Time Base (days)', min_value=0.5, max_value=20.0, value=2.0)

q_sim_manual = run_gr4j(rain, pet, x1, x2, x3, x4)

section_break()
st.subheader('Observed vs Simulated (mm/d) - using exploration parameters')

col1, col2 = st.columns(2)
col1.metric('KGE', f'{kge(q_obs_mmd, q_sim_manual):.3f}')
col2.metric('NSE', f'{nse(q_obs_mmd, q_sim_manual):.3f}')

fig, _ = plot_hydrograph(dates, q_obs_mmd, [(q_sim_manual, C_SIM, 'GR4J', 2)])
show(fig)

st.subheader('Residuals - using exploration parameters')
show(plot_log_residuals(dates, q_obs_mmd, q_sim_manual, C_SIM))

st.subheader('Observed vs Simulated Scatter - using exploration parameters')
show(plot_scatter(q_obs_mmd, q_sim_manual, C_SIM, 'Observed (mm/d)', 'Simulated (mm/d)'))

# %% 3. calibration
section_break()
st.subheader('3. Model Calibration')
st.write('This section calibrates X1, X2, X3 and X4, producing a suite of outputs. Choose from '
         'either the Kling-Gupta Efficiency (KGE) or Nash-Sutcliffe Efficiency (NSE) as the '
         'objective function.')

objective = st.selectbox('Objective Function', ['KGE', 'NSE'])
warmup_days = int(st.number_input('Warm-up Days', value=730, min_value=0))

st.write('Models within this distance of the best objective score are retained as behavioural '
         'models. The models within this delta value will be retained for uncertainty analyses '
         '(up to 200 model configurations).')
behavioural_delta = st.number_input('Behavioural Model Delta', value=0.05, min_value=0.001,
                                    max_value=0.50, step=0.01)

with st.expander('Advanced Calibration Settings'):
    maxiter = int(st.number_input('Maximum Iterations', value=25, min_value=1))
    popsize = int(st.number_input('Population Size', value=12, min_value=1))

if st.button('Calibrate GR4J'):

    cal_bar = st.progress(0.0, text='Calibrating GR4J...')
    generation = {'n': 0}

    def report_progress(params, convergence):
        generation['n'] += 1
        cal_bar.progress(min(generation['n'] / max(maxiter, 1), 1.0),
                         text=f"Calibrating GR4J, generation {generation['n']} of {maxiter}")

    cal_results = calibrate_gr4j(precip=rain, pet=pet, q_obs=q_obs_mmd, warmup_days=warmup_days,
                                 objective=objective, maxiter=maxiter, popsize=popsize,
                                 behavioural_delta=behavioural_delta,
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
        ensemble[i] = simulate(rain, pet, {p: row[p] for p in ['X1', 'X2', 'X3', 'X4']})
        ens_bar.progress((i + 1) / len(behavioural_df))

    ens_bar.empty()

    # Everything the rest of the app needs goes into session_state in one dict.
    # Session state survives the rerun that Streamlit triggers on any widget
    # interaction, including the download button, so the results persist.
    #
    # Only summary arrays and the ensemble members that are actually exported
    # are retained. The full ensemble is a 200 by record-length array, and
    # holding it, then rebuilding the flow duration curve array from it on every
    # rerun, is what pushes a multi-decade record past the container memory
    # limit and gets the app restarted.
    #
    # The flow units and catchment area are stored alongside the results, so
    # exports are converted with the same values the calibration used even if
    # the widgets are changed afterwards.
    obs_mask = np.isfinite(q_obs_mmd)
    fdc_ensemble = np.array([fdc(sim[obs_mask])[1] for sim in ensemble])

    st.session_state['cal'] = {
        'data_key': data_key,
        'objective': objective,
        'flow_units': flow_units,
        'area_km2': float(area_km2),
        'forcing_interpolated': forcing_interpolated,
        'dates': dates,
        'q_obs': q_obs_mmd,
        'behavioural_df': behavioural_df,
        'n_behavioural': len(behavioural_df),
        'best_params': best_params,
        'best_score': cal_results['best_score'],
        'q_cal': simulate(rain, pet, best_params),
        'q05': np.nanpercentile(ensemble, 5, axis=0),
        'q50': np.nanpercentile(ensemble, 50, axis=0),
        'q95': np.nanpercentile(ensemble, 95, axis=0),
        'fdc05': np.nanpercentile(fdc_ensemble, 5, axis=0),
        'fdc50': np.nanpercentile(fdc_ensemble, 50, axis=0),
        'fdc95': np.nanpercentile(fdc_ensemble, 95, axis=0),
        'ensemble_export': ensemble[:min(MAX_EXPORT_MODELS, len(ensemble))].copy(),
    }

    del ensemble, fdc_ensemble
    gc.collect()

# %% 3b. calibration results, rendered from session_state on every rerun
cal = st.session_state.get('cal')

if cal is None:
    st.info('Run a calibration to produce the behavioural ensemble, the gap filled series, and the '
            'downloadable workbook. Note that results are held for the current browser session '
            'only, so refreshing the page clears them.')
    st.stop()

if cal['data_key'] != data_key:
    st.warning('The input data, column selections, or catchment settings have changed since this '
               'calibration was run. The results and download below still refer to the earlier '
               'configuration. Re-run the calibration to update them.')
    if st.button('Clear calibration results'):
        del st.session_state['cal']
        st.rerun()

cal_dates = cal['dates']
q_obs = cal['q_obs']
cal_units = cal['flow_units']
cal_area = cal['area_km2']
behavioural_df = cal['behavioural_df']
best_params = cal['best_params']
q_cal = cal['q_cal']
q05, q50, q95 = cal['q05'], cal['q50'], cal['q95']

st.write(f"Behavioural Models Retained: {cal['n_behavioural']}")
st.write(f"Best {cal['objective']}: {cal['best_score']:.3f}")
st.dataframe(behavioural_df.head(20))

st.subheader('Behavioural Parameter Summary')
st.dataframe(behavioural_df[['X1', 'X2', 'X3', 'X4', 'Score']].describe())

st.subheader('Calibration Results')
st.json(best_params)

for param, (lo, hi) in PARAM_BOUNDS.items():
    tol = 0.001 * (hi - lo)
    if best_params[param] <= lo + tol:
        st.warning(f'{param} reached lower bound')
    if best_params[param] >= hi - tol:
        st.warning(f'{param} reached upper bound')

col1, col2 = st.columns(2)
col1.metric('Best model KGE', f'{kge(q_obs, q_cal):.3f}')
col2.metric('Best model NSE', f'{nse(q_obs, q_cal):.3f}')

# hydrograph
st.subheader('Behavioural Ensemble Hydrograph')
fig_cal, ax = plot_hydrograph(cal_dates, q_obs, [(q50, C_CAL, 'Behavioural median', 1.5)])
ax.fill_between(cal_dates, q05, q95, color=C_CAL, alpha=0.25, label='5-95% behavioural range')
ax.legend()
show(fig_cal)

# residuals
st.subheader('Behavioural Median Residuals')
show(plot_log_residuals(cal_dates, q_obs, q50, C_CAL))

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
show(fig_fdc)

# scatter
st.subheader('Best Model Scatter Plot')
show(plot_scatter(q_obs, q_cal, C_CAL, 'Observed (mm/d)', 'Calibrated GR4J (mm/d)'))

# parameter distributions
st.subheader('Behavioural Parameter Distributions')
fig_hist = plt.figure(figsize=(17 / 2.54, 12 / 2.54))

for i, (param, colour) in enumerate(zip(['X1', 'X2', 'X3', 'X4'], C_PARAM)):
    ax = fig_hist.add_subplot(2, 2, i + 1)
    ax.hist(behavioural_df[param], bins=20, color=colour, edgecolor='black', linewidth=0.5)
    ax.axvline(best_params[param], color='black', linestyle='--', linewidth=1.5,
               label=f'{best_params[param]:.2f}')
    ax.set_title(param)
    ax.set_xlabel(PARAM_LABELS[param])
    ax.set_ylabel('Count')
    ax.legend(fontsize=7, frameon=False)

fig_hist.subplots_adjust(hspace=0.45, wspace=0.20)
show(fig_hist)

with st.expander('Advanced Parameter Diagnostics'):
    st.subheader('Behavioural Parameter Correlation Matrix')
    st.dataframe(behavioural_df[['X1', 'X2', 'X3', 'X4', 'Score']].corr())
    st.caption('The behavioural set is drawn from the differential evolution trajectory rather '
               'than a random sample of parameter space, so the spread reflects local sensitivity '
               'around the optimum rather than a formal predictive uncertainty.')

# %% 4. gap filling
section_break()
st.subheader('4. Gap Filling')

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
ax.legend()
show(fig_gap)

# %% 5. export
section_break()
st.subheader('5. Download Results')

native_label = f'{cal_units} only (as uploaded)'
both_label = f'Both mm/d and {cal_units}'

if cal_units == 'mm/d':
    st.write('The flow column was uploaded in mm/d, so the export is in mm/d and no conversion is '
             'applied.')
    export_choice = 'mm/d only'
else:
    st.write(f'GR4J works in mm/d. Flow was uploaded in {cal_units} and can be written back in '
             f'either unit system, converted using the {cal_area:g} km² catchment area recorded at '
             'calibration time.')
    export_choice = st.radio('Export units', [both_label, 'mm/d only', native_label], index=0)

include_mmd = export_choice in ('mm/d only', both_label)
include_native = export_choice in (native_label, both_label) and cal_units != 'mm/d'

# Ensemble hydrographs are written in one unit system only, to keep that sheet
# to 30 columns rather than 60 on a multi-decade record.
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
                                     else cal_units if c.endswith(UNIT_SUFFIX[cal_units])
                                     and cal_units != 'mm/d'
                                     else 'mm/d'
                                     for c in output_df.columns]}),
             hide_index=True)

workbook_bytes = build_workbook(output_df, behavioural_df, ensemble_df, metadata_df)

st.download_button(
    label=f'Download Results Workbook ({sheet_units})',
    data=workbook_bytes,
    file_name=f'gr4j_gapfill_results_{file_tag}.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    key='download_workbook',
)

st.caption(f'Four sheets: GapFilled in {sheet_units}, BehaviouralModels holding the '
           f'{cal["n_behavioural"]} retained parameter sets, EnsembleHydrographs holding '
           f'{len(cal["ensemble_export"])} members in {ensemble_units}, and Metadata recording the '
           'units, catchment area and calibration settings. Switch the export units above and '
           'download again if you want a second copy in the other unit system.')
