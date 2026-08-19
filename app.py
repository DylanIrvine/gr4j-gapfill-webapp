import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from core.metrics import kge
from core.metrics import nse
from core.gr4j import simulate
from core.units import cumecs_to_mmd
from core.calibration import calibrate_gr4j

#%% plot settings
plt.style.use('default')
plt.rc('axes', linewidth=0.5)
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'legend.labelspacing': 0.1})
plt.matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

#%% helper functions
def section_break():
    st.markdown('---')


#%% Main code
st.title('GR4J Gap Filling Tool')

st.write('The GR4J model (Modèle du Génie Rural à 4 paramètres Journalier) is a simple, lumped conceptual rainfall-runoff model. It simulates daily streamflow using only catchment-averaged daily precipitation and potential evapotranspiration data. The model was originally published Perrin et al. (2003).\n\n The  gr4j-gapfill-webapp provides users with an online version of the tool that can be applied with no coding required. Simply upload your file, following the workflow, and you will have calibrated models and/or gap-filled hydrographs.\n\n Original reference \n\nPerrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement of a parsimonious model for streamflow simulation." Journal of Hydrology 279, no. 1 (2003): 275-289.')
    
st.subheader('1. Upload Data')

st.write('Upload a csv containing date, rainfall, PET, and streamflow. Dates must be in dd/mm/yyyy format. Rain and PET must be in mm/d, but flow can be m3/s, or ML/d, or mm/d.')

uploaded_file = st.file_uploader(
    'Upload CSV',
    type=['csv']
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    section_break()
    st.subheader('Data Preview')

    st.dataframe(df.head())

    columns = df.columns.tolist()

    date_col = st.selectbox(
        'Date Column',
        columns
    )

    rain_col = st.selectbox(
        'Rain Column',
        columns
    )

    pet_col = st.selectbox(
        'PET Column',
        columns
    )

    flow_col = st.selectbox(
        'Flow Column',
        columns
    )
    section_break()
    st.subheader('Catchment Information')

    area_km2 = st.number_input(
        'Catchment Area (km²)',
        min_value=0.001,
        value=1000.0,
        step=1.0
    )

    flow_units = st.selectbox(
        'Flow Units',
        [
            'm3/s',
            'ML/d',
            'mm/d'
        ]
    )

    try:

        dates = pd.to_datetime(
            df[date_col],
            dayfirst=True
        )

        rain = df[rain_col].to_numpy()

        pet = df[pet_col].to_numpy()

        flow = df[flow_col].to_numpy()

        if flow_units == 'm3/s':

            q_obs_mmd = cumecs_to_mmd(
                flow,
                area_km2
            )

        elif flow_units == 'mm/d':

            q_obs_mmd = flow

        else:

            q_obs_mmd = flow / area_km2
            
        section_break()
        st.subheader('Data Summary')

        st.write(
            f'Record Length: {len(df)} days'
        )

        st.write(
            f'Flow Missing Values: {pd.isna(flow).sum()}'
        )

        st.write(
            f'Rain Missing Values: {pd.isna(rain).sum()}'
        )

        st.write(
            f'PET Missing Values: {pd.isna(pet).sum()}'
        )

        st.write(
            f'Record starts: {dates.min()}'
        )

        st.write(
            f'Record ends: {dates.max()}'
        )

        section_break()
        st.subheader('2. Manual GR4J Simulation')
        
        st.write(
            'Use this section to manually adjust the GR4J parameters and assess model behaviour using the hydrograph, residuals, scatter plot, KGE, and NSE before running automatic calibration.'
        )

        x1 = st.number_input(
            'X1 Production Store Capacity (mm)',
            min_value=1.0,
            max_value=3000.0,
            value=500.0
        )

        x2 = st.number_input(
            'X2 Groundwater Exchange (mm/d)',
            min_value=-20.0,
            max_value=5.0,
            value=0.0
        )

        x3 = st.number_input(
            'X3 Routing Store Capacity (mm)',
            min_value=1.0,
            max_value=1000.0,
            value=100.0
        )

        x4 = st.number_input(
            'X4 Time Base (days)',
            min_value=0.5,
            max_value=20.0,
            value=2.0
        )

        params = {
            'X1': x1,
            'X2': x2,
            'X3': x3,
            'X4': x4
        }

        q_sim_uploaded = simulate(
            rain,
            pet,
            params
        )

        kge_value = kge(
            q_obs_mmd,
            q_sim_uploaded
        )
        
        nse_value = nse(
            q_obs_mmd,
            q_sim_uploaded
        )
        
        df_sim = pd.DataFrame({
            'Date': dates,
            'Observed_mm_d': q_obs_mmd,
            'Simulated_mm_d': q_sim_uploaded
        })

        section_break()
        st.subheader('Observed vs Simulated (mm/d) - using exploration parameters')

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                'KGE',
                f'{kge_value:.3f}'
            )
        
        with col2:
            st.metric(
                'NSE',
                f'{nse_value:.3f}'
            )
        
        fig = plt.figure(
            figsize=(17/2.54, 8/2.54)
        )
        
        ax = plt.axes(
            [0.10, 0.15, 0.85, 0.75]
        )
        
        ax.plot(
            dates,
            q_obs_mmd,
            color='black',
            alpha=0.6,
            linewidth=1,
            label='Observed'
        )
        
        ax.plot(
            dates,
            q_sim_uploaded,
            color='royalblue',
            alpha=0.6,
            linewidth=1,
            label='GR4J'
        )
        
        ax.legend()
        
        ax.set_ylabel('Flow (mm/d)')
        ax.set_xlabel('Date')
        
        st.pyplot(fig)

        residuals = (
            q_obs_mmd
            - q_sim_uploaded
        )

        eps = 0.01
        
        log_residuals = (
            np.log(q_obs_mmd + eps)
            -
            np.log(q_sim_uploaded + eps)
        )
                
        fig_res = plt.figure(
            figsize=(17/2.54, 6/2.54)
        )
        
        ax = plt.axes(
            [0.10, 0.18, 0.85, 0.72]
        )
        
        ax.plot(
            dates,
            log_residuals,
            color='firebrick',
            linewidth=0.8
        )
        
        ax.axhline(
            0,
            color='black',
            linewidth=0.8
        )
        
        ax.set_ylabel(
            'Log Residual'
        )
        
        ax.set_xlabel(
            'Date'
        )
        
        st.subheader('Residuals - using exploration parameters')
        
        st.pyplot(fig_res)

        st.subheader('Observed vs Simulated Scatter - using exploration parameter')
        
        mask = (
            np.isfinite(q_obs_mmd)
            & np.isfinite(q_sim_uploaded)
            & (q_obs_mmd > 0)
            & (q_sim_uploaded > 0)
        )
        
        fig_scatter = plt.figure(figsize=(8/2.54, 8/2.54))
        ax = fig_scatter.add_axes([0.18, 0.18, 0.72, 0.72])
        
        ax.scatter(
            q_obs_mmd[mask],
            q_sim_uploaded[mask],
            s=5,
            alpha=0.3
        )
        
        lim_lo = min(
            np.nanmin(q_obs_mmd[mask]),
            np.nanmin(q_sim_uploaded[mask])
        )
        
        lim_hi = max(
            np.nanmax(q_obs_mmd[mask]),
            np.nanmax(q_sim_uploaded[mask])
        )
        
        ax.plot(
            [lim_lo, lim_hi],
            [lim_lo, lim_hi],
            'k--'
        )
        
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        
        ax.set_xlabel('Observed (mm/d)')
        ax.set_ylabel('Simulated (mm/d)')
        
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        
        st.pyplot(fig_scatter)
        
        if np.isfinite(q_obs_mmd).sum() < 730:
        
            st.warning(
                'Less than two years of observed flow available.'
            )
            
        section_break()
        st.subheader('3. Model Calibration')
        st.write('This section calibrates X1, X2, X3 and X4, producing a suite of outputs. Set your objective functionn and the number of warm up days below.')

        objective = st.selectbox(
            'Objective Function',
            [
                'KGE',
                'NSE'
            ]
        )

        warmup_days = st.number_input(
            'Warm-up Days',
            value=730
        )

        st.write( 'Models within this distance of the best objective score are retained as behavioural models.')
        behavioural_delta = st.number_input(
            'Behavioural Model Delta',
            value=0.05,
            min_value=0.001,
            max_value=0.50,
            step=0.01
        )
        
        with st.expander(
            'Advanced Calibration Settings'
        ):

            maxiter = st.number_input(
                'Maximum Iterations',
                value=25,
                min_value=1
            )

            popsize = st.number_input(
                'Population Size',
                value=12,
                min_value=1
            )

        run_calibration = st.button(
            'Calibrate GR4J'
        )

        if run_calibration:

            with st.spinner(
                'Calibrating GR4J...'
            ):

                cal_results = calibrate_gr4j(
                    precip=rain,
                    pet=pet,
                    q_obs=q_obs_mmd,
                    warmup_days=warmup_days,
                    objective=objective,
                    maxiter=maxiter,
                    popsize=popsize,
                    behavioural_delta=behavioural_delta
                )
            
            best_params = cal_results['best_params']
            best_score = cal_results['best_score']
            behavioural_df = cal_results['behavioural_df']

            st.write(f'Behavioural Models Retained: {len(behavioural_df)}')
            st.write(f'Best {objective}: {best_score:.3f}')
            st.dataframe(behavioural_df.head(20))

            st.write( f'Behavioural Models Retained: {len(behavioural_df)}')
            st.write( f'Best {objective}: {best_score:.3f}')

            st.subheader(
                'Calibration Results'
            )

            st.json(best_params)

            if best_params['X1'] <= 1.01:
                st.warning('X1 reached lower bound')
            
            if best_params['X1'] >= 2999:
                st.warning('X1 reached upper bound')
            
            if best_params['X2'] <= -19.99:
                st.warning('X2 reached lower bound')
            
            if best_params['X2'] >= 4.99:
                st.warning('X2 reached upper bound')
            
            if best_params['X3'] <= 1.01:
                st.warning('X3 reached lower bound')
            
            if best_params['X3'] >= 999:
                st.warning('X3 reached upper bound')
            
            if best_params['X4'] <= 0.51:
                st.warning('X4 reached lower bound')
            
            if best_params['X4'] >= 19.9:
                st.warning('X4 reached upper bound')            
            
            q_cal = simulate(
                rain,
                pet,
                best_params
            )

            kge_cal = kge(
                q_obs_mmd,
                q_cal
            )

            nse_cal = nse(
                q_obs_mmd,
                q_cal
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    'Calibrated KGE',
                    f'{kge_cal:.3f}'
                )

            with col2:

                st.metric(
                    'Calibrated NSE',
                    f'{nse_cal:.3f}'
                )

            fig_cal = plt.figure(
                figsize=(17/2.54, 8/2.54)
            )

            ax = plt.axes(
                [0.10, 0.15, 0.85, 0.75]
            )

            ax.plot(
                dates,
                q_obs_mmd,
                color='black',
                alpha=0.6,
                linewidth=1,
                label='Observed'
            )

            ax.plot(
                dates,
                q_cal,
                color='#0DB14B',
                alpha=0.6,
                linewidth=1,
                label='Calibrated GR4J'
            )

            ax.legend()

            ax.set_ylabel(
                'Flow (mm/d)'
            )

            ax.set_xlabel(
                'Date'
            )

            st.subheader(
                'Calibrated Hydrograph'
            )

            st.pyplot(fig_cal)        

            # calibrated residual plot
            cal_log_residuals = (
                np.log(q_obs_mmd + eps)
                -
                np.log(q_cal + eps)
            )
            
            fig_cal_res = plt.figure(figsize=(17/2.54, 6/2.54))
            ax = fig_cal_res.add_axes([0.10, 0.18, 0.85, 0.72])
            
            ax.plot(
                dates,
                cal_log_residuals,
                color='#0DB14B'
            )
            
            ax.axhline(
                0,
                color='black',
                linewidth=0.8
            )
            
            ax.set_ylabel('Log Residual')
            ax.set_xlabel('Date')
            
            st.subheader('Calibrated Residuals')
            
            st.pyplot(fig_cal_res)
            
            # calibrated scatter plot
            st.subheader('Calibrated Scatter Plot')
            
            mask = (
                np.isfinite(q_obs_mmd)
                & np.isfinite(q_cal)
                & (q_obs_mmd > 0)
                & (q_cal > 0)
            )
            
            fig_cal_scatter = plt.figure(figsize=(8/2.54, 8/2.54))
            ax = fig_cal_scatter.add_axes([0.10, 0.18, 0.85, 0.72])
            
            ax.scatter(
                q_obs_mmd[mask],
                q_cal[mask],
                s=5,
                alpha=0.3
            )
            
            lim_lo = min(
                np.nanmin(q_obs_mmd[mask]),
                np.nanmin(q_cal[mask])
            )
            
            lim_hi = max(
                np.nanmax(q_obs_mmd[mask]),
                np.nanmax(q_cal[mask])
            )
            
            ax.plot(
                [lim_lo, lim_hi],
                [lim_lo, lim_hi],
                'k--'
            )
            
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
            
            ax.set_xlabel('Observed (mm/d)')
            ax.set_ylabel('Calibrated GR4J (mm/d)')
            
            st.pyplot(fig_cal_scatter)
            
            
            
            # calibrated flow duration curve
            def fdc(q):
            
                q = q[np.isfinite(q)]
                q = np.sort(q)[::-1]
            
                exceedance = (
                    np.arange(1, len(q)+1)
                    /
                    (len(q)+1)
                    * 100
                )
            
                return exceedance, q
            
            ex_obs, q_obs_fdc = fdc(q_obs_mmd)
            ex_cal, q_cal_fdc = fdc(q_cal)
            
            fig_fdc = plt.figure(figsize=(10/2.54, 8/2.54))
            ax = fig_fdc.add_axes([0.15, 0.15, 0.75, 0.75])
            
            ax.plot(ex_obs, q_obs_fdc, label='Observed')
            ax.plot(ex_cal, q_cal_fdc, label='Calibrated')
            
            ax.set_yscale('log')
            
            ax.set_xlabel('Exceedance (%)')
            ax.set_ylabel('Flow (mm/d)')
            
            ax.legend()
            
            st.subheader('Flow Duration Curve')
            
            st.pyplot(fig_fdc)
    
    except Exception as e:
    
        st.error(str(e))
