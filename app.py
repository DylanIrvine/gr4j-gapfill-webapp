import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from core.metrics import kge
from core.metrics import nse
from core.gr4j import simulate
from core.units import cumecs_to_mmd
from core.calibration import calibrate_gr4j

st.title('GR4J Gap Filling Tool')

st.subheader('Upload Data')

uploaded_file = st.file_uploader(
    'Upload CSV',
    type=['csv']
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

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

        st.subheader('GR4J Parameters')

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

        st.subheader(
            'Observed vs Simulated (mm/d)'
        )

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
        
        st.subheader(
            'Residuals'
        )
        
        st.pyplot(fig_res)

        if np.isfinite(q_obs_mmd).sum() < 730:
        
            st.warning(
                'Less than two years of observed flow available.'
            )

            fig_scatter = plt.figure(
                figsize=(8/2.54, 8/2.54)
            )
            
            ax = plt.axes(
                [0.18,0.18,0.72,0.72]
            )
            
            mask = (
                np.isfinite(q_obs_mmd)
                &
                np.isfinite(q_sim_uploaded)
            )
            
            ax.scatter(
                q_obs_mmd[mask],
                q_sim_uploaded[mask],
                s=5,
                alpha=0.3
            )
            
            lim = np.nanmax(
                [
                    np.nanmax(q_obs_mmd),
                    np.nanmax(q_sim_uploaded)
                ]
            )
            
            ax.plot(
                [0, lim],
                [0, lim],
                'k--'
            )
            
            ax.set_xlabel(
                'Observed (mm/d)'
            )
            
            ax.set_ylabel(
                'Simulated (mm/d)'
            )
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            st.pyplot(fig_scatter)

        
        st.subheader('Calibration')

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

        run_calibration = st.button(
            'Calibrate GR4J'
        )


        if run_calibration:

            with st.spinner(
                'Calibrating GR4J...'
            ):

                best_params = calibrate_gr4j(
                    precip=rain,
                    pet=pet,
                    q_obs=q_obs_mmd,
                    warmup_days=warmup_days,
                    objective=objective
                )

            st.subheader(
                'Calibration Results'
            )

            st.json(best_params)

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
                color='green',
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
    
    except Exception as e:
    
        st.error(str(e))
