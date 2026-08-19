import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.metrics import kge
from core.metrics import nse
from core.gr4j import simulate
from core.units import cumecs_to_mmd

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

        df_sim = pd.DataFrame({
            'Date': dates,
            'Observed_mm_d': q_obs_mmd,
            'Simulated_mm_d': q_sim_uploaded
        })

        st.subheader(
            'Observed vs Simulated (mm/d)'
        )

        st.line_chart(
            df_sim.set_index('Date')
        )

    except Exception as e:

        st.error(str(e))
