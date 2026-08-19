import streamlit as st
import numpy as np
import pandas as pd

from core.gr4j import simulate

st.title('GR4J Gap Filling Tool')

# synthetic climate series

n_days = 365

precip = np.full(n_days, 5.0)
pet = np.full(n_days, 3.0)

params = {
    'X1': 500,
    'X2': 0,
    'X3': 100,
    'X4': 2
}

q_sim = simulate(
    precip,
    pet,
    params
)

df = pd.DataFrame({
    'Day': np.arange(n_days),
    'Qsim_mm_d': q_sim
})

st.line_chart(
    df.set_index('Day')
)

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

    # Everything below here should also
    # be inside the same block

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

    st.write(
        f'Catchment Area: {area_km2:.1f} km²'
    )

    st.write(
        f'Flow Units: {flow_units}'
    )

    try:

        dates = pd.to_datetime(
            df[date_col]
        )

        rain = df[rain_col].to_numpy()

        pet = df[pet_col].to_numpy()

        flow = df[flow_col].to_numpy()

        #---------------------------------
        # Test GR4J on uploaded data
        #---------------------------------

        params = {
            'X1': 500,
            'X2': 0,
            'X3': 100,
            'X4': 2
        }

        q_sim_uploaded = simulate(
            rain,
            pet,
            params
        )

        st.subheader('Uploaded Data GR4J Test')

        df_sim = pd.DataFrame({
            'Date': dates,
            'Qsim_mm_d': q_sim_uploaded
        })

        st.line_chart(
            df_sim.set_index('Date')
        )

        
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

    except Exception as e:

        st.error(str(e))
