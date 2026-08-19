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

    df_upload = pd.read_csv(uploaded_file)

    st.subheader('Data Preview')

    st.dataframe(
        df_upload.head()
    )

    st.subheader('Column Selection')

    columns = df_upload.columns.tolist()

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

    st.write('Selected Columns')

    st.write({
        'Date': date_col,
        'Rain': rain_col,
        'PET': pet_col,
        'Flow': flow_col
    })
