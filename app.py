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



st.title('GR4J Gap Filling Tool')

uploaded_file = st.file_uploader(
    'Upload CSV',
    type=['csv']
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader('Preview')

    st.dataframe(df.head())

    st.subheader('Columns')

    st.write(df.columns.tolist())
