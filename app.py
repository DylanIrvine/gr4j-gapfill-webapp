import streamlit as st
import numpy as np

from core.gapfill import identify_gaps
from core.units import cumecs_to_mmd

st.title('GR4J Gap Filling Tool')

example = [1, 2, np.nan, np.nan, 5]

gaps = identify_gaps(example)

st.subheader('Gap Detection')

st.write(gaps)

st.subheader('Unit Conversion')

q_mmd = cumecs_to_mmd(
    q=10,
    area_km2=100
)

st.write(
    f'10 m³/s over 100 km² = {q_mmd:.2f} mm/d'
)
