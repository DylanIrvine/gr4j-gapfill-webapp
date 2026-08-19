import streamlit as st
import numpy as np

from core.gapfill import identify_gaps

st.title('GR4J Gap Filling Tool')

example = [1, 2, np.nan, np.nan, 5]

gaps = identify_gaps(example)

st.write('Gap detection test')
st.write(gaps)
