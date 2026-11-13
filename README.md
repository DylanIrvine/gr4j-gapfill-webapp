# HydroSTITCH

Hydrological Signatures, Time-series Infilling and Tools for Catchment Hydrology

https://gr4j-gapfill-webapp.streamlit.app/

Calibrates a lumped rainfall-runoff model to a daily streamflow record, uses it
to fill the gaps, and runs a set of common hydrological analyses on the result.
No coding needed: upload a csv, follow the steps, download the outputs. Built to
sit alongside HydroSILO and the Hydrochem Workbench.

Written by Dylan Irvine, Charles Darwin University.

## Run

    pip install -r requirements.txt
    streamlit run app.py

Tests: `pip install -r requirements-dev.txt` then `pytest`.

## What it does

1. **Upload.** One csv with a date column, daily rainfall and PET (mm/d) and
   daily streamflow (m3/s, ML/d or mm/d). The columns are picked from dropdowns
   and the date format is detected, or can be set.
2. **Models.** GR4J, GR5J, GR6J and SIMHYD. A manual simulation step lets the
   user try parameters by hand before calibrating.
3. **Calibration.** Differential evolution against KGE or NSE, with an optional
   flow transform (none, square root, log, inverse) to weight high or low flows,
   and an optional calibration/validation split. The good parameter sets found
   along the way are kept as a behavioural ensemble, which gives the P05, P50
   and P95 hydrographs.
4. **Gap filling.** Five methods: the ensemble median, the median snapped to the
   observations either side of a gap, a Gaussian process, an AR(1) model and an
   ensemble Kalman smoother.
5. **Analysis (optional).** Model evaluation against the observed record,
   Lyne-Hollick baseflow separation and recession analysis, water year, seasonal
   and monthly summaries, flow duration and regime indices, cease-to-flow
   spells, trends, and rainfall and runoff anomalies.
6. **Export.** A workbook (gap-filled series, behavioural parameter sets,
   ensemble hydrographs and a metadata sheet recording every setting), or a zip
   that adds csv products and every figure. Each figure also has its own
   download button, at 300 or 600 dpi.

## Privacy

Uploaded data are processed in memory on the server and are not written to a
database or to permanent storage. Results are cached in server memory for up to
an hour so the page redraws quickly, then discarded. The one thing the app
records is a running total of completed calibration runs (`core/usage.py`). For
that total to survive a redeploy on Streamlit Cloud, add Upstash Redis secrets:

    [redis]
    rest_url = "https://....upstash.io"
    rest_token = "..."

The key defaults to `hydrostitch:runs`. `.streamlit/config.toml` sets the theme
and turns off Streamlit's browser usage statistics.

## Layout

    app.py             Streamlit front end
    core/models.py     GR4J, GR5J, GR6J and SIMHYD
    core/calibration.py  differential evolution and the behavioural ensemble
    core/gapfill.py    the five gap-filling methods
    core/metrics.py    KGE, NSE and the flow transforms
    core/baseflow.py, signatures.py, indices.py, rainfall.py, evaluation.py
                       the analyses
    core/usage.py      the run counter
    docs/              logo, favicon and model structure diagrams
    tests/             pytest: models against reference values, gap filling,
                       dates, and a headless run of the app

`core/` has no Streamlit in it and can be used from Spyder directly.

## Deploying

After pushing an update, reboot the app from "Manage app" on Streamlit Cloud.
Streamlit can otherwise run the new `app.py` against old copies of the `core`
modules. Bump `APP_VERSION` in `app.py` with each release.

## How to cite

Irvine, D.J. (2026). HydroSTITCH: Hydrological Signatures, Time-series Infilling
and Tools for Catchment Hydrology (version 1.0) [Web application]. Charles
Darwin University. https://gr4j-gapfill-webapp.streamlit.app/

Please also cite the model you used (references are listed in the app).

## Licence

GPL-2 (see `LICENSE`). The GR models here were transcribed from the Fortran in
the airGR R package, which is GPL-2, and SIMHYD from the hydromad R package, so
this project is released under the same terms. You are free to use, modify and
share it; if you distribute a modified version you must release its source
under the GPL as well. There is no warranty: the outputs are an aid to
interpretation and should be checked before they are relied on.

## Credits

- **airGR**, Coron et al. (2017), Environmental Modelling and Software 94,
  166-171. Source of the GR4J, GR5J and GR6J code (transcribed and verified
  against it).
- **hydromad**, Andrews et al. (2011), Environmental Modelling and Software
  26(10), 1171-1185. Source of the SIMHYD code (transcribed).
- SciPy (differential evolution), scikit-learn (Gaussian process), Numba,
  Streamlit, pandas, NumPy and Matplotlib.
