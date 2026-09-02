"""Naive, independent SIMHYD reference for testing core.models._simhyd_loop.

Transcribed line for line from hydromad's SIMHYD:

  * src/simhyd.cpp  (the compiled path, hydromad's default)
  * R/simhyd.R      (the pure-R fallback; identical arithmetic)

hydromad repository cloned 2026-09-03; model after Chiew et al. (2009),
"Estimating climate change impact on runoff across southeast Australia",
Water Resources Research 45, W10414.

This file exists so the test compares the production kernel against a second
implementation written from the same source but with no shared code. It is
deliberately a plain Python loop: readable, slow, no numba.

Preserved hydromad quirks (see also the comment in core/models.py):
  * soil-store overflow is discarded, not recharged (the `REC += SMS - SMSC`
    line runs after `SMS = SMSC`, so it adds zero) -- unless overflow_to_gw is
    set, which restores the Chiew et al. (2009) path;
  * no store is clipped at zero and the total is not clipped at zero;
  * etmult is 1.0 here -- this app supplies real PET in mm/d.
"""

import numpy as np


def simhyd_reference(precip, pet, INSC, COEFF, SQ, SMSC, SUB, CRAK, K,
                     GWt0=0.0, SMSt0=0.5, overflow_to_gw=False):
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    n = len(precip)

    total = np.empty(n)
    surface = np.empty(n)
    interflow = np.empty(n)
    baseflow = np.empty(n)

    SMSt1 = SMSt0 * SMSC
    GWt1 = float(GWt0)

    for t in range(n):
        P = precip[t]
        E = pet[t]

        IMAX = min(INSC, E)
        INT = min(IMAX, P)
        INR = P - INT

        RMO = min(COEFF * np.exp(-SQ * SMSt1 / SMSC), INR)
        IRUN = INR - RMO
        SRUN = SUB * SMSt1 / SMSC * RMO
        REC = CRAK * SMSt1 / SMSC * (RMO - SRUN)
        SMF = RMO - SRUN - REC

        POT = E - INT
        ET = min(10.0 * SMSt1 / SMSC, POT)

        SMS = SMSt1 + SMF - ET
        if SMS > SMSC:
            if overflow_to_gw:
                REC = REC + SMS - SMSC      # Chiew et al. 2009: excess recharges GW
            SMS = SMSC
            # hydromad default: the `REC += SMS - SMSC` here would add zero
        SMSt1 = SMS

        BAS = K * GWt1
        GW = GWt1 + REC - BAS
        GWt1 = GW

        surface[t] = IRUN
        interflow[t] = SRUN
        baseflow[t] = BAS
        total[t] = IRUN + SRUN + BAS

    return {"total": total, "surface": surface,
            "interflow": interflow, "baseflow": baseflow}
