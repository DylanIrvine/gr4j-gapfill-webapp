#==== The main GR4J, 5J and 6J functions
import math
import numpy as np
from math import tanh

def _s1(tt, x4):
    if tt <= 0:
        return 0.0
    return (tt / x4) ** 2.5 if tt < x4 else 1.0


def _s2(tt, x4):
    if tt <= 0:
        return 0.0
    if tt < x4:
        return 0.5 * (tt / x4) ** 2.5
    if tt < 2 * x4:
        return 1.0 - 0.5 * (2 - tt / x4) ** 2.5
    return 1.0


def simulate(precip, pet, params):
    """
    Daily GR4J. precip and pet in mm/d, returns simulated runoff in mm/d.
    params is a dict or 4-sequence of X1..X4.
    G5: stores initialised at 0.3*X1 and 0.5*X3 rather than zero.
    """
    if not isinstance(params, dict):
        params = dict(zip(("X1", "X2", "X3", "X4"), params))
    X1, X2, X3, X4 = params["X1"], params["X2"], params["X3"], params["X4"]
    n1 = int(math.ceil(X4))
    n2 = int(math.ceil(2.0 * X4))
    o1 = [_s1(t, X4) - _s1(t - 1, X4) for t in range(1, n1 + 1)]
    o2 = [_s2(t, X4) - _s2(t - 1, X4) for t in range(1, n2 + 1)]
    UH1 = [0.0] * n1
    UH2 = [0.0] * n2
    S = 0.3 * X1
    R = 0.5 * X3
    out = np.empty(len(precip))
    for i in range(len(precip)):
        P = precip[i]
        E = pet[i]
        if P > E:
            net_evap = 0.0
            snp = min((P - E) / X1, 13.0)
            th = tanh(snp)
            prod = (X1 * (1 - (S / X1) ** 2) * th) / (1 + S / X1 * th)
            pat = P - E - prod
        else:
            sne = min((E - P) / X1, 13.0)
            th = tanh(sne)
            net_evap = S * ((2 - S / X1) * th) / (1 + (1 - S / X1) * th)
            prod = 0.0
            pat = 0.0
        S = S - net_evap + prod
        perc = S / (1 + (S / 2.25 / X1) ** 4) ** 0.25
        pat = pat + (S - perc)
        S = perc
        for j in range(n1 - 1):
            UH1[j] = UH1[j + 1] + o1[j] * pat
        UH1[-1] = o1[-1] * pat
        for j in range(n2 - 1):
            UH2[j] = UH2[j + 1] + o2[j] * pat
        UH2[-1] = o2[-1] * pat
        gwe = X2 * (R / X3) ** 3.5
        R = max(0.0, R + UH1[0] * 0.9 + gwe)
        R2 = R / (1 + (R / X3) ** 4) ** 0.25
        QR = R - R2
        R = R2
        QD = max(0.0, UH2[0] * 0.1 + gwe)
        out[i] = QR + QD
    return out

