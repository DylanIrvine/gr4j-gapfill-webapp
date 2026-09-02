# tests/_reference_params.py
# Parameter sets used by both the fixture generator and the regression tests, so
# the two cannot drift apart.

GR_PARAM_SETS = {
    "GR4J": [
        {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7},
        {"X1": 1200.0, "X2": 1.0, "X3": 250.0, "X4": 5.5},
    ],
    "GR5J": [
        {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.3},
        {"X1": 1200.0, "X2": 1.0, "X3": 250.0, "X4": 5.5, "X5": 0.0},
    ],
    "GR6J": [
        {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.3, "X6": 15.0},
        {"X1": 1200.0, "X2": 1.0, "X3": 250.0, "X4": 5.5, "X5": 0.0, "X6": 8.0},
    ],
}

SIMHYD_PARAM_SETS = [
    # INSC, COEFF, SQ, SMSC, SUB, CRAK, K
    {"INSC": 1.0, "COEFF": 200.0, "SQ": 2.0, "SMSC": 300.0,
     "SUB": 0.5, "CRAK": 0.5, "K": 0.1},
    {"INSC": 3.5, "COEFF": 80.0, "SQ": 4.0, "SMSC": 120.0,
     "SUB": 0.2, "CRAK": 0.8, "K": 0.02},
    {"INSC": 0.5, "COEFF": 350.0, "SQ": 1.0, "SMSC": 600.0,
     "SUB": 0.9, "CRAK": 0.1, "K": 0.25},
]
