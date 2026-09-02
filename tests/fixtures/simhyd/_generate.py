"""Regenerate the SIMHYD regression fixtures from the CURRENT core.models code.

Run only deliberately, after a reviewed change to the SIMHYD kernel.

    python tests/fixtures/simhyd/_generate.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.models import simulate_simhyd  # noqa: E402
from tests.conftest import _synthetic_forcing  # noqa: E402
from tests._reference_params import SIMHYD_PARAM_SETS  # noqa: E402

HERE = pathlib.Path(__file__).parent


def main():
    rain, pet = _synthetic_forcing()
    for i, params in enumerate(SIMHYD_PARAM_SETS):
        q = simulate_simhyd(rain, pet, params)
        out = HERE / f"simhyd_{i}.npy"
        np.save(out, q)
        print(f"wrote {out.name}  ({len(q)} days, mean {q.mean():.4f} mm/d)")


if __name__ == "__main__":
    main()
