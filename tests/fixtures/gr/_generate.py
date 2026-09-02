"""Regenerate the GR regression fixtures from the CURRENT core.models code.

Run this only deliberately, when an intended change to the GR simulators has been
reviewed. It pins today's output so an accidental future change is caught.

    python tests/fixtures/gr/_generate.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.models import simulate  # noqa: E402
from tests.conftest import _synthetic_forcing  # noqa: E402
from tests._reference_params import GR_PARAM_SETS  # noqa: E402

HERE = pathlib.Path(__file__).parent


def main():
    rain, pet = _synthetic_forcing()
    for model, sets in GR_PARAM_SETS.items():
        for i, params in enumerate(sets):
            q = simulate(rain, pet, params, model=model)
            out = HERE / f"{model}_{i}.npy"
            np.save(out, q)
            print(f"wrote {out.name}  ({len(q)} days, mean {q.mean():.4f} mm/d)")


if __name__ == "__main__":
    main()
