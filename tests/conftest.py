# tests/conftest.py
# Shared fixtures. The synthetic forcing is deterministic (seeded) so that
# reference outputs committed under tests/fixtures/ stay reproducible.

import numpy as np
import pytest


def _synthetic_forcing(n_days=5 * 365 + 2, seed=12345):
    """A plausible daily rainfall and PET series for a temperate catchment.

    Rain is an intermittent gamma process, PET a smooth seasonal cycle with a
    little noise. Values are in mm/d. Nothing here is calibrated to a real
    place; the point is a repeatable, non-trivial input.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_days)
    doy = t % 365

    # rainfall: Bernoulli wet/dry with a seasonally varying wet probability,
    # gamma-distributed depths on wet days. Tuned to roughly 1300 mm/yr so the
    # catchment actually generates runoff and the reference series are not all
    # sitting near zero.
    wet_prob = 0.40 + 0.15 * np.sin(2 * np.pi * (doy - 200) / 365)
    is_wet = rng.random(n_days) < wet_prob
    depth = rng.gamma(shape=0.9, scale=10.0, size=n_days)
    rain = np.where(is_wet, depth, 0.0)

    # PET: seasonal cosine, ~0.5 mm/d in winter to ~6 mm/d in summer, ~1200 mm/yr
    pet = 3.2 - 2.7 * np.cos(2 * np.pi * (doy - 15) / 365)
    pet = np.clip(pet + rng.normal(0.0, 0.3, n_days), 0.1, None)

    return rain.astype(float), pet.astype(float)


@pytest.fixture(scope="session")
def forcing():
    """(rain, pet) in mm/d, ~5 years of daily data."""
    return _synthetic_forcing()


@pytest.fixture(scope="session")
def short_forcing():
    """A shorter slice, for tests that do not need the full record."""
    rain, pet = _synthetic_forcing()
    return rain[:800], pet[:800]
