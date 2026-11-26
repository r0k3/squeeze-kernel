"""Shared fixtures for squeeze_kernel tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def returns_small(rng):
    """Small synthetic returns: 200 days, 5 assets."""
    return rng.normal(0, 0.01, size=(200, 5))


@pytest.fixture
def returns_medium(rng):
    """Medium synthetic returns: 500 days, 30 assets."""
    return rng.normal(0, 0.01, size=(500, 30))


@pytest.fixture
def returns_with_nans(rng):
    """Returns with 10% missing values."""
    x = rng.normal(0, 0.01, size=(300, 10))
    mask = rng.random(x.shape) < 0.10
    x[mask] = np.nan
    return x


@pytest.fixture
def returns_correlated(rng):
    """Returns with known correlation structure."""
    n, t = 5, 500
    factor = rng.normal(0, 0.01, size=t)
    idio = rng.normal(0, 0.005, size=(t, n))
    return factor[:, None] * np.ones(n) + idio
