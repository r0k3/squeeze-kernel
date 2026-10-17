"""Metamorphic properties of the estimator: exact invariances of the
algorithm, checked on the final forecast."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernel


def _final(X):
    sk = SqueezeKernel()
    for r in X:
        sk.update(r)
    return sk.covariance(), sk.correlation()


@pytest.fixture
def x(rng):
    return rng.normal(0, 0.01, size=(300, 12))


def test_permutation_equivariance(x):
    """Permuting assets permutes the output identically."""
    p = np.random.default_rng(0).permutation(x.shape[1])
    cov, corr = _final(x)
    cov_p, corr_p = _final(x[:, p])
    np.testing.assert_allclose(cov_p, cov[np.ix_(p, p)], rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(corr_p, corr[np.ix_(p, p)], rtol=1e-10, atol=1e-12)


def test_rescaling(x):
    """Rescaling asset returns rescales covariance bilinearly and leaves
    correlation unchanged, up to the additive epsilon floors, which are
    denominated in return units (exact only in the eps -> 0 limit)."""
    c = np.linspace(0.5, 3.0, x.shape[1])
    cov, corr = _final(x)
    cov_s, corr_s = _final(x * c)
    np.testing.assert_allclose(corr_s, corr, rtol=0, atol=5e-3)
    expected = cov * np.outer(c, c)
    np.testing.assert_allclose(cov_s, expected, rtol=0, atol=5e-3 * np.abs(expected).max())


def test_sign_flip(x):
    """Multiplying all returns by -1 leaves the forecast unchanged."""
    cov, corr = _final(x)
    cov_f, corr_f = _final(-x)
    np.testing.assert_array_equal(cov_f, cov)
    np.testing.assert_array_equal(corr_f, corr)


def test_all_missing_day(x):
    """An all-missing day carries no trading time and keeps the forecast PSD."""
    sk = SqueezeKernel()
    for r in x:
        sk.update(r)
    assert sk.update(np.full(x.shape[1], np.nan)) == 0.0
    cov = sk.covariance()
    assert np.isfinite(cov).all() and np.linalg.eigvalsh(cov).min() >= -1e-12
