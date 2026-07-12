"""Metamorphic properties of the estimator (rework directive §14).

Each test states an exact invariance of the algorithm and checks it on
both the single-scale published path and the ladder+cluster path.
"""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator

CONFIGS = {
    "single": {},
    "ladder_cluster": dict(corr_half_lives=[43.0, 173.0, 693.0],
                           corr_theta=0.25, shrinkage_target="cluster"),
}


def _final_cov(X, n, **kw):
    est = SqueezeKernelEstimator(n, **kw)
    for r in X:
        est.update(r)
    return est.get_cov(), est.get_corr()


@pytest.fixture
def x(rng):
    return rng.normal(0, 0.01, size=(300, 12))


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
class TestMetamorphic:
    def test_permutation_equivariance(self, x, cfg):
        """Permuting assets permutes the output identically."""
        n = x.shape[1]
        p = np.random.default_rng(0).permutation(n)
        cov, corr = _final_cov(x, n, **CONFIGS[cfg])
        cov_p, corr_p = _final_cov(x[:, p], n, **CONFIGS[cfg])
        np.testing.assert_allclose(cov_p, cov[np.ix_(p, p)], rtol=1e-12, atol=0)
        np.testing.assert_allclose(corr_p, corr[np.ix_(p, p)], rtol=1e-12, atol=0)

    def test_rescaling(self, x, cfg):
        """Rescaling asset returns rescales covariance bilinearly and
        leaves correlation unchanged — up to the additive epsilon floors
        (var init r^2 + eps and z = r/(vol + eps) are denominated in
        return units, so the invariance is exact only in the eps -> 0
        limit; leakage is O(eps / r_0^2) ~ 1e-3 at half scale for
        1e-2-sized returns)."""
        n = x.shape[1]
        c = np.linspace(0.5, 3.0, n)
        cov, corr = _final_cov(x, n, **CONFIGS[cfg])
        cov_s, corr_s = _final_cov(x * c, n, **CONFIGS[cfg])
        np.testing.assert_allclose(corr_s, corr, rtol=0, atol=5e-3)
        expected = cov * np.outer(c, c)
        # matrix-scale tolerance: near-zero entries carry the same absolute
        # eps-floor leakage as large ones
        np.testing.assert_allclose(cov_s, expected, rtol=0,
                                   atol=5e-3 * np.abs(expected).max())

    def test_sign_flip(self, x, cfg):
        """Multiplying all returns by -1 leaves the covariance unchanged."""
        n = x.shape[1]
        cov, corr = _final_cov(x, n, **CONFIGS[cfg])
        cov_f, corr_f = _final_cov(-x, n, **CONFIGS[cfg])
        np.testing.assert_array_equal(cov_f, cov)
        np.testing.assert_array_equal(corr_f, corr)

    def test_all_missing_day_is_no_information(self, x, cfg):
        """An all-missing day adds no information: with fixed shrinkage the
        forecast is bit-identical; with adaptive shrinkage the correlation
        state is untouched (only the information count decays)."""
        n = x.shape[1]
        est = SqueezeKernelEstimator(n, shrinkage=0.0, **CONFIGS[cfg])
        for r in x:
            est.update(r)
        before = est.get_cov()
        w = est.update(np.full(n, np.nan))
        assert w == 0.0
        np.testing.assert_array_equal(est.get_cov(), before)

    def test_adaptive_shrinkage_drifts_up_on_missing(self, x, cfg):
        """With shrinkage='auto', an all-missing day decays S and cannot
        decrease the shrinkage intensity."""
        n = x.shape[1]
        est = SqueezeKernelEstimator(n, shrinkage="auto", **CONFIGS[cfg])
        for r in x:
            est.update(r)
        a0 = est.shrinkage_intensity
        est.update(np.full(n, np.nan))
        assert est.shrinkage_intensity >= a0
        ev = np.linalg.eigvalsh(est.get_cov())
        assert ev.min() >= -1e-12
