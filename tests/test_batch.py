"""Tests for the batch estimation function."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernel, estimate_squeeze_cov


class TestBatchConsistency:
    def test_batch_matches_streaming(self, returns_small):
        sk = SqueezeKernel(lam=0.99)
        cov_ref = np.empty((len(returns_small),) + (returns_small.shape[1],) * 2)
        for t in range(len(returns_small)):
            sk.update(returns_small[t])
            cov_ref[t] = sk.covariance()
        cov_batch, _, _ = estimate_squeeze_cov(returns_small, lam=0.99, with_corr=False)
        assert np.allclose(cov_ref, cov_batch, atol=1e-12)

    def test_batch_with_corr_and_weights(self, returns_small):
        cov, corr, weights = estimate_squeeze_cov(
            returns_small, with_corr=True, with_weights=True)
        assert corr is not None and corr.shape == cov.shape
        assert weights is not None and weights.shape == (len(returns_small),)
        assert np.all(weights >= 0) and np.all(weights < 1)
        assert np.isfinite(cov).all()


class TestBatchValidation:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            estimate_squeeze_cov(np.zeros(10))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            estimate_squeeze_cov(np.zeros((10, 5, 3)))
