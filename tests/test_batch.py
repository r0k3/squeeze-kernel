"""Tests for the batch estimation function."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator, estimate_squeeze_cov, kernel_exponential


class TestBatchConsistency:
    def test_batch_matches_streaming(self, returns_small):
        """Batch API must produce identical results to streaming class."""
        n = returns_small.shape[1]
        est = SqueezeKernelEstimator(n, kappa=1.5, shrinkage="none")
        cov_ref = np.empty((len(returns_small), n, n))
        for t in range(len(returns_small)):
            est.update(returns_small[t])
            cov_ref[t] = est.get_cov()

        cov_batch, _, _ = estimate_squeeze_cov(
            returns_small, kappa=1.5, shrinkage="none", with_corr=False,
        )
        assert np.allclose(cov_ref, cov_batch, atol=1e-12)

    def test_batch_with_corr(self, returns_small):
        cov, corr, _ = estimate_squeeze_cov(returns_small, kappa=1.5, with_corr=True)
        assert corr is not None
        assert corr.shape == cov.shape

    def test_batch_with_weights(self, returns_small):
        _, _, weights = estimate_squeeze_cov(
            returns_small, kappa=1.5, with_weights=True,
        )
        assert weights is not None
        assert weights.shape == (len(returns_small),)
        assert np.all(weights >= 0)
        assert np.all(weights < 1)

    def test_batch_with_shrinkage(self, returns_small):
        cov_auto, _, _ = estimate_squeeze_cov(returns_small, kappa=1.5, shrinkage="auto")
        cov_none, _, _ = estimate_squeeze_cov(returns_small, kappa=1.5, shrinkage="none")
        # At n=5 the early timesteps have shrinkage (S_t is small);
        # by the end, shrinkage ≈ 0 and final estimates should converge
        assert np.allclose(cov_auto[-1], cov_none[-1], atol=1e-6)

    def test_batch_matches_streaming_with_custom_kernel(self, returns_small):
        n = returns_small.shape[1]
        kwargs = {"gamma": 1.5}

        est = SqueezeKernelEstimator(
            n,
            kernel_fn=kernel_exponential,
            kernel_kwargs=kwargs,
            shrinkage="none",
        )
        cov_ref = np.empty((len(returns_small), n, n))
        for t in range(len(returns_small)):
            est.update(returns_small[t])
            cov_ref[t] = est.get_cov()

        cov_batch, _, _ = estimate_squeeze_cov(
            returns_small,
            kernel_fn=kernel_exponential,
            kernel_kwargs=kwargs,
            shrinkage="none",
            with_corr=False,
        )
        assert np.allclose(cov_ref, cov_batch, atol=1e-12)


class TestBatchValidation:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            estimate_squeeze_cov(np.zeros(10), kappa=1.5)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            estimate_squeeze_cov(np.zeros((10, 5, 3)), kappa=1.5)
