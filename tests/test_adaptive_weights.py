"""Tests for the CUSUM-gated adaptive ladder weights."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator

LADDER = [43.0, 173.0, 693.0]


class TestAdaptiveWeights:
    def test_default_unchanged(self, returns_medium):
        a = SqueezeKernelEstimator(30, corr_half_lives=LADDER)
        b = SqueezeKernelEstimator(30, corr_half_lives=LADDER,
                                   adaptive_weights=None)
        for r in returns_medium:
            a.update(r)
            b.update(r)
        assert np.array_equal(a.get_cov(), b.get_cov())

    def test_validation(self):
        with pytest.raises(ValueError, match="adaptive_weights"):
            SqueezeKernelEstimator(5, corr_half_lives=LADDER,
                                   adaptive_weights="bogus")
        with pytest.raises(ValueError, match="requires corr_half_lives"):
            SqueezeKernelEstimator(5, adaptive_weights="cusum")

    def test_psd_and_unit_diag(self, returns_medium):
        est = SqueezeKernelEstimator(30, corr_half_lives=LADDER,
                                     shrinkage_target="cluster",
                                     adaptive_weights="cusum")
        for r in returns_medium:
            est.update(r)
            assert np.linalg.eigvalsh(est.get_cov()).min() >= -1e-10
        assert np.allclose(np.diag(est.get_corr()), 1.0)

    def test_no_alarm_equals_fixed(self, rng):
        """With the detector's tilt at zero the blend equals the fixed
        theta-prior blend exactly (state paths are identical)."""
        x = rng.normal(0, 0.01, size=(120, 8))
        a = SqueezeKernelEstimator(8, corr_half_lives=LADDER,
                                   adaptive_weights="cusum")
        b = SqueezeKernelEstimator(8, corr_half_lives=LADDER)
        for r in x:
            a.update(r)
            b.update(r)
            if a._aw_tilt == 0.0:
                np.testing.assert_allclose(a.get_cov(), b.get_cov(),
                                           rtol=1e-12, atol=0)

    def test_tilt_blend_geometry(self, rng):
        """A forced tilt moves weights toward the declared tilt vector."""
        x = rng.normal(0, 0.01, size=(80, 8))
        est = SqueezeKernelEstimator(8, corr_half_lives=LADDER,
                                     adaptive_weights="cusum")
        for r in x:
            est.update(r)
        est._aw_tilt = 0.5
        est._materialize_adaptive()
        cov_tilted = est.get_cov()
        est._aw_tilt = 0.0
        est._materialize_adaptive()
        cov_prior = est.get_cov()
        assert not np.allclose(cov_tilted, cov_prior)

    def test_masked_updates_run(self, returns_with_nans):
        est = SqueezeKernelEstimator(10, corr_half_lives=LADDER,
                                     adaptive_weights="cusum")
        for r in returns_with_nans:
            est.update(r)
        assert np.isfinite(est.get_cov()).all()
