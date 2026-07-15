"""Tests for the CUSUM-gated adaptive ladder weights (integral for K >= 2)."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator

LADDER = [43.0, 173.0, 693.0]


class TestAdaptiveWeights:
    def test_one_rung_ladder_has_no_detector(self, returns_medium):
        """A one-element ladder is a single-scale estimator: no detector
        state, and the lazy extraction path is used."""
        est = SqueezeKernelEstimator(30, corr_half_lives=[173.0])
        assert not est._adaptive
        for r in returns_medium:
            est.update(r)
        assert np.isfinite(est.get_cov()).all()

    def test_multi_rung_ladder_runs_detector(self, returns_medium):
        est = SqueezeKernelEstimator(30, corr_half_lives=LADDER)
        assert est._adaptive
        for r in returns_medium:
            est.update(r)
        assert est._aw_prev_sig is not None
        assert len(est._aw_prev_sig) == len(LADDER)

    def test_psd_and_unit_diag(self, returns_medium):
        est = SqueezeKernelEstimator(30, corr_half_lives=LADDER,
                                     shrinkage_target="cluster")
        for r in returns_medium:
            est.update(r)
            assert np.linalg.eigvalsh(est.get_cov()).min() >= -1e-10
        assert np.allclose(np.diag(est.get_corr()), 1.0)

    def test_no_alarm_equals_prior_blend(self, rng):
        """With the detector's tilt at zero the blend equals the theta-prior
        blend exactly: against a twin whose threshold never fires (state
        paths are identical; the tilt only affects extraction)."""
        x = rng.normal(0, 0.01, size=(120, 8))
        a = SqueezeKernelEstimator(8, corr_half_lives=LADDER)
        b = SqueezeKernelEstimator(8, corr_half_lives=LADDER)
        b._aw_b = float("inf")          # detector can never alarm
        for r in x:
            a.update(r)
            b.update(r)
            if a._aw_tilt == 0.0:
                np.testing.assert_allclose(a.get_cov(), b.get_cov(),
                                           rtol=1e-12, atol=0)

    def test_tilt_blend_geometry(self, rng):
        """A forced tilt moves weights toward the declared tilt vector."""
        x = rng.normal(0, 0.01, size=(80, 8))
        est = SqueezeKernelEstimator(8, corr_half_lives=LADDER)
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
        est = SqueezeKernelEstimator(10, corr_half_lives=LADDER)
        for r in returns_with_nans:
            est.update(r)
        assert np.isfinite(est.get_cov()).all()

    def test_score_paths_agree(self, monkeypatch, rng):
        """The SciPy Cholesky score path and the NumPy slogdet+solve
        fallback compute the same quantities: identical covariance paths
        and detector states on the same input."""
        import squeeze_kernel.estimator as mod
        if mod._cho_factor is None:
            pytest.skip("SciPy not installed; only the fallback path exists")
        x = rng.normal(0, 0.01, size=(150, 12))
        x[40] *= 8.0                       # a shock so scores move
        a = SqueezeKernelEstimator(12, corr_half_lives=LADDER)
        for r in x:
            a.update(r)
        cov_scipy, state_scipy = a.get_cov(), (a._aw_gp, a._aw_gm, a._aw_tilt)
        monkeypatch.setattr(mod, "_cho_factor", None)
        b = SqueezeKernelEstimator(12, corr_half_lives=LADDER)
        for r in x:
            b.update(r)
        np.testing.assert_allclose(b.get_cov(), cov_scipy, rtol=1e-9, atol=0)
        np.testing.assert_allclose(
            (b._aw_gp, b._aw_gm, b._aw_tilt), state_scipy, rtol=1e-9, atol=1e-12)
