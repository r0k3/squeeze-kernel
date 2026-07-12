"""Tests for the min_obs usability gate."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator


class TestMinObs:
    def test_default_all_usable(self, returns_small):
        est = SqueezeKernelEstimator(5)
        for r in returns_small:
            est.update(r)
        assert est.usable_mask.all()

    def test_validation(self):
        for bad in (0, -3, 2.5):
            with pytest.raises(ValueError, match="min_obs"):
                SqueezeKernelEstimator(5, min_obs=bad)

    def test_gate_lifts_at_threshold(self, rng):
        """A late-listing asset becomes usable exactly on its min_obs-th
        finite observation; NaN days do not count."""
        x = rng.normal(0, 0.01, size=(60, 3))
        x[:20, 2] = np.nan                      # asset 2 lists on day 20
        x[25, 2] = np.nan                       # one holiday after listing
        est = SqueezeKernelEstimator(3, min_obs=10)
        for t, r in enumerate(x):
            est.update(r)
            n_seen = np.isfinite(x[: t + 1, 2]).sum()
            assert est.usable_mask[2] == (n_seen >= 10)
        assert est.usable_mask.all()            # everyone warm by the end

    def test_gate_is_diagnostic_only(self, returns_with_nans):
        """min_obs must not change any estimate: outputs are bit-identical
        to an ungated estimator on the same path."""
        a = SqueezeKernelEstimator(10, min_obs=50)
        b = SqueezeKernelEstimator(10)
        for r in returns_with_nans:
            a.update(r)
            b.update(r)
        assert np.array_equal(a.get_cov(), b.get_cov())
        assert np.array_equal(a.get_corr(), b.get_corr())

    def test_subsetting_recipe(self, rng):
        """The documented recipe yields a PSD submatrix of the right size."""
        x = rng.normal(0, 0.01, size=(120, 6))
        x[:100, 4:] = np.nan                    # two assets list late
        est = SqueezeKernelEstimator(6, min_obs=60,
                                     corr_half_lives=[43.0, 173.0, 693.0],
                                     shrinkage_target="cluster")
        for r in x:
            est.update(r)
        m = est.usable_mask
        assert m.sum() == 4 and not m[4] and not m[5]
        sub = est.get_cov()[np.ix_(m, m)]
        assert sub.shape == (4, 4)
        assert np.linalg.eigvalsh(sub).min() >= -1e-12
