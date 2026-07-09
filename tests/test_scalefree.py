"""Tests for the opt-in scale-free correlation memory (corr_half_lives)."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator

LADDER = [43.0, 173.0, 693.0]


def _run(est, X):
    for r in X:
        est.update(r)
    return est.get_cov()


class TestDefaultUnchanged:
    def test_none_is_bit_for_bit_default(self, returns_medium):
        """corr_half_lives=None must be byte-identical to the published path."""
        a = _run(SqueezeKernelEstimator(30), returns_medium)
        b = _run(SqueezeKernelEstimator(30, corr_half_lives=None), returns_medium)
        assert np.array_equal(a, b)

    def test_one_element_ladder_reduces_to_single_scale(self, returns_medium):
        """A one-rung ladder at half-life h equals a single-scale estimator at
        lambda_corr = 2**(-1/h), exactly (theta is irrelevant for K=1)."""
        h = 173.0
        single = _run(SqueezeKernelEstimator(30, lambda_corr=2.0 ** (-1.0 / h)),
                      returns_medium)
        ladder = _run(SqueezeKernelEstimator(30, corr_half_lives=[h], corr_theta=0.5),
                      returns_medium)
        assert np.array_equal(single, ladder)


class TestScaleFreeBehaviour:
    def test_psd_every_step(self, returns_medium):
        est = SqueezeKernelEstimator(30, corr_half_lives=LADDER)
        for r in returns_medium:
            est.update(r)
            ev = np.linalg.eigvalsh(est.get_cov())
            assert ev.min() >= -1e-10

    def test_corr_unit_diagonal(self, returns_medium):
        est = SqueezeKernelEstimator(30, corr_half_lives=LADDER)
        _run(est, returns_medium)
        assert np.allclose(np.diag(est.get_corr()), 1.0)

    def test_masked_updates(self, returns_with_nans):
        """The masked path must run and stay PSD under NaNs."""
        est = SqueezeKernelEstimator(10, corr_half_lives=LADDER)
        for r in returns_with_nans:
            est.update(r)
        assert np.linalg.eigvalsh(est.get_cov()).min() >= -1e-10

    def test_cluster_composition(self, returns_medium):
        """Cluster target composes with the ladder and stays PSD; at the low
        concentration of this panel it reduces toward the equicorrelation
        result (the companion paper's second-order property)."""
        clu = _run(SqueezeKernelEstimator(30, corr_half_lives=LADDER,
                                          shrinkage_target="cluster"), returns_medium)
        equi = _run(SqueezeKernelEstimator(30, corr_half_lives=LADDER), returns_medium)
        assert np.linalg.eigvalsh(clu).min() >= -1e-10
        assert np.allclose(clu, equi, rtol=1e-3)

    def test_weights_sum_to_one(self):
        est = SqueezeKernelEstimator(5, corr_half_lives=LADDER, corr_theta=0.25)
        assert est._corr_w.sum() == pytest.approx(1.0)
        # theta>0 puts more weight on the longer half-life
        assert est._corr_w[-1] > est._corr_w[0]

    def test_theta_zero_is_equal_weight(self):
        est = SqueezeKernelEstimator(5, corr_half_lives=LADDER, corr_theta=0.0)
        assert np.allclose(est._corr_w, 1.0 / len(LADDER))


class TestValidation:
    def test_rejects_lambda_corr_fast(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SqueezeKernelEstimator(5, corr_half_lives=LADDER, lambda_corr_fast=0.99)

    def test_rejects_nonpositive_half_life(self):
        with pytest.raises(ValueError, match="positive half-lives"):
            SqueezeKernelEstimator(5, corr_half_lives=[43.0, -1.0])

    def test_rejects_empty_ladder(self):
        with pytest.raises(ValueError, match="non-empty"):
            SqueezeKernelEstimator(5, corr_half_lives=[])

    def test_rejects_negative_theta(self):
        with pytest.raises(ValueError, match="corr_theta"):
            SqueezeKernelEstimator(5, corr_half_lives=LADDER, corr_theta=-0.5)
