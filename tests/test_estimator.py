"""Tests for the SqueezeKernelEstimator class."""

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator, kernel_exponential


class TestBasicAPI:
    def test_update_returns_weight(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage="none")
        w = est.update(returns_small[0])
        assert isinstance(w, float)
        assert 0.0 <= w < 1.0

    def test_get_cov_shape(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5)
        est.update(returns_small[0])
        cov = est.get_cov()
        assert cov.shape == (5, 5)

    def test_get_corr_shape(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5)
        est.update(returns_small[0])
        corr = est.get_corr()
        assert corr.shape == (5, 5)

    def test_get_cov_before_update_raises(self):
        est = SqueezeKernelEstimator(5)
        with pytest.raises(RuntimeError, match="update"):
            est.get_cov()

    def test_wrong_shape_raises(self):
        est = SqueezeKernelEstimator(5)
        with pytest.raises(ValueError, match="shape"):
            est.update(np.zeros(3))

    def test_custom_kernel_runs(self, returns_small):
        est = SqueezeKernelEstimator(
            5,
            kernel_fn=kernel_exponential,
            kernel_kwargs={"gamma": 1.5},
            shrinkage="none",
        )
        w = est.update(returns_small[0])
        assert isinstance(w, float)
        assert 0.0 <= w < 1.0

    def test_kappa_with_custom_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel_kwargs"):
            SqueezeKernelEstimator(5, kappa=1.0, kernel_fn=kernel_exponential)


class TestPSD:
    """The covariance must be PSD at every time step (Theorem 1)."""

    def test_psd_all_steps(self, returns_medium):
        est = SqueezeKernelEstimator(30, kappa=1.5)
        for t in range(returns_medium.shape[0]):
            est.update(returns_medium[t])
            if t % 20 == 0:
                cov = est.get_cov()
                eigvals = np.linalg.eigvalsh(cov)
                assert eigvals.min() >= -1e-10, f"PSD violation at t={t}: min_eig={eigvals.min()}"

    def test_psd_with_nans(self, returns_with_nans):
        est = SqueezeKernelEstimator(10, kappa=1.5)
        for t in range(returns_with_nans.shape[0]):
            est.update(returns_with_nans[t])
        cov = est.get_cov()
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() >= -1e-10

    def test_psd_with_shrinkage(self, returns_medium):
        est = SqueezeKernelEstimator(30, kappa=1.5, shrinkage="auto")
        for t in range(returns_medium.shape[0]):
            est.update(returns_medium[t])
        cov = est.get_cov()
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() >= -1e-10


class TestSymmetry:
    def test_cov_symmetric(self, returns_medium):
        est = SqueezeKernelEstimator(30, kappa=1.5)
        for t in range(returns_medium.shape[0]):
            est.update(returns_medium[t])
        cov = est.get_cov()
        assert np.allclose(cov, cov.T, atol=1e-12)

    def test_corr_symmetric(self, returns_medium):
        est = SqueezeKernelEstimator(30, kappa=1.5)
        for t in range(returns_medium.shape[0]):
            est.update(returns_medium[t])
        corr = est.get_corr()
        assert np.allclose(corr, corr.T, atol=1e-12)

    def test_corr_diagonal_ones(self, returns_medium):
        est = SqueezeKernelEstimator(30, kappa=1.5)
        for t in range(returns_medium.shape[0]):
            est.update(returns_medium[t])
        corr = est.get_corr()
        assert np.allclose(np.diag(corr), 1.0, atol=1e-10)


class TestFiniteness:
    def test_outputs_finite(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5)
        for t in range(returns_small.shape[0]):
            w = est.update(returns_small[t])
            assert np.isfinite(w)
        assert np.all(np.isfinite(est.get_cov()))
        assert np.all(np.isfinite(est.get_corr()))

    def test_all_nan_day(self, rng):
        """All-NaN day should not crash; output unchanged from previous."""
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage="none")
        for _ in range(10):
            est.update(rng.normal(0, 0.01, 5))
        w = est.update(np.full(5, np.nan))
        assert w == 0.0
        # Covariance should scale (lambda_c decay) but corr ratio unchanged
        assert np.all(np.isfinite(est.get_cov()))


class TestHolidayInvariance:
    def test_holiday_invariance_no_shrinkage(self, rng):
        """On all-NaN day, M/S ratio is unchanged → raw corr is invariant."""
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage="none")
        for _ in range(50):
            est.update(rng.normal(0, 0.01, 5))
        corr_before = est.get_corr()
        est.update(np.full(5, np.nan))
        assert np.allclose(est.get_corr(), corr_before, atol=1e-12)


class TestAdaptiveShrinkage:
    def test_shrinkage_zero_at_small_n(self, returns_small):
        """At n=5 with enough data, shrinkage should be zero."""
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage="auto")
        for t in range(returns_small.shape[0]):
            est.update(returns_small[t])
        assert est.shrinkage_intensity == 0.0

    def test_shrinkage_positive_at_large_n(self, rng):
        """At n=100 with limited data, shrinkage should activate."""
        n = 100
        est = SqueezeKernelEstimator(n, kappa=1.5, shrinkage="auto")
        for _ in range(50):
            est.update(rng.normal(0, 0.01, n))
        assert est.shrinkage_intensity > 0.0

    def test_shrinkage_none_disables(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage="none")
        for t in range(returns_small.shape[0]):
            est.update(returns_small[t])
        assert est.shrinkage_intensity == 0.0

    def test_shrinkage_fixed(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5, shrinkage=0.5)
        for t in range(returns_small.shape[0]):
            est.update(returns_small[t])
        assert est.shrinkage_intensity == 0.5


class TestProperties:
    def test_effective_sample_size_grows(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5)
        s_prev = est.effective_sample_size
        for t in range(returns_small.shape[0]):
            est.update(returns_small[t])
        assert est.effective_sample_size > s_prev

    def test_weight_range(self, returns_small):
        est = SqueezeKernelEstimator(5, kappa=1.5)
        for t in range(returns_small.shape[0]):
            w = est.update(returns_small[t])
            assert 0.0 <= w < 1.0


class TestMissingData:
    def test_partial_nan_runs(self, returns_with_nans):
        est = SqueezeKernelEstimator(10, kappa=1.5)
        for t in range(returns_with_nans.shape[0]):
            est.update(returns_with_nans[t])
        cov = est.get_cov()
        assert np.all(np.isfinite(cov))
        assert cov.shape == (10, 10)

    def test_imputation_improves(self, rng):
        """Imputation should reduce cross-covariance error."""
        t_total = 400
        base = rng.normal(0, 0.012, t_total)
        a0 = base + rng.normal(0, 0.001, t_total)
        a1 = 1.2 * base + rng.normal(0, 0.001, t_total)
        a2 = rng.normal(0, 0.015, t_total)
        full = np.column_stack([a0, a1, a2])
        masked = full.copy()
        masked[60::2, 1] = np.nan

        est_no = SqueezeKernelEstimator(3, kappa=1.5, shrinkage="none")
        est_yes = SqueezeKernelEstimator(3, kappa=1.5, shrinkage="none", impute_missing=True)
        est_ref = SqueezeKernelEstimator(3, kappa=1.5, shrinkage="none")

        for t in range(t_total):
            est_ref.update(full[t])
            est_no.update(masked[t])
            est_yes.update(masked[t])

        ref_01 = est_ref.get_cov()[0, 1]
        err_no = abs(est_no.get_cov()[0, 1] - ref_01)
        err_yes = abs(est_yes.get_cov()[0, 1] - ref_01)
        assert err_yes < err_no


class TestExtensions:
    """Opt-in extensions: Mahalanobis weight statistic and score-driven memory."""

    def _run(self, est, x):
        for r in x:
            est.update(r)
        return est.get_cov()

    def test_defaults_unchanged(self, rng):
        """Extension defaults must reproduce the base estimator bit-for-bit."""
        x = rng.normal(0, 0.01, size=(300, 5))
        base = self._run(SqueezeKernelEstimator(5, kappa=1.0), x)
        ext = self._run(SqueezeKernelEstimator(5, kappa=1.0, weight_statistic="marginal",
                                               lambda_corr_fast=None), x)
        assert np.array_equal(base, ext)

    def test_lambda_fast_equal_slow_is_identity(self, rng):
        """lambda_corr_fast == lambda_corr must be exactly the base estimator."""
        x = rng.normal(0, 0.01, size=(300, 5))
        base = self._run(SqueezeKernelEstimator(5, kappa=1.0, lambda_corr=0.99), x)
        same = self._run(SqueezeKernelEstimator(5, kappa=1.0, lambda_corr=0.99,
                                                lambda_corr_fast=0.99), x)
        assert np.allclose(base, same)

    def test_mahalanobis_first_step_matches_marginal(self, rng):
        """Before a correlation estimate exists, mahalanobis falls back to marginal."""
        est_m = SqueezeKernelEstimator(4, kappa=1.0, weight_statistic="mahalanobis")
        est_b = SqueezeKernelEstimator(4, kappa=1.0)
        r0 = rng.normal(0, 0.01, 4)
        assert est_m.update(r0) == pytest.approx(est_b.update(r0))

    def test_mahalanobis_psd_and_divergence(self, rng):
        """Mahalanobis variant stays PSD every step and eventually differs from base."""
        x = rng.normal(0, 0.01, size=(400, 6))
        x[:, 1] = 0.8 * x[:, 0] + 0.2 * x[:, 1]  # give the panel real correlation
        est_m = SqueezeKernelEstimator(6, kappa=1.0, weight_statistic="mahalanobis")
        est_b = SqueezeKernelEstimator(6, kappa=1.0)
        for t in range(400):
            est_m.update(x[t])
            est_b.update(x[t])
            eig = np.linalg.eigvalsh(est_m.get_cov())
            assert eig.min() >= -1e-12
        assert not np.allclose(est_m.get_cov(), est_b.get_cov())

    def test_scorem_psd_and_divergence(self, rng):
        """Score-driven memory stays PSD every step and differs from base."""
        x = rng.normal(0, 0.01, size=(400, 6))
        est_f = SqueezeKernelEstimator(6, kappa=1.0, lambda_corr=0.996, lambda_corr_fast=0.99)
        est_b = SqueezeKernelEstimator(6, kappa=1.0, lambda_corr=0.996)
        for t in range(400):
            est_f.update(x[t])
            est_b.update(x[t])
            eig = np.linalg.eigvalsh(est_f.get_cov())
            assert eig.min() >= -1e-12
        assert not np.allclose(est_f.get_cov(), est_b.get_cov())

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            SqueezeKernelEstimator(3, weight_statistic="bogus")
        with pytest.raises(ValueError):
            SqueezeKernelEstimator(3, lambda_corr_fast=1.5)

    def test_mahalanobis_with_missing_data(self, rng):
        """Masked updates keep working under the mahalanobis statistic."""
        x = rng.normal(0, 0.01, size=(300, 5))
        x[50::7, 2] = np.nan
        est = SqueezeKernelEstimator(5, kappa=1.0, weight_statistic="mahalanobis")
        for t in range(300):
            w = est.update(x[t])
            assert 0.0 <= w < 1.0
        assert np.isfinite(est.get_cov()).all()
