"""Tests for kernel functions and calibration."""

import numpy as np
import pytest

from squeeze_kernel.kernels import (
    kernel_fisher, kernel_exponential, calibrate_kappa, extract_d2_series,
)
from squeeze_kernel import SqueezeKernelEstimator


class TestKernelFisher:
    def test_zero(self):
        assert kernel_fisher(0.0, kappa=1.0) == 0.0

    def test_positive_kappa_required(self):
        with pytest.raises(ValueError, match="kappa"):
            kernel_fisher(1.0, kappa=0.0)

    def test_monotone(self):
        vals = [kernel_fisher(d2, kappa=1.5) for d2 in [0.1, 0.5, 1.0, 2.0, 5.0]]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_bounded(self):
        assert 0.0 <= kernel_fisher(100.0, kappa=1.5) < 1.0

    def test_kappa_effect(self):
        w_low = kernel_fisher(1.0, kappa=0.5)
        w_high = kernel_fisher(1.0, kappa=5.0)
        assert w_low > w_high  # Lower kappa → higher weight


class TestKernelExponential:
    def test_zero(self):
        assert kernel_exponential(0.0, gamma=1.0) == 0.0

    def test_bounded(self):
        w = kernel_exponential(100.0, gamma=1.0)
        assert 0.0 < w <= 1.0

    def test_positive_gamma_required(self):
        with pytest.raises(ValueError, match="gamma"):
            kernel_exponential(1.0, gamma=0.0)


class TestCalibrateKappa:
    def test_calibration(self, rng):
        d2 = rng.exponential(1.0, size=1000)
        target = 0.6
        kappa = calibrate_kappa(d2, target)
        actual = float(np.mean(d2 / (d2 + kappa)))
        assert abs(actual - target) < 0.01

    def test_class_method(self, rng):
        returns = rng.normal(0, 0.01, size=(500, 10))
        kappa = SqueezeKernelEstimator.calibrate_kappa(returns, target_weight=0.5)
        assert kappa > 0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="10"):
            calibrate_kappa(np.array([1.0, 2.0]), 0.5)


class TestExtractD2:
    def test_shape(self, returns_small):
        d2 = extract_d2_series(returns_small, lambda_vol=0.94)
        assert d2.shape == (returns_small.shape[0],)

    def test_positive(self, returns_small):
        d2 = extract_d2_series(returns_small)
        assert np.all(d2[np.isfinite(d2)] >= 0)
