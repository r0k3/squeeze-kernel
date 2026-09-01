"""Tests for kernel functions and calibration."""

import pytest

from squeeze_kernel.kernels import kernel_fisher


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
