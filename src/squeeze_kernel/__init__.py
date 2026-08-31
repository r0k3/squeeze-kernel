"""
Squeeze Kernel Covariance Estimator
====================================

Streaming, PSD-by-construction covariance estimator. The 2.0 API is one
number and two booleans; everything else derives from the half-life or is
a frozen structural constant.

Quick start::

    import numpy as np
    from squeeze_kernel import SqueezeKernel

    returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 30))

    sk = SqueezeKernel(half_life=173, detector=True, cluster_target=True)
    for r_t in returns:
        sk.update(r_t)          # NaN marks missing assets

    cov = sk.covariance()
    corr = sk.correlation()

The published v1 estimator (all legacy knobs) remains available as
``SqueezeKernelEstimator`` or ``SqueezeKernel.v1(...)``; see MIGRATION.md.
"""

from squeeze_kernel.core import CONSTANTS, SqueezeKernel, StructuralConstants
from squeeze_kernel.estimator import SqueezeKernelEstimator
from squeeze_kernel.kernels import kernel_fisher, kernel_exponential, kernel_chi2_cdf
from squeeze_kernel.batch import estimate_squeeze_cov

__all__ = [
    "SqueezeKernel",
    "StructuralConstants",
    "CONSTANTS",
    "SqueezeKernelEstimator",
    "estimate_squeeze_cov",
    "kernel_fisher",
    "kernel_exponential",
    "kernel_chi2_cdf",
]

__version__ = "2.0.0.dev0"
