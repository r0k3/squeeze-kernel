"""
Squeeze Kernel Covariance Estimator
====================================

Streaming, PSD-by-construction covariance estimator with adaptive
equicorrelation shrinkage and pluggable kernel weighting.

Quick start::

    import numpy as np
    from squeeze_kernel import SqueezeKernelEstimator

    returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 3))

    # Defaults to the Fisher kernel; pass kernel_fn/kernel_kwargs
    # to use alternatives such as kernel_exponential.
    est = SqueezeKernelEstimator(n_assets=3, kappa=1.5)
    for r_t in returns:
        est.update(r_t)

    cov = est.get_cov()
    corr = est.get_corr()
"""

from squeeze_kernel.estimator import SqueezeKernelEstimator
from squeeze_kernel.kernels import kernel_fisher, kernel_exponential, kernel_chi2_cdf
from squeeze_kernel.batch import estimate_squeeze_cov

__all__ = [
    "SqueezeKernelEstimator",
    "estimate_squeeze_cov",
    "kernel_fisher",
    "kernel_exponential",
    "kernel_chi2_cdf",
]

__version__ = "0.1.0"
