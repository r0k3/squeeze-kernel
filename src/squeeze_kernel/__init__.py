"""
Squeeze Kernel Covariance Estimator
====================================

Streaming, PSD-by-construction covariance estimator with adaptive
equicorrelation shrinkage and pluggable kernel weighting.

Quick start::

    import numpy as np
    from squeeze_kernel import SqueezeKernelEstimator

    returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 3))

    # Defaults (lambda_vol=0.98, lambda_corr=0.996, kappa=0.25) are the
    # paper-recommended settings for panels of daily financial returns.
    est = SqueezeKernelEstimator(n_assets=3)
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

__version__ = "0.5.0"
