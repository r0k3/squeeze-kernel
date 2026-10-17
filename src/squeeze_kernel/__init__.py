"""
Squeeze Kernel Covariance Estimator
====================================

Streaming, PSD-by-construction covariance estimator whose public surface
is one number: everything else derives from that decay, is a structural
constant of the method, or is state computed online -- including per-asset
market clocks read from the correlation structure itself.

Quick start::

    import numpy as np
    from squeeze_kernel import SqueezeKernel

    returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 30))

    sk = SqueezeKernel(lam=0.996)        # the entire public surface
    for r_t in returns:
        sk.update(r_t)                   # NaN marks missing assets

    cov = sk.covariance()
    corr = sk.correlation()
"""

from squeeze_kernel.core import SqueezeKernel
from squeeze_kernel.batch import estimate_squeeze_cov

__all__ = ["SqueezeKernel", "estimate_squeeze_cov"]

__version__ = "4.0.0"
