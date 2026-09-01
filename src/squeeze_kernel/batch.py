"""Batch estimation over an entire returns panel."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from squeeze_kernel.core import SqueezeKernel


def estimate_squeeze_cov(
    returns: ArrayLike,
    *,
    lam: float = 0.996,
    with_corr: bool = True,
    with_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run :class:`SqueezeKernel` over a panel and collect the estimate path.

    Parameters
    ----------
    returns : array-like, shape (T, n)
        Daily returns; NaN marks missing observations.
    lam : float
        The estimator's single parameter (default 0.996).
    with_corr : bool
        Also return the correlation path.
    with_weights : bool
        Also return the per-day kernel weight.

    Returns
    -------
    cov : ndarray, shape (T, n, n)
    corr : ndarray or None, shape (T, n, n)
    weights : ndarray or None, shape (T,)
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D returns, got shape {values.shape}.")
    t_total, n_assets = values.shape
    sk = SqueezeKernel(lam=lam)
    cov = np.empty((t_total, n_assets, n_assets), dtype=np.float64)
    corr = np.empty_like(cov) if with_corr else None
    weights = np.empty(t_total, dtype=np.float64) if with_weights else None
    for t in range(t_total):
        w_t = sk.update(values[t])
        cov[t] = sk.covariance()
        if corr is not None:
            corr[t] = sk.correlation()
        if weights is not None:
            weights[t] = w_t
    return cov, corr, weights
