"""Batch estimation over an entire returns panel."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from squeeze_kernel.estimator import SqueezeKernelEstimator


def estimate_squeeze_cov(
    returns: ArrayLike,
    *,
    with_corr: bool = True,
    with_weights: bool = False,
    **estimator_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Estimate streaming covariance over an entire returns panel.

    Runs one :class:`SqueezeKernelEstimator` over the panel day by day and
    collects the estimate path.  ``n_assets`` is taken from the panel shape;
    every other keyword argument is forwarded unchanged to the estimator, so
    batch mode reaches the full estimator surface — ``kappa``, ``shrinkage``,
    ``shrinkage_target``, ``corr_half_lives``, ``weight_statistic``,
    ``vol_anchor_phi``, ``min_obs``, custom kernels, and so on (see the
    estimator's docstring for the complete list and defaults).

    Parameters
    ----------
    returns : array-like, shape (T, n)
        2D return matrix.  May contain NaN for missing observations.
    with_corr : bool
        If True, also return the correlation tensor.
    with_weights : bool
        If True, also return per-timestamp kernel weights.
    **estimator_kwargs
        Passed through to ``SqueezeKernelEstimator``.

    Returns
    -------
    cov : ndarray, shape (T, n, n)
        The covariance estimate after each day.
    corr : ndarray or None, shape (T, n, n)
    weights : ndarray or None, shape (T,)
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D returns, got shape {values.shape}.")
    if "n_assets" in estimator_kwargs:
        raise ValueError(
            "n_assets is derived from the panel shape and cannot be passed."
        )
    t_total, n_assets = values.shape

    est = SqueezeKernelEstimator(n_assets, **estimator_kwargs)

    cov = np.empty((t_total, n_assets, n_assets), dtype=np.float64)
    corr = np.empty_like(cov) if with_corr else None
    weights = np.empty(t_total, dtype=np.float64) if with_weights else None

    for t in range(t_total):
        w_t = est.update(values[t])
        cov[t] = est.get_cov()
        if corr is not None:
            corr[t] = est.get_corr()
        if weights is not None:
            weights[t] = w_t

    return cov, corr, weights
