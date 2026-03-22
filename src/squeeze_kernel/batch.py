"""Batch estimation over an entire returns panel."""

from __future__ import annotations

import numpy as np

from squeeze_kernel.estimator import SqueezeKernelEstimator
from squeeze_kernel.kernels import KernelFn


def estimate_squeeze_cov(
    returns,
    *,
    lambda_vol: float = 0.94,
    lambda_corr: float = 0.99,
    kappa: float | None = None,
    kernel_fn: KernelFn | None = None,
    kernel_kwargs: dict[str, object] | None = None,
    epsilon: float = 1e-8,
    shrinkage: str | float = "auto",
    shrinkage_delta: float = 0.10,
    impute_missing: bool = False,
    impute_threshold: float = 0.6,
    with_corr: bool = True,
    with_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Estimate streaming covariance over an entire returns panel.

    Parameters
    ----------
    returns : array-like, shape (T, n)
        2D return matrix.  May contain NaN for missing observations.
    kappa : float, optional
        Saturation parameter for the default Fisher kernel.
    kernel_fn : callable, optional
        Custom kernel ``(d2, *, n_observed, **kw) -> float``.
    kernel_kwargs : dict, optional
        Extra keyword arguments forwarded to ``kernel_fn``.
    with_corr : bool
        If True, also return the correlation tensor.
    with_weights : bool
        If True, also return per-timestamp kernel weights.

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

    est = SqueezeKernelEstimator(
        n_assets,
        lambda_vol=lambda_vol,
        lambda_corr=lambda_corr,
        kappa=kappa,
        kernel_fn=kernel_fn,
        kernel_kwargs=kernel_kwargs,
        epsilon=epsilon,
        shrinkage=shrinkage,
        shrinkage_delta=shrinkage_delta,
        impute_missing=impute_missing,
        impute_threshold=impute_threshold,
    )

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
