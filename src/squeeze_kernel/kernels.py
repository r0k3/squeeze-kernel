"""Built-in kernel weight functions for the Squeeze Kernel estimator."""

from __future__ import annotations

import math
from typing import Callable

KernelFn = Callable[..., float]


def kernel_fisher(d2: float, /, *, kappa: float, **kw) -> float:
    """Fisher information saturation kernel: w = d² / (d² + κ).

    Motivated by the signal-to-noise structure of the Gaussian score.
    Default kernel for the Squeeze Kernel estimator.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be > 0.")
    return d2 / (d2 + kappa)


def kernel_exponential(d2: float, /, *, gamma: float, **kw) -> float:
    """Rayleigh survival (exponential) kernel: w = 1 − exp(−d²/(2γ²))."""
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0.")
    return 1.0 - math.exp(-d2 / (2.0 * gamma * gamma))


def kernel_chi2_cdf(d2: float, /, *, n_observed: int, **kw) -> float:
    """Chi-squared CDF kernel: w = F_χ²_N(N·d²).  Requires scipy."""
    try:
        from scipy.stats import chi2
    except ImportError as exc:
        raise ImportError(
            "kernel_chi2_cdf() requires SciPy. Install the optional "
            "'squeeze-kernel[full]' dependencies."
        ) from exc
    return float(chi2.cdf(n_observed * d2, df=n_observed))


def calibrate_kappa(d2_samples, target_mean_weight: float = 0.5) -> float:
    """Find κ such that E[d²/(d²+κ)] ≈ target_mean_weight on burn-in data.

    This implements the closed-form initializer from Proposition 1 of the paper,
    refined by one-dimensional root finding.

    Parameters
    ----------
    d2_samples : array-like
        Observed d̄²_t values from a burn-in window (use ``extract_d2_series``).
    target_mean_weight : float
        Target average kernel weight in (0, 1).

    Returns
    -------
    float
        Calibrated κ value.
    """
    import numpy as np
    try:
        from scipy.optimize import brentq
    except ImportError as exc:
        raise ImportError(
            "calibrate_kappa() requires SciPy. Install the optional "
            "'squeeze-kernel[full]' dependencies."
        ) from exc

    d2 = np.asarray(d2_samples, dtype=np.float64)
    d2 = d2[np.isfinite(d2) & (d2 > 0)]
    if len(d2) < 10:
        raise ValueError("Need at least 10 finite positive d² samples for calibration.")

    mu = float(d2.mean())
    # Closed-form initializer (Proposition 1): κ₀ = μ·(1/w₀ − 1)
    kappa_init = mu * (1.0 / target_mean_weight - 1.0)

    # Refine via root finding
    def residual(kappa):
        return float(np.mean(d2 / (d2 + kappa))) - target_mean_weight

    lo = max(kappa_init * 0.01, 1e-6)
    hi = kappa_init * 100.0
    # Ensure bracket
    while residual(lo) < 0:
        lo *= 0.1
    while residual(hi) > 0:
        hi *= 10.0

    return float(brentq(residual, lo, hi, xtol=1e-8))


def extract_d2_series(
    returns, lambda_vol: float = 0.94, epsilon: float = 1e-8,
):
    """Extract the d̄²_t series from a returns panel for κ calibration.

    Parameters
    ----------
    returns : array-like, shape (T, n)
        Daily return matrix.
    lambda_vol : float
        Volatility decay factor.

    Returns
    -------
    ndarray, shape (T,)
        Average squared standardized return at each timestep.
    """
    import numpy as np

    x = np.asarray(returns, dtype=np.float64)
    t_total, n = x.shape
    var_t = np.zeros(n, dtype=np.float64)
    var_init = np.zeros(n, dtype=bool)
    d2_out = np.empty(t_total, dtype=np.float64)
    one_minus_lv = 1.0 - lambda_vol

    for t in range(t_total):
        rt = x[t]
        finite = np.isfinite(rt)
        first = finite & ~var_init
        repeat = finite & var_init

        if np.any(first):
            var_t[first] = rt[first] ** 2 + epsilon
            var_init[first] = True
        if np.any(repeat):
            var_t[repeat] = lambda_vol * var_t[repeat] + one_minus_lv * rt[repeat] ** 2

        vol = np.sqrt(var_t)
        obs = finite & var_init
        n_obs = int(obs.sum())
        if n_obs > 0:
            z = rt[obs] / (vol[obs] + epsilon)
            d2_out[t] = float(z @ z) / n_obs
        else:
            d2_out[t] = float("nan")

    return d2_out
