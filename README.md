# Squeeze Kernel Covariance Estimator

[![CI](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A streaming, PSD-by-construction covariance estimator with adaptive shrinkage for financial applications.

**Paper:** "The Squeeze Kernel Covariance Estimator: Dual-Timescale Tracking with Adaptive Shrinkage" (Kende, 2026)

## Features

- **PSD by construction** at every time step (no eigenvalue clipping needed)
- **Adaptive shrinkage** automatically calibrates regularisation to the dimension/sample-size ratio
- **Native missing data handling** via masked EWMA updates
- **O(n²) streaming** — single-pass, no matrix decomposition during online operation
- **Pluggable kernels** — Fisher (default), exponential, χ² CDF, or custom

## Installation

```bash
# From PyPI (once released)
pip install squeeze-kernel
pip install "squeeze-kernel[full]"

# Directly from GitHub
pip install "git+https://github.com/r0k3/squeeze-kernel.git"
pip install "squeeze-kernel[full] @ git+https://github.com/r0k3/squeeze-kernel.git"

# From a local clone
pip install .
pip install ".[full]"
```

## Simple Usage

```python
import numpy as np

from squeeze_kernel import SqueezeKernelEstimator

# Synthetic daily returns for 3 assets over 250 days
rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, size=(250, 3))

# Stream returns one day at a time
est = SqueezeKernelEstimator(n_assets=3, kappa=1.5)
for r_t in returns:
    est.update(r_t)

# Get current estimates
cov = est.get_cov()
corr = est.get_corr()

print(cov.shape)                  # (3, 3)
print(corr.shape)                 # (3, 3)
print(est.effective_sample_size)  # kernel-weighted sample size
print(est.weight)                 # last kernel weight
```

Missing observations can be passed as `np.nan`; the estimator handles them
without breaking positive semi-definiteness.

## Quick Start

If you already have a returns matrix `daily_returns` with shape `(T, n_assets)`:

```python
from squeeze_kernel import SqueezeKernelEstimator

est = SqueezeKernelEstimator(n_assets=50, kappa=0.25)
for r_t in daily_returns:
    est.update(r_t)

cov = est.get_cov()
corr = est.get_corr()
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa` | 1.5 | Fisher kernel saturation used when `kernel_fn` is omitted |
| `kernel_fn` | `None` | Optional alternative kernel function |
| `kernel_kwargs` | `None` | Keyword arguments forwarded to `kernel_fn` |
| `lambda_vol` | 0.94 | Volatility EWMA decay (half-life ≈ 12 days) |
| `lambda_corr` | 0.99 | Correlation EWMA decay (half-life ≈ 69 days) |
| `shrinkage` | `"auto"` | `"auto"`, `"none"`, or fixed float in [0, 1] |
| `shrinkage_delta` | 0.10 | Threshold for adaptive shrinkage activation |

## Calibrating κ from Data

```python
# Calibrate κ so the average kernel weight is ~0.6 on burn-in data
kappa = SqueezeKernelEstimator.calibrate_kappa(
    burn_in_returns, target_weight=0.6
)
est = SqueezeKernelEstimator(n_assets=50, kappa=kappa)
```

`calibrate_kappa()` and `kernel_chi2_cdf()` require SciPy. Install the `full`
extra if you want those helpers.

## Alternative Kernels

The estimator defaults to the Fisher kernel. To use another built-in kernel,
supply it explicitly with `kernel_fn` and pass its parameters via
`kernel_kwargs`.

```python
from squeeze_kernel import SqueezeKernelEstimator, kernel_exponential

est = SqueezeKernelEstimator(
    n_assets=3,
    kernel_fn=kernel_exponential,
    kernel_kwargs={"gamma": 1.0},
)
```

`gamma` is not an estimator argument; it belongs to the exponential kernel.

## Batch Mode

```python
from squeeze_kernel import estimate_squeeze_cov

cov_tensor, corr_tensor, weights = estimate_squeeze_cov(
    returns_matrix,  # shape (T, n)
    kappa=0.25,
    with_corr=True,
    with_weights=True,
)
```

## Examples

The `examples/` directory contains end-to-end scripts:

- `quickstart.py` for a compact API walkthrough
- `portfolio_voltarget.py` for a simple volatility-targeted portfolio backtest

## How It Works

The estimator combines three components:

1. **Dual-timescale EWMA**: Fast volatility tracking (λ_vol) separated from slow correlation tracking (λ_corr)
2. **Fisher kernel weighting**: w_t = d̄²_t / (d̄²_t + κ) filters uninformative calm-day observations
3. **Adaptive equicorrelation shrinkage**: α_t = max(0, n/(2·S_t) − δ) automatically regularises when the effective sample size is small relative to dimension

The complete update is a **natural gradient** step on the Gaussian log-likelihood (see paper, Appendix C).

## Development

```bash
uv sync --extra full --extra dev
uv run python -m pytest
uv run python -m ruff check .
uv build
```

## Citation

```bibtex
@article{kende2026squeeze,
  title={The Squeeze Kernel Covariance Estimator: Dual-Timescale Tracking with Adaptive Shrinkage},
  author={Kende, Robert},
  year={2026}
}
```

## License

MIT
