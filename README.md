# Squeeze Kernel Covariance Estimator

[![CI](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/squeeze-kernel.svg)](https://pypi.org/project/squeeze-kernel/)
[![Python](https://img.shields.io/pypi/pyversions/squeeze-kernel.svg)](https://pypi.org/project/squeeze-kernel/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **streaming covariance estimator for panels of daily financial returns**. One `O(n²)` update per day, positive semi-definite **by construction** at every step, missing values handled **natively**, and defaults that require no tuning. Only dependency: NumPy.

Reference: *"The Squeeze Kernel Covariance Estimator: Dual-Timescale Tracking with Adaptive Shrinkage"* (Kende, 2026) — [SSRN abstract 6455918](https://ssrn.com/abstract=6455918).

## Why

Rolling-window estimators (Ledoit–Wolf, nonlinear shrinkage, RMT denoising) refit over a fixed window each day and cannot adapt within it; multivariate GARCH (DCC) adapts but needs multi-stage estimation and a fragile news coefficient. The Squeeze Kernel is a single streaming recursion that:

- **is PSD at every step, structurally** — never needs eigenvalue clipping, nearest-PSD projection, or a solver;
- **adapts fastest exactly when it matters** — a Fisher-information kernel up-weights high-dispersion (stress) days, when correlation regimes actually move;
- **regularises itself** — an adaptive equicorrelation shrinkage activates automatically as the asset count approaches the effective sample size, with a provable condition-number bound;
- **ingests missing values natively** — listings, delistings, and halts enter as `NaN`; no imputation or complete-case subsetting;
- **is fast** — a full 30-year daily pass takes ~0.75 s at n=100 and ~3.4 s at n=300 (single-threaded), 30–40× faster than rolling-window baselines at scale.

On a 30-year S&P 500 panel (n=100, ~7,600 out-of-sample days) it statistically ties DCC on one-step density forecasts and beats EWMA, Ledoit–Wolf, OAS, nonlinear shrinkage, RMT denoising, and the Gerber statistic — and it is the only method in the 90% model confidence set together with DCC. At n=300 it leads every competitor that remains statistically viable.

## Installation

```bash
pip install squeeze-kernel          # NumPy only
pip install "squeeze-kernel[full]"  # + SciPy (kappa calibration, chi² kernel)
```

## Quickstart

```python
import numpy as np
from squeeze_kernel import SqueezeKernelEstimator

# daily_returns: array of shape (T, n) — may contain NaN for missing assets
est = SqueezeKernelEstimator(n_assets=daily_returns.shape[1])

for r_t in daily_returns:          # stream one day at a time
    est.update(r_t)

cov  = est.get_cov()               # (n, n) covariance, PSD by construction
corr = est.get_corr()              # (n, n) correlation
```

That is the whole API for most uses. The defaults (`lambda_vol=0.98`, `lambda_corr=0.996`, `kappa=0.25`) are the paper-recommended settings for daily returns, selected by time-series cross-validation and robust across a 50× parameter sweep — deploy them as-is.

Batch mode, if you prefer the full path in one call:

```python
from squeeze_kernel import estimate_squeeze_cov

cov_path, corr_path, weights = estimate_squeeze_cov(daily_returns, with_weights=True)
# cov_path: (T, n, n) — the estimate after each day
```

A complete runnable walkthrough (streaming, missing data, batch) is in [`examples/quickstart.py`](examples/quickstart.py).

## Missing values

Pass `NaN` for any asset not observed on a given day — nothing else to do:

```python
r_t = np.array([0.004, np.nan, -0.011])   # asset 2 not trading today
est.update(r_t)                            # PSD preserved, no imputation
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `lambda_vol` | `0.98` | volatility EWMA decay (half-life ≈ 34 days) |
| `lambda_corr` | `0.996` | correlation EWMA decay (half-life ≈ 173 days, T_eff ≈ 250) |
| `kappa` | `0.25` | Fisher kernel saturation; higher = stronger calm-day filtering |
| `shrinkage` | `"auto"` | adaptive equicorrelation shrinkage (`"none"` or a float to override) |
| `shrinkage_delta` | `0.10` | concentration threshold at which shrinkage activates |

Useful read-only state after each `update()`: `est.weight` (last kernel weight), `est.effective_sample_size` (kernel-weighted T_eff), `est.shrinkage_intensity` (current α).

To recalibrate `kappa` for a different asset class (requires the `full` extra):

```python
kappa = SqueezeKernelEstimator.calibrate_kappa(burn_in_returns, target_weight=0.6)
```

## Advanced options

**Score-exact weighting** (`weight_statistic="mahalanobis"`, use with `kappa=1.0`): drives the kernel with the Mahalanobis surprise `z'C⁻¹z/N` against the estimator's own correlation instead of the marginal dispersion. Improves accuracy in the moderate-concentration regime — use only when `n / T_eff ≲ 0.5` (e.g. n ≤ 100 at the default `lambda_corr`); at higher concentration the estimated inverse degrades it and the default is strictly better.

```python
est = SqueezeKernelEstimator(n_assets=100, kappa=1.0, weight_statistic="mahalanobis")
```

**Score-driven memory** (`lambda_corr_fast=0.99`): lets stress days also *shorten* the correlation memory (decay slides from `lambda_corr` toward `lambda_corr_fast` as the kernel weight rises). Do **not** combine with the Mahalanobis option — they act on the same channel and the combination degrades accuracy.

**OU volatility anchor** (`vol_anchor_phi=0.995`): mean-reverts each asset's variance prediction toward a slow per-asset anchor (a ~1000-day EWMA of squared returns) before the daily update — a two-timescale, component-style volatility structure. One global parameter with a clean interpretation (deviation half-life ≈ ln 2/(1−φ) days; φ=0.995 ≈ 139 d). On the S&P 500 n=100 benchmark this improved held-out one-step NLL by 3.3 points (4.3 at φ=0.99) and five-step NLL by 3.9 (5.0), with no degradation at n=300. `None` (default) or φ=1 reproduces the published estimator exactly.

```python
est = SqueezeKernelEstimator(n_assets=100, vol_anchor_phi=0.995)
```

**Cluster shrinkage target** (`shrinkage_target="cluster"`): generalizes the equicorrelation shrinkage target to respect the correlation matrix's own block/cluster structure — with **no clustering algorithm**. The target morphs with the shrinkage intensity, T = (1−α)·T_equi + α·[(1−γ)I + γ·(C∘C)], where C∘C is the Hadamard square of the current correlation (positive semi-definite by the Schur product theorem; entries are pairwise shared-variance fractions) and γ is level-matched automatically. Zero added parameters, still O(n²), and as α→0 it reduces exactly to the default estimator. Held-out one-step NLL on the S&P 500 benchmark: ±0.1 at n=100, **−4.2 at n=200, −25.0 at n=300** — recommended whenever the universe size approaches the effective sample size.

```python
est = SqueezeKernelEstimator(n_assets=300, shrinkage_target="cluster")
```

**Alternative kernels**: pass `kernel_fn=kernel_exponential` (with `kernel_kwargs={"gamma": ...}`) or `kernel_chi2_cdf`, or any callable `(d2, *, n_observed, **kw) -> float` mapping to `[0, 1)`. The PSD guarantee holds for any such kernel.

## How it works

Three mechanisms in one recursion:

1. **Dual-timescale EWMA** — fast per-asset volatility (`lambda_vol`) is separated from slow correlation dynamics (`lambda_corr`), so variance shocks don't contaminate the correlation estimate.
2. **Fisher kernel weighting** — each day's standardized outer product enters with weight `w = d²/(d² + kappa)`, where `d²` is the mean squared standardized return: calm days contribute little, dispersion shocks contribute fully.
3. **Adaptive equicorrelation shrinkage** — `alpha = min(1, max(0, n/(2·S) − delta))` blends toward an equicorrelation target using the estimator's own kernel-weighted sample size `S`; it is a no-op at low dimension and provides provably bounded conditioning at high dimension.

The complete update is a natural-gradient step on the Gaussian log-likelihood, with the kernel weight acting as an adaptive Riemannian learning rate (paper, Appendix B).

## Development

```bash
uv sync --extra full --extra dev
uv run python -m pytest        # test suite
uv run python -m ruff check .  # lint
uv build                       # build sdist + wheel
```

Releases: publishing a GitHub release from a `v*` tag triggers the [publish workflow](.github/workflows/publish.yml), which builds and uploads to PyPI via trusted publishing.

## Citation

```bibtex
@article{kende2026squeeze,
  title  = {The Squeeze Kernel Covariance Estimator: Dual-Timescale Tracking with Adaptive Shrinkage},
  author = {Kende, Robert},
  year   = {2026},
  note   = {Available at SSRN: \url{https://ssrn.com/abstract=6455918}}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT
