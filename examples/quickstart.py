"""Squeeze Kernel quickstart — a basic but complete tour of the API.

Runs with numpy only:  python examples/quickstart.py
"""

import numpy as np

from squeeze_kernel import SqueezeKernelEstimator, estimate_squeeze_cov

# ── Synthetic daily returns: 2 correlated assets + 1 independent, 500 days ──
rng = np.random.default_rng(42)
factor = rng.normal(0, 0.01, 500)
returns = np.column_stack([
    factor + rng.normal(0, 0.005, 500),
    0.8 * factor + rng.normal(0, 0.006, 500),
    rng.normal(0, 0.012, 500),
])

# ── 1. Streaming usage (the primary API) ─────────────────────────────────────
# The defaults (lambda_vol=0.98, lambda_corr=0.996, kappa=0.25) are the
# paper-recommended settings for panels of daily financial returns —
# no tuning required.
est = SqueezeKernelEstimator(n_assets=3)

for r_t in returns:
    est.update(r_t)          # one O(n^2) update per day; returns the kernel weight

print("Covariance matrix:")
print(est.get_cov().round(8))
print("\nCorrelation matrix (PSD by construction, no projection ever needed):")
print(est.get_corr().round(3))
print(f"\nLast kernel weight:        {est.weight:.3f}")
print(f"Effective sample size:     {est.effective_sample_size:.1f}")
print(f"Shrinkage intensity:       {est.shrinkage_intensity:.4f}  (self-activates when n approaches T_eff)")

# ── 2. Missing values are handled natively ───────────────────────────────────
# Newly listed, delisted, or halted assets: just pass NaN. No imputation,
# no zero-filling, no complete-case subsetting — and PSD is preserved.
returns_gappy = returns.copy()
returns_gappy[100:180, 2] = np.nan          # asset 3 goes dark for 80 days

est2 = SqueezeKernelEstimator(n_assets=3)
for r_t in returns_gappy:
    est2.update(r_t)

print("\nCorrelation with an 80-day gap in asset 3 (still PSD, still finite):")
print(est2.get_corr().round(3))

# ── 3. Batch mode: full panel in, covariance path out ────────────────────────
cov_path, corr_path, weights = estimate_squeeze_cov(
    returns, with_corr=True, with_weights=True,
)
print(f"\nBatch output: cov {cov_path.shape}, corr {corr_path.shape}, weights {weights.shape}")
print(f"Average kernel weight over the sample: {weights.mean():.3f}")
