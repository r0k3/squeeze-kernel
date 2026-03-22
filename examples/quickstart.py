"""Quick start: Squeeze Kernel on synthetic data."""

import numpy as np
from squeeze_kernel import SqueezeKernelEstimator, estimate_squeeze_cov

# Generate synthetic returns (3 correlated assets, 500 days)
rng = np.random.default_rng(42)
factor = rng.normal(0, 0.01, 500)
returns = np.column_stack([
    factor + rng.normal(0, 0.005, 500),
    0.8 * factor + rng.normal(0, 0.006, 500),
    rng.normal(0, 0.012, 500),  # uncorrelated
])

# ── Basic usage (default Fisher kernel via kappa) ──
est = SqueezeKernelEstimator(n_assets=3, kappa=1.5)
for r_t in returns:
    est.update(r_t)

print("Final covariance matrix:")
print(est.get_cov())
print("\nFinal correlation matrix:")
print(est.get_corr())
print(f"\nEffective sample size: {est.effective_sample_size:.1f}")
print(f"Shrinkage intensity: {est.shrinkage_intensity:.4f}")
print(f"Last kernel weight: {est.weight:.4f}")

# ── Calibrate kappa from burn-in data ──
kappa_cal = SqueezeKernelEstimator.calibrate_kappa(
    returns[:100], target_weight=0.6, lambda_vol=0.94,
)
print(f"\nCalibrated kappa (target weight=0.6): {kappa_cal:.3f}")

# ── Batch mode ──
cov_tensor, corr_tensor, weights = estimate_squeeze_cov(
    returns, kappa=1.5, with_corr=True, with_weights=True,
)
print(f"\nBatch output shapes: cov={cov_tensor.shape}, weights={weights.shape}")
print(f"Average kernel weight: {weights.mean():.3f}")
