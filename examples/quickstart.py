"""Squeeze Kernel quickstart — the complete public API in one screen.

Runs with numpy only:  python examples/quickstart.py
"""

import numpy as np

from squeeze_kernel import SqueezeKernel, estimate_squeeze_cov

# ── Synthetic daily returns: 2 correlated assets + 1 independent, 500 days ──
rng = np.random.default_rng(42)
factor = rng.normal(0, 0.01, 500)
returns = np.column_stack([
    factor + rng.normal(0, 0.005, 500),
    0.8 * factor + rng.normal(0, 0.006, 500),
    rng.normal(0, 0.012, 500),
])

# ── 1. Streaming usage — one number, no tuning ──────────────────────────────
sk = SqueezeKernel(lam=0.996)      # decay of the anchor correlation timescale
for r_t in returns:
    sk.update(r_t)                 # one O(n^2) update per day

print("Covariance matrix:")
print(sk.covariance().round(8))
print("\nCorrelation matrix (PSD by construction, no projection ever needed):")
print(sk.correlation().round(3))
st = sk.state()
print(f"\nkernel scale (state): {st['kappa_t']:.3f}   "
      f"effective sizes per timescale: {[round(v) for v in st['rung_nu']]}")

# ── 2. Missing values are native ─────────────────────────────────────────────
returns_gappy = returns.copy()
returns_gappy[100:180, 2] = np.nan          # asset 3 goes dark for 80 days
sk2 = SqueezeKernel()
for r_t in returns_gappy:
    sk2.update(r_t)
print("\nCorrelation with an 80-day gap in asset 3 (still PSD, still finite):")
print(sk2.correlation().round(3))

# ── 3. Batch mode: full panel in, covariance path out ────────────────────────
cov_path, corr_path, weights = estimate_squeeze_cov(
    returns, with_corr=True, with_weights=True,
)
print(f"\nBatch output: cov {cov_path.shape}, corr {corr_path.shape}, "
      f"weights {weights.shape}; mean weight {weights.mean():.3f}")
