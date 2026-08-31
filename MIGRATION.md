# Migrating to squeeze-kernel 2.0

2.0 introduces `SqueezeKernel` — a one-number public surface. The v1
estimator is unchanged and remains available as `SqueezeKernelEstimator`
(or `SqueezeKernel.v1(...)`); nothing you run today breaks, and the v1
golden tests still pin the published numbers bit-for-bit.

```python
from squeeze_kernel import SqueezeKernel

sk = SqueezeKernel(half_life=173, detector=True, cluster_target=True)
for r_t in returns:          # NaN marks missing assets
    sk.update(r_t)
cov = sk.covariance()
```

## Where every v1 knob went

| v1 parameter | 2.0 status |
|---|---|
| `n_assets` | inferred from the first `update` |
| `lambda_vol` | frozen structural constant 0.98 (`CONSTANTS.lambda_vol`). The `h/b` derivation was falsified by crisis sub-periods; 0.98 is the single remaining empirically set constant, disclosed as such |
| `lambda_corr` (single scale) | removed — the ladder is the estimator; single-scale remains a v1 mode |
| `corr_half_lives` | derived: `half_life * (43, 173, 693) / 173` |
| `corr_theta` | frozen at ½ (rung weights ∝ √h; θ=0 fails the n=300 veto) |
| `kappa` | state, not parameter: κ_t = ⅓·EWMA_h(d²) (χ²-null constant ⅓) |
| `shrinkage` / `shrinkage_delta` | self-tuning per-rung intensity α = min(1,c)·g̃²/(g̃²+(1−g̃)²·max(0,1/c−1)) from the online concentration c = n/ν and de-noised equicorrelation-explained fraction g̃ — δ is gone |
| `shrinkage_target` | `cluster_target` boolean; the cluster target is the γ=1 Hadamard square (p=2, veto-selected; PSD by the Schur product theorem) |
| detector (always-on in v1 ladder mode) | `detector` boolean |
| `weight_statistic="mahalanobis"` | removed from 2.0 (post-mortem: the equicorrelation surprise discount is crisis-regressive); v1 opt-in unchanged |
| `lambda_corr_fast`, `vol_anchor_phi/decay` | v1 opt-ins, unchanged there; not part of 2.0 |
| `impute_missing`, `min_obs`, `epsilon` | v1 options; 2.0 handles missing via NaN/mask natively with `epsilon` structural |

## Scope

2.0 is tuned for concentration (n from ~50 into the hundreds against
effective sample sizes of months–years). The self-tuning intensity makes
it safe on low-concentration, weakly-factored panels too (measured on
signed multi-asset futures; see the papers' evaluation section) — it
shrinks each rung only as much as the online bias–variance evidence
supports.

## Reproducing v1

```python
est = SqueezeKernel.v1(n_assets, lambda_vol=0.98, kappa=0.25,
                       shrinkage="auto", shrinkage_delta=0.10,
                       corr_half_lives=(43., 173., 693.), corr_theta=0.25)
```
reproduces the published adaptive-headline numbers exactly (CI-pinned).
