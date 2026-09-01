# Changelog

## 2.0.0 (2026-09-01)

Breaking: the public estimator is `SqueezeKernel(lam=0.996)` — one
number, the exponential decay of the anchor correlation timescale.
Everything else derives from `lam`, is a frozen structural constant
(`CONSTANTS`), or is self-tuning state.

- Per-asset market clocks: each asset's trading time advances with the
  squared surprise of its correlation neighbourhood (row-normalised
  Schur-square weighting), so clusters get their own clocks with no
  cluster identification; rung update is the PSD diagonal congruence
  `Q_ij <- sqrt((1-eta_i)(1-eta_j)) Q_ij + sqrt(eta_i eta_j) z_i z_j`,
  exactly the published chord when clocks coincide.
- Self-tuning shrinkage intensity per timescale from the online
  concentration `n/nu` and the de-noised target-fit fraction; the
  `shrinkage_delta` offset no longer exists.
- Kernel scale as state (`kappa_t = EWMA(activity)/3`); timescale
  ladder `(lam**4, lam, lam**(1/4))`; theta = 1/2; gamma = 1 Schur-square
  cluster target; surprise detector always on.
- `SqueezeKernelEstimator` (v1) is unchanged and bit-exact; the 2.0
  mechanics are its opt-ins `clock`, `kappa_mode`, `alpha_rule`,
  `level_match`, `detector`, plus the research variant
  `alpha_rule="selftuning-target"`.
- Cross-engine golden tests pin the 2.0 default and three ablation
  variants to the research pipeline's per-day log-scores (<1e-8).
- See MIGRATION.md for the mapping of every v1 knob.

## 0.7.1

Last 1.x release (published configuration; see git history).
