# Changelog

## 3.1.1 — 2026-09-26

Performance only; every output is bit-identical to 3.1.0.

- **The blend gradient skips its eigenvalue floor on certified days.**
  The scored blend was floored at ``lambda_min >= 1e-8`` through a full
  symmetric eigenvalue computation every day, although the floor acts
  only in the first days of a panel.  The inverse the gradient needs
  anyway certifies ``lambda_min >= 1/||Sigma^-1||_F``; a day certified
  at ten times the floor is factorised once and is bit-identical to the
  floored path, every other day takes the floor exactly as before.
  Measured: the eigenvalue computation now runs on 3 of 7,840 days of
  the S&P n = 100 panel (5 of 7,648 at n = 300); on panels whose returns
  are small enough for the floor to bind (short-rate futures) it still
  runs, as it must, and such a day costs one extra Cholesky factorisation
  (the attempt that fails to certify).  Research engine and library stay
  bit-identical.
- New test ``tests/test_floor_certificate.py`` pins the exactness,
  including a panel on which the floor binds every day.
- New test ``tests/test_gram_representation.py`` pins the closed form of
  the correlation flow: ``Q_k = S^{-1/2} M_k S^{-1/2}`` with ``M_k`` the
  exponentially weighted Gram matrix of the clock-weighted returns
  ``sqrt(w) * z`` (missing assets and late listings included).

## 3.1.0 — 2026-09-19

The learned shrinkage split.  Same public surface, ``SqueezeKernel(lam)``;
the shrunk correlation of each timescale changes, so a minor version
with a behaviour change (the 3.0 path is ``split_learn=False`` on
``SqueezeKernelEstimator``, bit-for-bit).

- **Each timescale's shrunk correlation is a learned pool.**  Three PSD
  unit-diagonal experts -- the raw correlation, the equicorrelation
  matrix at its mean level, the Schur square -- are combined with
  weights whose prior is the intensity rule's morph ``(1-a, a(1-a),
  a^2)`` (exactly the 3.0 target) and which the blend gradient corrects
  online: the same exponentiated-gradient step as the timescale weights,
  studentised within the timescale, null temperature, fastest rung's
  memory, fixed share to the prior.  No new constant; one added state
  per timescale (three weights and a scale).
- Measured (research record, squeeze_cov R0K-11/12): equities 2022+
  held out −0.6 / −0.4 / −2.3 / −3.1 NLL per day vs 3.0 at n = 50 / 100 /
  200 / 300; diversified futures held out −1.7 per day; Boyd FF49 −0.55;
  known-truth Monte Carlo better on all 18 design × size cells.
- ``state()`` gains ``split`` (K x 3 learned weights).
- Blend-gradient gate: a day on which an observed asset has no variance
  state yet (it is first observed the next day) abstains, as the
  covariance factorisation always did; the split path scored such days
  before this fix.  Research engine and library agree to 1e-19 per day
  on staggered-missing panels; goldens regenerated.
- Cost: per day, 3K expert covariances and 3K gradient traces from the
  blend's existing factorisation, all O(n^2), no extra factorisation;
  in wall time 101 s for thirty years at n = 300 single-threaded against
  87 s for 3.0 (the gradients run on expert correlations against one
  whitened outer product; nothing per expert is materialised).

## 3.0.0 — 2026-09-08

The self-adapting estimator.  Same public surface, ``SqueezeKernel(lam)``;
different behaviour, so a major version.

- **Volatility on a ladder.**  The marginal variance is tracked on three
  decays derived from the anchor (half-lives h/16, h/4, h) and pooled
  with panel-wide weights: Bayes at the null-calibrated temperature on
  saturated (tanh) evidence, the fastest rung's memory, a uniform prior.
  Replaces the fitted constant ``lambda_vol``, which is now ignored by
  the front-end (still honoured by ``SqueezeKernelEstimator`` when
  ``vol_ladder=False``).
- **Timescale mixture by exponentiated gradient on the blend.**  The rung
  covariances are mixed by EG on the blend's own log score at the null
  temperature, fast-rung memory, prior ∝ √h.  Replaces the two-sided
  CUSUM detector and its constants (drift, snap, thresholds, fast/slow
  priors); ``weights="cusum"`` keeps the 2.x behaviour bit-for-bit.
- No fitted constant remains.  Cost: one extra triangular solve per day
  for the blend gradient (about 2.3x the 2.x update at n=300).
- Equivalence to the research engine's promoted row is pinned to
  0.00e+00 on the synthetic golden path.

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
