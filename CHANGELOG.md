# Changelog

## 1.0.0 — 2026-11-27

First release of the estimator under this version line.

- ``SqueezeKernel(lam)``: streaming covariance and correlation forecasts
  with per-asset market clocks and self-tuning shrinkage; one number,
  ``lam``, default 0.996.  NaN- and mask-native.
- ``estimate_squeeze_cov``: the same estimator over a whole returns panel.
- Read-only diagnostics: ``rung_correlations()`` and, in ``state()``,
  ``m_t``, ``n_t``, ``clock_t`` and ``asset_nu``.
