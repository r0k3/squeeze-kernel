# Squeeze Kernel Covariance Estimator

[![CI](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/r0k3/squeeze-kernel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/squeeze-kernel.svg)](https://pypi.org/project/squeeze-kernel/)
[![Python](https://img.shields.io/pypi/pyversions/squeeze-kernel.svg)](https://pypi.org/project/squeeze-kernel/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **streaming covariance estimator for panels of financial returns** whose entire public surface is **one number** — the decay `lam` of the anchor correlation timescale. Every other quantity is derived from it, fixed by a structural argument, or computed online from the estimator's own state. One `O(n²)` update per day, positive semi-definite **by construction**, missing values handled **natively**, no tuning, no refits. Only dependency: NumPy.

```python
from squeeze_kernel import SqueezeKernel

sk = SqueezeKernel(lam=0.996)        # the entire public surface
for r_t in returns:                  # NaN marks missing assets
    sk.update(r_t)
cov = sk.covariance()
```

Reference: *"The Squeeze Kernel Covariance Estimator: Dual-Timescale Tracking with Adaptive Shrinkage"* (Kende, 2026) — [SSRN abstract 6455918](https://ssrn.com/abstract=6455918); the 2.0 estimator is described in the paper's current revision.

## Why

Markets do not keep calendar time. Following Mandelbrot, the estimator treats a panel as a collection of partially coupled markets, **each advancing on its own activity-driven clock** — and reads those clocks from the panel's own correlation structure, so a hot cluster (say precious metals and FX) advances its correlation state while an idle one (agriculture) does not, without anyone identifying a cluster. On those clocks it runs a single recursion that:

- **is PSD at every step, structurally** — the correlation state evolves by a diagonal-congruence flow (a congruence plus a rank-one term); no eigenvalue clipping, no nearest-PSD repair, no solver on the online path;
- **learns in market time and forgets in calendar time** — observations enter with a saturating, self-studentising weight (no day counts more than one unit of trading time); memory decays at fixed per-day rates on a geometric ladder of three timescales `(lam⁴, lam, lam^¼)`;
- **regularises itself** — each timescale's shrinkage intensity is computed from two online statistics, the concentration `n/ν` (dimension per unit trading time) and the de-noised fraction of correlation dispersion the target explains; the target is the Hadamard square of the running correlation (cluster-respecting, PSD by the Schur product theorem);
- **adapts its memory to regime breaks** — a Page-CUSUM detector on the inter-timescale score drift, under an explicit two-year false-alarm budget, reallocates weight across timescales and self-silences where no break signatures exist;
- **ingests missing values natively** — listings, delistings, halts enter as `NaN`;
- **is fast** — a thirty-year daily pass at n=300 takes ~40 s single-threaded, two orders of magnitude under daily rolling-window refits.

**Evidence.** On thirty years of S&P 500 constituents against an eleven-method field (EWMA, DCC, Ledoit–Wolf, OAS, nonlinear shrinkage, RMT filtering, Gerber, IEWMA, CM-IEWMA, and the published v1 estimator) it leads at every universe size from 50 to 300 and is the **sole member of the 90% model confidence set at every size**. Carried **zero-shot** to a diversified panel of 121 futures across eight asset classes it beats the same field *calibrated on that panel's own history* — matched-backbone IEWMA by 6.9 NLL/day (p = 4·10⁻⁴), calibrated DCC by 17.9 — out-of-time.

## See the difference

A passive strategy any allocator would recognize: long-only minimum-variance over 300 liquid US stocks, scaled to a 15% volatility target, rebalanced monthly, 5 bps costs. Two runs on identical data; the only difference is the covariance matrix. The Squeeze Kernel arm runs `SqueezeKernel(lam=0.996)` — **nothing tuned on this panel**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/r0k3/squeeze-kernel/main/examples/figures/vol_targeted_portfolio_dark.png">
  <img alt="Vol-targeted long-only minimum-variance portfolio on 300 US equities: Squeeze Kernel vs Ledoit-Wolf equity curves, drawdown, realized volatility, and risk/return profile" src="https://raw.githubusercontent.com/r0k3/squeeze-kernel/main/examples/figures/vol_targeted_portfolio.png">
</picture>

| Method | CAGR | Vol | Sharpe | MaxDD | Calmar | Vol-target RMSE |
|---|---|---|---|---|---|---|
| **Squeeze Kernel (default)** | **12.4%** | **13.5%** | **0.91** | **-34.6%** | **0.36** | **7.10%** |
| Ledoit-Wolf (252d) | 11.7% | 14.6% | 0.80 | -38.7% | 0.30 | 7.51% |

Reproduce from the repo alone (the 300-stock panel ships as a parquet; survivorship and provenance are documented in the script):

```bash
pip install squeeze-kernel pandas pyarrow scikit-learn matplotlib
python examples/vol_targeted_portfolio.py    # ~2 minutes
```

## Installation

```bash
pip install squeeze-kernel          # NumPy only
pip install "squeeze-kernel[full]"  # + SciPy (faster detector factorisations)
```

## Quickstart

```python
import numpy as np
from squeeze_kernel import SqueezeKernel, estimate_squeeze_cov

returns = np.random.default_rng(42).normal(0.0, 0.01, size=(500, 30))

sk = SqueezeKernel()                 # lam=0.996 (anchor half-life ~173 days)
for r_t in returns:
    w = sk.update(r_t)               # returns the day's kernel weight
cov, corr = sk.covariance(), sk.correlation()
sk.state()                           # kernel scale, per-timescale effective sizes, detector tilt

# batch mode: full panel in, covariance path out
cov_path, corr_path, weights = estimate_squeeze_cov(returns, with_weights=True)
```

Missing values: pass `NaN` (or `mask=` on `update`). Newly listed, delisted or halted assets need no imputation and no complete-case subsetting.

## What derives from `lam`

| quantity | value |
|---|---|
| timescale ladder | decays `(lam⁴, lam, lam^¼)` — half-lives `(h/4, h, 4h)`, `h = -1/log2(lam)` |
| kernel scale | state: `κ_t = ⅓ · EWMA(activity)` at the anchor rate |
| shrinkage intensity | per timescale, `α = min(1,c) · g̃²/(g̃² + (1−g̃)²·max(0, 1/c − 1))` from the online concentration `c = n/ν` and target-fit `g̃` |
| timescale weights | prior ∝ √h, tilted by the surprise detector |
| structural constants | K=3, b=4, θ=½, κ-scale ⅓, Schur power 2, detector budget — each bracketed by ablation in the paper |
| the one empirical constant | volatility clock `λ_v = 0.98`, disclosed |

`from squeeze_kernel import CONSTANTS` exposes the structural constants for research. The published v1 estimator (all its knobs) remains available as `SqueezeKernelEstimator` / `SqueezeKernel.v1(...)`; every 2.0 mechanism is also an estimator-level switch for ablation. See [MIGRATION.md](MIGRATION.md).

## How it works

One daily update: variance EWMA per asset → standardised surprise → per-asset clock increments from the Schur-square-weighted neighbourhood mean of squared surprises → diagonal-congruence update of each timescale's correlation state on those clocks → per-timescale self-tuning shrinkage toward the Hadamard-square target → surprise-gated blend across timescales → covariance. The paper gives the derivations, guarantees (PSD, conditioning floor, exact reductions to the published special cases), and the full evaluation.

## Development

```bash
uv sync --extra full --extra dev
uv run python -m pytest        # test suite
uv run python -m ruff check .  # lint
uv run mypy                      # strict type check (src/squeeze_kernel)
uv build                       # build sdist + wheel
```

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
