"""Example: Volatility-targeted portfolio using the Squeeze Kernel.

Constructs a GMV portfolio from the Squeeze Kernel covariance forecast,
scaled to a target annualised volatility, with weekly rebalancing.
"""

import numpy as np
from squeeze_kernel import SqueezeKernelEstimator

# ── Generate synthetic multi-asset returns ──
rng = np.random.default_rng(123)
n_assets, n_days = 20, 2000
# Factor model: 2 factors + idiosyncratic
loadings = rng.normal(0, 1, (n_assets, 2))
factors = rng.normal(0, 0.005, (n_days, 2))
returns = factors @ loadings.T + rng.normal(0, 0.008, (n_days, n_assets))

# ── Parameters ──
VOL_TARGET = 0.10        # 10% annualised target
REBAL_FREQ = 5           # Weekly rebalancing
LOOKBACK = 252           # Burn-in period
TC_BPS = 2.0             # Transaction costs (bps per unit turnover)

# ── Run backtest ──
est = SqueezeKernelEstimator(n_assets, kappa=0.25, lambda_vol=0.96, lambda_corr=0.995)
portfolio_returns = []
current_weights = None
tc_rate = TC_BPS / 10_000

for t in range(n_days - 1):
    est.update(returns[t])

    if t < LOOKBACK - 1:
        continue

    is_rebal = ((t - LOOKBACK + 1) % REBAL_FREQ == 0) or current_weights is None

    if is_rebal:
        cov = est.get_cov()
        # GMV weights: w = Σ⁻¹·1 / (1'·Σ⁻¹·1)
        ones = np.ones(n_assets)
        w = np.linalg.solve(cov, ones)
        w /= w.sum()

        # Vol-target: scale to σ_target
        sigma_forecast = np.sqrt(w @ cov @ w) * np.sqrt(252)
        leverage = min(VOL_TARGET / max(sigma_forecast, 1e-8), 5.0)
        new_weights = w * leverage

        # Transaction costs
        tc = 0.0 if current_weights is None else tc_rate * np.abs(new_weights - current_weights).sum()
        current_weights = new_weights
    else:
        tc = 0.0

    # Realised return
    ret = float(current_weights @ returns[t + 1]) - tc
    portfolio_returns.append(ret)

# ── Summary statistics ──
rets = np.array(portfolio_returns)
ann_ret = rets.mean() * 252
ann_vol = rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
cumret = np.cumprod(1 + rets)
maxdd = float(np.min(cumret / np.maximum.accumulate(cumret)) - 1)

print(f"Portfolio summary ({n_assets} assets, {len(rets)} days)")
print(f"  Ann. return:  {ann_ret*100:+.2f}%")
print(f"  Ann. vol:     {ann_vol*100:.2f}%  (target: {VOL_TARGET*100:.0f}%)")
print(f"  Sharpe:       {sharpe:.3f}")
print(f"  Max drawdown: {maxdd*100:.1f}%")
print(f"  Eff. sample:  {est.effective_sample_size:.1f}")
print(f"  Shrinkage α:  {est.shrinkage_intensity:.4f}")
