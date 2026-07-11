"""Deterministic input paths for the golden regression tests.

Each scenario returns a (T, n) float array (NaN = missing) built from a
seeded generator, so the inputs are reproducible from source. The golden
outputs in tests/golden/*.npz pin the exact covariance path of the
combined cluster + scale-free estimator (the papers' headline
configuration) as shipped in v0.5.0.
"""

import numpy as np

N = 40
LADDER = [43.0, 173.0, 693.0]
THETA = 0.25

ESTIMATOR_KWARGS = dict(
    corr_half_lives=LADDER,
    corr_theta=THETA,
    shrinkage_target="cluster",
)


def _block_returns(rng, t, within=0.6, cross=0.0):
    """Two 20-asset blocks with the given within/cross correlations."""
    corr = np.full((N, N), cross)
    corr[:20, :20] = within
    corr[20:, 20:] = within
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr + 1e-10 * np.eye(N))
    return 0.01 * rng.standard_normal((t, N)) @ chol.T


def complete_blocks():
    """Complete observations, positive two-block structure, T short enough
    that the per-rung shrinkage (and hence the cluster morph) is active."""
    return _block_returns(np.random.default_rng(1), 250)


def staggered_missing():
    """Late-listing block, MCAR holes, and one all-missing day."""
    rng = np.random.default_rng(2)
    x = _block_returns(rng, 300)
    x[:100, 30:] = np.nan                       # assets 30-39 list late
    x[rng.random(x.shape) < 0.05] = np.nan      # 5% MCAR
    x[150, :] = np.nan                          # market holiday
    return x


def negative_blocks():
    """Anti-correlated blocks so mean off-diagonal correlation hovers just
    below zero, exercising the equicorrelation fallback of the cluster
    target (population mean off-diagonal = -0.019; min eigenvalue 0.25)."""
    return _block_returns(np.random.default_rng(3), 250, within=0.15, cross=-0.18)


def crisis_shock():
    """Volatility regime shift: a 20-day window of 6x returns."""
    x = _block_returns(np.random.default_rng(4), 250)
    x[120:140] *= 6.0
    return x


def high_concentration():
    """Very short history (n > effective sample size): alpha at/near cap."""
    return _block_returns(np.random.default_rng(5), 100)


SCENARIOS = {
    "complete_blocks": complete_blocks,
    "staggered_missing": staggered_missing,
    "negative_blocks": negative_blocks,
    "crisis_shock": crisis_shock,
    "high_concentration": high_concentration,
}
