"""Theorem 1 of the paper: the diagonal-congruence flow is a clock-weighted
Gram matrix, Q_k = S^{-1/2} M_k S^{-1/2} with M_k <- lam_k M_k + u u',
u = sqrt(w) o z, so corr(Q_k) = corr(M_k), missing assets included."""

import numpy as np

from squeeze_kernel import SqueezeKernel


def test_flow_is_clock_weighted_gram(rng):
    x = rng.normal(0, 0.01, size=(300, 20)) + rng.normal(0, 0.01, size=(300, 1))
    x[rng.random(x.shape) < 0.1] = np.nan
    x[:40, 3] = np.nan                      # a late listing
    sk = SqueezeKernel()
    sk.update(x[0])
    est = sk._est
    lam = est._corr_lam
    M = [np.sqrt(np.outer(est._S_asset[k], est._S_asset[k])) * est._Q_list[k]
         for k in range(lam.size)]
    for r in x[1:]:
        s_old = est._S_asset.copy()
        sk.update(r)
        w = est._S_asset[0] - lam[0] * s_old[0]          # the day's clock increments
        fin = np.isfinite(r)
        z = np.zeros_like(r)
        z[fin] = r[fin] / (est._vol_t[fin] + est.epsilon)
        u = np.sqrt(w) * z
        for k in range(lam.size):
            M[k] = lam[k] * M[k] + np.outer(u, u)
            S = est._S_asset[k]
            np.testing.assert_allclose(M[k] / np.sqrt(np.outer(S, S)), est._Q_list[k],
                                       rtol=1e-10, atol=1e-12)
