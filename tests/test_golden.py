"""Golden regression tests: the estimator's covariance and correlation on
five deterministic scenarios (complete, staggered listings with holes and
a market holiday, negative blocks, a volatility shock, high concentration).
Any statistical change fails here; output-equivalent refactors keep them green.
"""

from pathlib import Path

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernel

from golden_scenarios import LAM, SCENARIOS

GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_golden_path(name):
    ref = np.load(GOLDEN / f"{name}.npz")
    x = SCENARIOS[name]()
    t_mid = x.shape[0] // 2
    sk = SqueezeKernel(lam=LAM)
    cov_mid = None
    for t, r in enumerate(x):
        sk.update(r)
        if t == t_mid:
            cov_mid = sk.covariance()
    np.testing.assert_allclose(cov_mid, ref["cov_mid"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sk.covariance(), ref["cov_final"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sk.correlation(), ref["corr_final"], rtol=1e-8, atol=1e-10)
