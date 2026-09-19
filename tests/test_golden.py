"""Golden regression tests for the combined cluster + scale-free path.

The .npz references were generated at B0 (v0.5.0, commit 0d44ecf) by
tests/golden/generate.py. Any statistical change to the combined path
fails here; output-equivalent refactors must keep these green.
"""

from pathlib import Path

import numpy as np
import pytest

from squeeze_kernel import SqueezeKernelEstimator

from golden_scenarios import ESTIMATOR_KWARGS, PROMOTED_KWARGS, SPLIT_KWARGS, N, SCENARIOS

GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_path(name):
    ref = np.load(GOLDEN / f"{name}.npz")
    x = SCENARIOS[name]()
    t_mid = x.shape[0] // 2
    est = SqueezeKernelEstimator(N, **ESTIMATOR_KWARGS)
    cov_mid = None
    for t, r in enumerate(x):
        est.update(r)
        if t == t_mid:
            cov_mid = est.get_cov()
    np.testing.assert_allclose(cov_mid, ref["cov_mid"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_cov(), ref["cov_final"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_corr(), ref["corr_final"], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_promoted_golden_path(name):
    """3.0 promoted configuration, pinned on every scenario."""
    ref = np.load(GOLDEN / f"promoted_{name}.npz")
    x = SCENARIOS[name]()
    t_mid = x.shape[0] // 2
    est = SqueezeKernelEstimator(N, **PROMOTED_KWARGS)
    cov_mid = None
    for t, r in enumerate(x):
        est.update(r)
        if t == t_mid:
            cov_mid = est.get_cov()
    np.testing.assert_allclose(cov_mid, ref["cov_mid"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_cov(), ref["cov_final"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_corr(), ref["corr_final"], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_split_golden_path(name):
    """3.1 learned-split configuration, pinned on every scenario."""
    ref = np.load(GOLDEN / f"split_{name}.npz")
    x = SCENARIOS[name]()
    t_mid = x.shape[0] // 2
    est = SqueezeKernelEstimator(N, **SPLIT_KWARGS)
    cov_mid = None
    for t, r in enumerate(x):
        est.update(r)
        if t == t_mid:
            cov_mid = est.get_cov()
    np.testing.assert_allclose(cov_mid, ref["cov_mid"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_cov(), ref["cov_final"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(est.get_corr(), ref["corr_final"], rtol=1e-8, atol=1e-10)
