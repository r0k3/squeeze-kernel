"""The blend-gradient's spectral floor is skipped only on certified days,
and a certified day is bit-identical to the floored path (3.1.1)."""

import numpy as np
import pytest

import squeeze_kernel.estimator as E
from squeeze_kernel import SqueezeKernel


def _path(x, cert):
    old = E._BlendGradient._CERT
    E._BlendGradient._CERT = cert
    calls = [0]
    orig = E.np.linalg.eigvalsh

    def spy(a):
        calls[0] += 1
        return orig(a)

    E.np.linalg.eigvalsh = spy
    try:
        sk = SqueezeKernel()
        covs = []
        for r in x:
            sk.update(r)
            covs.append(sk.covariance())
    finally:
        E.np.linalg.eigvalsh = orig
        E._BlendGradient._CERT = old
    return np.array(covs), calls[0]


@pytest.mark.parametrize("scale", [1.0, 1e-4])   # 1e-4: the floor binds every day
def test_certificate_is_exact(rng, scale):
    f = rng.normal(0, 0.01, size=(400, 1))
    x = (0.6 * f + 0.8 * rng.normal(0, 0.01, size=(400, 25))) * scale
    x[rng.random(x.shape) < 0.05] = np.nan
    fast, n_fast = _path(x, E._BlendGradient._CERT)
    ref, n_ref = _path(x, np.inf)          # every day takes the floored path
    np.testing.assert_array_equal(fast, ref)
    if scale == 1.0:
        assert n_fast < n_ref // 10        # the eigenvalue floor is skipped
