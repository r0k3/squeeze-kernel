"""Regenerate the golden .npz files from the current estimator.

Run ONLY from a commit whose behaviour is the accepted baseline (v0.6.0
headline: adaptive-blend ladder + cluster target, detector integral):
uv run python tests/golden/generate.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from golden_scenarios import ESTIMATOR_KWARGS, PROMOTED_KWARGS, SPLIT_KWARGS, N, SCENARIOS  # noqa: E402
from squeeze_kernel import SqueezeKernelEstimator  # noqa: E402


def main():
    import sys
    only = sys.argv[1:] or None
    for name, build in SCENARIOS.items():
        x = build()
        t_mid = x.shape[0] // 2
        est = SqueezeKernelEstimator(N, **ESTIMATOR_KWARGS)
        if only is not None and name not in only:
            continue
        cov_mid = None
        for t, r in enumerate(x):
            est.update(r)
            if t == t_mid:
                cov_mid = est.get_cov()
        np.savez_compressed(
            HERE / f"{name}.npz",
            cov_mid=cov_mid,
            cov_final=est.get_cov(),
            corr_final=est.get_corr(),
        )
        print(f"{name}: wrote cov_mid/cov_final/corr_final "
              f"(trace final = {np.trace(est.get_cov()):.6e})")


def promoted():
    """Golden for the 3.0 promoted configuration on every scenario."""
    for name, build in SCENARIOS.items():
        x = build()
        t_mid = x.shape[0] // 2
        est = SqueezeKernelEstimator(N, **PROMOTED_KWARGS)
        cov_mid = None
        for t, r in enumerate(x):
            est.update(r)
            if t == t_mid:
                cov_mid = est.get_cov()
        np.savez_compressed(HERE / f"promoted_{name}.npz", cov_mid=cov_mid,
                            cov_final=est.get_cov(), corr_final=est.get_corr())
        print(f"promoted_{name}: wrote (trace final = {np.trace(est.get_cov()):.6e})")


def split():
    """Golden for the 3.1 learned-split configuration on every scenario."""
    for name, build in SCENARIOS.items():
        x = build()
        t_mid = x.shape[0] // 2
        est = SqueezeKernelEstimator(N, **SPLIT_KWARGS)
        cov_mid = None
        for t, r in enumerate(x):
            est.update(r)
            if t == t_mid:
                cov_mid = est.get_cov()
        np.savez_compressed(HERE / f"split_{name}.npz", cov_mid=cov_mid,
                            cov_final=est.get_cov(), corr_final=est.get_corr())
        print(f"split_{name}: wrote (trace final = {np.trace(est.get_cov()):.6e})")


if __name__ == "__main__":
    if "--promoted" in sys.argv:
        promoted()
    else:
        main()
