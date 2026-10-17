"""Regenerate the golden .npz files from the current estimator.

Run ONLY from a commit whose behaviour is the accepted baseline:
uv run python tests/golden/generate.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from golden_scenarios import LAM, SCENARIOS  # noqa: E402
from squeeze_kernel import SqueezeKernel  # noqa: E402


def main():
    only = sys.argv[1:] or None
    for name, build in SCENARIOS.items():
        if only is not None and name not in only:
            continue
        x = build()
        t_mid = x.shape[0] // 2
        sk = SqueezeKernel(lam=LAM)
        cov_mid = None
        for t, r in enumerate(x):
            sk.update(r)
            if t == t_mid:
                cov_mid = sk.covariance()
        np.savez_compressed(HERE / f"{name}.npz", cov_mid=cov_mid,
                            cov_final=sk.covariance(), corr_final=sk.correlation())
        print(f"{name}: wrote (trace final = {np.trace(sk.covariance()):.6e})")


if __name__ == "__main__":
    main()
