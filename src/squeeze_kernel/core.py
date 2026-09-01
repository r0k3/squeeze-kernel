"""squeeze-kernel 2.0 public API: the one-number estimator.

``SqueezeKernel`` exposes the v2 configuration of the estimator — every
constant either derives from ``half_life`` or is a frozen structural
constant (``CONSTANTS``) — with a public surface of one number and two
booleans.  The v1 estimator remains available unchanged as
``SqueezeKernelEstimator`` (or via :meth:`SqueezeKernel.v1`).

Configuration (research record: squeeze_cov V2_STATUS.md, branch v2):

- correlation ladder at ``half_life * (43/173, 1, 693/173)`` — the
  canonical rungs at the default half-life; rung weights ``pi ~ sqrt(h)``
  (theta = 1/2, structural),
- volatility EWMA at ``lambda_vol = 0.98`` — the single remaining
  empirically frozen constant (the ``h/b`` derivation is falsified by
  crisis sub-periods; see the paper's intensity section),
- kernel scale as state, not parameter: ``kappa_t = (1/3) EWMA_h(d^2)``
  (chi-squared-null constant),
- self-tuning shrinkage intensity per rung from the online concentration
  ``c = n / nu`` and the de-noised equicorrelation-explained fraction
  ``g``:  ``alpha = min(1, c) g^2 / (g^2 + (1-g)^2 max(0, 1/c - 1))``,
- shrinkage target: gamma = 1 Schur-square cluster target (PSD by the
  Schur product theorem) or equicorrelation,
- sequential surprise-gated rung weights (Page CUSUM on the studentised
  inter-rung predictive-score drift), switchable via ``detector``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .estimator import SqueezeKernelEstimator

__all__ = ["SqueezeKernel", "StructuralConstants", "CONSTANTS"]


@dataclass(frozen=True)
class StructuralConstants:
    """Frozen structural constants of the v2 estimator.

    Importable for research; not constructor arguments.
    """

    b: float = 4.0                  # ladder spacing: rungs (h/b, h, h*b)
    theta: float = 0.5              # rung weights ~ h^theta
    lambda_vol: float = 0.98        # frozen empirical (not derived from h)
    kappa_c: float = 1.0 / 3.0      # kernel scale: kappa = kappa_c * EWMA(activity)
    schur_p: int = 2                # Hadamard power of the cluster target
    epsilon: float = 1e-8


CONSTANTS = StructuralConstants()


class SqueezeKernel:
    """Streaming covariance estimator whose public surface is one number.

    Parameters
    ----------
    half_life : float
        The single tunable: the anchor correlation half-life in trading
        days.  The timescale ladder ``(h/b, h, h*b)``, the kernel-scale
        clock, and the rung weights all derive from it.  Default 173.

    Everything else is structural or self-tuning state: per-asset market
    clocks read from the correlation neighborhood (row-normalized Schur-
    square weighting of squared surprises), a per-timescale shrinkage
    intensity from the online concentration and target-fit, the Schur-
    square cluster target, and the surprise-gated timescale weights.
    The number of assets is inferred from the first ``update`` call.
    The published v1 estimator and every ablation switch remain available
    on ``SqueezeKernelEstimator``.
    """

    def __init__(self, half_life: float = 173.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive.")
        self.half_life = float(half_life)
        self._est: SqueezeKernelEstimator | None = None
        self._t = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def _build(self, n_assets: int) -> SqueezeKernelEstimator:
        c = CONSTANTS
        h = self.half_life
        ladder = (h / c.b, h, h * c.b)
        return SqueezeKernelEstimator(
            n_assets,
            lambda_vol=c.lambda_vol,
            shrinkage="auto",
            shrinkage_target="cluster",
            corr_half_lives=ladder,
            corr_theta=c.theta,
            kappa_mode="adaptive",
            alpha_rule="selftuning",
            level_match=False,
            detector=True,
            clock="asset",
            epsilon=c.epsilon,
        )

    # ── public API ───────────────────────────────────────────────────────

    def update(self, r_t: ArrayLike, mask: ArrayLike | None = None) -> float:
        """Process one return vector; returns the kernel weight w_t.

        ``r_t`` may contain NaN for missing assets; ``mask`` (optional
        boolean, True = observed) is an alternative way to mark them.
        """
        r = np.asarray(r_t, dtype=np.float64)
        if r.ndim != 1:
            raise ValueError("r_t must be one-dimensional.")
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != r.shape:
                raise ValueError("mask must match r_t's shape.")
            r = np.where(m, r, np.nan)
        if self._est is None:
            self._est = self._build(r.size)
        self._t += 1
        return self._est.update(r)

    def covariance(self) -> np.ndarray:
        """Current covariance estimate (n x n)."""
        if self._est is None:
            raise RuntimeError("Call update() at least once first.")
        return self._est.get_cov()

    def correlation(self) -> np.ndarray:
        """Current correlation estimate (n x n)."""
        if self._est is None:
            raise RuntimeError("Call update() at least once first.")
        return self._est.get_corr()

    def state(self) -> dict[str, object]:
        """Diagnostics: update count, last weight, kernel scale, per-rung
        effective sizes, detector tilt."""
        if self._est is None:
            return {"n_assets": None, "t": 0}
        est = self._est
        det = est._detector
        S = list(est._S_list or [])
        J = list(est._J_list or [])
        nu = [s * s / max(j, est.epsilon) for s, j in zip(S, J)]
        return {
            "n_assets": est.n_assets,
            "t": self._t,
            "last_weight": est.weight,
            "kappa_t": getattr(est, "_last_kappa",
                               CONSTANTS.kappa_c * est._kap_state),
            "rung_S": S,
            "rung_nu": nu,
            "detector_tilt": 0.0 if det is None else det.tilt,
        }

    # ── v1 escape hatch ──────────────────────────────────────────────────

    @classmethod
    def v1(cls, n_assets: int, **kwargs: object) -> SqueezeKernelEstimator:
        """The published v1 estimator, unchanged (all legacy knobs)."""
        return SqueezeKernelEstimator(n_assets, **kwargs)  # type: ignore[arg-type]
