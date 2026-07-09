"""Core streaming Squeeze Kernel covariance estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from squeeze_kernel.kernels import (
    KernelFn, kernel_fisher, calibrate_kappa, extract_d2_series,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class SqueezeKernelEstimator:
    """Streaming robust covariance estimator with pluggable kernel weighting.

    The estimator is positive semi-definite by construction at every time step,
    handles missing observations natively, and separates volatility and
    correlation dynamics through dual-timescale EWMAs.  An adaptive
    equicorrelation shrinkage rule automatically calibrates regularisation
    to the concentration ratio n / T_eff.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    lambda_vol : float
        Decay factor for per-asset volatility EWMA (default 0.98,
        half-life ≈ 34 trading days).
    lambda_corr : float
        Decay factor for correlation EWMA (default 0.996, half-life
        ≈ 173 trading days, effective sample size ≈ 250).
    kappa : float, optional
        Saturation parameter for the default Fisher kernel (default 0.25).
        The defaults for ``lambda_vol``, ``lambda_corr`` and ``kappa`` are
        the values recommended in the paper for panels of daily financial
        returns; they were selected by time-series cross-validation and are
        insensitive to moderate perturbation.
    kernel_fn : callable, optional
        Custom kernel ``(d2, *, n_observed, **kw) -> float``.
        If omitted, the estimator uses ``kernel_fisher``.
    kernel_kwargs : dict, optional
        Extra keyword arguments forwarded to ``kernel_fn``.
        Use this to configure alternative kernels such as
        ``kernel_exponential(gamma=...)``.
    epsilon : float
        Numerical floor (default 1e-8).
    shrinkage : str or float
        ``'auto'`` (default) for adaptive shrinkage,
        ``0`` or ``'none'`` to disable, or a float in [0, 1] for fixed intensity.
    shrinkage_delta : float
        Threshold for adaptive shrinkage (default 0.10).
    impute_missing : bool
        If True, impute missing standardized returns from correlated assets.
    impute_threshold : float
        Minimum |correlation| for imputation donors (default 0.6).
    weight_statistic : str
        Statistic fed to the kernel.  ``'marginal'`` (default) uses the mean
        squared standardized return d² = z'z/N — the published estimator.
        ``'mahalanobis'`` uses the score-exact surprise z'C⁻¹z/N measured
        against the estimator's own previous correlation matrix (one linear
        solve per update).  With κ ≈ 1 (the parameter-free default, since
        E[z'C⁻¹z/N] = 1 under a correct C) this improved one-step NLL by
        ≈2.5 points on the S&P-500 n=100 benchmark, holdout-confirmed.
        IMPORTANT — regime-dependent: the advantage inverts in the
        high-concentration regime (≈12 NLL worse at n=200 and ≈61 worse at
        n=300 with T_eff = 250), because the estimated inverse inflates the
        statistic as n approaches T_eff, saturating the kernel and weakening
        the adaptive shrinkage.  Use only when n/T_eff ≲ 0.5; the marginal
        default is the concentration-robust choice at every dimension.
    lambda_corr_fast : float or None
        If set, enables score-driven memory: the correlation decay becomes
        λ_t = lambda_corr + (lambda_corr_fast − lambda_corr)·w_t, shortening
        the memory on high-weight (stress) days.  PSD is preserved.  Pass
        ``None`` (default) for the published constant-λ behaviour.
        Do not combine with ``weight_statistic='mahalanobis'`` — the two
        mechanisms act on the same reactivity channel and their combination
        degraded out-of-sample accuracy in testing.
    vol_anchor_phi : float or None
        If set, enables the OU volatility anchor: each asset's variance
        prediction mean-reverts toward a slow per-asset anchor before the
        measurement update, v_pred = v̄ + φ·(v − v̄), with the anchor v̄ a
        slow EWMA of squared returns (see ``vol_anchor_decay``).  φ is the
        per-step retention of deviations from the anchor (deviation
        half-life ≈ ln 2 / (1 − φ) days); φ = 1 or ``None`` (default)
        reproduces the published estimator exactly.  Recommended φ = 0.995
        (conservative; the range [0.99, 0.995] is robust).  On the S&P-500
        n=100 benchmark this improved held-out one-step NLL by 3.3 points
        (φ=0.995; 4.3 at φ=0.99) and five-step NLL by 3.9 (5.0), with no
        degradation at n=300.  Mechanism: a two-timescale (component-style)
        volatility structure — it changes persistence, not shock response.
        Validated with the default marginal kernel; interaction with the
        correlation-side extensions above is untested.
    vol_anchor_decay : float
        Decay of the slow per-asset variance anchor (default 0.999,
        effective memory ≈ 1000 trading days).  Only used when
        ``vol_anchor_phi`` is set.
    shrinkage_target : str
        Geometry of the adaptive shrinkage target.  ``'equicorrelation'``
        (default) is the published single-factor target.  ``'cluster'``
        uses the concentration-morphing cluster target
        T = (1−α)·T_equi + α·[(1−γ)I + γ·(C∘C)], where C∘C is the Hadamard
        square of the current correlation (PSD by the Schur product
        theorem) and γ = min(1, ρ̄/mean-offdiag(C∘C)) level-matches the
        target to the equicorrelation mass.  Respects the correlation
        matrix's own block/cluster structure without any clustering
        algorithm; adds no parameters and stays O(n²).  As α → 0 it
        reduces exactly to the published estimator, so behaviour at low
        concentration is unchanged.  Held-out one-step NLL on the S&P-500
        benchmark: +0.14 (negligible) at n=100, −4.2 at n=200, −25.0 at
        n=300.  Recommended when n approaches the effective sample size.
    corr_half_lives : sequence of float, optional
        Scale-free correlation memory.  When set, the single correlation
        timescale is replaced by a positive combination of EWMAs on the
        given geometric half-life ladder (in trading days, e.g.
        ``(43, 173, 693)``): each scale is normalised and adaptively
        shrunk against its own effective sample size, and the resulting
        covariances are blended with weights ∝ half-life\\ :sup:`corr_theta`.
        By Bernstein's theorem this approximates the power-law memory of
        financial correlations (the streaming analogue of HAR).  PSD by
        construction (positive combination of PSD matrices).  ``None``
        (default) is the published single-scale estimator, bit-for-bit;
        a one-element ladder reduces to a single-scale estimator at that
        half-life.  Mutually exclusive with ``lambda_corr_fast``; composes
        with ``shrinkage_target='cluster'``.  Cost is O(K·n²) per update.
        Held-out one-step NLL on the S&P-500 benchmark improves at every
        dimension (−3.9 at n=100, up to −18 at n=300 before the cluster
        target); the $90\\%$ model confidence set collapses to this
        configuration alone.  Recommended default: the base-centred ladder
        ``(43, 173, 693)`` with ``corr_theta=0.25``.
    corr_theta : float
        Long-memory exponent controlling the ladder weights (default
        0.25).  Only used when ``corr_half_lives`` is set.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 3))
    >>> est = SqueezeKernelEstimator(n_assets=3)
    >>> for r_t in returns:
    ...     est.update(r_t)
    >>> cov = est.get_cov()
    >>> corr = est.get_corr()
    """

    def __init__(
        self,
        n_assets: int,
        *,
        lambda_vol: float = 0.98,
        lambda_corr: float = 0.996,
        kappa: float | None = None,
        kernel_fn: KernelFn | None = None,
        kernel_kwargs: dict[str, object] | None = None,
        epsilon: float = 1e-8,
        shrinkage: str | float = "auto",
        shrinkage_delta: float = 0.10,
        impute_missing: bool = False,
        impute_threshold: float = 0.6,
        weight_statistic: str = "marginal",
        lambda_corr_fast: float | None = None,
        vol_anchor_phi: float | None = None,
        vol_anchor_decay: float = 0.999,
        shrinkage_target: str = "equicorrelation",
        corr_half_lives: "Sequence[float] | None" = None,
        corr_theta: float = 0.25,
    ):
        self.n_assets = n_assets
        self.lambda_vol = lambda_vol
        self.lambda_corr = lambda_corr
        self.epsilon = epsilon
        self.impute_missing = impute_missing
        self.impute_threshold = impute_threshold
        self.shrinkage_delta = shrinkage_delta

        # Opt-in extensions (defaults preserve the published estimator exactly)
        if weight_statistic not in ("marginal", "mahalanobis"):
            raise ValueError("weight_statistic must be 'marginal' or 'mahalanobis'.")
        self.weight_statistic = weight_statistic
        if lambda_corr_fast is not None and not (0.0 < lambda_corr_fast < 1.0):
            raise ValueError("lambda_corr_fast must be in (0, 1).")
        self.lambda_corr_fast = lambda_corr_fast
        if vol_anchor_phi is not None and not (0.0 < vol_anchor_phi <= 1.0):
            raise ValueError("vol_anchor_phi must be in (0, 1].")
        if not (0.0 < vol_anchor_decay < 1.0):
            raise ValueError("vol_anchor_decay must be in (0, 1).")
        self.vol_anchor_phi = vol_anchor_phi
        self.vol_anchor_decay = vol_anchor_decay
        if shrinkage_target not in ("equicorrelation", "cluster"):
            raise ValueError("shrinkage_target must be 'equicorrelation' or 'cluster'.")
        self.shrinkage_target = shrinkage_target

        # Scale-free correlation memory (opt-in): replace the single correlation
        # timescale by a positive combination of EWMAs on a geometric half-life
        # ladder, blended per-scale (Mode A). None => single-scale, published
        # behaviour bit-for-bit. See ``corr_half_lives`` in the class docstring.
        self.corr_half_lives = None
        self.corr_theta = corr_theta
        self._corr_lam: np.ndarray | None = None
        self._corr_w: np.ndarray | None = None
        self._M_list: list[np.ndarray] | None = None
        self._S_list: list[float] | None = None
        if corr_half_lives is not None:
            hl = np.asarray(corr_half_lives, dtype=np.float64)
            if hl.ndim != 1 or hl.size < 1 or np.any(hl <= 0.0):
                raise ValueError("corr_half_lives must be a non-empty sequence of positive half-lives.")
            if corr_theta < 0.0:
                raise ValueError("corr_theta must be >= 0.")
            if lambda_corr_fast is not None:
                raise ValueError(
                    "corr_half_lives and lambda_corr_fast are mutually exclusive "
                    "correlation-memory mechanisms; set at most one."
                )
            self.corr_half_lives = hl
            self._corr_lam = 2.0 ** (-1.0 / hl)
            w = hl ** corr_theta
            self._corr_w = w / w.sum()
            self._M_list = [np.eye(n_assets, dtype=np.float64) * epsilon for _ in hl]
            self._S_list = [float(epsilon) for _ in hl]

        # Resolve shrinkage
        if isinstance(shrinkage, str):
            self._shrinkage_alpha = -1.0 if shrinkage == "auto" else 0.0
        else:
            self._shrinkage_alpha = float(shrinkage)

        # Resolve kernel
        self._kernel_fn, self._kernel_kwargs = _resolve_kernel(kappa, kernel_fn, kernel_kwargs)
        self.kappa = self._kernel_kwargs.get("kappa") if self._kernel_fn is kernel_fisher else None

        # State
        self._var_t: np.ndarray | None = None
        self._var_init: np.ndarray | None = None
        self._var_anchor: np.ndarray | None = None
        self._M_t = np.eye(n_assets, dtype=np.float64) * epsilon
        self._S_t = float(epsilon)
        self._cov: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._last_weight: float = 0.0

        # Cached scratch buffers reused per ``update()`` to avoid per-step
        # allocator churn. These are intentionally module-private and
        # never escape the estimator.
        self._scratch_outer = np.empty((n_assets, n_assets), dtype=np.float64)
        self._scratch_corr = np.empty((n_assets, n_assets), dtype=np.float64)
        self._n_off = float(n_assets * (n_assets - 1)) if n_assets > 1 else 1.0

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, r_t) -> float:
        """Process one return vector and update the covariance estimate.

        Parameters
        ----------
        r_t : array-like, shape (n_assets,)
            Return vector.  May contain NaN for missing assets.

        Returns
        -------
        float
            Kernel weight w_t assigned to this observation.
        """
        r_t = np.asarray(r_t, dtype=np.float64)
        n = self.n_assets
        eps = self.epsilon
        if r_t.shape != (n,):
            raise ValueError(f"Expected shape ({n},), got {r_t.shape}.")

        finite = np.isfinite(r_t)

        # ── Volatility update ──
        if self._var_t is None:
            self._var_t = np.zeros(n, dtype=np.float64)
            self._var_init = np.zeros(n, dtype=bool)
            if self.vol_anchor_phi is not None:
                self._var_anchor = np.zeros(n, dtype=np.float64)

        first = finite & ~self._var_init
        repeat = finite & self._var_init
        if np.any(first):
            self._var_t[first] = r_t[first] ** 2 + eps
            self._var_init[first] = True
            if self._var_anchor is not None:
                self._var_anchor[first] = self._var_t[first]
        if np.any(repeat):
            if self.vol_anchor_phi is None:
                self._var_t[repeat] = (
                    self.lambda_vol * self._var_t[repeat]
                    + (1.0 - self.lambda_vol) * r_t[repeat] ** 2
                )
            else:
                # OU anchor: mean-revert the variance prediction toward a slow
                # per-asset anchor before the measurement update, then update
                # the anchor itself (order matters and matches the validated
                # experiment: prediction uses the *old* anchor).
                phi = self.vol_anchor_phi
                lam_bar = self.vol_anchor_decay
                anchor = self._var_anchor[repeat]
                v_pred = anchor + phi * (self._var_t[repeat] - anchor)
                self._var_t[repeat] = (
                    self.lambda_vol * v_pred
                    + (1.0 - self.lambda_vol) * r_t[repeat] ** 2
                )
                self._var_anchor[repeat] = (
                    lam_bar * anchor + (1.0 - lam_bar) * r_t[repeat] ** 2
                )

        vol_t = np.zeros(n, dtype=np.float64)
        vol_t[self._var_init] = np.sqrt(self._var_t[self._var_init])

        # ── Standardized returns ──
        z_t = np.zeros(n, dtype=np.float64)
        n_obs = int(finite.sum())
        if n_obs > 0:
            z_t[finite] = r_t[finite] / (vol_t[finite] + eps)
            d2 = float(z_t[finite] @ z_t[finite]) / n_obs
            if self.weight_statistic == "mahalanobis" and self._corr is not None:
                # Score-exact surprise against the estimator's own previous
                # correlation; falls back to the marginal d² on the first
                # step or a (rare) singular observed submatrix.
                try:
                    c_sub = self._corr[np.ix_(finite, finite)]
                    d2 = float(z_t[finite] @ np.linalg.solve(c_sub, z_t[finite])) / n_obs
                except np.linalg.LinAlgError:
                    pass
            w_t = self._kernel_fn(d2, n_observed=n_obs, **self._kernel_kwargs)
        else:
            w_t = 0.0

        # ── Imputation ──
        if self.impute_missing and 0 < n_obs < n:
            self._impute(z_t, finite)

        if self._corr_lam is None:
            # ── Single-scale correlation EWMA (published path, unchanged) ──
            lam_c = self.lambda_corr
            if self.lambda_corr_fast is not None:
                # Score-driven memory: stress days (w_t → 1) shorten the memory
                # toward lambda_corr_fast; calm days keep the slow decay.
                lam_c = self.lambda_corr + (self.lambda_corr_fast - self.lambda_corr) * w_t
            self._S_t = lam_c * self._S_t + w_t
            self._M_t *= lam_c
            if w_t > 0.0 and n_obs > 0:
                # np.multiply.outer with out= avoids the temporary that
                # np.outer otherwise allocates each step.
                np.multiply.outer(z_t, z_t, out=self._scratch_outer)
                self._M_t += w_t * self._scratch_outer
            self._cov, self._corr = self._extract(vol_t)
        else:
            # ── Scale-free ladder (Mode A) ──
            # Update K correlation accumulators on the geometric half-life
            # ladder; normalise and adaptively shrink each against its OWN
            # effective sample size, then blend the per-scale covariances.
            add = w_t > 0.0 and n_obs > 0
            if add:
                np.multiply.outer(z_t, z_t, out=self._scratch_outer)
            cov = np.zeros((n, n), dtype=np.float64)
            s_eff = 0.0
            for k in range(self._corr_lam.size):
                self._S_list[k] = self._corr_lam[k] * self._S_list[k] + w_t
                self._M_list[k] *= self._corr_lam[k]
                if add:
                    self._M_list[k] += w_t * self._scratch_outer
                cov_k, _ = self._extract(vol_t, self._M_list[k], self._S_list[k])
                cov += self._corr_w[k] * cov_k
                s_eff += self._corr_w[k] * self._S_list[k]
            cov = 0.5 * (cov + cov.T)
            self._cov = cov
            self._S_t = s_eff                        # blended effective size (for the property)
            d = np.sqrt(np.maximum(np.diagonal(cov), eps))
            self._corr = cov / np.outer(d, d)
            np.fill_diagonal(self._corr, 1.0)
        self._last_weight = w_t
        return w_t

    def get_cov(self) -> np.ndarray:
        """Return the current covariance matrix estimate (n x n)."""
        if self._cov is None:
            raise RuntimeError("Call update() at least once before get_cov().")
        return self._cov.copy()

    def get_corr(self) -> np.ndarray:
        """Return the current correlation matrix estimate (n x n)."""
        if self._corr is None:
            raise RuntimeError("Call update() at least once before get_corr().")
        return self._corr.copy()

    @property
    def weight(self) -> float:
        """Kernel weight assigned to the most recent observation."""
        return self._last_weight

    @property
    def effective_sample_size(self) -> float:
        """Kernel-weighted effective sample size S_t."""
        return self._S_t

    @property
    def shrinkage_intensity(self) -> float:
        """Current adaptive shrinkage intensity alpha_t."""
        if self._shrinkage_alpha >= 0:
            return self._shrinkage_alpha
        n = self.n_assets
        return max(0.0, min(1.0, n / (2.0 * max(self._S_t, self.epsilon)) - self.shrinkage_delta))

    @staticmethod
    def calibrate_kappa(
        returns, target_weight: float = 0.5, lambda_vol: float = 0.98,
    ) -> float:
        """Calibrate κ from burn-in data so E[w_t] ≈ target_weight.

        Parameters
        ----------
        returns : array-like, shape (T, n)
            Burn-in return data.
        target_weight : float
            Target average kernel weight in (0, 1).
        lambda_vol : float
            Volatility decay factor used for standardization.

        Returns
        -------
        float
            Calibrated κ value.
        """
        d2 = extract_d2_series(returns, lambda_vol=lambda_vol)
        return calibrate_kappa(d2, target_weight)

    # ── Private helpers ───────────────────────────────────────────────────

    def _impute(self, z_t: np.ndarray, finite: np.ndarray) -> None:
        eps = self.epsilon
        denom = max(self._S_t, eps)
        sigma_z = self._M_t / denom
        diag_z = np.diag(sigma_z)
        inv_diag = 1.0 / np.sqrt(np.maximum(diag_z, eps))
        missing = ~finite
        for i in range(self.n_assets):
            if not missing[i]:
                continue
            num = den = 0.0
            for j in range(self.n_assets):
                if not finite[j]:
                    continue
                c_ij = sigma_z[i, j] * inv_diag[i] * inv_diag[j]
                if abs(c_ij) < self.impute_threshold:
                    continue
                num += c_ij * z_t[j]
                den += abs(c_ij)
            if den > 0.0:
                z_t[i] = num / den

    def _extract(self, vol_t: np.ndarray, M_t=None, S_in=None) -> tuple[np.ndarray, np.ndarray]:
        eps = self.epsilon
        M_t = self._M_t if M_t is None else M_t
        S_t = max(self._S_t if S_in is None else S_in, eps)
        n = self.n_assets

        # Normalised standardised covariance matrix sigma_z = M_t / S_t.
        # Compute correlations directly into the cached scratch buffer to
        # avoid two intermediate allocations (sigma_z and corr).
        diag_z = np.diagonal(M_t).copy()
        diag_z /= S_t                                # in-place
        inv_diag = 1.0 / np.sqrt(np.maximum(diag_z, eps))
        # corr_ij = (M_ij / S_t) * inv_diag_i * inv_diag_j; this writes
        # the rescaled outer-product into _scratch_corr in one pass.
        np.multiply.outer(inv_diag, inv_diag, out=self._scratch_corr)
        corr = self._scratch_corr
        corr *= M_t                                  # in-place; corr = sigma_z * outer(inv_diag,inv_diag)
        corr *= (1.0 / S_t)                          # absorb the M_t / S_t scale
        np.fill_diagonal(corr, np.where(diag_z > eps, 1.0, 0.0))

        # Adaptive shrinkage: blend toward the equicorrelation target
        # T = (1 - rho_bar) I + rho_bar 11'.  We avoid materialising T by
        # blending the off-diagonal toward rho_bar in place and resetting
        # the diagonal to 1.
        alpha = self._shrinkage_alpha
        if alpha < 0:
            alpha = max(0.0, min(1.0, n / (2.0 * S_t) - self.shrinkage_delta))
        if alpha > 0.0 and n > 1:
            # Off-diagonal mean: O(n^2) sum, no mask allocation.
            rho_bar = (corr.sum() - corr.trace()) / self._n_off
            if self.shrinkage_target == "equicorrelation" or rho_bar <= 0.0:
                corr *= (1.0 - alpha)
                corr += alpha * rho_bar
                np.fill_diagonal(corr, 1.0)
            else:
                # Cluster (concentration-morphing) target:
                #   T = (1-alpha) T_equi + alpha [(1-gamma) I + gamma (C o C)]
                # C o C is the Hadamard square of the raw correlation (PSD by
                # the Schur product theorem, unit diagonal for free); gamma is
                # level-matched so the target carries the same average
                # correlation mass as the equicorrelation target.  As
                # alpha -> 0 this reduces exactly to the published estimator.
                had = corr * corr                       # Hadamard square, O(n^2)
                mean_off = (had.sum() - np.trace(had)) / self._n_off
                gamma = min(1.0, rho_bar / max(mean_off, eps))
                corr *= (1.0 - alpha)
                corr += (alpha * (1.0 - alpha)) * rho_bar
                corr += (alpha * alpha * gamma) * had
                np.fill_diagonal(corr, 1.0)

        # Build covariance from corr and vol_t.  We need an output array
        # that the caller can keep, so allocate one cov here (cannot reuse
        # corr buffer because both are returned).
        cov = corr * np.outer(vol_t, vol_t)
        # Defensive symmetrisation against floating-point asymmetry.
        cov += cov.T
        cov *= 0.5
        # Return a copy of corr so callers see a stable snapshot even if
        # the next update() overwrites the scratch buffer.
        corr_out = corr.copy()
        corr_out += corr_out.T
        corr_out *= 0.5
        return cov, corr_out


# ── Kernel resolution ─────────────────────────────────────────────────────────

def _resolve_kernel(
    kappa: float | None,
    kernel_fn: KernelFn | None,
    kernel_kwargs: dict[str, object] | None,
) -> tuple[KernelFn, dict[str, object]]:
    kw = dict(kernel_kwargs) if kernel_kwargs else {}

    if kernel_fn is not None:
        if kappa is not None:
            raise ValueError(
                "Pass kernel-specific parameters via kernel_kwargs when kernel_fn is set."
            )
        return kernel_fn, kw

    if "kappa" in kw and kappa is not None:
        raise ValueError("Pass kappa either as a top-level argument or in kernel_kwargs, not both.")

    resolved_kappa = float(kw.get("kappa", 0.25 if kappa is None else kappa))
    if resolved_kappa <= 0.0:
        raise ValueError("kappa must be > 0.")
    kw["kappa"] = resolved_kappa
    return kernel_fisher, kw
