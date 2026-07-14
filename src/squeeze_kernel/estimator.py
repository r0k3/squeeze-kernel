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
    adaptive_weights : str or None
        Sequential surprise-gated adaptation of the ladder blend weights
        (requires ``corr_half_lives``).  ``"cusum"`` runs a two-sided Page
        CUSUM on the studentised fast-vs-slow per-rung predictive-score
        drift (reference drift 0.5, threshold 4.9721 = Siegmund
        average-run-length ~2 years); an alarm applies a half-magnitude
        tilt of the theta-prior toward the inverse-horizon vector (fast
        alarms, w ~ 1/h) or the square-root-horizon vector (slow alarms,
        w ~ h^0.5), decaying at the fastest rung's half-life.  Weights
        equal the prior on all non-alarmed days; PSD is untouched (the
        blend stays convex).  Adds five scalars of state and one Cholesky
        per rung per update for the scores.  Validated across S&P panels
        (held-out +0.5-0.6 NLL/day), an external industry panel incl. an
        out-of-time seal (+0.53/day, p=1e-4), a multi-asset futures panel,
        and synthetic regime/null suites; a one-at-a-time sensitivity
        sweep over all structural constants is sign-stable.  ``None``
        (default) keeps the fixed theta-prior blend bit-for-bit.
    min_obs : int or None
        Usability gate for newly listed assets.  When set, the property
        ``usable_mask`` marks an asset usable only once it has delivered
        at least ``min_obs`` finite observations.  The gate is purely
        diagnostic: state evolution and ``get_cov``/``get_corr`` are
        unchanged (the states keep warming during the gated window, so an
        asset is fully warm when the gate lifts).  Motivation: on an
        expanding multi-asset universe, forecast rows for assets in their
        first ~100 observations are dominated by the single-observation
        variance initialisation and are unusable for scoring or portfolio
        construction (measured ≈ +2,800 NLL/day on days whose scored set
        included such assets).  Deployment recipe::

            m = est.usable_mask
            cov_usable = est.get_cov()[np.ix_(m, m)]

        Recommended ``min_obs`` ≈ 60–100 for daily data.  ``None``
        (default) disables the gate (``usable_mask`` is all-True).

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
        min_obs: int | None = None,
        adaptive_weights: str | None = None,
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
        if min_obs is not None and (not isinstance(min_obs, int) or min_obs < 1):
            raise ValueError("min_obs must be a positive integer or None.")
        self.min_obs = min_obs
        self._obs_count = np.zeros(n_assets, dtype=np.int64)
        if adaptive_weights not in (None, "cusum"):
            raise ValueError("adaptive_weights must be None or 'cusum'.")
        if adaptive_weights is not None and corr_half_lives is None:
            raise ValueError("adaptive_weights requires corr_half_lives.")
        self.adaptive_weights = adaptive_weights
        if adaptive_weights is not None:
            hl_arr = np.asarray(corr_half_lives, dtype=np.float64)
            self._aw_pi_fast = (1.0 / hl_arr) / (1.0 / hl_arr).sum()
            self._aw_pi_slow = hl_arr ** 0.5 / (hl_arr ** 0.5).sum()
            self._aw_lam_tilt = 2.0 ** (-1.0 / float(hl_arr.min()))
            self._aw_gamma_scale = 2.0 ** (-1.0 / float(np.median(hl_arr)))
            # Page CUSUM: drift 0.5, threshold from Siegmund's ARL
            # approximation at ARL0 = 504 trading days (~2 years).
            self._aw_drift, self._aw_b, self._aw_snap = 0.5, 4.9721088583, 0.5
            self._aw_gp = self._aw_gm = 0.0
            self._aw_tilt = 0.0
            self._aw_scale = 1.0
            self._aw_prev_sig: list[np.ndarray] | None = None

        # Scale-free correlation memory (opt-in): replace the single correlation
        # timescale by a positive combination of EWMAs on a geometric half-life
        # ladder, blended per-scale (Mode A). None => single-scale, published
        # behaviour bit-for-bit. See ``corr_half_lives`` in the class docstring.
        self.corr_half_lives = None
        self.corr_theta = corr_theta
        self._corr_lam: np.ndarray | None = None
        self._corr_w: np.ndarray | None = None
        self._Q_list: list[np.ndarray] | None = None
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
            self._Q_list = [np.eye(n_assets, dtype=np.float64) for _ in hl]
            self._S_list = [float(epsilon) for _ in hl]

        # Resolve shrinkage
        if isinstance(shrinkage, str):
            self._shrinkage_alpha = -1.0 if shrinkage == "auto" else 0.0
        else:
            self._shrinkage_alpha = float(shrinkage)

        # Resolve kernel
        self._kernel_fn, self._kernel_kwargs = _resolve_kernel(kappa, kernel_fn, kernel_kwargs)
        self.kappa = self._kernel_kwargs.get("kappa") if self._kernel_fn is kernel_fisher else None

        # State. The correlation memory is stored NORMALISED: Q_t = M_t / S_t
        # with the recursion Q_t = (1 - eta_t) Q_{t-1} + eta_t z_t z_t',
        # eta_t = w_t / S_t after S_t <- lam S_{t-1} + w_t. This is
        # algebraically identical to the raw-mass form (M init eps*I, S init
        # eps => Q init I), keeps the matrix state well scaled, and makes the
        # PSD convex-combination recursion explicit.
        self._var_t: np.ndarray | None = None
        self._var_init: np.ndarray | None = None
        self._var_anchor: np.ndarray | None = None
        self._vol_t: np.ndarray | None = None
        # No single-scale state is allocated in ladder mode.
        self._Q_t = (np.eye(n_assets, dtype=np.float64)
                     if self._corr_lam is None else None)
        self._S_t = float(epsilon)
        self._cov: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._last_weight: float = 0.0
        # Extraction (normalise + shrink + vol application) is deferred until
        # get_cov()/get_corr(); _dirty marks state newer than _cov/_corr.
        self._dirty = False

        # Cached scratch buffers reused per ``update()`` to avoid per-step
        # allocator churn. These are intentionally module-private and
        # never escape the estimator.
        self._scratch_outer = np.empty((n_assets, n_assets), dtype=np.float64)
        self._scratch_corr = np.empty((n_assets, n_assets), dtype=np.float64)
        self._scratch_had = np.empty((n_assets, n_assets), dtype=np.float64)
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
        self._obs_count[finite] += 1

        # ── Adaptive-weight detector: score r_t under yesterday's per-rung
        # forecasts, then advance the CUSUM (weights used below therefore
        # reflect information through r_t only — causal). ──
        if (self.adaptive_weights is not None and self._aw_prev_sig is not None
                and finite.any()):
            oidx = np.flatnonzero(finite)
            r_o = r_t[oidx]
            ell = np.empty(len(self._aw_prev_sig))
            ok = True
            for k, sig in enumerate(self._aw_prev_sig):
                sub = sig[np.ix_(oidx, oidx)]
                sub = (sub + sub.T) * 0.5
                sign, logdet = np.linalg.slogdet(sub)
                if sign <= 0:
                    ok = False               # degenerate day (e.g. a fresh
                    break                    # listing): no clean score
                try:
                    quad = float(r_o @ np.linalg.solve(sub, r_o))
                except np.linalg.LinAlgError:
                    ok = False
                    break
                ell[k] = -0.5 * (oidx.size * np.log(2 * np.pi) + logdet + quad)
            if not ok:
                self._aw_tilt *= self._aw_lam_tilt
                ell = None
        else:
            ell = None
        if ell is not None:
            dd = ell - ell.mean()
            rms = float(np.sqrt((dd @ dd) / ell.size))
            self._aw_scale = (self._aw_gamma_scale * self._aw_scale
                              + (1.0 - self._aw_gamma_scale) * rms)
            zc = np.clip(dd / (self._aw_scale + 1e-12), -3.0, 3.0)
            k_fast = int(np.argmin(self.corr_half_lives))
            k_slow = int(np.argmax(self.corr_half_lives))
            zfs = float(zc[k_fast] - zc[k_slow])
            self._aw_gp = max(0.0, self._aw_gp + zfs - self._aw_drift)
            self._aw_gm = max(0.0, self._aw_gm - zfs - self._aw_drift)
            if self._aw_gp > self._aw_b:
                self._aw_tilt, self._aw_gp = self._aw_snap, 0.0
            elif self._aw_gm > self._aw_b:
                self._aw_tilt, self._aw_gm = -self._aw_snap, 0.0
            else:
                self._aw_tilt *= self._aw_lam_tilt

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
            if self.weight_statistic == "mahalanobis" and self._vol_t is not None:
                # Score-exact surprise against the estimator's own previous
                # correlation; falls back to the marginal d² on the first
                # step or a (rare) singular observed submatrix. Extraction is
                # lazy, so bring _corr up to the t-1 state first.
                if self._dirty:
                    self._materialize()
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

        add = w_t > 0.0 and n_obs > 0
        if add:
            # np.multiply.outer with out= avoids the temporary that
            # np.outer otherwise allocates each step. zz' is computed once
            # and shared by every rung.
            np.multiply.outer(z_t, z_t, out=self._scratch_outer)
        if self._corr_lam is None:
            # ── Single-scale correlation EWMA (published path) ──
            lam_c = self.lambda_corr
            if self.lambda_corr_fast is not None:
                # Score-driven memory: stress days (w_t → 1) shorten the memory
                # toward lambda_corr_fast; calm days keep the slow decay.
                lam_c = self.lambda_corr + (self.lambda_corr_fast - self.lambda_corr) * w_t
            self._S_t = lam_c * self._S_t + w_t
            if add:
                # Q <- (1 - eta) Q + eta zz'. With w_t = 0 both S and M decay
                # by lam_c, so Q is unchanged — no matrix work at all.
                eta = w_t / self._S_t
                self._Q_t *= 1.0 - eta
                self._scratch_outer *= eta
                self._Q_t += self._scratch_outer
        else:
            # ── Scale-free ladder (Mode A) ──
            # Update K normalised correlation states on the geometric
            # half-life ladder; extraction (normalise + shrink + blend)
            # happens lazily in _materialize().
            s_eff = 0.0
            for k in range(self._corr_lam.size):
                self._S_list[k] = self._corr_lam[k] * self._S_list[k] + w_t
                if add:
                    eta = w_t / self._S_list[k]
                    self._Q_list[k] *= 1.0 - eta
                    np.multiply(self._scratch_outer, eta, out=self._scratch_had)
                    self._Q_list[k] += self._scratch_had
                s_eff += self._corr_w[k] * self._S_list[k]
            self._S_t = s_eff                        # blended effective size (for the property)
        self._vol_t = vol_t
        self._dirty = True
        if self.adaptive_weights is not None and self._corr_lam is not None:
            self._materialize_adaptive()
        self._last_weight = w_t
        return w_t

    def get_cov(self) -> np.ndarray:
        """Return the current covariance matrix estimate (n x n)."""
        if self._vol_t is None:
            raise RuntimeError("Call update() at least once before get_cov().")
        if self._dirty:
            self._materialize()
        return self._cov.copy()

    def get_corr(self) -> np.ndarray:
        """Return the current correlation matrix estimate (n x n)."""
        if self._vol_t is None:
            raise RuntimeError("Call update() at least once before get_corr().")
        if self._dirty:
            self._materialize()
        return self._corr.copy()

    @property
    def weight(self) -> float:
        """Kernel weight assigned to the most recent observation."""
        return self._last_weight

    @property
    def usable_mask(self) -> np.ndarray:
        """Boolean mask of assets with at least ``min_obs`` observations.

        All-True when ``min_obs`` is None. Purely diagnostic — estimates
        are not affected; subset the outputs with it (see class docstring).
        """
        if self.min_obs is None:
            return np.ones(self.n_assets, dtype=bool)
        return self._obs_count >= self.min_obs

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
        if self._Q_t is None:
            # Ladder mode: imputation reads the single-scale state, which
            # was never updated on this path — historically a silent no-op
            # (all correlations below threshold); keep it an explicit one.
            return
        eps = self.epsilon
        sigma_z = self._Q_t
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

    def _shrunk_corr_from_Q(self, Q: np.ndarray, S_t: float) -> np.ndarray:
        """Normalise one Q state to a correlation and shrink it in place.

        Returns ``self._scratch_corr`` — valid only until the next call.
        """
        eps = self.epsilon
        n = self.n_assets
        S_t = max(S_t, eps)

        # corr_ij = Q_ij * inv_diag_i * inv_diag_j; the scalar S_t cancels
        # in the normalisation, so Q needs no rescaling pass.
        diag_z = np.diagonal(Q).copy()
        inv_diag = 1.0 / np.sqrt(np.maximum(diag_z, eps))
        np.multiply.outer(inv_diag, inv_diag, out=self._scratch_corr)
        corr = self._scratch_corr
        corr *= Q                                    # in-place
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
                # rho_bar <= 0 also covers the q = meanoff(C o C) = 0 corner:
                # C o C has nonnegative entries, so q = 0 forces C = I and
                # hence rho_bar = 0 — the equicorrelation fallback applies.
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
                had = self._scratch_had
                np.multiply(corr, corr, out=had)        # Hadamard square, O(n^2)
                mean_off = (had.sum() - np.trace(had)) / self._n_off
                gamma = min(1.0, rho_bar / max(mean_off, eps))
                corr *= (1.0 - alpha)
                corr += (alpha * (1.0 - alpha)) * rho_bar
                had *= alpha * alpha * gamma
                corr += had
                np.fill_diagonal(corr, 1.0)
        return corr

    def _materialize_adaptive(self) -> None:
        """Adaptive-weight extraction: build per-rung shrunk covariances
        (kept for the next update's detector scores), blend with the
        CUSUM-tilted weights, derive _cov/_corr. Runs eagerly."""
        eps = self.epsilon
        vol_t = self._vol_t
        vv = np.multiply.outer(vol_t, vol_t)
        sig_k = []
        for k in range(self._corr_lam.size):
            corr_k = self._shrunk_corr_from_Q(self._Q_list[k], self._S_list[k])
            sig_k.append(corr_k * vv)
        self._aw_prev_sig = sig_k
        t = self._aw_tilt
        pi = self._corr_w
        if t >= 0:
            w = (1.0 - t) * pi + t * self._aw_pi_fast
        else:
            w = (1.0 + t) * pi + (-t) * self._aw_pi_slow
        cov = w[0] * sig_k[0]
        for k in range(1, len(sig_k)):
            cov = cov + w[k] * sig_k[k]
        cov = (cov + cov.T) * 0.5
        self._cov = cov
        d = np.sqrt(np.maximum(np.diagonal(cov), eps))
        self._corr = cov / np.outer(d, d)
        np.fill_diagonal(self._corr, 1.0)
        self._dirty = False

    def _materialize(self) -> None:
        """Extract _cov/_corr from the current state (lazy, on demand)."""
        eps = self.epsilon
        vol_t = self._vol_t
        if self._corr_lam is None:
            # ── Single scale: shrunk correlation IS the correlation output ──
            corr = self._shrunk_corr_from_Q(self._Q_t, self._S_t)
            np.multiply.outer(vol_t, vol_t, out=self._scratch_outer)
            cov = corr * self._scratch_outer
            cov += cov.T
            cov *= 0.5
            corr_out = corr.copy()
            corr_out += corr_out.T
            corr_out *= 0.5
            self._cov, self._corr = cov, corr_out
        else:
            # ── Ladder: blend per-rung shrunk correlations, then apply the
            # (shared) volatilities once — algebraically identical to
            # blending per-rung covariances, K-1 fewer O(n^2) passes.
            mix = np.zeros((self.n_assets, self.n_assets), dtype=np.float64)
            for k in range(self._corr_lam.size):
                corr_k = self._shrunk_corr_from_Q(self._Q_list[k], self._S_list[k])
                corr_k *= self._corr_w[k]
                mix += corr_k
            cov = mix
            np.multiply.outer(vol_t, vol_t, out=self._scratch_outer)
            cov *= self._scratch_outer
            cov += cov.T
            cov *= 0.5
            self._cov = cov
            # Correlation is re-derived from the blended covariance (not the
            # blended correlation mix) to keep the published dead-asset
            # semantics: rows of never-observed assets renormalise to zero.
            d = np.sqrt(np.maximum(np.diagonal(cov), eps))
            self._corr = cov / np.outer(d, d)
            np.fill_diagonal(self._corr, 1.0)
        self._dirty = False


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
