"""Core streaming Squeeze Kernel covariance estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    # SciPy's direct LAPACK bindings factorise an SPD matrix ~4x faster than
    # the NumPy slogdet+solve route (one dpotrf vs two LU factorisations) and
    # compute the identical quantities; the detector falls back to the
    # NumPy-only path when SciPy is absent.
    from scipy.linalg import cho_factor as _cho_factor, cho_solve as _cho_solve
except ImportError:  # pragma: no cover
    _cho_factor = _cho_solve = None

from squeeze_kernel.kernels import kernel_fisher
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Sequence


class _SurpriseDetector:
    """Two-sided Page CUSUM on the studentised fast-vs-slow per-rung
    predictive-score drift.

    Gates the blend weights of the scale-free correlation ladder: on each
    update the day is scored under every rung's previous (t-1) covariance
    — causal, since ``record`` is called with the rung states *after* the
    update — and the drift of the centred rung scores advances the CUSUM.
    An alarm sets a half-magnitude tilt of the theta-prior toward the
    inverse-horizon vector (fast alarm) or the square-root-horizon vector
    (slow alarm); on all other days the tilt decays at the fastest rung's
    half-life.  State: five scalars (``gp``, ``gm``, ``tilt``, ``scale``
    and the cached ``prev_sig`` rung covariances).
    """

    _DRIFT = 0.5
    _THRESHOLD = 4.9721088583   # Siegmund ARL approximation at ~2 years
    _SNAP = 0.5
    _CLIP = 3.0
    _JITTER = 1e-12

    def __init__(self, half_lives: np.ndarray) -> None:
        hl = half_lives
        self.pi_fast = (1.0 / hl) / (1.0 / hl).sum()
        self.pi_slow = hl ** 0.5 / (hl ** 0.5).sum()
        self._lam_tilt = 2.0 ** (-1.0 / float(hl.min()))
        self._gamma_scale = 2.0 ** (-1.0 / float(np.median(hl)))
        self._k_fast = int(np.argmin(hl))
        self._k_slow = int(np.argmax(hl))
        self.gp = 0.0
        self.gm = 0.0
        self.tilt = 0.0
        self.scale = 1.0
        self.prev_sig: list[np.ndarray] | None = None

    def score(self, r_t: np.ndarray, finite: np.ndarray) -> np.ndarray | None:
        """Per-rung Gaussian log-likelihood of ``r_t`` under ``prev_sig``,
        restricted to the observed assets.  Returns None when there is no
        previous state, nothing is observed, or the day is degenerate
        (e.g. a fresh listing); a degenerate day also decays the tilt."""
        if self.prev_sig is None or not finite.any():
            return None
        oidx = np.flatnonzero(finite)
        r_o = r_t[oidx]
        ell = np.empty(len(self.prev_sig))
        for k, sig in enumerate(self.prev_sig):
            sub = sig[np.ix_(oidx, oidx)]
            sub = (sub + sub.T) * 0.5
            if _cho_factor is not None:
                # One Cholesky per rung: logdet from the factor's
                # diagonal, quadratic form via triangular solves.
                try:
                    cf = _cho_factor(sub, lower=True, check_finite=False)
                except np.linalg.LinAlgError:
                    self.tilt *= self._lam_tilt
                    return None
                logdet = 2.0 * np.log(np.diagonal(cf[0])).sum()
                quad = float(r_o @ _cho_solve(cf, r_o, check_finite=False))
            else:
                sign, logdet = np.linalg.slogdet(sub)
                if sign <= 0:
                    self.tilt *= self._lam_tilt
                    return None
                try:
                    quad = float(r_o @ np.linalg.solve(sub, r_o))
                except np.linalg.LinAlgError:
                    self.tilt *= self._lam_tilt
                    return None
            ell[k] = -0.5 * (oidx.size * np.log(2 * np.pi) + logdet + quad)
        return ell

    def advance(self, ell: np.ndarray | None) -> None:
        """Advance the CUSUM with the rung scores for this day.  ``None``
        (no clean score) is a no-op: the degenerate-day tilt decay already
        happened inside ``score``."""
        if ell is None:
            return
        dd = ell - ell.mean()
        rms = float(np.sqrt((dd @ dd) / ell.size))
        self.scale = (self._gamma_scale * self.scale
                      + (1.0 - self._gamma_scale) * rms)
        zc = np.clip(dd / (self.scale + self._JITTER), -self._CLIP, self._CLIP)
        zfs = float(zc[self._k_fast] - zc[self._k_slow])
        self.gp = max(0.0, self.gp + zfs - self._DRIFT)
        self.gm = max(0.0, self.gm - zfs - self._DRIFT)
        if self.gp > self._THRESHOLD:
            self.tilt, self.gp = self._SNAP, 0.0
        elif self.gm > self._THRESHOLD:
            self.tilt, self.gm = -self._SNAP, 0.0
        else:
            self.tilt *= self._lam_tilt

    def record(self, rung_covs: list[np.ndarray]) -> None:
        """Cache this step's rung covariances as next step's forecasts."""
        self.prev_sig = rung_covs

    def blend(self, prior: np.ndarray) -> np.ndarray:
        """Blend weights for the rung covariances: the theta-prior on
        non-alarmed days, tilted half-magnitude toward the horizon vectors
        while an alarm is live.  Convex in both regimes."""
        t = self.tilt
        if t >= 0:
            return np.asarray((1.0 - t) * prior + t * self.pi_fast)
        return np.asarray((1.0 + t) * prior + (-t) * self.pi_slow)


class _BlendGradient:
    """Exponentiated gradient on the BLEND's log score, at the
    null-calibrated temperature.

    The emitted covariance is the linear pool Sigma(w) = sum_k w_k Sigma_k
    of the rung covariances.  Its Gaussian log score has gradient

        d ell / d w_k = -1/2 tr(Sigma^-1 Sigma_k) + 1/2 u' Sigma_k u,
        u = Sigma^-1 r,

    so the update moves the blend rather than selecting a rung (Bayesian
    model averaging over rung likelihoods selects, and collapses).  The
    gradient is studentised across rungs by its running rms and scaled by
    sqrt(2 eps), eps the fixed-share rate: the accumulated log-odds then
    have unit variance under uninformative evidence, so the mixture stays
    within a factor e of its prior when there is nothing to learn and
    moves linearly in a persistent advantage.  Evidence memory: the
    fastest rung (a regime can change as fast as the fastest rung can
    follow).  State: the weights, one scalar scale, and the cached rung
    covariances of the previous step.  Causal: ``score`` reads the
    previous step's covariances, ``advance`` updates the weights used for
    the next blend.
    """

    _JITTER = 1e-12
    _FLOOR = 1e-8   # spectral floor of the scored blend
    _CERT = 10.0    # certify lambda_min >= _CERT * _FLOOR before skipping the floor

    @staticmethod
    def _solve(sig: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sig^-1 r, sig^-1) from one Cholesky factorisation."""
        if _cho_factor is not None:
            cf = _cho_factor(sig, lower=True, check_finite=False)
            return (_cho_solve(cf, r, check_finite=False),
                    _cho_solve(cf, np.eye(sig.shape[0]), check_finite=False))
        sinv = np.linalg.inv(sig)
        return sinv @ r, sinv

    def __init__(self, half_lives: np.ndarray, prior: np.ndarray,
                 split: bool = False, n_experts: int = 3) -> None:
        hl = np.asarray(half_lives, dtype=np.float64)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.w = self.prior.copy()
        self.eps = float(1.0 - 2.0 ** (-1.0 / float(hl.min())))
        self._gamma_scale = 2.0 ** (-1.0 / float(np.median(hl)))
        self.scale = 1.0
        self.prev_sig: list[np.ndarray] | None = None
        self.prev_w: np.ndarray | None = None
        # Learned shrinkage split (3.1): each timescale's correlation is a
        # convex combination of three PSD unit-diagonal experts -- raw,
        # equicorrelation at its mean level, Schur square -- whose prior
        # is the intensity rule's morph (1-a, a(1-a), a^2) and whose
        # weights are moved by the same gradient step as the timescale
        # weights, studentised within the timescale.  Starts uniform and
        # forgets toward the prior at the fixed-share rate.
        self.split = bool(split)
        self.n_experts = int(n_experts)
        self.v = np.full((hl.size, self.n_experts), 1.0 / self.n_experts)
        self.scale_v = np.ones(hl.size)
        self.prev_comps: list[tuple[np.ndarray, ...]] | None = None
        self.prev_prior: np.ndarray | None = None
        self.prev_cov: np.ndarray | None = None
        self.prev_vol: np.ndarray | None = None
        self.prev_v: np.ndarray | None = None
        self._gc: np.ndarray | None = None

    def score(self, r_t: np.ndarray, finite: np.ndarray) -> np.ndarray | None:
        """Gradient of the previous blend's log score at ``r_t`` (observed
        subvector), one entry per rung; ``None`` on a degenerate day."""
        if self.prev_sig is None or self.prev_w is None:
            return None
        idx = np.flatnonzero(finite)
        if idx.size < 2:
            return None
        K = len(self.prev_sig)
        full = idx.size == self.prev_sig[0].shape[0]
        # Gate exactly as the research engine does: the day counts only if
        # every rung's observed sub-block is positive definite (a newly
        # listed asset enters with a zero row and fails this); otherwise
        # the mixture keeps its weights for the day.  With the split the
        # rung matrices held here are correlations (vol > 0 on observed
        # assets, so PD of the correlation block <=> PD of the covariance).
        for k in range(K):
            sub = self.prev_sig[k] if full else self.prev_sig[k][np.ix_(idx, idx)]
            if not self.split:
                sub = 0.5 * (sub + sub.T)
            try:
                if _cho_factor is not None:
                    _cho_factor(sub, lower=True, check_finite=False)
                else:
                    np.linalg.cholesky(sub)
            except (np.linalg.LinAlgError, ValueError):
                return None
        if self.split:
            prev_vol, prev_cov = self.prev_vol, self.prev_cov
            assert prev_vol is not None and prev_cov is not None
            # an asset observed today but without a variance state at the
            # forecast has a zero covariance row: abstain, as the covariance
            # factorisation did
            if not full and np.any(prev_vol[idx] <= 0.0):
                return None
            sig = prev_cov if full else prev_cov[np.ix_(idx, idx)]
            sig = 0.5 * (sig + sig.T)
        else:
            sig = np.zeros((idx.size, idx.size))
            for k in range(K):
                sig += self.prev_w[k] * self.prev_sig[k][np.ix_(idx, idx)]
            sig = 0.5 * (sig + sig.T)
        # The spectrum is floored at _FLOOR before factorising, exactly as
        # the research engine's scoring path does: a near-singular early
        # blend would otherwise hand the running scale one enormous
        # gradient and silence the mixer for years.  The floor acts only
        # when lambda_min < _FLOOR, so factorise first and certify: the
        # inverse gives lambda_min >= 1/||sig^-1||_F, and a certified day
        # is bit-identical to the floored path (no floor, same factor).
        # Uncertified or failed days take the eigenvalue floor as before.
        sol: tuple[np.ndarray, np.ndarray] | None = None
        try:
            sol = self._solve(sig, r_t[idx])
            fro = float(np.linalg.norm(sol[1]))
            if not (np.isfinite(fro) and fro * self._CERT * self._FLOOR <= 1.0):
                sol = None
        except (np.linalg.LinAlgError, ValueError):
            sol = None
        if sol is None:
            min_eig = float(np.linalg.eigvalsh(sig).min())
            if min_eig < self._FLOOR:
                sig = sig + np.eye(idx.size) * (self._FLOOR - min_eig)
            try:
                sol = self._solve(sig, r_t[idx])
            except (np.linalg.LinAlgError, ValueError):
                return None
        u, sinv = sol
        g = np.empty(K)
        self._gc = None
        if self.split and self.prev_comps is not None:
            prev_comps, prev_vol, prev_v = self.prev_comps, self.prev_vol, self.prev_v
            assert prev_vol is not None and prev_v is not None
            # Gradients on the expert CORRELATIONS:
            #   tr(Sinv (T o vv)) = sum((Sinv o vv) * T),
            #   u'(T o vv) u = (u o s)' T (u o s);
            # the rung gradient follows by linearity of the pool.
            vo = prev_vol if full else prev_vol[idx]
            mw = sinv * np.multiply.outer(vo, vo)
            ut = u * vo
            J = self.n_experts
            gc = np.empty((K, J))
            for k in range(K):
                for j in range(J):
                    t_ = prev_comps[k][j]
                    ts = t_ if full else t_[np.ix_(idx, idx)]
                    gc[k, j] = -0.5 * float((mw * ts).sum()) + 0.5 * float(ut @ ts @ ut)
                g[k] = float(prev_v[k] @ gc[k])
            if not np.all(np.isfinite(g)):
                return None
            self._gc = gc
            return g
        for k in range(K):
            sk = self.prev_sig[k][np.ix_(idx, idx)]
            g[k] = -0.5 * float((sinv * sk).sum()) + 0.5 * float(u @ sk @ u)
        if not np.all(np.isfinite(g)):
            return None
        return g

    def advance(self, g: np.ndarray | None) -> None:
        if g is not None:
            dd = g - g.mean()
            rms = float(np.sqrt((dd @ dd) / dd.size))
            self.scale = (self._gamma_scale * self.scale
                          + (1.0 - self._gamma_scale) * rms)
            ell = (dd / (self.scale + self._JITTER)) * np.sqrt(2.0 * self.eps)
            lw = np.log(np.maximum(self.w, 1e-300)) + ell
            lw -= lw.max()
            w = np.exp(lw)
            self.w = w / w.sum()
            if self.split and self._gc is not None and self.prev_prior is not None:
                for k in range(self.v.shape[0]):
                    gc = self._gc[k]
                    if not np.all(np.isfinite(gc)):
                        continue
                    ddv = gc - gc.mean()
                    rmsv = float(np.sqrt((ddv @ ddv) / float(self.n_experts)))
                    self.scale_v[k] = (self._gamma_scale * self.scale_v[k]
                                       + (1.0 - self._gamma_scale) * rmsv)
                    lv = (np.log(np.maximum(self.v[k], 1e-300))
                          + (ddv / (self.scale_v[k] + self._JITTER)) * np.sqrt(2.0 * self.eps))
                    lv -= lv.max()
                    vk = np.exp(lv)
                    vk /= vk.sum()
                    self.v[k] = (1.0 - self.eps) * vk + self.eps * self.prev_prior[k]
        self.w = (1.0 - self.eps) * self.w + self.eps * self.prior

    def record(self, rung_covs: list[np.ndarray],
               comps: list[tuple[np.ndarray, ...]] | None = None,
               prior: np.ndarray | None = None,
               cov: np.ndarray | None = None,
               vol: np.ndarray | None = None) -> None:
        """Cache this step's forecast for next step's ``score``.  Without
        the split: the rung covariances and the weights that blend them.
        With the split: the rung CORRELATIONS (mixed), the expert
        correlations, the rule's prior split, the emitted covariance and
        the volatility vector; nothing per expert is materialised."""
        self.prev_sig = rung_covs
        self.prev_w = self.w.copy()
        self.prev_comps = comps
        self.prev_prior = None if prior is None else prior.copy()
        self.prev_cov = cov
        self.prev_vol = vol
        self.prev_v = self.v.copy()

    def blend(self, prior: np.ndarray) -> np.ndarray:
        return np.asarray(self.w)

    @property
    def tilt(self) -> float:
        """Diagnostic analogue of the CUSUM tilt: signed departure of the
        blend from its prior toward the fast (+) or slow (-) rung."""
        return float(self.w[0] - self.prior[0] - (self.w[-1] - self.prior[-1]))


class SqueezeKernelEstimator:
    """The Squeeze Kernel estimator with its full switch surface.

    ``SqueezeKernel`` (one number, ``lam``) is the public front-end; this
    class is the engine behind it and the home of every reproduction and
    ablation switch.  The defaults reproduce the published v1 single-scale
    estimator bit-for-bit; ``corr_half_lives``, ``shrinkage_target`` and
    ``corr_theta`` give the published v1 adaptive configuration; the
    ``kappa_mode``, ``alpha_rule``, ``level_match``, ``detector`` and
    ``clock`` switches are the 2.0 mechanics.  Positive semi-definite by
    construction, missing values (NaN) native, O(K n^2) per update.

    Parameters
    ----------
    n_assets : int
    lambda_vol : float
        Per-asset variance EWMA decay (default 0.98).
    lambda_corr : float
        Single-scale correlation decay (default 0.996); ignored when a
        ladder is configured.
    kappa : float
        Fixed kernel scale of the saturating weight w = d²/(d² + kappa)
        (default 0.25); ignored under ``kappa_mode='adaptive'``.
    epsilon : float
        Numerical floor (default 1e-8).
    shrinkage : str or float
        ``'auto'`` (default) for adaptive intensity, ``'none'``/``0`` to
        disable, or a fixed float in [0, 1].
    shrinkage_delta : float
        Offset of the published intensity rule ``min(1, n/2S) - delta``
        (default 0.10); unused under ``alpha_rule='selftuning'``.
    shrinkage_target : str
        ``'equicorrelation'`` (default) or ``'cluster'`` (Hadamard-square
        target, PSD by the Schur product theorem).
    corr_half_lives : sequence of float, optional
        Correlation timescale ladder in trading days, e.g. ``(43, 173,
        693)``; ``None`` is the single-scale estimator.  Two or more rungs
        enable the surprise detector.
    corr_theta : float
        Rung-weight exponent, prior weights ∝ half-life**theta (default
        0.25; 2.0 uses 0.5).
    min_obs : int or None
        Diagnostic usability gate: ``usable_mask`` marks assets with at
        least ``min_obs`` observations; estimates are unaffected.
    kappa_mode : str
        ``'fixed'`` (default) or ``'adaptive'``: kernel scale as state,
        ``kappa_t = EWMA(activity) / 3`` at the median rung's half-life.
    alpha_rule : str
        ``'published'`` (default), ``'selftuning'`` (intensity from the
        online concentration and de-noised market fit) or
        ``'selftuning-target'`` (target-family fit variant).
    level_match : bool
        Level-match the cluster target (default True; 2.0 uses False,
        the pure Schur-square target).
    detector : bool
        Surprise-gated rung weights when a ladder is configured (default
        True).
    clock : str
        ``'global'`` (default) or ``'asset'``: per-asset market clocks from
        the correlation neighbourhood, with the PSD diagonal-congruence
        rung update.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 3))
    >>> est = SqueezeKernelEstimator(n_assets=3)
    >>> for r_t in returns:
    ...     est.update(r_t)
    >>> cov = est.get_cov()
    """

    def __init__(
        self,
        n_assets: int,
        *,
        lambda_vol: float = 0.98,
        lambda_corr: float = 0.996,
        kappa: float = 0.25,
        epsilon: float = 1e-8,
        shrinkage: str | float = "auto",
        shrinkage_delta: float = 0.10,
        shrinkage_target: str = "equicorrelation",
        corr_half_lives: "Sequence[float] | None" = None,
        corr_theta: float = 0.25,
        min_obs: int | None = None,
        kappa_mode: str = "fixed",
        alpha_rule: str = "published",
        level_match: bool = True,
        detector: bool = True,
        clock: str = "global",
        vol_ladder: bool = False,
        weights: str = "cusum",
        split_learn: bool = False,
    ):
        self.n_assets = n_assets
        if weights not in ("cusum", "eg_blend"):
            raise ValueError("weights must be 'cusum' or 'eg_blend'.")
        self.weights = weights
        if split_learn and weights != "eg_blend":
            raise ValueError("split_learn requires weights='eg_blend'.")
        self.split_learn = bool(split_learn)
        self.vol_ladder = bool(vol_ladder)
        self.lambda_vol = lambda_vol
        self.lambda_corr = lambda_corr
        self.epsilon = epsilon
        self.shrinkage_delta = shrinkage_delta
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0.")
        self.kappa = float(kappa)
        if shrinkage_target not in ("equicorrelation", "cluster"):
            raise ValueError("shrinkage_target must be 'equicorrelation' or 'cluster'.")
        self.shrinkage_target = shrinkage_target
        if min_obs is not None and (not isinstance(min_obs, int) or min_obs < 1):
            raise ValueError("min_obs must be a positive integer or None.")
        self.min_obs = min_obs
        self._obs_count = np.zeros(n_assets, dtype=np.int64)

        # v2 self-tuning opt-ins (defaults preserve v1 bit-for-bit).
        if kappa_mode not in ("fixed", "adaptive"):
            raise ValueError("kappa_mode must be 'fixed' or 'adaptive'.")
        if alpha_rule not in ("published", "selftuning", "selftuning-target"):
            raise ValueError(
                "alpha_rule must be 'published', 'selftuning' or "
                "'selftuning-target'."
            )
        if (kappa_mode == "adaptive" or alpha_rule == "selftuning") \
                and corr_half_lives is None:
            raise ValueError(
                "kappa_mode='adaptive' and alpha_rule='selftuning' require "
                "the correlation ladder (corr_half_lives)."
            )
        if clock not in ("global", "asset"):
            raise ValueError("clock must be 'global' or 'asset'.")
        if clock == "asset" and corr_half_lives is None:
            raise ValueError("clock='asset' requires the correlation ladder.")
        self.clock = clock
        self.kappa_mode = kappa_mode
        self.alpha_rule = alpha_rule
        self.level_match = level_match
        self._kappa_c = 1.0 / 3.0          # chi-squared-null constant
        self._kap_state = 1.0              # chi-squared null mean of d^2
        self._kap_lam = 0.0                # set with the ladder below
        self._J_list: list[float] | None = None
        self._offmask_cache: np.ndarray | None = None
        # v2 per-asset market clocks (clock='asset'): each asset's trading
        # time advances with the squared surprise of its correlation
        # neighborhood (row-normalized Schur-square weighting of z^2); the
        # rung update becomes the PSD diagonal congruence
        #   Q_ij <- sqrt((1-eta_i)(1-eta_j)) Q_ij + sqrt(eta_i eta_j) z_i z_j,
        # exactly the chord when clocks equalize. Missing assets keep the
        # global clock; nu/alpha/detector stay on the global clock.
        self._S_asset: np.ndarray | None = None
        self._kap_asset: np.ndarray | None = None
        self._C_prev: np.ndarray | None = None
        self._corr_blend_buf: np.ndarray | None = None
        if clock == "asset":
            khl = np.asarray(corr_half_lives, dtype=np.float64)
            self._S_asset = np.full((khl.size, n_assets), float(epsilon))
            self._kap_asset = np.ones(n_assets)
            self._C_prev = np.eye(n_assets)

        # Scale-free correlation memory (opt-in): replace the single correlation
        # timescale by a positive combination of EWMAs on a geometric half-life
        # ladder, blended per-scale (Mode A). None => single-scale, published
        # behaviour bit-for-bit. See ``corr_half_lives`` in the class docstring.
        self.corr_half_lives = None
        self.corr_theta = corr_theta
        self._corr_lam: np.ndarray | None = None
        self._corr_w: np.ndarray | None = None
        self._detector: _SurpriseDetector | _BlendGradient | None = None
        self._Q_list: list[np.ndarray] | None = None
        self._S_list: list[float] | None = None
        self._adaptive = False
        if corr_half_lives is not None:
            hl = np.asarray(corr_half_lives, dtype=np.float64)
            if hl.ndim != 1 or hl.size < 1 or np.any(hl <= 0.0):
                raise ValueError("corr_half_lives must be a non-empty sequence of positive half-lives.")
            if corr_theta < 0.0:
                raise ValueError("corr_theta must be >= 0.")
            self.corr_half_lives = hl
            self._corr_lam = 2.0 ** (-1.0 / hl)
            w = hl ** corr_theta
            self._corr_w = w / w.sum()
            self._Q_list = [np.eye(n_assets, dtype=np.float64) for _ in hl]
            self._S_list = [float(epsilon) for _ in hl]
            self._J_list = [float(epsilon) for _ in hl]
            self._kap_lam = 2.0 ** (-1.0 / float(np.median(hl)))
            # Surprise-gated blend weights (integral for K >= 2): see
            # ``_SurpriseDetector``. ``detector=False`` opts out (theta-prior
            # weights; the v2 ablation switch).
            self._adaptive = detector and hl.size >= 2
            if self._adaptive:
                self._detector = (_BlendGradient(hl, self._corr_w, split=self.split_learn,
                                                 n_experts=2 if shrinkage_target == "equicorrelation" else 3)
                                  if weights == "eg_blend"
                                  else _SurpriseDetector(hl))

        # Resolve shrinkage
        if isinstance(shrinkage, str):
            self._shrinkage_alpha = -1.0 if shrinkage == "auto" else 0.0
        else:
            self._shrinkage_alpha = float(shrinkage)

        # State. The correlation memory is stored NORMALISED: Q_t = M_t / S_t
        # with the recursion Q_t = (1 - eta_t) Q_{t-1} + eta_t z_t z_t',
        # eta_t = w_t / S_t after S_t <- lam S_{t-1} + w_t. This is
        # algebraically identical to the raw-mass form (M init eps*I, S init
        # eps => Q init I), keeps the matrix state well scaled, and makes the
        # PSD convex-combination recursion explicit.
        self._var_t: np.ndarray | None = None
        self._var_init: np.ndarray | None = None
        self._vol_t: np.ndarray | None = None
        # Self-adapting volatility memory: a ladder of decays two octaves
        # below the correlation ladder, pooled with panel-wide weights that
        # are Bayes at the null-calibrated temperature on SATURATED
        # (tanh) evidence -- one day cannot hand a stale rung weeks of
        # weight -- with the fastest rung's memory and a uniform prior.
        # Replaces the fitted constant ``lambda_vol``.
        self._var_l: np.ndarray | None = None
        if self.vol_ladder:
            if self.corr_half_lives is None:
                raise ValueError("vol_ladder requires corr_half_lives.")
            h = float(np.median(self.corr_half_lives))
            self._hl_v = np.array([h / 16.0, h / 4.0, h])
            self._lv_l = 2.0 ** (-1.0 / self._hl_v)
            self._eps_v = float(1.0 - self._lv_l[0])
            self._pi_v = np.full(3, 1.0 / 3.0)
            self._gamma_v = 2.0 ** (-1.0 / h)
            self._w_p = self._pi_v.copy()
            self._scale_p = 1.0
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

    def update(self, r_t: ArrayLike) -> float:
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
        ell = None
        detector = self._detector
        if detector is not None:
            ell = detector.score(r_t, finite)
            detector.advance(ell)

        # ── Volatility update ──
        # Locals alias the variance states: the guard initialises the pair
        # together, so both are non-None afterwards, and local aliases keep
        # the narrowing visible to mypy.
        var_t = self._var_t
        var_init = self._var_init
        if var_t is None or var_init is None:
            var_t = np.zeros(n, dtype=np.float64)
            var_init = np.zeros(n, dtype=bool)
            self._var_t = var_t
            self._var_init = var_init

        first = finite & ~var_init
        repeat = finite & var_init
        if np.any(first):
            var_t[first] = r_t[first] ** 2 + eps
            var_init[first] = True
            if self.vol_ladder:
                if self._var_l is None:
                    self._var_l = np.full((3, n), eps)
                self._var_l[:, first] = r_t[first] ** 2 + eps
        if np.any(repeat):
            if self.vol_ladder:
                assert self._var_l is not None
                r2 = r_t[repeat] ** 2
                vr = self._var_l[:, repeat]
                vr_safe = np.maximum(vr, eps)
                ell_v = -0.5 * (np.log(vr_safe) + r2 / vr_safe)
                ep = ell_v.sum(axis=1)
                dp = ep - ep.mean()
                rp = float(np.sqrt((dp @ dp) / 3.0))
                self._scale_p = (self._gamma_v * self._scale_p
                                 + (1.0 - self._gamma_v) * rp)
                zp = np.tanh(dp / (self._scale_p + 1e-12))
                lw = np.log(np.maximum(self._w_p, 1e-300)) + zp * np.sqrt(2.0 * self._eps_v)
                lw -= lw.max()
                w_p = np.exp(lw)
                w_p /= w_p.sum()
                self._w_p = (1.0 - self._eps_v) * w_p + self._eps_v * self._pi_v
                lv = self._lv_l[:, None]
                self._var_l[:, repeat] = lv * vr + (1.0 - lv) * r2
                var_t[repeat] = (self._w_p[:, None] * self._var_l[:, repeat]).sum(axis=0)
            else:
                var_t[repeat] = (
                    self.lambda_vol * var_t[repeat]
                    + (1.0 - self.lambda_vol) * r_t[repeat] ** 2
                )

        vol_t = np.zeros(n, dtype=np.float64)
        vol_t[var_init] = np.sqrt(var_t[var_init])

        # ── Standardized returns ──
        z_t = np.zeros(n, dtype=np.float64)
        n_obs = int(finite.sum())
        if n_obs > 0:
            z_t[finite] = r_t[finite] / (vol_t[finite] + eps)
            d2 = float(z_t[finite] @ z_t[finite]) / n_obs
            if self.kappa_mode == "adaptive":
                # kappa_t = c * EWMA_h(d^2): state, not parameter. Today's
                # weight uses the state BEFORE absorbing today's d^2.
                kappa_t = self._kappa_c * self._kap_state
                self._kap_state = (self._kap_lam * self._kap_state
                                   + (1.0 - self._kap_lam) * d2)
                w_t = d2 / (d2 + kappa_t)
                self._last_kappa = kappa_t
            else:
                w_t = kernel_fisher(d2, kappa=self.kappa)
        else:
            w_t = 0.0

        add = w_t > 0.0 and n_obs > 0
        if add:
            # np.multiply.outer with out= avoids the temporary that
            # np.outer otherwise allocates each step. zz' is computed once
            # and shared by every rung.
            np.multiply.outer(z_t, z_t, out=self._scratch_outer)
        corr_lam = self._corr_lam
        if corr_lam is None:
            # ── Single-scale correlation EWMA (published path) ──
            Q_t = self._Q_t
            assert Q_t is not None     # the single-scale state exists exactly
            lam_c = self.lambda_corr   # when the ladder is not configured
            self._S_t = lam_c * self._S_t + w_t
            if add:
                # Q <- (1 - eta) Q + eta zz'. With w_t = 0 both S and M decay
                # by lam_c, so Q is unchanged — no matrix work at all.
                eta = w_t / self._S_t
                Q_t *= 1.0 - eta
                self._scratch_outer *= eta
                Q_t += self._scratch_outer
        else:
            # ── Scale-free ladder (Mode A) ──
            # Update K normalised correlation states on the geometric
            # half-life ladder; extraction (normalise + shrink + blend)
            # happens lazily in _materialize().
            Q_list = self._Q_list
            S_list = self._S_list
            corr_w = self._corr_w
            assert Q_list is not None and S_list is not None
            assert corr_w is not None  # allocated together with corr_lam
            J_list = self._J_list
            assert J_list is not None
            w_vec = None
            if self.clock == "asset" and n_obs > 0:
                C_prev = self._C_prev
                kap_a = self._kap_asset
                assert C_prev is not None and kap_a is not None
                H = C_prev * C_prev
                denom = H @ finite.astype(np.float64)
                a_act = (H @ (z_t * z_t)) / np.maximum(denom, eps)
                w_vec = np.where(finite,
                                 a_act / (a_act + self._kappa_c * kap_a),
                                 w_t)
                self._kap_asset = np.where(
                    finite,
                    self._kap_lam * kap_a + (1.0 - self._kap_lam) * a_act,
                    kap_a)
            s_eff = 0.0
            for k in range(corr_lam.size):
                S_list[k] = corr_lam[k] * S_list[k] + w_t
                J_list[k] = corr_lam[k] * corr_lam[k] * J_list[k] + w_t
                if w_vec is not None:
                    S_a = self._S_asset
                    assert S_a is not None
                    S_a[k] = corr_lam[k] * S_a[k] + w_vec
                    eta_v = w_vec / S_a[k]
                    o = np.sqrt(1.0 - eta_v)
                    u = np.sqrt(eta_v) * z_t
                    np.multiply.outer(o, o, out=self._scratch_had)
                    Q_list[k] *= self._scratch_had
                    np.multiply.outer(u, u, out=self._scratch_had)
                    Q_list[k] += self._scratch_had
                elif add:
                    eta = w_t / S_list[k]
                    Q_list[k] *= 1.0 - eta
                    np.multiply(self._scratch_outer, eta, out=self._scratch_had)
                    Q_list[k] += self._scratch_had
                s_eff += corr_w[k] * S_list[k]
            self._S_t = s_eff                        # blended effective size (for the property)
        self._vol_t = vol_t
        self._dirty = True
        if self._adaptive:
            self._materialize_adaptive()
        elif self.clock == "asset":
            self._materialize()          # C_prev must advance daily
        self._last_weight = w_t
        return w_t

    def get_cov(self) -> np.ndarray:
        """Return the current covariance matrix estimate (n x n)."""
        if self._vol_t is None:
            raise RuntimeError("Call update() at least once before get_cov().")
        if self._dirty:
            self._materialize()
        cov = self._cov
        assert cov is not None
        return cov.copy()

    def get_corr(self) -> np.ndarray:
        """Return the current correlation matrix estimate (n x n)."""
        if self._vol_t is None:
            raise RuntimeError("Call update() at least once before get_corr().")
        if self._dirty:
            self._materialize()
        corr = self._corr
        assert corr is not None
        return corr.copy()

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

    # ── Private helpers ───────────────────────────────────────────────────

    def _shrunk_corr_from_Q(self, Q: np.ndarray, S_t: float,
                            J_t: float | None = None) -> np.ndarray:
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
        alpha = self._intensity(corr, S_t, J_t)
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
                if self.level_match:
                    mean_off = (had.sum() - np.trace(had)) / self._n_off
                    gamma = min(1.0, rho_bar / max(mean_off, eps))
                else:
                    gamma = 1.0            # v2: pure Schur-square target
                corr *= (1.0 - alpha)
                corr += (alpha * (1.0 - alpha)) * rho_bar
                had *= alpha * alpha * gamma
                corr += had
                np.fill_diagonal(corr, 1.0)
        return corr

    def _intensity(self, corr: np.ndarray, S_t: float,
                   J_t: float | None = None) -> float:
        """Shrinkage intensity of one timescale from its normalised raw
        correlation ``corr`` and its information masses (published rule or
        the self-tuning rule); ``corr`` is read, not modified."""
        eps = self.epsilon
        n = self.n_assets
        S_t = max(S_t, eps)
        alpha = self._shrinkage_alpha
        if alpha < 0:
            if self.alpha_rule != "published" and J_t is not None:
                # v2 self-tuning intensity (WP7): alpha from the online
                # concentration c = n/nu (nu = S^2/J, fractional-info ESS)
                # and the de-noised equicorrelation-explained fraction
                # g = rho^2 / (mo - vhat):
                #   alpha = min(1,c) * g^2 / (g^2 + (1-g)^2 max(0, 1/c - 1))
                # Zero free constants; arithmetic mirrors the research
                # engine entry-for-entry (offmask means).
                if self._offmask_cache is None:
                    self._offmask_cache = ~np.eye(n, dtype=bool)
                off = corr[self._offmask_cache]
                nu = S_t * S_t / max(J_t, eps)
                c_k = n / max(nu, eps)
                vhat = float(((1.0 - off * off) ** 2).mean()) / max(nu, eps)
                mo_r = float((off * off).mean())
                rho_r = float(off.mean())
                if self.alpha_rule == "selftuning-target":
                    # Research variant (f7): gate by the target-family fit
                    # R^2 of offdiag(C) on {1, C o C} = corr(x, x^2)^2 —
                    # closes the strong-cluster/zero-mean-correlation corner
                    # at a measured cost on dispersed signed panels (see the
                    # papers' ablation); the shipped rule uses the market
                    # fit below.
                    m3 = float((off * off * off).mean())
                    m4 = float((off ** 4).mean())
                    varx = max(mo_r - rho_r * rho_r, 1e-12)
                    vary = max(m4 - mo_r * mo_r, 1e-12)
                    cov_xy = m3 - rho_r * mo_r
                    gt = min(1.0, cov_xy * cov_xy / (varx * vary))
                else:
                    gt = min(1.0, rho_r * rho_r / max(mo_r - vhat, 1e-6))
                r_ = (1.0 - gt) / max(gt, 1e-6)
                s_ = max(0.0, 1.0 / max(c_k, eps) - 1.0)
                alpha = min(1.0, c_k) / (1.0 + r_ * r_ * s_)
            else:
                alpha = max(0.0, min(1.0, n / (2.0 * S_t) - self.shrinkage_delta))
        return alpha

    def _rung_experts(self, Q: np.ndarray, S_t: float, J_t: float
                      ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """The three PSD unit-diagonal experts of one timescale -- raw
        correlation, equicorrelation at its mean level, Schur square --
        and the rule's prior split (1-a, a(1-a), a^2)."""
        eps = self.epsilon
        n = self.n_assets
        diag_z = np.diagonal(Q).copy()
        inv_diag = 1.0 / np.sqrt(np.maximum(diag_z, eps))
        corr = np.multiply.outer(inv_diag, inv_diag)
        corr *= Q
        np.fill_diagonal(corr, np.where(diag_z > eps, 1.0, 0.0))
        a = self._intensity(corr, S_t, J_t)
        rho = (corr.sum() - n) / self._n_off
        rho = float(min(max(rho, -0.99 / max(n - 1, 1)), 0.999))
        E = np.full((n, n), rho)
        np.fill_diagonal(E, 1.0)
        if self.shrinkage_target == "equicorrelation":
            # target off: the pool is the equicorrelation shrinkage alone
            return (corr, E), np.array([1.0 - a, a])
        return (corr, E, corr * corr), np.array([1.0 - a, a * (1.0 - a), a * a])

    def _materialize_adaptive(self) -> None:
        """Adaptive-weight extraction: build per-rung shrunk covariances
        (kept for the next update's detector scores), blend with the
        CUSUM-tilted weights, derive _cov/_corr. Runs eagerly."""
        detector = self._detector
        corr_lam = self._corr_lam
        Q_list = self._Q_list
        S_list = self._S_list
        corr_w = self._corr_w
        vol_t = self._vol_t
        assert detector is not None      # only called when the detector exists
        assert corr_lam is not None and Q_list is not None
        assert S_list is not None and corr_w is not None
        assert vol_t is not None         # set on every update before this runs
        eps = self.epsilon
        vv = np.multiply.outer(vol_t, vol_t)
        sig_k = []
        J_list = self._J_list
        assert J_list is not None
        K = corr_lam.size
        buf = self._corr_blend_buf
        if self.clock == "asset" and buf is None:
            buf = np.empty((K, self.n_assets, self.n_assets))
            self._corr_blend_buf = buf
        comps: list[tuple[np.ndarray, ...]] | None = None
        prior: np.ndarray | None = None
        v: np.ndarray | None = None
        if self.split_learn:
            assert isinstance(detector, _BlendGradient)   # the split rides on the blend gradient
            comps = []
            v = detector.v
            n_exp = 2 if self.shrinkage_target == "equicorrelation" else 3
            if v.shape[1] != n_exp:
                # the target was switched after construction; the split has
                # not learned anything yet, so re-shape its state
                assert detector.prev_comps is None, "cannot switch the target mid-stream"
                detector.n_experts = n_exp
                detector.v = np.full((K, n_exp), 1.0 / n_exp)
                v = detector.v
            prior = np.empty(v.shape)
        for k in range(K):
            if self.split_learn:
                assert comps is not None and prior is not None and v is not None
                experts, prior[k] = self._rung_experts(Q_list[k], S_list[k], J_list[k])
                corr_k = v[k, 0] * experts[0]
                for j in range(1, len(experts)):
                    corr_k = corr_k + v[k, j] * experts[j]
                comps.append(experts)
                sig_k.append(corr_k)          # mixed correlation; vv applied once below
            else:
                corr_k = self._shrunk_corr_from_Q(Q_list[k], S_list[k], J_list[k])
                sig_k.append(corr_k * vv)
            if buf is not None:
                buf[k][:] = corr_k
        w = detector.blend(corr_w)
        if self.split_learn:
            cb = w[0] * sig_k[0]
            for k in range(1, K):
                cb = cb + w[k] * sig_k[k]
            if buf is not None:
                self._C_prev = cb
            cov = cb * vv
            cov = (cov + cov.T) * 0.5
            detector.record(sig_k, comps, prior, cov, vol_t)  # type: ignore[call-arg]
        else:
            detector.record(sig_k)
            if buf is not None:
                # next day's neighborhood weighting: the emitted blend of the
                # shrunk rung correlations
                self._C_prev = np.tensordot(w, buf, axes=1)
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
        assert vol_t is not None   # get_cov/get_corr guard this before calling
        if self._corr_lam is None:
            # ── Single scale: shrunk correlation IS the correlation output ──
            Q_t = self._Q_t
            assert Q_t is not None   # the ladder is not configured on this path
            corr = self._shrunk_corr_from_Q(Q_t, self._S_t)
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
            corr_lam = self._corr_lam
            Q_list = self._Q_list
            S_list = self._S_list
            corr_w = self._corr_w
            assert Q_list is not None and S_list is not None
            assert corr_w is not None  # allocated together with corr_lam
            J_list = self._J_list
            assert J_list is not None
            mix = np.zeros((self.n_assets, self.n_assets), dtype=np.float64)
            for k in range(corr_lam.size):
                corr_k = self._shrunk_corr_from_Q(Q_list[k], S_list[k],
                                                  J_list[k])
                corr_k *= corr_w[k]
                mix += corr_k
            if self.clock == "asset":
                self._C_prev = mix.copy()
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


