"""The Squeeze Kernel covariance estimator.

One public number, ``lam``; everything else derives from it, is a
structural constant of the method, or is state computed online:

- a ladder of three correlation timescales with decays
  ``(lam**4, lam, lam**(1/4))`` and prior weights proportional to the
  square root of the half-life;
- per-asset variance on a ladder of half-lives ``(h/16, h/4, h)``,
  pooled panel-wide by weights that move on tanh-saturated evidence;
- per-asset market clocks read from the correlation neighbourhood (the
  Schur-square-weighted mean of squared surprises) through the saturating
  kernel ``w = a / (a + kappa)``, with the kernel scale
  ``kappa = EWMA(activity) / 3`` as state;
- a diagonal-congruence update of each timescale's correlation state on
  those clocks, the day's innovation scaled by ``omega = n/m`` with ``m``
  its Mahalanobis distance under the previous forecast;
- per timescale, a learned pool of the raw correlation, its
  equicorrelation level and its Hadamard square, with the self-tuning
  intensity rule's split ``(1-a, a(1-a), a**2)`` as the prior;
- timescale weights and pool splits moved by exponentiated gradient on
  the blend's own log score at the null-calibrated temperature.

Positive semi-definite by construction, missing values (NaN) native,
``O(K n^2)`` state work plus one Cholesky factorisation of the blend and
``K`` positive-definiteness checks a day.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

try:
    # SciPy's LAPACK bindings factorise an SPD matrix several times faster
    # than the NumPy route and compute the identical quantities.
    from scipy.linalg import cho_factor as _cho_factor, cho_solve as _cho_solve
except ImportError:  # pragma: no cover
    _cho_factor = _cho_solve = None

__all__ = ["SqueezeKernel"]

# Structural constants of the method (each bracketed by ablation in the paper).
_B = 4.0             # ladder spacing: timescales (h/b, h, h*b)
_THETA = 0.5         # timescale prior weights ~ h**theta
_KAPPA_C = 1.0 / 3.0  # kernel scale: kappa = kappa_c * EWMA(activity)
_EPS = 1e-8          # numerical floor, squared-return units


class _BlendGradient:
    """Exponentiated gradient on the BLEND's log score, at the
    null-calibrated temperature.

    The emitted covariance is the pool of the timescales' correlations,
    each itself a pool of three experts, times the volatilities.  The
    Gaussian log score of the blend has gradient

        d ell / d w_k = -1/2 tr(Sigma^-1 Sigma_k) + 1/2 u' Sigma_k u,
        u = Sigma^-1 r,

    so the update moves the blend rather than selecting a timescale.  The
    gradient is studentised across timescales (and across experts within
    one) by its running rms and scaled by sqrt(2 eps), eps the fixed-share
    rate: the accumulated log-odds then have unit variance under
    uninformative evidence.  Evidence memory: the fastest timescale.
    Causal: ``score`` reads the previous step's forecast, ``advance``
    updates the weights used for the next blend.
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
                 n_experts: int = 3) -> None:
        hl = np.asarray(half_lives, dtype=np.float64)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.w = self.prior.copy()
        self.eps = float(1.0 - 2.0 ** (-1.0 / float(hl.min())))
        self._gamma_scale = 2.0 ** (-1.0 / float(np.median(hl)))
        self.scale = 1.0
        self.prev_sig: list[np.ndarray] | None = None
        self.prev_w: np.ndarray | None = None
        # Each timescale's correlation is a convex combination of three PSD
        # unit-diagonal experts -- raw, equicorrelation at its mean level,
        # Schur square -- whose prior is the intensity rule's split
        # (1-a, a(1-a), a^2) and whose weights are moved by the same
        # gradient step, studentised within the timescale.  Starts uniform
        # and forgets toward the prior at the fixed-share rate.
        self.n_experts = int(n_experts)
        self.v = np.full((hl.size, self.n_experts), 1.0 / self.n_experts)
        self.scale_v = np.ones(hl.size)
        self.prev_comps: list[tuple[np.ndarray, ...]] | None = None
        self.prev_prior: np.ndarray | None = None
        self.prev_cov: np.ndarray | None = None
        self.prev_vol: np.ndarray | None = None
        self.prev_v: np.ndarray | None = None
        self._gc: np.ndarray | None = None
        # The day's Mahalanobis distance under the forecast, m = r' Sigma^-1 r
        # on the observed subvector, and that subvector's size: read by the
        # observation weight; NaN on a day the gradient abstains.
        self.m_last = float("nan")
        self.n_last = 0

    def score(self, r_t: np.ndarray, finite: np.ndarray) -> np.ndarray | None:
        """Gradient of the previous blend's log score at ``r_t`` (observed
        subvector), one entry per timescale; ``None`` on a degenerate day."""
        self.m_last = float("nan")
        self.n_last = 0
        if self.prev_sig is None or self.prev_w is None:
            return None
        idx = np.flatnonzero(finite)
        if idx.size < 2:
            return None
        K = len(self.prev_sig)
        full = idx.size == self.prev_sig[0].shape[0]
        # The day counts only if every timescale's observed sub-block is
        # positive definite (a newly listed asset enters with a zero row and
        # fails this); otherwise the mixture keeps its weights for the day.
        # The matrices held here are correlations (vol > 0 on observed
        # assets, so PD of the correlation block <=> PD of the covariance).
        for k in range(K):
            sub = self.prev_sig[k] if full else self.prev_sig[k][np.ix_(idx, idx)]
            try:
                if _cho_factor is not None:
                    _cho_factor(sub, lower=True, check_finite=False)
                else:
                    np.linalg.cholesky(sub)
            except (np.linalg.LinAlgError, ValueError):
                return None
        prev_vol, prev_cov = self.prev_vol, self.prev_cov
        assert prev_vol is not None and prev_cov is not None
        # an asset observed today but without a variance state at the
        # forecast has a zero covariance row: abstain, as the covariance
        # factorisation would
        if not full and np.any(prev_vol[idx] <= 0.0):
            return None
        sig = prev_cov if full else prev_cov[np.ix_(idx, idx)]
        sig = 0.5 * (sig + sig.T)
        # The spectrum is floored at _FLOOR before factorising: a
        # near-singular early blend would otherwise hand the running scale
        # one enormous gradient and silence the mixer for years.  The floor
        # acts only when lambda_min < _FLOOR, so factorise first and
        # certify: the inverse gives lambda_min >= 1/||sig^-1||_F, and a
        # certified day is bit-identical to the floored path (no floor, same
        # factor).  Uncertified or failed days take the eigenvalue floor.
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
        self.m_last = float(r_t[idx] @ u)
        self.n_last = int(idx.size)
        g = np.empty(K)
        self._gc = None
        prev_comps, prev_v = self.prev_comps, self.prev_v
        assert prev_comps is not None and prev_v is not None
        # Gradients on the expert CORRELATIONS:
        #   tr(Sinv (T o vv)) = sum((Sinv o vv) * T),
        #   u'(T o vv) u = (u o s)' T (u o s);
        # the timescale gradient follows by linearity of the pool.
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
            if self._gc is not None and self.prev_prior is not None:
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

    def record(self, rung_corrs: list[np.ndarray],
               comps: list[tuple[np.ndarray, ...]],
               prior: np.ndarray, cov: np.ndarray, vol: np.ndarray) -> None:
        """Cache this step's forecast for next step's ``score``: the
        timescale correlations (pooled), the expert correlations, the
        rule's prior split, the emitted covariance and the volatilities."""
        self.prev_sig = rung_corrs
        self.prev_w = self.w.copy()
        self.prev_comps = comps
        self.prev_prior = prior.copy()
        self.prev_cov = cov
        self.prev_vol = vol
        self.prev_v = self.v.copy()

    @property
    def tilt(self) -> float:
        """Signed departure of the blend from its prior toward the fast (+)
        or slow (-) timescale."""
        return float(self.w[0] - self.prior[0] - (self.w[-1] - self.prior[-1]))


class _Engine:
    """State and daily update for a panel of ``n_assets``."""

    def __init__(self, n_assets: int, lam: float) -> None:
        n = n_assets
        eps = _EPS
        self.n_assets = n
        self.epsilon = eps
        h = float(-1.0 / np.log2(lam))
        hl = np.asarray((h / _B, h, h * _B), dtype=np.float64)
        self._corr_lam = 2.0 ** (-1.0 / hl)
        w = hl ** _THETA
        self._corr_w = w / w.sum()
        K = hl.size
        self._Q_list = [np.eye(n, dtype=np.float64) for _ in hl]
        self._S_list = [float(eps) for _ in hl]
        self._J_list = [float(eps) for _ in hl]
        self._kap_lam = 2.0 ** (-1.0 / float(np.median(hl)))
        self._kap_state = 1.0              # chi-squared null mean of d^2
        self._last_kappa = _KAPPA_C * self._kap_state
        # per-asset market clocks: each asset's trading time advances with
        # the squared surprise of its correlation neighbourhood
        self._S_asset = np.full((K, n), float(eps))
        self._kap_asset = np.ones(n)
        self._C_prev = np.eye(n)
        self._detector = _BlendGradient(hl, self._corr_w)
        self._last_omega = 1.0
        # variance ladder two octaves below the correlation ladder
        hv = float(np.median(hl))
        self._lv_l = 2.0 ** (-1.0 / np.array([hv / 16.0, hv / 4.0, hv]))
        self._eps_v = float(1.0 - self._lv_l[0])
        self._pi_v = np.full(3, 1.0 / 3.0)
        self._gamma_v = 2.0 ** (-1.0 / hv)
        self._w_p = self._pi_v.copy()
        self._scale_p = 1.0
        self._var_t = np.zeros(n, dtype=np.float64)
        self._var_init = np.zeros(n, dtype=bool)
        self._var_l: np.ndarray | None = None
        self._vol_t: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._last_weight = 0.0
        self._offmask = ~np.eye(n, dtype=bool)
        self._scratch_had = np.empty((n, n), dtype=np.float64)
        self._n_off = float(n * (n - 1)) if n > 1 else 1.0

    def update(self, r_t: np.ndarray) -> float:
        n = self.n_assets
        eps = self.epsilon
        finite = np.isfinite(r_t)

        # ── score r_t under yesterday's forecast, move the blend weights ──
        detector = self._detector
        detector.advance(detector.score(r_t, finite))

        # ── volatility ladder ──
        var_t, var_init = self._var_t, self._var_init
        first = finite & ~var_init
        repeat = finite & var_init
        if np.any(first):
            var_t[first] = r_t[first] ** 2 + eps
            var_init[first] = True
            if self._var_l is None:
                self._var_l = np.full((3, n), eps)
            self._var_l[:, first] = r_t[first] ** 2 + eps
        if np.any(repeat):
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

        vol_t = np.zeros(n, dtype=np.float64)
        vol_t[var_init] = np.sqrt(var_t[var_init])

        # ── standardised surprise and the panel kernel weight ──
        z_t = np.zeros(n, dtype=np.float64)
        n_obs = int(finite.sum())
        if n_obs > 0:
            z_t[finite] = r_t[finite] / (vol_t[finite] + eps)
            d2 = float(z_t[finite] @ z_t[finite]) / n_obs
            # kappa_t = c * EWMA(d^2): state, not parameter; today's weight
            # uses the state BEFORE absorbing today's d^2.
            kappa_t = _KAPPA_C * self._kap_state
            self._kap_state = (self._kap_lam * self._kap_state
                               + (1.0 - self._kap_lam) * d2)
            w_t = d2 / (d2 + kappa_t)
            self._last_kappa = kappa_t
        else:
            w_t = 0.0

        # ── per-asset clocks and the diagonal-congruence update ──
        corr_lam = self._corr_lam
        Q_list, S_list, J_list = self._Q_list, self._S_list, self._J_list
        w_vec = None
        if n_obs > 0:
            kap_a = self._kap_asset
            H = self._C_prev * self._C_prev
            denom = H @ finite.astype(np.float64)
            a_act = (H @ (z_t * z_t)) / np.maximum(denom, eps)
            w_vec = np.where(finite,
                             a_act / (a_act + _KAPPA_C * kap_a),
                             w_t)
            self._kap_asset = np.where(
                finite,
                self._kap_lam * kap_a + (1.0 - self._kap_lam) * a_act,
                kap_a)
        # observation weight: sqrt(omega) on the innovation, omega = n/m
        # from the gradient's own solve of the forecast at r_t; one on a
        # day the gradient abstained.
        sq_omega = 1.0
        if w_vec is not None:
            m_day = detector.m_last
            if np.isfinite(m_day) and m_day > 1e-12:
                sq_omega = float(np.sqrt(detector.n_last / m_day))
        self._last_omega = sq_omega * sq_omega
        for k in range(corr_lam.size):
            S_list[k] = corr_lam[k] * S_list[k] + w_t
            J_list[k] = corr_lam[k] * corr_lam[k] * J_list[k] + w_t
            if w_vec is not None:
                S_a = self._S_asset
                S_a[k] = corr_lam[k] * S_a[k] + w_vec
                eta_v = w_vec / S_a[k]
                o = np.sqrt(1.0 - eta_v)
                u = np.sqrt(eta_v) * z_t
                if sq_omega != 1.0:
                    u = u * sq_omega
                np.multiply.outer(o, o, out=self._scratch_had)
                Q_list[k] *= self._scratch_had
                np.multiply.outer(u, u, out=self._scratch_had)
                Q_list[k] += self._scratch_had
        self._vol_t = vol_t
        self._materialize()
        self._last_weight = w_t
        return w_t

    def _intensity(self, corr: np.ndarray, S_t: float, J_t: float) -> float:
        """Self-tuning shrinkage intensity of one timescale from its raw
        correlation and information masses: alpha from the online
        concentration c = n/nu (nu = S^2/J) and the de-noised
        equicorrelation-explained fraction g = rho^2 / (mo - vhat),
        alpha = min(1,c) g^2 / (g^2 + (1-g)^2 max(0, 1/c - 1))."""
        eps = self.epsilon
        n = self.n_assets
        S_t = max(S_t, eps)
        off = corr[self._offmask]
        nu = S_t * S_t / max(J_t, eps)
        c_k = n / max(nu, eps)
        vhat = float(((1.0 - off * off) ** 2).mean()) / max(nu, eps)
        mo_r = float((off * off).mean())
        rho_r = float(off.mean())
        gt = min(1.0, rho_r * rho_r / max(mo_r - vhat, 1e-6))
        r_ = (1.0 - gt) / max(gt, 1e-6)
        s_ = max(0.0, 1.0 / max(c_k, eps) - 1.0)
        return min(1.0, c_k) / (1.0 + r_ * r_ * s_)

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
        return (corr, E, corr * corr), np.array([1.0 - a, a * (1.0 - a), a * a])

    def _materialize(self) -> None:
        """Pool each timescale's experts, blend the timescales, apply the
        volatilities; cache the forecast for the next day's gradient."""
        detector = self._detector
        vol_t = self._vol_t
        assert vol_t is not None
        eps = self.epsilon
        vv = np.multiply.outer(vol_t, vol_t)
        K = self._corr_lam.size
        v = detector.v
        comps: list[tuple[np.ndarray, ...]] = []
        prior = np.empty(v.shape)
        sig_k = []
        for k in range(K):
            experts, prior[k] = self._rung_experts(self._Q_list[k], self._S_list[k], self._J_list[k])
            corr_k = v[k, 0] * experts[0]
            for j in range(1, len(experts)):
                corr_k = corr_k + v[k, j] * experts[j]
            comps.append(experts)
            sig_k.append(corr_k)          # pooled correlation; vv applied once below
        w = np.asarray(detector.w)
        cb = w[0] * sig_k[0]
        for k in range(1, K):
            cb = cb + w[k] * sig_k[k]
        self._C_prev = cb                 # next day's neighbourhood weighting
        cov = cb * vv
        cov = (cov + cov.T) * 0.5
        detector.record(sig_k, comps, prior, cov, vol_t)
        self._cov = cov
        d = np.sqrt(np.maximum(np.diagonal(cov), eps))
        self._corr = cov / np.outer(d, d)
        np.fill_diagonal(self._corr, 1.0)


class SqueezeKernel:
    """Streaming covariance estimator whose public surface is one number.

    Parameters
    ----------
    lam : float
        The exponential decay of the anchor correlation timescale per
        trading day, in the classical EWMA convention (default 0.996, a
        half-life of about 173 days).  The timescale ladder, the variance
        ladder, the kernel-scale clock and the timescale prior all derive
        from it.

    The number of assets is inferred from the first ``update`` call.
    Returns are expected in decimal units (``0.01`` = 1%).

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.default_rng(42).normal(0.0, 0.01, size=(250, 3))
    >>> sk = SqueezeKernel()
    >>> for r_t in returns:
    ...     _ = sk.update(r_t)
    >>> sk.covariance().shape
    (3, 3)
    """

    def __init__(self, lam: float = 0.996) -> None:
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1).")
        self.lam = float(lam)
        self._est: _Engine | None = None
        self._t = 0

    @property
    def half_life(self) -> float:
        """Anchor half-life in trading days implied by ``lam``."""
        return float(-1.0 / np.log2(self.lam))

    def update(self, r_t: ArrayLike, mask: ArrayLike | None = None) -> float:
        """Process one return vector; returns the day's kernel weight.

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
            self._est = _Engine(r.size, self.lam)
        elif r.size != self._est.n_assets:
            raise ValueError(f"Expected {self._est.n_assets} assets, got {r.size}.")
        self._t += 1
        return self._est.update(r)

    def covariance(self) -> np.ndarray:
        """Current covariance forecast (n x n)."""
        if self._est is None or self._est._cov is None:
            raise RuntimeError("Call update() at least once first.")
        return self._est._cov.copy()

    def correlation(self) -> np.ndarray:
        """Current correlation forecast (n x n)."""
        if self._est is None or self._est._corr is None:
            raise RuntimeError("Call update() at least once first.")
        return self._est._corr.copy()

    def state(self) -> dict[str, object]:
        """Diagnostics: update count, last kernel weight, kernel scale,
        per-timescale information masses and effective sizes, blend tilt,
        pool splits and the last observation weight."""
        if self._est is None:
            return {"n_assets": None, "t": 0}
        est = self._est
        det = est._detector
        S = list(est._S_list)
        J = list(est._J_list)
        nu = [s * s / max(j, est.epsilon) for s, j in zip(S, J)]
        return {
            "n_assets": est.n_assets,
            "t": self._t,
            "last_weight": est._last_weight,
            "kappa_t": est._last_kappa,
            "rung_S": S,
            "rung_nu": nu,
            "detector_tilt": det.tilt,
            "split": det.v.copy(),
            "omega_t": est._last_omega,
        }
