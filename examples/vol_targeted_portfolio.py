"""Vol-targeted long-only portfolio on 300 US equities — Squeeze Kernel vs Ledoit-Wolf.

A deliberately simple passive allocation rule, identical in every respect except
the covariance matrix that drives it:

  * long-only minimum-variance weights over 300 liquid US stocks
    (unconstrained GMV, negative weights clipped and renormalized —
    Jagannathan & Ma 2003), the classic passive "min-vol" allocation,
  * scaled to a 15% annualized ex-ante volatility target with gross leverage
    capped at 1 — the book deleverages to cash when forecast vol exceeds the
    target but never levers up (each arm uses its OWN covariance for both the
    weights and the scaling),
  * rebalanced monthly at the close, 5 bps proportional cost on turnover,
  * positions drift with prices between rebalances (no intra-month trading).

Covariance arms:
  * Squeeze Kernel   — streaming, one O(n^2) update per day, run with its
                       single public number at the library default,
                       SqueezeKernel(lam=0.996).  Nothing is tuned on this
                       panel.
  * Ledoit-Wolf      — scikit-learn's own implementation, refit on a trailing
                       252-day window at every rebalance (the classical
                       "well-conditioned estimator" baseline).
Reporting window: stats and charts cover REPORT_START (2010) onward; the
strategy trades — and the streaming estimator warms — from 2001, so no
estimator sees the reported window cold.

An equal-risk-contribution (ERC) weight rule is included below for comparison.
On a homogeneous large-cap panel ERC is nearly inverse-volatility weighting,
so it consumes almost no correlation information and all estimators score
within noise of each other — minimum variance is where covariance quality
becomes visible.  Swap WEIGHT_RULE to "erc" to reproduce that finding.

Data: examples/data/equity_panel_returns.parquet — 6,686 days x 300 symbols of
dividend-adjusted daily returns (2000-2026), materialised from Yahoo EOD data by
examples/data/build_equity_panel.py (provenance + caveats documented there).
The universe is selected with hindsight (survivors), so ABSOLUTE numbers are
optimistic; all arms run on the identical panel, so the COMPARISON is fair.
Sharpe uses rf=0; the uninvested sleeve earns nothing.

Run:  pip install squeeze-kernel pandas pyarrow scikit-learn matplotlib
      python examples/vol_targeted_portfolio.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from squeeze_kernel import SqueezeKernel

# ── Configuration ────────────────────────────────────────────────────────────
VOL_TARGET = 0.15          # annualized ex-ante volatility target
LEVERAGE_CAP = 1.0         # long-only: deleverage to cash, never lever up
TC_BPS = 5.0               # proportional cost per unit of L1 turnover
LW_LOOKBACK = 252          # trailing window for Ledoit-Wolf refits
WARMUP = 252               # trading days before the first rebalance
REPORT_START = "2010-01-01"    # stats/charts report from here; the strategy
                               # trades (and the estimators warm) from 2001
WEIGHT_RULE = "min_variance"   # "min_variance" or "erc"
ANN = 252.0

DATA = Path(__file__).resolve().parent / "data" / "equity_panel_returns.parquet"
FIGDIR = Path(__file__).resolve().parent / "figures"

ARMS = {  # name -> (color_light, color_dark)
    "Squeeze Kernel": ("#2a78d6", "#3987e5"),
    "Ledoit-Wolf": ("#eb6834", "#d95926"),
}


# ── Weight rules ─────────────────────────────────────────────────────────────
def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum variance: clip negative GMV weights, renormalize."""
    ones = np.ones(cov.shape[0])
    try:
        w = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        w = ones
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 1e-12 else ones / len(ones)


def erc_weights(cov: np.ndarray, max_iter: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """Long-only equal-risk-contribution weights (fixed-point iteration)."""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        rc = w * (cov @ w)                       # per-asset risk contribution
        target = rc.sum() / n
        w_new = w * (target / np.maximum(rc, 1e-12))
        w_new = np.clip(w_new, 1e-12, None)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


WEIGHT_FN = min_variance_weights if WEIGHT_RULE == "min_variance" else erc_weights


# ── Backtest engine ──────────────────────────────────────────────────────────
def run_arm(returns: np.ndarray, rebal_idx: np.ndarray,
            covs: dict[int, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Trade the vol-targeted rule off one covariance stream.

    Weights are decided at the close of each rebalance day from a covariance
    that has seen data only up to that close, and applied from the next day.
    Positions drift with prices between rebalances.
    """
    n_days = returns.shape[0]
    first = int(rebal_idx[0])
    rebal_set = set(int(i) for i in rebal_idx)

    port_val = 1.0
    positions = np.zeros(returns.shape[1])       # dollar positions
    cash = port_val
    daily = np.full(n_days, np.nan)
    lev_hist, turn_hist = [], []

    for d in range(first, n_days):
        if d > first:                            # accrue day d
            prev = port_val
            positions *= 1.0 + returns[d]
            port_val = cash + positions.sum()
            daily[d] = port_val / prev - 1.0
        if d in rebal_set and d < n_days - 1:    # trade at the close of day d
            cov = covs[d]
            w = WEIGHT_FN(cov)
            ex_ante = np.sqrt(max(w @ cov @ w, 1e-18) * ANN)
            lev = min(VOL_TARGET / ex_ante, LEVERAGE_CAP)
            w_tgt = lev * w
            w_cur = positions / port_val
            turnover = float(np.abs(w_tgt - w_cur).sum())
            port_val -= (TC_BPS / 1e4) * turnover * port_val
            positions = w_tgt * port_val
            cash = port_val - positions.sum()
            lev_hist.append(lev)
            turn_hist.append(turnover)

    r = daily[first + 1:]
    meta = {"leverage": np.asarray(lev_hist),
            "turnover": np.asarray(turn_hist),
            "rebal_days": rebal_idx[:len(turn_hist)],
            "start": first + 1}
    return r, meta


def perf_stats(r: np.ndarray, turnover: np.ndarray,
               leverage: np.ndarray) -> dict:
    wealth = np.cumprod(1.0 + r)
    years = r.size / ANN
    cagr = wealth[-1] ** (1.0 / years) - 1.0
    vol = r.std(ddof=1) * np.sqrt(ANN)
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    roll = pd.Series(r).rolling(63).std() * np.sqrt(ANN)
    track = float(np.sqrt(np.nanmean((roll - VOL_TARGET) ** 2)))
    return {"CAGR": cagr, "Vol": vol, "Sharpe": cagr / vol,
            "MaxDD": float(dd.min()), "Calmar": cagr / abs(float(dd.min())),
            "VolTrackRMSE": track, "Turnover/mo": float(turnover.mean()),
            "AvgLev": float(leverage.mean())}


# ── Covariance streams ───────────────────────────────────────────────────────
def squeeze_covs(returns: np.ndarray, rebal_idx: np.ndarray) -> dict[int, np.ndarray]:
    sk = SqueezeKernel()                      # lam=0.996, nothing else
    need = set(int(i) for i in rebal_idx)
    out: dict[int, np.ndarray] = {}
    for d in range(returns.shape[0]):
        sk.update(returns[d])
        if d in need:
            out[d] = sk.covariance()
    return out


def ledoit_wolf_covs(returns: np.ndarray,
                     rebal_idx: np.ndarray) -> dict[int, np.ndarray]:
    from sklearn.covariance import LedoitWolf
    out: dict[int, np.ndarray] = {}
    for d in rebal_idx:
        window = returns[int(d) - LW_LOOKBACK + 1: int(d) + 1]
        out[int(d)] = LedoitWolf().fit(window).covariance_
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    panel = pd.read_parquet(DATA).astype(np.float64)
    returns = panel.to_numpy()
    dates = panel.index

    # last trading day of each month, after warm-up, never the final day
    month = pd.PeriodIndex(dates, freq="M")
    is_month_end = np.r_[month[:-1] != month[1:], True]
    rebal_idx = np.flatnonzero(is_month_end)
    rebal_idx = rebal_idx[(rebal_idx >= WARMUP) & (rebal_idx < len(dates) - 1)]
    print(f"panel: {returns.shape[0]} days x {returns.shape[1]} assets, "
          f"{rebal_idx.size} monthly rebalances "
          f"({dates[rebal_idx[0]].date()} -> {dates[rebal_idx[-1]].date()})")

    t0 = time.perf_counter()
    covs_sq = squeeze_covs(returns, rebal_idx)
    t_sq = time.perf_counter() - t0
    t0 = time.perf_counter()
    covs_lw = ledoit_wolf_covs(returns, rebal_idx)
    t_lw = time.perf_counter() - t0
    print(f"covariance passes: squeeze {t_sq:.1f}s (full 26y stream), "
          f"ledoit-wolf {t_lw:.1f}s ({rebal_idx.size} window refits)")

    # trade from 2001, report from REPORT_START (estimators fully warmed)
    all_dates = dates[int(rebal_idx[0]) + 1:]
    cut = int(all_dates.searchsorted(pd.Timestamp(REPORT_START, tz=all_dates.tz)))
    eval_dates = all_dates[cut:]

    results, stats = {}, {}
    for name, covs in [("Squeeze Kernel", covs_sq),
                       ("Ledoit-Wolf", covs_lw)]:
        r, meta = run_arm(returns, rebal_idx, covs)
        r = r[cut:]
        post = meta["rebal_days"] >= int(rebal_idx[0]) + 1 + cut
        results[name] = r
        stats[name] = perf_stats(r, meta["turnover"][post],
                                 meta["leverage"][post])

    print(f"\nreporting window: {eval_dates[0].date()} -> "
          f"{eval_dates[-1].date()} ({len(eval_dates)} days)")
    print("\n| Method | CAGR | Vol | Sharpe | MaxDD | Calmar | "
          "Vol-target RMSE | Turnover/mo |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        print(f"| {name} | {s['CAGR']:.1%} | {s['Vol']:.1%} | "
              f"{s['Sharpe']:.2f} | {s['MaxDD']:.1%} | {s['Calmar']:.2f} | "
              f"{s['VolTrackRMSE']:.2%} | {s['Turnover/mo']:.2f} |")

    make_figures(eval_dates, results, stats)


# ── Figures ──────────────────────────────────────────────────────────────────
def make_figures(dates: pd.DatetimeIndex, results: dict, stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    FIGDIR.mkdir(exist_ok=True)
    for mode, suffix in [("light", ""), ("dark", "_dark")]:
        dark = mode == "dark"
        ink = "#ffffff" if dark else "#0b0b0b"
        ink2 = "#c3c2b7" if dark else "#52514e"
        muted = "#898781"
        grid = "#2c2c2a" if dark else "#e1e0d9"
        colors = {n: c[1] if dark else c[0] for n, c in ARMS.items()}

        bg = "#0d1117" if dark else "#ffffff"
        fig = plt.figure(figsize=(12.8, 8.4), dpi=200)
        fig.patch.set_facecolor(bg)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0],
                              hspace=0.34, wspace=0.28,
                              left=0.055, right=0.985, top=0.845, bottom=0.075)
        ax_eq = fig.add_subplot(gs[0, :])
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_vol = fig.add_subplot(gs[1, 1])
        ax_rr = fig.add_subplot(gs[1, 2])
        axes = [ax_eq, ax_dd, ax_vol, ax_rr]

        for ax in axes:
            ax.set_facecolor("none")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.grid(True, color=grid, linewidth=0.7)
            ax.tick_params(colors=muted, labelsize=8.5, length=0)

        order = ["Ledoit-Wolf", "Squeeze Kernel"]

        # Equity curves (log scale)
        finals = {}
        for name in order:
            r = results[name]
            wealth = np.cumprod(1.0 + r)
            finals[name] = wealth[-1]
            lw_ = 2.2 if name == "Squeeze Kernel" else 1.6
            ax_eq.plot(dates, wealth, color=colors[name], lw=lw_, label=name,
                       zorder=3 if name == "Squeeze Kernel" else 2)
        # end labels: spread vertically so they never collide, keyed by a dot
        slots = np.linspace(0.93, 0.79, len(order))
        for name, frac in zip(sorted(order, key=lambda n: -finals[n]), slots):
            ax_eq.scatter([0.962], [frac + 0.012], transform=ax_eq.transAxes,
                          s=28, color=colors[name], edgecolors="none",
                          clip_on=False, zorder=5)
            ax_eq.annotate(f"×{finals[name]:.1f}",
                           xy=(0.972, frac), xycoords="axes fraction",
                           ha="left", color=ink, fontsize=10, fontweight="bold")
        ax_eq.set_yscale("log")
        ax_eq.minorticks_off()
        top = max(finals.values()) * 1.25
        ticks = [t for t in (1, 2, 4, 8, 16, 32) if t <= top]
        ax_eq.set_ylim(0.9, top)
        ax_eq.set_yticks(ticks)
        ax_eq.set_yticklabels([str(t) for t in ticks])
        ax_eq.set_title("Growth of $1  (log scale)", loc="left", fontsize=10,
                        color=ink2, pad=8)
        handles, labels = ax_eq.get_legend_handles_labels()
        leg = ax_eq.legend(handles[::-1], labels[::-1], loc="upper left",
                           frameon=False, fontsize=9.5,
                           handlelength=1.6, borderaxespad=0.2)
        for t in leg.get_texts():
            t.set_color(ink)

        # Drawdown (wash only on the featured arm; lines carry the rest)
        for name in order:
            r = results[name]
            wealth = np.cumprod(1.0 + r)
            dd = wealth / np.maximum.accumulate(wealth) - 1.0
            ax_dd.plot(dates, dd, color=colors[name], lw=1.4,
                       zorder=3 if name == "Squeeze Kernel" else 2)
            if name == "Squeeze Kernel":
                ax_dd.fill_between(dates, dd, 0.0, color=colors[name],
                                   alpha=0.10, linewidth=0)
        ax_dd.set_title("Drawdown", loc="left", fontsize=10, color=ink2, pad=8)
        ax_dd.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

        # Rolling realized vol vs the target
        for name in order:
            r = results[name]
            roll = pd.Series(r, index=dates).rolling(63).std() * np.sqrt(ANN)
            ax_vol.plot(dates, roll, color=colors[name], lw=1.4,
                        zorder=3 if name == "Squeeze Kernel" else 2)
        ax_vol.axhline(VOL_TARGET, color=muted, lw=1.0, ls=(0, (4, 3)))
        ax_vol.annotate("15% target", (0.52, VOL_TARGET),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", color=muted, fontsize=8.5)
        ax_vol.set_title("Realized volatility (63-day, annualized)", loc="left",
                         fontsize=10, color=ink2, pad=8)
        ax_vol.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax_vol.set_ylim(0, None)
        ax_vol.margins(y=0.08)

        # Risk/return profile: one dot per calendar year + full-sample marker
        rs = {n: pd.Series(results[n], index=dates) for n in order}
        for name in order:
            by_year = rs[name].groupby(rs[name].index.year)
            yv = by_year.std() * np.sqrt(ANN)
            yr = by_year.apply(lambda x: (1 + x).prod() ** (ANN / len(x)) - 1)
            full_ok = by_year.count() > 60
            ax_rr.scatter(yv[full_ok], yr[full_ok], s=26, color=colors[name],
                          alpha=0.75, edgecolors="none", zorder=3)
            s = stats[name]
            ax_rr.scatter([s["Vol"]], [s["CAGR"]], s=150, marker="D",
                          color=colors[name], edgecolors=ink, linewidths=1.2,
                          zorder=4)
        ax_rr.axvline(VOL_TARGET, color=muted, lw=1.0, ls=(0, (4, 3)))
        ax_rr.axhline(0.0, color=grid, lw=0.7)
        ax_rr.set_title("Risk / return by year  (diamond = full sample)",
                        loc="left", fontsize=10, color=ink2, pad=8)
        ax_rr.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax_rr.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

        for ax in (ax_eq, ax_dd, ax_vol):
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.margins(x=0.01)

        s_sq, s_lw = stats["Squeeze Kernel"], stats["Ledoit-Wolf"]
        fig.text(0.055, 0.960,
                 "Same portfolio rule, different covariance matrix",
                 fontsize=15, fontweight="bold", color=ink)
        fig.text(0.055, 0.925,
                 f"300 US equities  ·  long-only minimum variance  ·  "
                 f"{VOL_TARGET:.0%} vol target  ·  monthly rebalance  ·  "
                 f"{TC_BPS:.0f} bps costs  ·  2010–2026, warmed up on 2000–2009",
                 fontsize=9.5, color=ink2)
        fig.text(0.055, 0.895,
                 f"Sharpe {s_sq['Sharpe']:.2f} vs {s_lw['Sharpe']:.2f}  ·  "
                 f"max drawdown {s_sq['MaxDD']:.0%} vs {s_lw['MaxDD']:.0%}  ·  "
                 f"realized vol {s_sq['Vol']:.1%} vs {s_lw['Vol']:.1%}",
                 fontsize=9.5, fontweight="bold", color=ink)

        out = FIGDIR / f"vol_targeted_portfolio{suffix}.png"
        fig.savefig(out, facecolor=bg)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
