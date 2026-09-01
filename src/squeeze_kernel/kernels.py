"""Kernel weight function of the Squeeze Kernel estimator."""

from __future__ import annotations


def kernel_fisher(d2: float, /, *, kappa: float) -> float:
    """Saturating information kernel: w = d² / (d² + κ)."""
    if kappa <= 0.0:
        raise ValueError("kappa must be > 0.")
    return d2 / (d2 + kappa)
