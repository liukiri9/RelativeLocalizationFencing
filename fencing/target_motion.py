"""Joint position/velocity/acceleration differentiator from one position signal.

The recursive differentiator follows Jiang et al., TASE 2025, Eq. (14),
with m=2. The source forms its input from target-relative position plus
integrated own measured velocity; no target motion reference enters this module.
"""

from __future__ import annotations
import numpy as np


def joint_gains(jerk_bound, mu0=3.0, mu1=1.5, mu2=1.1):
    """Return equivalent homogeneous gains for the recursive formulation."""
    if not np.isfinite(jerk_bound) or jerk_bound <= 0:
        raise ValueError("jerk bound must be finite and positive")
    if not all(np.isfinite(v) and v > 0 for v in (mu0, mu1, mu2)):
        raise ValueError("differentiator gains must be finite and positive")
    return (
        mu0 * jerk_bound ** (1.0 / 3.0),
        mu1 * np.sqrt(mu0) * jerk_bound ** (2.0 / 3.0),
        mu2 * jerk_bound,
    )


def implicit_joint_step(z0, z1, z2, measurement_next, h, jerk_bound, mu0=3.0, mu1=1.5, mu2=1.1):
    """One fully implicit backward-Euler inclusion step.

    Eliminating the two derivative states gives a monotone scalar equation
    for e=z0_next-measurement_next in each component. Its nonzero branch is
    a cubic in x=abs(e)**(1/3), solved by bounded bisection. Its zero branch
    uses the admissible set-valued sign, rather than clipping a sign surrogate.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("step must be finite and positive")
    arrays = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (z0, z1, z2, measurement_next))
    )
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("differentiator states and input must be finite")
    z0, z1, z2, measured = arrays
    k0, k1, k2 = joint_gains(jerk_bound, mu0, mu1, mu2)
    predictor = z0 + h * z1 + h * h * z2 - measured
    magnitude = np.abs(predictor)
    threshold = h**3 * k2
    sliding = magnitude <= threshold
    sign = np.sign(predictor)
    # The strictly increasing polynomial has one nonnegative root. A cbrt
    # upper bracket works because all remaining coefficients are positive.
    lo = np.zeros_like(predictor)
    hi = np.cbrt(magnitude)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        polynomial = mid**3 + h * k0 * mid**2 + h * h * k1 * mid + threshold
        lower = polynomial < magnitude
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    root = np.where(sliding, 0.0, 0.5 * (lo + hi))
    sigma = np.where(sliding, predictor / threshold, sign)
    z2_next = z2 - h * k2 * sigma
    z1_next = z1 + h * (z2_next - k1 * sign * root)
    z0_next = measured + sign * root**3
    return z0_next, z1_next, z2_next
