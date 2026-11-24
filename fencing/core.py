"""Range--velocity localization, target observation, and matched fencing feedback."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EDGE_SIZE = 18


class DomainError(ValueError):
    """The measured edge is outside the open barrier domain."""


@dataclass(frozen=True)
class Config:
    kp: float = 0.12
    kv: float = 1.5
    ks: float = 2.0
    communication_weight: float = 1.0
    pinning: float = 2.0
    filter_lambda: float = 1.0
    observer_gamma: float = 0.2
    history_gain: float = 2000.0
    adaptation_gain: float = 0.3
    barrier_A: float = 0.5
    barrier_B: float = 0.1
    safe_distance: float = 0.45
    outer_radius: float = 12.0
    dt: float = 0.005
    duration: float = 240.0
    output_period: float = 0.1
    Lj: float = 0.02
    mu0: float = 3.0
    mu1: float = 1.5
    mu2: float = 1.1
    acceleration_compensation: bool = True
    seed: int = 20260916
    beta: float = 0.05  # Retained for compatibility with the original parameter set.
    history_enabled: bool = True  # False is an explicitly labelled ablation.
    history_mode: str = (
        "retained"  # "instantaneous" keeps ell but uses only the current regressor.
    )

    def __post_init__(self):
        positive = (
            "kp",
            "kv",
            "ks",
            "communication_weight",
            "pinning",
            "filter_lambda",
            "observer_gamma",
            "history_gain",
            "adaptation_gain",
            "barrier_A",
            "barrier_B",
            "safe_distance",
            "dt",
            "duration",
            "output_period",
            "Lj",
            "mu0",
            "mu1",
            "mu2",
            "beta",
        )
        if any(not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0 for k in positive):
            raise ValueError("all gains, distances and time scales must be finite and positive")
        if not all(np.isfinite(x) for x in (self.outer_radius,)):
            raise ValueError("radii and super-twisting gains must be finite")
        if not self.safe_distance < self.outer_radius:
            raise ValueError("require 0 < safe_distance < outer_radius")
        if self.mu2 <= 1:
            raise ValueError("final differentiator coefficient must exceed the jerk bound")
        if self.history_mode not in ("retained", "instantaneous"):
            raise ValueError("history_mode must be retained or instantaneous")

    @property
    def effective_history_gain(self):
        return self.history_gain if self.history_enabled else 0.0


@dataclass(frozen=True)
class SensorFrame:
    """No positions, bearings, target velocity, or true edge parameters."""

    velocities: np.ndarray
    ranges: np.ndarray
    source_relative: np.ndarray


def initial_edges(m):
    x = np.zeros((m, EDGE_SIZE))
    x[:, 6:10] = np.eye(2).ravel()
    return x


def split_edges(x):
    return (
        x[:, :2],
        x[:, 2],
        x[:, 3],
        x[:, 4:6],
        x[:, 6:10].reshape(-1, 2, 2),
        x[:, 10:14].reshape(-1, 2, 2),
        x[:, 14:16],
        x[:, 16:18],
    )


def barrier(distance, cfg):
    d2 = np.asarray(distance) ** 2
    low, high = d2 - cfg.safe_distance**2, cfg.outer_radius**2 - d2
    if np.any(low <= 0) or np.any(high <= 0) or not np.all(np.isfinite(d2)):
        raise DomainError("active range crossed a barrier: no clipping is permitted")
    W = cfg.barrier_A / low + cfg.barrier_B / high
    omega = -2 * cfg.barrier_A / low**2 + 2 * cfg.barrier_B / high**2
    return W, omega


def incidence(n, edges):
    D = np.zeros((n, len(edges)))
    for col, (i, j) in enumerate(edges):
        D[i, col], D[j, col] = 1.0, -1.0
    return D


def controller_rhs(t, cfg, frame, shat, edge_state, initial_ranges, edges, source, vbar, abar):
    """Range-based observer and acceleration feedback.

    Each undirected edge has one shared oriented state. Gamma is adaptation_gain
    times I_2. G and h always integrate for diagnostics. In instantaneous mode,
    their values do not enter the parameter correction, whose gain ell is kept.
    """
    D = incidence(len(shat), edges)
    phi, rho, z, xi, Phi, G, h, theta = split_edges(edge_state)
    nu = D.T @ frame.velocities
    _, omega = barrier(frame.ranges, cfg)
    qhat = xi + np.einsum("mij,mj->mi", Phi, theta)
    Y = (
        0.5
        * (
            frame.ranges**2
            - np.exp(-cfg.filter_lambda * t) * initial_ranges**2
            - cfg.filter_lambda * rho
        )
        + z
    )
    psi = np.einsum("mji,mj->mi", Phi, phi)
    Z = Y - np.einsum("mi,mi->m", phi, xi)
    edot = np.empty_like(edge_state)
    edot[:, :2] = -cfg.filter_lambda * phi + nu
    edot[:, 2] = -cfg.filter_lambda * rho + frame.ranges**2
    edot[:, 3] = -cfg.filter_lambda * z + np.einsum("mi,mi->m", nu, phi)
    edot[:, 4:6] = nu - cfg.observer_gamma * phi * (np.einsum("mi,mi->m", phi, xi) - Y)[:, None]
    edot[:, 6:10] = (-cfg.observer_gamma * np.einsum("mi,mj,mjk->mik", phi, phi, Phi)).reshape(
        -1, 4
    )
    edot[:, 10:14] = np.einsum("mi,mj->mij", psi, psi).reshape(-1, 4)
    edot[:, 14:16] = psi * Z[:, None]
    if cfg.history_mode == "instantaneous":
        correction = (
            cfg.effective_history_gain * psi * (Z - np.einsum("mi,mi->m", psi, theta))[:, None]
        )
    else:
        correction = cfg.effective_history_gain * (h - np.einsum("mij,mj->mi", G, theta))
    edot[:, 16:18] = cfg.adaptation_gain * (
        correction + omega[:, None] * np.einsum("mji,mj->mi", Phi, nu)
    )
    # Ideal, delay-free network broadcast from the designated source.
    eta = cfg.communication_weight * D @ (D.T @ shat + qhat)
    eta[source] += cfg.pinning * (shat[source] - frame.source_relative)
    shdot = vbar - frame.velocities - cfg.ks * eta
    u = cfg.kv * (vbar - frame.velocities) + cfg.kp * shat - D @ (omega[:, None] * qhat)
    if cfg.acceleration_compensation:
        u = u + abar
    return u, shdot, edot, qhat, vbar
