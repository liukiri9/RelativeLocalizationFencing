"""Global-position fencing of Jiang et al., IEEE TASE 22 (2025).

Source: DOI 10.1109/TASE.2025.3596929, Eqs. (14), (17)--(23),
and (36)--(37), pp. 20102--20104.  This module implements the published
adaptive target-position/acceleration observers and dynamic-compensation
controller, not the proposed controller supplied with exact geometry.

The only external target signals are a source's measured target position
and its differentiated acceleration.  Neither target velocity nor target
acceleration truth is accepted.  Positions passed to ``rhs`` are global
measurements, whereas the proposed method uses range/velocity measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .target_motion import implicit_joint_step


@dataclass(frozen=True)
class BaselineConfig:
    """Published gains and explicit gains adapted to the common length scale.

    The default c1/c2/c3 factor as (s+1.5)(s²+1.5s+0.12), matching the
    proposed controller's centroid poles with one additional stable pole.
    ``matched_config`` also matches the repulsion/attraction ratio at a
    declared reference distance.  These are design choices, not the
    original paper's numerical parameter values.
    """

    safe_distance: float = 0.45
    outer_radius: float = 12.0
    c1: float = 3.0
    c2: float = 2.37
    c3: float = 0.18
    k_phi: float = 3.364
    c_p: float = 5.0
    gamma_p: float = 4.2
    c_a: float = 0.5
    gamma_a: float = 2.0
    sigma_initial: float = 1.0
    sigma_decay: float = 0.02
    pinning: float = 1.0
    jerk_bound: float = 15.0
    mu0: float = 2.0
    mu1: float = 1.5
    mu2: float = 1.1

    def __post_init__(self):
        if not all(np.isfinite(value) and value > 0 for value in self.__dict__.values()):
            raise ValueError("baseline parameters must be finite and positive")
        if self.outer_radius <= self.safe_distance:
            raise ValueError("outer radius must exceed the safety distance")
        if self.c1 * self.c2 <= self.c3:
            raise ValueError("published controller requires c1*c2 > c3")


def matched_config(
    kp=0.12,
    kv=1.5,
    safe_distance=0.45,
    outer_radius=12.0,
    separation_weight=0.5,
    cohesion_weight=0.1,
    reference_distance=2.5,
    extra_pole=None,
    **observer_parameters,
):
    """Choose gains before running a comparison, without fitting outcomes.

    Besides matching centroid poles, match repulsive-force/centering-gain
    ratios at ``reference_distance``.  The published inverse-distance
    force and dynamic compensation are retained without modification.
    """
    if not (kp > 0 and kv > 0 and safe_distance < reference_distance < outer_radius):
        raise ValueError("positive gains and an interior reference distance are required")
    pole = kv if extra_pole is None else extra_pole
    if pole <= 0:
        raise ValueError("additional pole must be positive")
    c1, c2, c3 = kv + pole, kp + kv * pole, kp * pole
    d = reference_distance
    proposed_repulsion = (
        2 * separation_weight * d / (d * d - safe_distance**2) ** 2
        - 2 * cohesion_weight * d / (outer_radius**2 - d * d) ** 2
    )
    published_unit_force = 1 / (d - safe_distance) - 1 / (outer_radius - safe_distance)
    k_phi = c2 / kp * proposed_repulsion / published_unit_force
    return BaselineConfig(
        safe_distance=safe_distance,
        outer_radius=outer_radius,
        c1=c1,
        c2=c2,
        c3=c3,
        k_phi=k_phi,
        **observer_parameters,
    )


def initial_state(measured_initial_positions):
    """Eqs. (23), (37): phat_i=p_i; ahat_i=theta_i=rho_i=kappa_i=0.

    A flattened n-by-8 state uses columns
    [phat_x, phat_y, ahat_x, ahat_y, theta, rho, kappa_x, kappa_y].
    """
    positions = np.asarray(measured_initial_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("measured initial positions must have shape (n,2)")
    result = np.zeros((len(positions), 8))
    result[:, :2] = positions
    return result.ravel()


def unpack(state):
    """Return observer and dynamic-compensation state views."""
    values = np.asarray(state, dtype=float).reshape(-1, 8)
    return values[:, :2], values[:, 2:4], values[:, 4], values[:, 5], values[:, 6:8]


def interaction(measured_positions, cfg):
    """Eq. (36): symmetric range-limited inverse-distance repulsion.

    ``adjacency`` has unit weights for measured distances strictly below r.
    A measured distance at or below d raises rather than silently clipping
    the singular published force.  Such an event must be logged by a runner.
    """
    positions = np.asarray(measured_positions, dtype=float)
    n = len(positions)
    relative = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(relative, axis=2)
    off_diagonal = ~np.eye(n, dtype=bool)
    if np.any(distance[off_diagonal] <= cfg.safe_distance):
        raise ValueError("published baseline measured separation reached d")
    active = off_diagonal & (distance < cfg.outer_radius)
    force_coefficient = np.zeros_like(distance)
    d = distance[active]
    force_coefficient[active] = (
        cfg.k_phi * (1 / (d - cfg.safe_distance) - 1 / (cfg.outer_radius - cfg.safe_distance)) / d
    )
    phi = np.einsum("ij,ijk->ik", force_coefficient, relative)
    return phi, active.astype(float)


def observer_rhs(
    t, state, adjacency, measured_target_position, source_acceleration_estimate, cfg, source=0
):
    """Published Eqs. (17), (18), (21), and (22), with one source.

    Only the source row contains a target-position or source-acceleration
    pin.  The vanishing positive sigma is the paper's exponential function.
    No projection, gain clipping, leakage, or exact target derivatives enter.
    """
    phat, ahat, theta, rho, _ = unpack(state)
    omega = np.sum(adjacency, axis=1)[:, None] * phat - adjacency @ phat
    varsigma = np.sum(adjacency, axis=1)[:, None] * ahat - adjacency @ ahat
    omega[source] += cfg.pinning * (phat[source] - measured_target_position)
    varsigma[source] += cfg.pinning * (ahat[source] - source_acceleration_estimate)
    sigma = cfg.sigma_initial * np.exp(-cfg.sigma_decay * t)
    denominator_p = np.hypot(np.linalg.norm(omega, axis=1), sigma)
    denominator_a = np.hypot(np.linalg.norm(varsigma, axis=1), sigma)
    phat_dot = -cfg.c_p * omega - theta[:, None] * omega / denominator_p[:, None]
    ahat_dot = -cfg.c_a * varsigma - rho[:, None] * varsigma / denominator_a[:, None]
    theta_dot = cfg.gamma_p * np.sum(omega * omega, axis=1) / denominator_p
    rho_dot = cfg.gamma_a * np.sum(varsigma * varsigma, axis=1) / denominator_a
    return phat_dot, ahat_dot, theta_dot, rho_dot


def rhs(
    t,
    state,
    measured_global_positions,
    measured_target_position,
    source_acceleration_estimate,
    cfg,
    source=0,
):
    """Return ``(acceleration_command, flat_state_derivative)``.

    The caller integrates its common double-integrator plant.  It also
    advances Eq. (14) via ``source_step`` from measured target position.
    There is deliberately no robot-velocity or true target-motion argument.
    """
    positions = np.asarray(measured_global_positions, dtype=float)
    phat, ahat, _, _, kappa = unpack(state)
    phi, adjacency = interaction(positions, cfg)
    dp, da, dtheta, drho = observer_rhs(
        t,
        state,
        adjacency,
        np.asarray(measured_target_position),
        np.asarray(source_acceleration_estimate),
        cfg,
        source,
    )
    position_correction = cfg.c2 * (phat - positions)
    command = position_correction + kappa + ahat + phi
    kappa_dot = -cfg.c1 * kappa - (cfg.c1 * cfg.c2 - cfg.c3) / cfg.c2 * (
        phi + position_correction
    )
    derivative = np.column_stack((dp, da, dtheta, drho, kappa_dot))
    return command, derivative.ravel()


def source_step(z0, z1, z2, measured_target_position_next, h, cfg):
    """Eq. (14), using the same implicit inclusion discretization as ours."""
    return implicit_joint_step(
        z0, z1, z2, measured_target_position_next, h, cfg.jerk_bound, cfg.mu0, cfg.mu1, cfg.mu2
    )


def required_substeps(t, state, cfg, dt, stability_limit=1.0):
    """Resolve the published observers' decreasing-sigma stiffness with RK4.

    ``state`` is the flat 8*n baseline state, without the physical plant.
    For a unit-weight undirected subgraph of a complete n-node graph, the
    grounded Laplacian's largest eigenvalue is at most n+pinning.  The
    Jacobian of x/sqrt(||x||²+sigma²) has norm at most 1/sigma.  This
    conservative bound sets a microstep count for the *unchanged* RHS.
    It does not clip estimates or alter the published observer gains.

    The source differentiator is already implicit: keep its original dt
    step and its output held across these RK4 microsteps.  Thus refining
    the observer/plant solver does not change the source differentiation
    filter or the common sensor-noise realization.
    """
    phat, ahat, theta, rho, _ = unpack(state)
    sigma_end = cfg.sigma_initial * np.exp(-cfg.sigma_decay * (t + dt))
    if not np.isfinite(sigma_end) or sigma_end <= 0:
        raise ValueError("sigma underflow prevents a finite stiffness bound")
    grounded_bound = len(phat) + cfg.pinning
    # Add a reserve for gain evolution within the solver step.  The frozen
    # Jacobian bound is conservative; this finite-step margin is a practical
    # step selector and is checked by refinement, not a global error bound.
    spread_p = np.max(np.linalg.norm(phat[:, None, :] - phat[None, :, :], axis=2))
    spread_a = np.max(np.linalg.norm(ahat[:, None, :] - ahat[None, :, :], axis=2))
    theta_bound = np.max(theta) + dt * cfg.gamma_p * grounded_bound * (1 + spread_p)
    rho_bound = np.max(rho) + dt * cfg.gamma_a * grounded_bound * (1 + spread_a)
    stiffness = grounded_bound * max(
        cfg.c_p + theta_bound / sigma_end, cfg.c_a + rho_bound / sigma_end
    )
    return max(1, int(np.ceil(dt * stiffness / stability_limit)))
