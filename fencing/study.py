"""Integrate the six-robot double-integrator plant and the selected controller."""

from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import json
import time
import numpy as np

from . import published_baseline as pb
from .core import SensorFrame, incidence, initial_edges, controller_rhs
from .geometry import hull_distance, signed_hull_margin
from .results import common_performance, json_ready, save_case
from .scenario import StudyConfig, HeldNoise, complex_target, initial_conditions
from .target_motion import implicit_joint_step


class RangeSimulator:
    """Six robots using a source motion estimate and pairwise range and velocity measurements."""

    source = 0
    target = staticmethod(complex_target)

    def __init__(self, cfg=None):
        self.cfg = cfg or StudyConfig()
        c = self.cfg
        self.n = c.robots
        self.pairs = tuple(combinations(range(self.n), 2))
        self.m = len(self.pairs)
        self.D = incidence(self.n, self.pairs)
        self.noise = HeldNoise(c)
        p, v = initial_conditions(c)
        self.truth_theta = self.D.T @ p
        self.ranges0 = np.linalg.norm(self.truth_theta, axis=1) + self.noise.at(0.0)[0]
        shat = np.zeros_like(p)
        shat[self.source] = self.target(0.0)[0] - p[self.source] + self.noise.at(0.0)[2]
        self.y = np.r_[
            p.ravel(), v.ravel(), shat.ravel(), initial_edges(self.m).ravel(), np.zeros(2)
        ]
        self.z0, self.z1, self.z2 = shat[self.source].copy(), np.zeros(2), np.zeros(2)
        self.global_min_pair, self.global_max_edge = np.inf, 0.0
        self.record = []
        self.snapshots = {
            key: []
            for key in (
                "time",
                "position",
                "velocity",
                "shat",
                "edge_state",
                "active",
                "edge_active",
                "z0",
                "z1",
                "z2",
                "chi",
                "vbar",
                "abar",
                "control",
                "target_position",
                "target_velocity",
                "target_acceleration",
            )
        }

    def measured(self, t, state):
        p, v, _, _ = self.unpack(state)
        nr, nv, nt, _ = self.noise.at(t)
        return SensorFrame(
            v + nv,
            np.linalg.norm(self.D.T @ p, axis=1) + nr,
            self.target(t)[0] - p[self.source] + nt,
        )

    def rhs(self, t, state):
        _, velocity, shat, edge = self.unpack(state)
        frame = self.measured(t, state)
        control, shdot, edot, _, _ = controller_rhs(
            t,
            self.cfg,
            frame,
            shat,
            edge,
            self.ranges0,
            self.pairs,
            self.source,
            self.z1,
            self.z2,
        )
        return np.r_[
            velocity.ravel(),
            control.ravel(),
            shdot.ravel(),
            edot.ravel(),
            frame.velocities[self.source],
        ]

    def unpack(self, state):
        b = 2 * self.n
        return (
            state[:b].reshape(self.n, 2),
            state[b : 2 * b].reshape(self.n, 2),
            state[2 * b : 3 * b].reshape(self.n, 2),
            state[3 * b : -2].reshape(self.m, 18),
        )

    def sample_extrema(self):
        p = self.unpack(self.y)[0]
        distance = np.linalg.norm(self.D.T @ p, axis=1)
        self.global_min_pair = min(self.global_min_pair, float(np.min(distance)))
        self.global_max_edge = max(self.global_max_edge, float(np.max(distance)))

    def log(self, t):
        p, v, shat, edge = self.unpack(self.y)
        p0, v0, a0 = self.target(t)
        u, _, _, qhat, vbar = controller_rhs(
            t,
            self.cfg,
            self.measured(t, self.y),
            shat,
            edge,
            self.ranges0,
            self.pairs,
            self.source,
            self.z1,
            self.z2,
        )
        q = self.D.T @ p
        distances = np.linalg.norm(q, axis=1)
        target_offset = p - p0
        velocity_error = v - v0
        information = edge[:, 10:14].reshape(self.m, 2, 2)
        self.record.append(
            {
                "time": t,
                "localization_max": float(np.max(np.linalg.norm(qhat - q, axis=1))),
                "observer_max": float(np.max(np.linalg.norm(shat + p - p0, axis=1))),
                "speed_max": float(np.max(np.linalg.norm(velocity_error, axis=1))),
                "broadcast_error": float(np.linalg.norm(vbar - v0)),
                "acceleration_broadcast_error": float(np.linalg.norm(self.z2 - a0)),
                "centroid_error": float(np.linalg.norm(np.mean(target_offset, axis=0))),
                "hull_distance": hull_distance(p, p0),
                "min_pair_distance": float(np.min(distances)),
                "max_edge_distance": float(np.max(distances)),
                "gramian_min": float(np.min(np.linalg.eigvalsh(information)[:, 0])),
                "control_max": float(np.max(np.linalg.norm(u, axis=1))),
                "signed_hull_margin": signed_hull_margin(p, p0),
                "speed_rms": float(
                    np.sqrt(np.mean(np.sum(velocity_error * velocity_error, axis=1)))
                ),
            }
        )
        values = (
            t,
            p.copy(),
            v.copy(),
            shat.copy(),
            edge.copy(),
            np.ones(self.n, bool),
            np.ones(self.m, bool),
            self.z0.copy(),
            self.z1.copy(),
            self.z2.copy(),
            self.y[-2:].copy(),
            vbar,
            self.z2.copy(),
            u,
            p0,
            v0,
            a0,
        )
        for key, value in zip(self.snapshots, values):
            self.snapshots[key].append(value)

    def run(self):
        start = time.perf_counter()
        c = self.cfg
        steps = int(round(c.duration / c.dt))
        stride = int(round(c.output_period / c.dt))
        self.log(0.0)
        self.sample_extrema()
        for k in range(steps):
            t, h, state = k * c.dt, c.dt, self.y
            k1 = self.rhs(t, state)
            k2 = self.rhs(t + h / 2, state + h * k1 / 2)
            k3 = self.rhs(t + h / 2, state + h * k2 / 2)
            k4 = self.rhs(t + h, state + h * k3)
            self.y = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not np.all(np.isfinite(self.y)):
                raise FloatingPointError(f"nonfinite trajectory at {t+h}")
            now = (k + 1) * h
            target_signal = self.measured(now, self.y).source_relative + self.y[-2:]
            self.z0, self.z1, self.z2 = implicit_joint_step(
                self.z0, self.z1, self.z2, target_signal, h, c.Lj, c.mu0, c.mu1, c.mu2
            )
            self.sample_extrema()
            if (k + 1) % stride == 0 or k + 1 == steps:
                self.log(now)
        arrays = {key: np.asarray(values) for key, values in self.snapshots.items()}
        summary = {
            "config": asdict(c),
            "final": self.record[-1],
            "performance": common_performance(self.record, arrays, c),
            "safety": {
                "minimum_pair_distance": self.global_min_pair,
                "maximum_active_edge_distance": self.global_max_edge,
            },
        }
        summary["runtime_seconds"] = time.perf_counter() - start
        return summary, arrays, self.record


class PublishedSimulator:
    """Common double-integrator plant with the published global-position law."""

    def __init__(self, cfg=None):
        self.cfg = cfg or StudyConfig()
        c = self.cfg
        self.n, self.source = c.robots, 0
        self.pairs = tuple(combinations(range(self.n), 2))
        self.D = incidence(self.n, self.pairs)
        self.noise = HeldNoise(c)
        p, v = initial_conditions(c)
        self.baseline_cfg = pb.matched_config(
            kp=c.kp,
            kv=c.kv,
            safe_distance=c.safe_distance,
            outer_radius=c.outer_radius,
            separation_weight=c.barrier_A,
            cohesion_weight=c.barrier_B,
            jerk_bound=c.Lj,
        )
        measured_p = p + self.noise.at(0.0)[3]
        target = complex_target(0.0)[0] + self.noise.at(0.0)[2]
        state = pb.initial_state(measured_p)
        # Eq. (15) permits other initializations. Both methods use a measured
        # position signal and zero derivative estimates; no target truth is used.
        self.z0, self.z1, self.z2 = target.copy(), np.zeros(2), np.zeros(2)
        self.y = np.r_[p.ravel(), v.ravel(), state]
        self.record, self.snapshots = [], {}
        self.min_pair, self.max_pair = np.inf, 0.0
        self.substep_total, self.substep_max = 0, 1

    def unpack(self, state):
        b = 2 * self.n
        return state[:b].reshape(self.n, 2), state[b : 2 * b].reshape(self.n, 2), state[2 * b :]

    def measured(self, t, state):
        p = self.unpack(state)[0]
        _, _, nt, np_ = self.noise.at(t)
        return p + np_, complex_target(t)[0] + nt

    def algorithm(self, t, state):
        measured_p, measured_t = self.measured(t, state)
        return pb.rhs(
            t, self.unpack(state)[2], measured_p, measured_t, self.z2, self.baseline_cfg
        )

    def rhs(self, t, state):
        _, v, _ = self.unpack(state)
        u, ds = self.algorithm(t, state)
        return np.r_[v.ravel(), u.ravel(), ds]

    def extrema(self):
        d = np.linalg.norm(self.D.T @ self.unpack(self.y)[0], axis=1)
        self.min_pair, self.max_pair = min(self.min_pair, float(np.min(d))), max(
            self.max_pair, float(np.max(d))
        )

    def log(self, t):
        p, v, state = self.unpack(self.y)
        p0, v0, a0 = complex_target(t)
        phat, ahat, _, _, _ = pb.unpack(state)
        u, _ = self.algorithm(t, self.y)
        distances = np.linalg.norm(self.D.T @ p, axis=1)
        row = dict(
            time=t,
            active_nodes=self.n,
            active_edges=len(self.pairs),
            localization_max=float("nan"),
            observer_max=float(np.max(np.linalg.norm(phat - p0, axis=1))),
            speed_max=float(np.max(np.linalg.norm(v - v0, axis=1))),
            broadcast_error=float(np.linalg.norm(self.z1 - v0)),
            acceleration_broadcast_error=float(np.linalg.norm(self.z2 - a0)),
            acceleration_observer_max=float(np.max(np.linalg.norm(ahat - a0, axis=1))),
            centroid_error=float(np.linalg.norm(np.mean(p, axis=0) - p0)),
            hull_distance=hull_distance(p, p0),
            signed_hull_margin=signed_hull_margin(p, p0),
            min_pair_distance=float(np.min(distances)),
            min_edge_distance=float(np.min(distances)),
            max_edge_distance=float(np.max(distances)),
            control_max=float(np.max(np.linalg.norm(u, axis=1))),
            speed_rms=float(np.sqrt(np.mean(np.sum((v - v0) ** 2, axis=1)))),
        )
        self.record.append(row)
        values = dict(
            time=t,
            position=p.copy(),
            velocity=v.copy(),
            shat=phat - p,
            baseline_state=state.copy(),
            edge_state=np.full((len(self.pairs), 18), np.nan),
            active=np.ones(self.n, bool),
            edge_active=np.ones(len(self.pairs), bool),
            z0=self.z0.copy(),
            z1=self.z1.copy(),
            z2=self.z2.copy(),
            chi=np.full(2, np.nan),
            vbar=self.z1.copy(),
            abar=self.z2.copy(),
            control=u,
            target_position=p0,
            target_velocity=v0,
            target_acceleration=a0,
            target_position_estimate=phat.copy(),
            target_acceleration_estimate=ahat.copy(),
        )
        for key, value in values.items():
            self.snapshots.setdefault(key, []).append(value)

    def run(self):
        start = time.perf_counter()
        c = self.cfg
        steps = int(round(c.duration / c.dt))
        stride = int(round(c.output_period / c.dt))
        self.log(0.0)
        self.extrema()
        for k in range(steps):
            t, h, state = k * c.dt, c.dt, self.y
            count = pb.required_substeps(
                t,
                self.unpack(state)[2],
                self.baseline_cfg,
                h,
                stability_limit=c.baseline_stability_limit,
            )
            self.substep_total += count
            self.substep_max = max(self.substep_max, count)
            hs = h / count
            for j in range(count):
                ts = t + j * hs
                k1 = self.rhs(ts, state)
                k2 = self.rhs(ts + hs / 2, state + hs * k1 / 2)
                k3 = self.rhs(ts + hs / 2, state + hs * k2 / 2)
                k4 = self.rhs(ts + hs, state + hs * k3)
                state = state + hs * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.y = state
            if not np.all(np.isfinite(self.y)):
                raise FloatingPointError(f"nonfinite baseline at {t+h}")
            self.z0, self.z1, self.z2 = pb.source_step(
                self.z0, self.z1, self.z2, self.measured(t + h, self.y)[1], h, self.baseline_cfg
            )
            self.extrema()
            if (k + 1) % stride == 0 or k + 1 == steps:
                self.log(t + h)
        arrays = {k: np.asarray(v) for k, v in self.snapshots.items()}
        summary = {
            "scenario": "six_robot_noisy_published_global_position_fencing",
            "source": "Jiang et al., IEEE TASE 22 (2025), DOI 10.1109/TASE.2025.3596929, Eqs. (14), (21), (22), (36)",
            "config": asdict(c),
            "baseline_config": asdict(self.baseline_cfg),
            "source_initialization": "measured target position; velocity and acceleration estimates zero (optional initialization in published Eq. (15))",
            "comparison_adaptations": "common target jerk bound and measured-signal initialization; centroid poles and repulsion/centering ratio matched before comparison",
            "integration": {
                "method": "RK4 on plant and published observer with conservative stiffness-controlled microsteps; implicit source differentiator once per outer step",
                "outer_dt": c.dt,
                "stability_limit": c.baseline_stability_limit,
                "maximum_microsteps_per_outer_step": self.substep_max,
                "total_microsteps": self.substep_total,
            },
            "final": self.record[-1],
            "performance": common_performance(self.record, arrays, c),
            "safety": {
                "minimum_pair_distance": self.min_pair,
                "maximum_active_edge_distance": self.max_pair,
                "ds": c.safe_distance,
                "R_e": c.outer_radius,
                "sampling": "Initial point and every integration endpoint; not continuous extrema.",
            },
            "not_applicable_metrics": ["localization_max", "edge_state", "chi"],
            "runtime_seconds": time.perf_counter() - start,
            "scope": "Published global-position controller on common noisy plant; no exact target velocity or acceleration is used.",
        }
        return summary, arrays, self.record


def run_study_case(case, cfg, output):
    if case not in ("proposed", "current_only", "published_global"):
        raise ValueError(case)
    cfg = replace(cfg, history_mode="instantaneous" if case == "current_only" else "retained")
    sim = PublishedSimulator(cfg) if case == "published_global" else RangeSimulator(cfg)
    try:
        summary, arrays, records = sim.run()
        summary["case"] = case
        summary["status"] = "completed"
        summary = json_ready(summary)
        save_case(output, cfg, summary, arrays, records)
        return summary
    except Exception as exc:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case,
            "status": "failed",
            "config": asdict(cfg),
            "exception": repr(exc),
            "last_saved_time": sim.record[-1]["time"] if sim.record else None,
        }
        (out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
