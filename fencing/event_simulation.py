"""Offline scheduled-event experiment on the paper's synthetic point-mass plant.

The event injector controls membership, sensor availability, and the independent
motion of the withdrawn point. It does not modify the two fencing control laws.
"""
from dataclasses import asdict
import time

import numpy as np

from . import published_baseline as pb
from .core import SensorFrame, controller_rhs, initial_edges
from .study import RangeSimulator, PublishedSimulator
from .scenario import complex_target
from .target_motion import implicit_joint_step


LEAVE, PARKED, REJOIN = 80., 100., 160.
BLIND, VISIBLE = 180., 190.
ROBOT = 4
WAIT = np.array([20., 3.])


def visible(t):
    return not (BLIND <= t < VISIBLE)


def departure(t, start, velocity):
    """C1 cubic Hermite departure plus a rounded sideways offset with zero endpoint slopes."""
    if t >= PARKED:
        return WAIT.copy(), np.zeros(2)
    duration = PARKED - LEAVE
    u = np.clip((t - LEAVE) / duration, 0., 1.)
    delta = WAIT - start
    side = 0.25 * np.array([-delta[1], delta[0]])
    p = start + (3*u*u - 2*u**3)*delta + duration*u*(1-u)**2*velocity + 16*u*u*(1-u)**2*side
    v = (6*u*(1-u)*delta + 32*u*(1-u)*(1-2*u)*side)/duration + (1-u)*(1-3*u)*velocity
    return p, v


class EventSimulation:
    def __init__(self, case, cfg):
        self.case, self.cfg = case, cfg
        self.global_case = case == 'published_global'
        self.sim = PublishedSimulator(cfg) if self.global_case else RangeSimulator(cfg)
        self.pairs = np.asarray(self.sim.pairs)
        self.incident = np.any(self.pairs == ROBOT, axis=1)
        self.epoch = np.zeros(len(self.pairs))
        self.active = np.ones(cfg.robots, bool)
        self.update_graph()
        self.samples = {}
        self.departure_state = None
        self.substeps = 0
        self.maximum_substeps = 1

    def update_graph(self):
        self.links = self.active[self.pairs].all(axis=1)
        self.working_edges = self.pairs[self.links]
        self.all_active = bool(np.all(self.active))

    def apply_membership(self, now):
        sim = self.sim
        p, v = sim.unpack(sim.y)[:2]
        if now == LEAVE:
            self.departure_state = (p[ROBOT].copy(), v[ROBOT].copy())
            self.active[ROBOT] = False
        elif now == REJOIN:
            # Enabling exchanges only after the former node is back online.
            nr, _, _, np_ = sim.noise.at(now)
            if self.global_case:
                measured_positions = p + np_
                measured_ranges = np.linalg.norm(sim.D.T @ measured_positions, axis=1)
            else:
                measured_ranges = np.linalg.norm(sim.D.T @ p, axis=1) + nr
            d = measured_ranges[self.incident]
            if np.any((d <= self.cfg.safe_distance) | (d >= self.cfg.outer_radius)):
                raise ValueError(f'Reentry at {now:g} s failed the measured edge admission test: {d}')
            if self.global_case:
                state = sim.unpack(sim.y)[2].reshape(sim.n, 8)
                state[ROBOT] = pb.initial_state(measured_positions[ROBOT:ROBOT+1]).reshape(8)
            else:
                _, _, shat, edge = sim.unpack(sim.y)
                shat[ROBOT] = 0.
                edge[self.incident] = initial_edges(int(self.incident.sum()))
                sim.ranges0[self.incident] = d
                self.epoch[self.incident] = now
            self.active[ROBOT] = True
        else:
            return
        self.update_graph()

    def commands(self, t, state, target_available):
        sim = self.sim
        p, v = sim.unpack(state)[:2]
        nr, nv, nt, np_ = sim.noise.at(t)
        if self.global_case:
            obs = state[4*sim.n:].reshape(sim.n, 8)
            current = obs if self.all_active else obs[self.active]
            measured_p = (p+np_)[self.active]
            # Feeding an observer its own current estimate cancels only the
            # unavailable pin. Global robot positions are still measured.
            p_input = complex_target(t)[0] + nt if target_available else current[0, :2]
            a_input = sim.z2 if target_available else current[0, 2:4]
            local_u, local_der = pb.rhs(t, current.ravel(), measured_p, p_input, a_input, sim.baseline_cfg)
            if self.all_active:
                return local_u, local_der
            u, deriv = np.zeros_like(v), np.zeros_like(obs)
            u[self.active], deriv[self.active] = local_u, local_der.reshape(-1, 8)
            return u, deriv.ravel()
        _, _, shat, edges = sim.unpack(state)
        relative = complex_target(t)[0] - p[0] + nt if target_available else shat[0]
        ranges = (np.linalg.norm(sim.D.T @ p, axis=1) + nr)[self.links]
        frame = SensorFrame(v+nv, ranges, relative)
        u, sdot, edot, _, _ = controller_rhs(t-self.epoch[self.links], self.cfg,
            frame, shat, edges[self.links], sim.ranges0[self.links], self.working_edges,
            0, sim.z1, sim.z2)
        if not self.all_active:
            u[ROBOT] = sdot[ROBOT] = 0.
            derivatives = np.zeros_like(edges)
            derivatives[self.links] = edot
            edot = derivatives
        return u, np.concatenate((sdot.ravel(), edot.ravel(), frame.velocities[0]))

    def rhs(self, t, state, target_available):
        sim = self.sim
        u, extras = self.commands(t, state, target_available and visible(t))
        v = sim.unpack(state)[1]
        return np.concatenate((v.ravel(), u.ravel(), extras))

    def set_departure_pose(self, now):
        if not self.all_active:
            p, v = self.sim.unpack(self.sim.y)[:2]
            p[ROBOT], v[ROBOT] = departure(now, *self.departure_state)

    def step_source(self, now):
        # Hold the differentiator states while the source has no measurement.
        # No search, extrapolated observation, or target truth replaces a missing sample.
        if not visible(now):
            return
        sim, c = self.sim, self.cfg
        p, _, _ = complex_target(now)
        z = p + sim.noise.at(now)[2]
        if self.global_case:
            sim.z0, sim.z1, sim.z2 = pb.source_step(sim.z0, sim.z1, sim.z2, z, c.dt, sim.baseline_cfg)
        else:
            z = z - sim.unpack(sim.y)[0][0] + sim.y[-2:]
            sim.z0, sim.z1, sim.z2 = implicit_joint_step(
                sim.z0, sim.z1, sim.z2, z, c.dt, c.Lj, c.mu0, c.mu1, c.mu2)

    def log(self, t):
        sim = self.sim
        p, v = sim.unpack(sim.y)[:2]
        p0, v0, a0 = complex_target(t)
        u, _ = self.commands(t, sim.y, visible(t))
        u[~self.active] = np.nan
        values = dict(time=t, position=p, velocity=v, active=self.active,
            edge_active=self.links, target_visible=visible(t), control=u,
            z0=sim.z0, z1=sim.z1, z2=sim.z2, vbar=sim.z1, abar=sim.z2,
            target_position=p0, target_velocity=v0, target_acceleration=a0)
        if self.global_case:
            obs = sim.unpack(sim.y)[2]
            phat, ahat, _, _, _ = pb.unpack(obs)
            values.update(baseline_state=obs, shat=phat-p, target_position_estimate=phat,
                target_acceleration_estimate=ahat, edge_state=np.full((len(self.pairs), 18), np.nan),
                chi=np.full(2, np.nan))
        else:
            _, _, shat, es = sim.unpack(sim.y)
            values.update(shat=shat, edge_state=es, chi=sim.y[-2:])
        for key, val in values.items():
            self.samples.setdefault(key, []).append(np.array(val, copy=True))

    def run(self):
        from .event_metrics import event_output
        before = time.perf_counter()
        sim, c = self.sim, self.cfg
        stride, steps = round(c.output_period/c.dt), round(c.duration/c.dt)
        for boundary in (c.output_period, c.duration, LEAVE, PARKED, REJOIN, BLIND, VISIBLE):
            if not np.isclose(round(boundary/c.dt)*c.dt, boundary, rtol=0., atol=1e-9):
                raise ValueError('Output times and event boundaries must align with the solver grid')
        self.log(0.)
        for k in range(steps):
            t, end = k*c.dt, (k+1)*c.dt
            state = sim.y
            count = 1
            avail = visible(t)
            if self.global_case:
                obs = sim.unpack(state)[2].reshape(sim.n, 8)[self.active]
                count = pb.required_substeps(t, obs.ravel(), sim.baseline_cfg, c.dt, c.baseline_stability_limit)
            self.substeps += count
            self.maximum_substeps = max(count, self.maximum_substeps)
            h = c.dt/count
            for j in range(count):
                tj = t + j*h
                k1 = self.rhs(tj, state, avail)
                k2 = self.rhs(tj+h/2, state+h*k1/2, avail)
                k3 = self.rhs(tj+h/2, state+h*k2/2, avail)
                k4 = self.rhs(tj+h, state+h*k3, avail)
                state = state+(h/6)*(k1+2*k2+2*k3+k4)
            if not np.isfinite(state).all():
                raise FloatingPointError(f'Nonfinite event trajectory at {end:g} s')
            sim.y = state
            self.set_departure_pose(end)
            self.apply_membership(end)
            self.step_source(end)
            if (k+1) % stride == 0 or k+1 == steps:
                self.log(end)
        arrays = {k: np.asarray(v) for k,v in self.samples.items()}
        summary, records = event_output(self.case, arrays, c)
        summary.update(config=asdict(c), scenario='six_robot_scheduled_membership_and_sensor_outage',
            runtime_seconds=time.perf_counter()-before, mission={
            'events': {'departing_robot':5, 'inactive_seconds':[LEAVE, REJOIN],
                'arrival_at_wait_point_seconds': PARKED, 'wait_position': WAIT.tolist(),
                'target_measurement_outage_seconds':[BLIND, VISIBLE]}},
            integration={'method':'RK4, implicit source update only with available measurements',
            'outer_dt':c.dt,'maximum_microsteps_per_outer_step':self.maximum_substeps,
            'total_microsteps':self.substeps})
        if self.global_case:
            summary['baseline_config'] = asdict(sim.baseline_cfg)
        return summary, arrays, records
