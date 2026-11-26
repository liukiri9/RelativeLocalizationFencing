"""Target trajectory, initial conditions and sampled measurement noise."""

from dataclasses import dataclass
import numpy as np

from .core import Config


@dataclass(frozen=True)
class StudyConfig(Config):
    robots: int = 6
    events: bool = True
    sensor_rate: float = 30.0
    range_std: float = 0.1
    velocity_std: float = 0.05
    source_position_std: float = 0.1
    global_position_std: float = 0.1
    initial_min_separation: float = 1.3
    initial_velocity_std: float = 0.12
    baseline_stability_limit: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.robots != 6:
            raise ValueError("the registered comparison uses six robots")
        for field in (
            "sensor_rate",
            "baseline_stability_limit",
            "initial_min_separation",
            "range_std",
            "velocity_std",
            "source_position_std",
            "global_position_std",
        ):
            if not np.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive and finite")
        if not np.isfinite(self.initial_velocity_std) or self.initial_velocity_std < 0:
            raise ValueError("initial_velocity_std must be nonnegative and finite")
        if self.seed < 0 or not isinstance(self.seed, int):
            raise ValueError("seed must be a nonnegative integer")
        if not np.isclose(round(self.duration / self.dt) * self.dt, self.duration):
            raise ValueError("duration must be a multiple of dt")
        stride = round(self.output_period / self.dt)
        if stride < 1 or not np.isclose(stride * self.dt, self.output_period):
            raise ValueError("output_period must be a positive multiple of dt")
        if self.initial_min_separation <= self.safe_distance:
            raise ValueError("positive sample rate and strictly safe initial separation required")


# Drift plus independent spatial harmonics. The target repeatedly changes
# curvature and occasionally reverses its x velocity, without circular motion.
DRIFT = np.array([0.16, 0.0])
AMPLITUDES = np.array([[4.0, 2.0, 0.0], [5.0, 3.0, 1.5]])
RATES = np.array([[0.035, 0.08, 0.0], [0.04, 0.075, 0.115]])
PHASES = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])


def complex_target(t):
    q = RATES * float(t) + PHASES
    position = DRIFT * float(t) + np.sum(AMPLITUDES * (np.sin(q) - np.sin(PHASES)), axis=1)
    velocity = DRIFT + np.sum(AMPLITUDES * RATES * np.cos(q), axis=1)
    acceleration = -np.sum(AMPLITUDES * RATES**2 * np.sin(q), axis=1)
    return position, velocity, acceleration


def initial_conditions(cfg):
    """Uniform Cartesian rejection sampling, without a polygon construction."""
    rng = np.random.default_rng(cfg.seed)
    positions, attempts = [], 0
    while len(positions) < cfg.robots:
        attempts += 1
        point = rng.uniform([1.0, -6.0], [7.0, 0.0])
        if all(
            np.linalg.norm(point - other) >= cfg.initial_min_separation for other in positions
        ):
            positions.append(point)
        if attempts > 100000:
            raise RuntimeError("initial-condition rejection sampler failed")
    return np.asarray(positions), rng.normal(0.0, cfg.initial_velocity_std, (cfg.robots, 2))


class HeldNoise:
    """Precomputed channel samples; querying never advances an RNG.

    The additive perturbation is held between exact sensor sample times.
    Noise channels are identical across paired cases, even when a case does
    not use a channel. The global-position baseline receives independent
    position errors with the same standard deviation as the source position.
    """

    def __init__(self, cfg):
        self.rate = cfg.sensor_rate
        count = int(np.ceil(cfg.duration * self.rate)) + 2
        m = cfg.robots * (cfg.robots - 1) // 2
        children = np.random.SeedSequence([cfg.seed, 916]).spawn(4)
        self.range = np.random.default_rng(children[0]).normal(0.0, cfg.range_std, (count, m))
        self.velocity = np.random.default_rng(children[1]).normal(
            0.0, cfg.velocity_std, (count, cfg.robots, 2)
        )
        self.target = np.random.default_rng(children[2]).normal(
            0.0, cfg.source_position_std, (count, 2)
        )
        self.position = np.random.default_rng(children[3]).normal(
            0.0, cfg.global_position_std, (count, cfg.robots, 2)
        )

    def index(self, t):
        return min(int(np.floor(max(float(t), 0.0) * self.rate + 1e-9)), len(self.range) - 1)

    def at(self, t):
        k = self.index(t)
        return self.range[k], self.velocity[k], self.target[k], self.position[k]
