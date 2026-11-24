"""Per-run data files and performance measures."""

import csv
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np

PERFORMANCE_METRICS = (
    "localization_max",
    "observer_max",
    "speed_max",
    "broadcast_error",
    "acceleration_broadcast_error",
    "centroid_error",
    "hull_distance",
    "signed_hull_margin",
    "control_max",
)


def common_performance(records, arrays, cfg):
    t = arrays["time"]
    columns = {key: np.array([r[key] for r in records]) for key in records[0]}
    tail = t >= t[-1] - 40.0 - 1e-9
    return {
        "full_mean": {k: float(np.mean(columns[k])) for k in PERFORMANCE_METRICS},
        "tail_window_seconds": min(40.0, cfg.duration),
        "tail_mean": {k: float(np.mean(columns[k][tail])) for k in PERFORMANCE_METRICS},
        "containment_fraction": float(np.mean(columns["hull_distance"] <= 1e-8)),
        "tail_containment_fraction": float(np.mean(columns["hull_distance"][tail] <= 1e-8)),
        "full_mean_speed_rms": float(np.mean(columns["speed_rms"])),
        "control_peak": float(np.max(columns["control_max"])),
        "target_path_length": float(
            np.sum(np.linalg.norm(np.diff(arrays["target_position"], axis=0), axis=1))
        ),
        "aggregation": "Arithmetic mean over all saved times including t=0; E_q/E_s/E_v are maxima over edges/robots at each time.",
        "sample_count": len(t),
        "sample_period_seconds": cfg.output_period,
    }


def save_case(directory, cfg, summary, arrays, records):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("summary", summary), ("config", asdict(cfg))]:
        (out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    edges = np.asarray(tuple(combinations(range(arrays["position"].shape[1]), 2)))
    np.savez_compressed(out / "trajectory.npz", **arrays, edges=edges)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def json_ready(value):
    """Use JSON null for undefined metrics; preserve NPZ/CSV NaN samples."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
