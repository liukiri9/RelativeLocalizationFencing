"""Score a synthetic fault trial from logged states, independently of the controller."""
from dataclasses import asdict
from itertools import combinations

import numpy as np

from .geometry import hull_distance, signed_hull_margin
from .results import PERFORMANCE_METRICS


def _active_extreme(values, mask, maximum=True):
    selected = values[mask]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError("invalid active-node or active-edge samples")
    if maximum:
        return np.where(mask, values, -np.inf).max(axis=1)
    return np.where(mask, values, np.inf).min(axis=1)


def _first_sustained(time, ok, start, seconds=2.):
    """Time after start until the first successful run spanning at least seconds."""
    indices = np.flatnonzero(ok & (time >= start))
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if len(run) and time[run[-1]] - time[run[0]] >= seconds - 1e-9:
            return float(time[run[0]] - start)
    return None


def _target_outage(time, visible, observer, hull):
    missing = np.flatnonzero(~visible)
    if not len(missing):
        return None
    if (np.diff(missing) != 1).any():
        raise ValueError("the specified trial has a single contiguous outage")
    i, j = missing[0], missing[-1]
    reference = (time >= time[i] - 10. - 1e-9) & (time < time[i]) & visible
    threshold = float(np.quantile(observer[reference], .95)) if np.any(reference) else None
    out = {
        "onset_s": float(time[i]), "unobserved_samples": int(len(missing)),
        "peak_position_error_m": float(observer[missing].max()),
        "peak_target_to_hull_m": float(hull[missing].max()),
        "outage_containment_fraction": float(np.mean(hull[missing] <= 1e-8)),
        "pre_outage_10s_position_error_p95_m": threshold,
        "recovery_dwell_s": 2.,
        "observation_restored_s": None, "position_error_recovery_delay_s": None,
        "containment_recovery_delay_s": None,
    }
    if j + 1 < len(time):
        start = float(time[j+1])
        out["observation_restored_s"] = start
        if threshold is not None:
            out["position_error_recovery_delay_s"] = _first_sustained(
                time, observer <= threshold, start)
        out["containment_recovery_delay_s"] = _first_sustained(time, hull <= 1e-8, start)
        out["containment_fraction_after_restore"] = float(np.mean(hull[j+1:] <= 1e-8))
    return out


def event_output(case, arrays, cfg):
    """Return summary and CSV rows. Only active members enter fencing metrics.

    Employed by all three methods. Absolute-position benchmark edge estimates are
    undefined; the per-sample CSV retains NaNs and the JSON writer emits nulls.
    Distances use the simulated plant; sensor noise enters the controller only.
    """
    t = np.asarray(arrays["time"])
    p, v = arrays["position"], arrays["velocity"]
    p0, v0, a0 = (arrays[key] for key in ("target_position", "target_velocity", "target_acceleration"))
    active = np.asarray(arrays["active"], dtype=bool)
    edge = np.asarray(arrays["edge_active"], dtype=bool)
    visible = np.asarray(arrays["target_visible"], dtype=bool).reshape(len(t))
    edges = np.array(list(combinations(range(p.shape[1]), 2)))
    i, j = edges.T
    if np.any(np.diff(t) <= 0) or np.any(active.sum(axis=1) < 3):
        raise ValueError("invalid time grid or active group")
    if not np.all(edge.any(axis=1)) or np.any(edge & ~(active[:, i] & active[:, j])):
        raise ValueError("active edge lacks two participating endpoints")
    count = np.sum(active, axis=1)
    q = p[:, i] - p[:, j]
    distances = np.linalg.norm(q, axis=2)
    pairs = active[:, i] & active[:, j]
    centroid = np.sum(np.where(active[:, :, None], p, 0.), axis=1) / count[:, None]
    speeds = np.linalg.norm(v - v0[:, None], axis=2)
    observer = _active_extreme(np.linalg.norm(arrays["shat"] + p - p0[:, None], axis=2), active)
    hull = np.array([hull_distance(pk[mask], target) for pk, mask, target in zip(p, active, p0)])
    margin = np.array([signed_hull_margin(pk[mask], target) for pk, mask, target in zip(p, active, p0)])

    localization = np.full(len(t), np.nan)
    information = np.full(len(t), np.nan)
    if case != "published_global":
        es = arrays["edge_state"]
        phi = es[:, :, 6:10].reshape(*edge.shape, 2, 2)
        estimates = es[:, :, 4:6] + np.einsum("tmij,tmj->tmi", phi, es[:, :, 16:18])
        localization = _active_extreme(np.linalg.norm(estimates - q, axis=2), edge)
        gram = es[:, :, 10:14].reshape(*edge.shape, 2, 2)
        eig = np.zeros(edge.shape)
        eig[edge] = np.linalg.eigvalsh(gram[edge])[:, 0]
        information = _active_extreme(eig, edge, maximum=False)

    columns = {
        "time": t, "active_nodes": count, "active_edges": edge.sum(axis=1),
        "target_visible": visible.astype(int),
        "localization_max": localization, "observer_max": observer,
        "speed_max": _active_extreme(speeds, active),
        "speed_rms": np.sqrt((np.where(active, speeds, 0.)**2).sum(axis=1) / count),
        "broadcast_error": np.linalg.norm(arrays["vbar"] - v0, axis=1),
        "acceleration_broadcast_error": np.linalg.norm(arrays["abar"] - a0, axis=1),
        "centroid_error": np.linalg.norm(centroid - p0, axis=1),
        "hull_distance": hull, "signed_hull_margin": margin,
        "min_pair_distance": _active_extreme(distances, pairs, maximum=False),
        "min_edge_distance": _active_extreme(distances, edge, maximum=False),
        "max_edge_distance": _active_extreme(distances, edge),
        "all_physical_min_pair_distance": distances.min(axis=1), "gramian_min": information,
        "control_max": _active_extreme(np.linalg.norm(arrays["control"], axis=2), active),
    }
    rows = [{key: value.item() for key, value in zip(columns, values)}
            for values in zip(*columns.values())]
    tail = t >= t[-1] - 40. - 1e-9
    means = {key: float(np.mean(columns[key])) for key in PERFORMANCE_METRICS}
    tail_means = {key: float(np.mean(columns[key][tail])) for key in PERFORMANCE_METRICS}
    performance = {
        "sample_count": len(t), "sample_period_seconds": cfg.output_period,
        "full_mean": means, "tail_window_seconds": min(40., float(t[-1] - t[0])),
        "tail_mean": tail_means, "containment_fraction": float(np.mean(hull <= 1e-8)),
        "tail_containment_fraction": float(np.mean(hull[tail] <= 1e-8)),
        "aggregation": "Arithmetic mean over all saved times; E_q, E_s and E_v first take the maximum over currently active edges or robots. E_c and hull distance use currently active robots.",
    }
    safety = {
        "minimum_pair_distance": float(columns["min_pair_distance"].min()),
        "minimum_active_edge_distance": float(columns["min_edge_distance"].min()),
        "maximum_active_edge_distance": float(columns["max_edge_distance"].max()),
        "minimum_all_physical_pairs_including_inactive": float(distances.min()),
        "active_edge_violation_samples": int(np.count_nonzero(
            (columns["min_edge_distance"] <= cfg.safe_distance) |
            (columns["max_edge_distance"] >= cfg.outer_radius))),
        "ds": cfg.safe_distance, "Ds": cfg.outer_radius,
        "sampling": "Actual geometry at saved output times only.",
    }
    return {
        "config": asdict(cfg), "sample_count": len(t), "performance": performance,
        "safety": safety, "target_outage": _target_outage(t, visible, observer, hull),
        "final": rows[-1],
    }, rows
