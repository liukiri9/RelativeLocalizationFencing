"""Independent planar convex-hull diagnostics."""

from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def hull_distance(points, target):
    points = np.asarray(points) - target
    try:
        hull = ConvexHull(points)
        if np.all(hull.equations[:, -1] <= 1e-12):
            return 0.0
        sides = hull.simplices
    except QhullError:
        sides = combinations(range(len(points)), 2)
    best = np.inf
    for i, j in sides:
        a, b = points[i], points[j]
        direction = b - a
        fraction = np.clip(-a @ direction / max(direction @ direction, 1e-30), 0.0, 1.0)
        best = min(best, np.linalg.norm(a + fraction * direction))
    return float(best)


def signed_hull_margin(points, target):
    try:
        hull = ConvexHull(np.asarray(points) - target)
        return float(
            np.min(-hull.equations[:, -1] / np.linalg.norm(hull.equations[:, :-1], axis=1))
        )
    except QhullError:
        return float("nan")
