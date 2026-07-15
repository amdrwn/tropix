from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .polynomial import TropicalPolynomial


def newton_polytope_vertices(f: TropicalPolynomial):
    points = np.asarray(f.newton_polytope_points(), dtype=float)

    if len(points) == 0:
        return points

    points = np.unique(points, axis=0)

    if points.shape[1] == 1:
        lo = points.min(axis=0)
        hi = points.max(axis=0)

        if np.array_equal(lo, hi):
            return lo.reshape(1, -1)

        return np.vstack([lo, hi])

    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except QhullError:
        return points


def newton_polytope(f: TropicalPolynomial):
    """
    Return a ConvexHull object when possible, otherwise return the
    underlying point set.
    """
    points = np.asarray(f.newton_polytope_points(), dtype=float)

    if len(points) == 0:
        return points

    points = np.unique(points, axis=0)

    if points.shape[1] == 1:
        return points

    try:
        return ConvexHull(points)
    except QhullError:
        return points


def minkowski_sum(points1, points2):
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)

    if len(points1) == 0 or len(points2) == 0:
        return np.empty((0, 0))

    if points1.ndim != 2 or points2.ndim != 2:
        raise ValueError("Point sets must be two-dimensional arrays")

    if points1.shape[1] != points2.shape[1]:
        raise ValueError("Dimension mismatch")

    sums = points1[:, None, :] + points2[None, :, :]
    sums = sums.reshape(-1, points1.shape[1])
    sums = np.unique(sums, axis=0)

    if sums.shape[1] == 1:
        lo = sums.min(axis=0)
        hi = sums.max(axis=0)

        if np.array_equal(lo, hi):
            return lo.reshape(1, -1)

        return np.vstack([lo, hi])

    try:
        hull = ConvexHull(sums)
        return sums[hull.vertices]
    except QhullError:
        return sums