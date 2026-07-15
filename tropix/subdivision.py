from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .polynomial import TropicalPolynomial


def lift_newton_polytope(f: TropicalPolynomial):
    if not f.terms:
        return (
            np.empty((0, f.nvars + 1)),
            np.empty((0, f.nvars)),
        )

    items = sorted(f.terms.items())

    exponents = np.asarray(
        [exp for exp, _ in items],
        dtype=float,
    )
    coefficients = np.asarray(
        [coeff for _, coeff in items],
        dtype=float,
    )

    lifted = np.column_stack([exponents, coefficients])

    return lifted, exponents


def _has_affine_lifting(exponents, coefficients, tol):
    """
    Check whether the lifting function is affine.

    Affine liftings induce the trivial regular subdivision consisting of
    a single cell containing every exponent.
    """
    design = np.column_stack(
        [exponents, np.ones(len(exponents))]
    )

    fitted, _, _, _ = np.linalg.lstsq(
        design,
        coefficients,
        rcond=None,
    )

    residual = coefficients - design @ fitted

    return np.all(np.abs(residual) <= tol)


def lower_convex_hull_faces(
    lifted_points,
    tol=1e-10,
):
    lifted_points = np.asarray(
        lifted_points,
        dtype=float,
    )

    if tol < 0:
        raise ValueError("tol must be non-negative")

    if lifted_points.ndim != 2:
        raise ValueError(
            "lifted_points must be a 2D array"
        )

    if len(lifted_points) == 0:
        return []

    points = np.unique(
        lifted_points,
        axis=0,
    )

    if len(points) == 1:
        return [np.array([0], dtype=int)]

    centre = points.mean(axis=0)
    centred = points - centre

    _, singular_values, vh = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    if len(singular_values) == 0:
        return [np.arange(len(points))]

    rank_tol = (
        tol
        * max(points.shape)
        * singular_values[0]
    )

    rank = int(
        np.sum(singular_values > rank_tol)
    )

    if rank <= 1:
        return [np.arange(len(points))]

    basis = vh[:rank].T
    reduced_points = centred @ basis

    try:
        hull = ConvexHull(reduced_points)
    except QhullError:
        return []

    lower_faces = []

    for simplex, equation in zip(
        hull.simplices,
        hull.equations,
    ):
        reduced_normal = equation[:-1]
        ambient_normal = basis @ reduced_normal

        if ambient_normal[-1] < -tol:
            lower_faces.append(simplex)

    return lower_faces


def regular_subdivision(
    f: TropicalPolynomial,
    tol=1e-10,
):
    if tol < 0:
        raise ValueError("tol must be non-negative")

    lifted, exponents = lift_newton_polytope(f)

    if len(lifted) == 0:
        return []

    trivial_cell = tuple(
        sorted(map(tuple, exponents))
    )

    if len(lifted) == 1:
        return [trivial_cell]

    coefficients = lifted[:, -1]

    if _has_affine_lifting(
        exponents,
        coefficients,
        tol,
    ):
        return [trivial_cell]

    faces = lower_convex_hull_faces(
        lifted,
        tol=tol,
    )

    if not faces:
        raise ValueError(
            "Could not compute the lower hull of the lifted points"
        )

    cells = []

    for face in faces:
        cell = tuple(
            sorted(
                map(
                    tuple,
                    lifted[face, :-1],
                )
            )
        )
        cells.append(cell)

    return sorted(set(cells))