from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError

from .polynomial import TropicalPolynomial


def tropical_hypersurface_points(
    f: TropicalPolynomial,
    bounds=(-5, 5),
    grid_size=400,
):
    """
    Approximate a two-dimensional tropical hypersurface on a regular grid.

    A point belongs to the tropical hypersurface when the minimum among
    the monomial values is attained at least twice. On a finite grid,
    this is approximated by checking whether the two smallest values
    differ by at most a numerical tolerance.
    """
    if f.nvars != 2:
        raise ValueError("Requires a polynomial in two variables")

    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be an increasing pair")

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    if not f.terms:
        return np.empty((0, 2))

    xs = np.linspace(bounds[0], bounds[1], grid_size)
    ys = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(xs, ys)

    grid_spacing = (bounds[1] - bounds[0]) / (grid_size - 1)
    tol = max(1e-10, grid_spacing)

    smallest = np.full_like(X, np.inf, dtype=float)
    second_smallest = np.full_like(X, np.inf, dtype=float)

    for exp, coeff in f.terms.items():
        values = coeff + exp[0] * X + exp[1] * Y

        lower = values < smallest

        second_smallest = np.where(
            lower,
            smallest,
            np.minimum(second_smallest, values),
        )

        smallest = np.where(
            lower,
            values,
            smallest,
        )

    mask = second_smallest - smallest <= tol

    return np.column_stack((X[mask], Y[mask]))


def plot_tropical_curve(
    f: TropicalPolynomial,
    bounds=(-5, 5),
    grid_size=400,
):
    """
    Plot a grid approximation of a tropical curve and its Newton polytope.
    """
    if f.nvars != 2:
        raise ValueError("Requires a polynomial in two variables")

    points = tropical_hypersurface_points(
        f,
        bounds=bounds,
        grid_size=grid_size,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_curve = axes[0]

    if len(points) > 0:
        ax_curve.scatter(
            points[:, 0],
            points[:, 1],
            s=1,
            alpha=0.6,
            rasterized=True,
        )

    ax_curve.set_xlim(bounds)
    ax_curve.set_ylim(bounds)
    ax_curve.set_title("Tropical hypersurface T(f)")
    ax_curve.set_xlabel("x")
    ax_curve.set_ylabel("y")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.set_aspect("equal")

    ax_newton = axes[1]

    newton_points = np.asarray(
        f.newton_polytope_points(),
        dtype=float,
    )

    if len(newton_points) > 0:
        ax_newton.scatter(
            newton_points[:, 0],
            newton_points[:, 1],
            s=50,
            zorder=5,
        )

    if len(newton_points) >= 3:
        try:
            hull = ConvexHull(newton_points)
            hull_points = newton_points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])

            ax_newton.plot(
                hull_points[:, 0],
                hull_points[:, 1],
            )
        except QhullError:
            pass

    for exp, coeff in f.terms.items():
        ax_newton.annotate(
            f"{coeff:g}",
            xy=exp,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax_newton.set_title("Newton polytope")
    ax_newton.set_xlabel("x exponent")
    ax_newton.set_ylabel("y exponent")
    ax_newton.grid(True, alpha=0.3)
    ax_newton.set_aspect("equal")

    fig.tight_layout()

    return fig


def inspect_point(
    f: TropicalPolynomial,
    point,
):
    """
    Print the ordered monomial values at a point.
    """
    if len(point) != f.nvars:
        raise ValueError(
            f"Expected a {f.nvars}-dimensional point, got {len(point)}"
        )

    values = []

    for exp, coeff in f.terms.items():
        value = coeff + sum(
            exponent * coordinate
            for exponent, coordinate in zip(exp, point)
        )
        values.append((exp, value))

    if not values:
        print(f"Point {point}: no terms")
        return

    values.sort(key=lambda item: item[1])

    print(f"Point {point}:")
    print(f"  min = ({values[0][0]}, {values[0][1]:.6f})")

    if len(values) > 1:
        gap = values[1][1] - values[0][1]

        print(f"  2nd = ({values[1][0]}, {values[1][1]:.6f})")
        print(f"  gap = {gap:.6f}")