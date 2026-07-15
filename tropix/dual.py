from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .polynomial import TropicalPolynomial
from .subdivision import regular_subdivision


TOL = 1e-9


def find_dual_vertex(cell, f: TropicalPolynomial, tol=TOL):
    exponents = [np.asarray(exp, dtype=float) for exp in cell]
    coefficients = [f.terms.get(exp, np.inf) for exp in cell]

    if any(np.isinf(c) for c in coefficients) or len(exponents) < 2:
        return None

    a0 = exponents[0]
    c0 = coefficients[0]

    A = []
    b = []

    for exponent, coefficient in zip(exponents[1:], coefficients[1:]):
        A.append(a0 - exponent)
        b.append(coefficient - c0)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if np.linalg.norm(A @ x - b) > tol:
        return None

    return x


def deduplicate_vertices(vertices, tol=TOL):
    unique = []
    index_map = {}

    for i, vertex in enumerate(vertices):
        for j, existing in enumerate(unique):
            if np.linalg.norm(vertex - existing) < tol:
                index_map[i] = j
                break
        else:
            index_map[i] = len(unique)
            unique.append(vertex)

    return unique, index_map


def _canonical_edge(edge):
    return tuple(sorted(edge))


def _cell_boundary_edges(cell):
    """
    Return the geometric boundary edges of a two-dimensional cell.
    """
    points = np.asarray(cell, dtype=float)

    if len(points) < 2:
        return []

    if len(points) == 2:
        return [_canonical_edge((cell[0], cell[1]))]

    try:
        hull = ConvexHull(points)
        order = hull.vertices
    except QhullError:
        centred = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]

        coordinates = points @ direction
        lo = int(np.argmin(coordinates))
        hi = int(np.argmax(coordinates))

        if lo == hi:
            return []

        return [_canonical_edge((cell[lo], cell[hi]))]

    edges = []

    for i, current in enumerate(order):
        following = order[(i + 1) % len(order)]
        edges.append(
            _canonical_edge((cell[current], cell[following]))
        )

    return edges


def _cell_edge_sets(cells):
    return [set(_cell_boundary_edges(cell)) for cell in cells]


def _edge_counts(cell_edges):
    counts = defaultdict(int)

    for edges in cell_edges:
        for edge in edges:
            counts[edge] += 1

    return counts


def tropical_edges(f: TropicalPolynomial, tol=TOL):
    """
    Compute the dual combinatorics of a two-dimensional tropical curve.

    Returns
    -------
    vertices : list[np.ndarray]
        Finite tropical vertices.
    edges : list[tuple[int, int]]
        Bounded edges between tropical vertices.
    boundary_edges : list[tuple]
        Boundary edges of the Newton subdivision, dual to unbounded rays.
    valid_cells : list[tuple]
        Subdivision cells with a finite dual vertex.
    index_map : dict[int, int]
        Mapping from valid-cell index to deduplicated tropical-vertex index.
    """
    if f.nvars != 2:
        raise NotImplementedError(
            "Dual edge extraction currently supports two variables only"
        )

    cells = regular_subdivision(f, tol=tol)

    if not cells:
        return [], [], [], [], {}

    raw_vertices = []
    valid_cells = []

    for cell in cells:
        vertex = find_dual_vertex(cell, f, tol=tol)

        if vertex is not None:
            raw_vertices.append(vertex)
            valid_cells.append(cell)

    vertices, index_map = deduplicate_vertices(
        raw_vertices,
        tol=tol,
    )

    cell_edges = _cell_edge_sets(valid_cells)
    edge_count = _edge_counts(cell_edges)

    bounded_edges = set()

    for i in range(len(valid_cells)):
        for j in range(i + 1, len(valid_cells)):
            shared_edges = cell_edges[i] & cell_edges[j]

            if not shared_edges:
                continue

            vertex_i = index_map[i]
            vertex_j = index_map[j]

            if vertex_i != vertex_j:
                bounded_edges.add(
                    (min(vertex_i, vertex_j), max(vertex_i, vertex_j))
                )

    boundary_edges = sorted(
        edge for edge, count in edge_count.items()
        if count == 1
    )

    return (
        vertices,
        sorted(bounded_edges),
        boundary_edges,
        valid_cells,
        index_map,
    )


def _perp(vector):
    return np.array([vector[1], -vector[0]], dtype=float)


def plot_tropical_curve_dual(
    f: TropicalPolynomial,
    ax=None,
    ray_length=5,
):
    import matplotlib.pyplot as plt

    if f.nvars != 2:
        raise NotImplementedError(
            "Plotting currently supports two variables only"
        )

    if ray_length <= 0:
        raise ValueError("ray_length must be positive")

    (
        vertices,
        edges,
        boundary_edges,
        valid_cells,
        index_map,
    ) = tropical_edges(f)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for i, j in edges:
        vertex_i = vertices[i]
        vertex_j = vertices[j]

        ax.plot(
            [vertex_i[0], vertex_j[0]],
            [vertex_i[1], vertex_j[1]],
            "b-",
            linewidth=2,
        )

    for vertex in vertices:
        ax.scatter(
            vertex[0],
            vertex[1],
            c="black",
            s=50,
            zorder=5,
        )

    newton_points = np.asarray(
        f.newton_polytope_points(),
        dtype=float,
    )

    if len(newton_points) == 0:
        return ax

    centroid = np.mean(newton_points, axis=0)
    cell_edges = _cell_edge_sets(valid_cells)

    for edge in boundary_edges:
        endpoint_a = np.asarray(edge[0], dtype=float)
        endpoint_b = np.asarray(edge[1], dtype=float)

        subdivision_direction = endpoint_b - endpoint_a
        direction = _perp(subdivision_direction)
        midpoint = 0.5 * (endpoint_a + endpoint_b)

        if np.dot(direction, midpoint - centroid) < 0:
            direction = -direction

        norm = np.linalg.norm(direction)

        if norm == 0:
            continue

        direction = direction / norm

        for cell_index, edges_of_cell in enumerate(cell_edges):
            if edge not in edges_of_cell:
                continue

            vertex_index = index_map[cell_index]
            dual_vertex = vertices[vertex_index]
            ray_end = dual_vertex + ray_length * direction

            ax.plot(
                [dual_vertex[0], ray_end[0]],
                [dual_vertex[1], ray_end[1]],
                "r--",
                linewidth=2,
            )
            break

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("T(f) - analytical dual")

    return ax