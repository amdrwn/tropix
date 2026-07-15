import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tropix.dual import plot_tropical_curve_dual, tropical_edges
from tropix.polynomial import TropicalPolynomial


def test_tropical_line_has_one_vertex_three_rays():
    """
    The standard tropical line min(x, y, 0) has one vertex at the
    origin and three unbounded rays.
    """
    f = TropicalPolynomial(
        {
            (1, 0): 0.0,
            (0, 1): 0.0,
            (0, 0): 0.0,
        },
        nvars=2,
    )

    vertices, edges, boundary_edges, valid_cells, index_map = (
        tropical_edges(f)
    )

    assert len(vertices) == 1
    np.testing.assert_array_almost_equal(
        vertices[0],
        [0.0, 0.0],
    )
    assert len(edges) == 0
    assert len(boundary_edges) == 3


def test_nonsimplicial_conic_has_three_vertices_two_edges_six_rays():
    """
    This tropical conic has a non-simplicial quadrilateral cell in its
    regular subdivision. Its dual curve therefore has three vertices,
    two bounded edges and six unbounded rays.
    """
    f = TropicalPolynomial(
        {
            (2, 0): 0.0,
            (0, 2): 0.0,
            (0, 0): 0.0,
            (1, 1): -1.0,
            (1, 0): -0.5,
            (0, 1): -0.5,
        },
        nvars=2,
    )

    vertices, edges, boundary_edges, valid_cells, index_map = (
        tropical_edges(f)
    )

    assert len(vertices) == 3
    assert len(edges) == 2
    assert len(boundary_edges) == 6
    assert len(valid_cells) == 3


def test_affine_coefficients_collapse_to_trivial_subdivision():
    """
    If the coefficients are an affine function of the exponents, the
    lifting is flat and induces a single trivial subdivision cell.
    """
    f = TropicalPolynomial(
        {
            (2, 0): 0.0,
            (1, 1): 0.0,
            (0, 2): 0.0,
            (1, 0): 1.0,
            (0, 1): 1.0,
            (0, 0): 2.0,
        },
        nvars=2,
    )

    vertices, edges, boundary_edges, valid_cells, index_map = (
        tropical_edges(f)
    )

    assert len(valid_cells) == 1
    assert len(valid_cells[0]) == 6


def test_dual_plot_runs_without_error():
    """The dual plotting function should run on a headless backend."""
    f = TropicalPolynomial(
        {
            (1, 0): 0.0,
            (0, 1): 0.0,
            (0, 0): 0.0,
        },
        nvars=2,
    )

    ax = plot_tropical_curve_dual(f)

    assert ax is not None
    plt.close(ax.figure)