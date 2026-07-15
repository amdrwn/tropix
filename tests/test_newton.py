import numpy as np
import pytest
from tropix.polynomial import TropicalPolynomial
from tropix.newton import newton_polytope_vertices, minkowski_sum


def test_minkowski_sum_equals_newton_product():
    f = TropicalPolynomial({(2, 0): 0, (0, 0): 0}, nvars=2)
    g = TropicalPolynomial({(1, 0): 0, (0, 1): 0}, nvars=2)

    fg = f * g
    newt_fg = newton_polytope_vertices(fg)

    newt_f = newton_polytope_vertices(f)
    newt_g = newton_polytope_vertices(g)
    mink = minkowski_sum(newt_f, newt_g)

    newt_fg_sorted = newt_fg[np.lexsort(newt_fg.T)]
    mink_sorted = mink[np.lexsort(mink.T)]

    np.testing.assert_array_almost_equal(newt_fg_sorted, mink_sorted)


def test_2d_vertices_exclude_interior_points():
    f = TropicalPolynomial(
        {
            (0, 0): 0.0,
            (2, 0): 0.0,
            (0, 2): 0.0,
            (2, 2): 0.0,
            (1, 1): -1.0,
        },
        nvars=2,
    )
    vertices = newton_polytope_vertices(f)
    assert len(vertices) == 4
    assert not any(np.array_equal(v, [1.0, 1.0]) for v in vertices)


def test_1d_vertices_are_just_the_two_extremes():
    f = TropicalPolynomial({(0,): 0.0, (1,): 0.0, (2,): 0.0}, nvars=1)
    vertices = newton_polytope_vertices(f)
    assert vertices.shape == (2, 1)
    np.testing.assert_array_equal(vertices, [[0.0], [2.0]])


def test_1d_single_point_returns_one_vertex():
    f = TropicalPolynomial({(1,): 0.0}, nvars=1)
    vertices = newton_polytope_vertices(f)
    assert vertices.shape == (1, 1)
    np.testing.assert_array_equal(vertices, [[1.0]])


def test_1d_minkowski_sum_is_also_just_two_extremes():
    f = TropicalPolynomial({(0,): 0.0, (2,): 0.0}, nvars=1)
    g = TropicalPolynomial({(0,): 0.0, (3,): 0.0}, nvars=1)
    pts_f = newton_polytope_vertices(f)
    pts_g = newton_polytope_vertices(g)
    result = minkowski_sum(pts_f, pts_g)
    np.testing.assert_array_equal(result, [[0.0], [5.0]])


def test_minkowski_sum_deduplicates_points():
    a = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0]])
    result = minkowski_sum(a, b)
    assert len(result) == 2
    assert any(np.array_equal(v, [0.0, 0.0]) for v in result)
    assert any(np.array_equal(v, [1.0, 0.0]) for v in result)


def test_empty_polynomial_gives_empty_vertices():
    f = TropicalPolynomial.zero(nvars=2)
    vertices = newton_polytope_vertices(f)
    assert len(vertices) == 0


def test_minkowski_sum_rejects_dimension_mismatch():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        minkowski_sum(a, b)


def test_minkowski_sum_rejects_malformed_1d_input():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([[1.0]])
    with pytest.raises(ValueError):
        minkowski_sum(a, b)


def test_minkowski_sum_with_empty_input_returns_empty():
    a = np.empty((0, 2))
    b = np.array([[1.0, 1.0]])
    result = minkowski_sum(a, b)
    assert len(result) == 0