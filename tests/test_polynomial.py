import math
import pytest
from tropix.polynomial import TropicalPolynomial


def test_evaluation_is_minimum_over_terms():
    # f = min(x, y, 0)
    f = TropicalPolynomial({(1, 0): 0.0, (0, 1): 0.0, (0, 0): 0.0}, nvars=2)
    assert f((2.0, 5.0)).value == 0.0
    assert f((-3.0, 5.0)).value == -3.0
    assert f((5.0, -3.0)).value == -3.0


def test_empty_polynomial_evaluates_to_tropical_zero():
    f = TropicalPolynomial.zero(nvars=2)
    assert f((1.0, 2.0)).is_infinite()


def test_addition_takes_termwise_minimum():
    f = TropicalPolynomial({(1, 0): 0.0}, nvars=2)
    g = TropicalPolynomial({(1, 0): 5.0, (0, 1): 2.0}, nvars=2)
    h = f + g
    assert h.terms[(1, 0)] == 0.0  # min(0, 5)
    assert h.terms[(0, 1)] == 2.0


def test_multiplication_adds_exponents_and_coefficients():
    f = TropicalPolynomial({(1, 0): 1.0}, nvars=2)
    g = TropicalPolynomial({(0, 1): 2.0}, nvars=2)
    h = f * g
    assert h.terms == {(1, 1): 3.0}


def test_polynomial_zero_is_absorbing_under_multiplication():
    f = TropicalPolynomial({(1, 0): 2.0}, nvars=2)
    zero = TropicalPolynomial.zero(nvars=2)
    assert f * zero == zero
    assert zero * f == zero


def test_argmin_terms_returns_tied_minimum_exponents():
    # f = min(x, y, 0), evaluated where x == y == 0 all tie
    f = TropicalPolynomial({(1, 0): 0.0, (0, 1): 0.0, (0, 0): 0.0}, nvars=2)
    tied = f.argmin_terms((0.0, 0.0))
    assert set(tied) == {(1, 0), (0, 1), (0, 0)}


def test_negative_exponent_is_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(-1, 0): 0.0})


def test_non_integer_exponent_is_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(1.5, 0): 0.0})


def test_nan_coefficient_is_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(1, 0): math.nan})


def test_negative_infinity_coefficient_is_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(1, 0): -math.inf})


def test_positive_infinity_coefficient_is_silently_dropped():
    """
    A tropical-zero (infinity) coefficient means the monomial isn't
    really present, so it should vanish rather than raise or persist.
    """
    f = TropicalPolynomial({(1, 0): math.inf, (0, 1): 0.0}, nvars=2)
    assert (1, 0) not in f.terms
    assert (0, 1) in f.terms


def test_mismatched_exponent_lengths_are_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(1, 0): 0.0, (1, 0, 0): 1.0})


def test_explicit_nvars_mismatch_is_rejected():
    with pytest.raises(ValueError):
        TropicalPolynomial({(1, 0): 0.0}, nvars=3)


def test_newton_polytope_points_are_sorted():
    f = TropicalPolynomial(
        {(2, 0): 0.0, (0, 0): 0.0, (0, 2): 0.0, (1, 1): -1.0}, nvars=2
    )
    points = f.newton_polytope_points()
    assert points == sorted(points)


def test_addition_keeps_minimum_coefficient_for_matching_exponents():
    f = TropicalPolynomial({(1, 0): 5.0}, nvars=2)
    g = TropicalPolynomial({(1, 0): 2.0}, nvars=2)
    h = f + g
    assert h.terms[(1, 0)] == 2.0


def test_equality():
    f = TropicalPolynomial({(1, 0): 0.0}, nvars=2)
    g = TropicalPolynomial({(1, 0): 0.0}, nvars=2)
    h = TropicalPolynomial({(1, 0): 1.0}, nvars=2)
    assert f == g
    assert f != h


def test_coercion_from_constant():
    f = TropicalPolynomial({(1, 0): 0.0}, nvars=2)
    g = f + 5.0
    # adding a constant should introduce the (0,0) term at value 5.0
    # only if it's smaller than any existing constant term, else no-op
    assert (0, 0) in g.terms