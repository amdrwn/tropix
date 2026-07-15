import math
import pytest
from tropix.arithmetic import TropicalNumber


def test_addition_is_min():
    a = TropicalNumber(3.0)
    b = TropicalNumber(5.0)
    assert (a + b).value == 3.0


def test_multiplication_is_addition():
    a = TropicalNumber(3.0)
    b = TropicalNumber(5.0)
    assert (a * b).value == 8.0


def test_zero_is_additive_identity():
    a = TropicalNumber(3.0)
    assert (a + TropicalNumber.zero()).value == a.value


def test_one_is_multiplicative_identity():
    a = TropicalNumber(3.0)
    assert (a * TropicalNumber.one()).value == a.value

def test_zero_is_absorbing_under_multiplication():
    a = TropicalNumber(3.0)
    assert a * TropicalNumber.zero() == TropicalNumber.zero()
    assert TropicalNumber.zero() * a == TropicalNumber.zero()


def test_equality_and_hash_are_consistent():
    """
    Equal TropicalNumbers must hash equally, or dict/set usage silently
    breaks. This was a real bug: __eq__ used to allow tolerant (isclose)
    comparison while __hash__ used the exact float, so two "equal"
    values could hash differently.
    """
    a = TropicalNumber(3.0)
    b = TropicalNumber(3.0)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_zeroth_power_of_infinity_is_multiplicative_identity():
    """
    x^0 should be the multiplicative identity for any x, including
    tropical zero (infinity), the same "empty product" convention
    as ordinary arithmetic's 0^0 = 1. This was a real bug: infinity
    raised to the zeroth power used to return infinity instead of
    the multiplicative identity.
    """
    result = TropicalNumber.zero() ** 0
    assert result == TropicalNumber.one()


def test_zeroth_power_of_finite_value_is_multiplicative_identity():
    result = TropicalNumber(5.0) ** 0
    assert result == TropicalNumber.one()


def test_power_matches_repeated_multiplication():
    a = TropicalNumber(2.0)
    assert (a ** 3).value == (a * a * a).value


def test_negative_infinity_is_rejected():
    """
    The min-plus tropical semiring's domain is R union {+infinity}.
    This used to silently flip -infinity to +infinity instead of
    raising, which could mask real bugs elsewhere.
    """
    with pytest.raises(ValueError):
        TropicalNumber(-math.inf)


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        TropicalNumber(math.nan)


def test_negative_exponent_is_rejected():
    with pytest.raises(ValueError):
        TropicalNumber(2.0) ** -1


def test_subtraction_does_not_exist():
    a = TropicalNumber(3.0)
    b = TropicalNumber(5.0)
    with pytest.raises(NotImplementedError):
        a - b


def test_is_infinite():
    assert TropicalNumber.zero().is_infinite()
    assert not TropicalNumber(3.0).is_infinite()


def test_coercion_from_plain_number():
    a = TropicalNumber(3.0)
    assert (a + 5).value == 3.0
    assert (a * 5).value == 8.0


def test_ordering():
    a = TropicalNumber(3.0)
    b = TropicalNumber(5.0)
    assert a < b
    assert b > a
    assert a <= a
    assert a >= a