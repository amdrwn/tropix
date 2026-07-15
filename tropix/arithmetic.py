from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass(frozen=True)
class TropicalNumber:
    value: float

    def __post_init__(self):
        value = float(self.value)

        if math.isnan(value):
            raise ValueError("Tropical numbers cannot be NaN")

        if value == -math.inf:
            raise ValueError(
                "Negative infinity is not part of the min-plus tropical semiring"
            )

        object.__setattr__(self, "value", value)

    @staticmethod
    def _coerce(other) -> TropicalNumber:
        if isinstance(other, TropicalNumber):
            return other
        return TropicalNumber(other)

    def __add__(self, other):
        other = self._coerce(other)
        return TropicalNumber(min(self.value, other.value))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._coerce(other)

        if self.is_infinite() or other.is_infinite():
            return TropicalNumber.zero()

        return TropicalNumber(self.value + other.value)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("Exponent must be an integer")

        if n < 0:
            raise ValueError(
                "Exponent must be non-negative in the tropical semiring"
            )

        if n == 0:
            return TropicalNumber.one()

        if self.is_infinite():
            return TropicalNumber.zero()

        return TropicalNumber(n * self.value)

    def __sub__(self, other):
        raise NotImplementedError(
            "Tropical subtraction does not exist because the tropical "
            "semiring has no additive inverses"
        )

    def __rsub__(self, other):
        raise NotImplementedError(
            "Tropical subtraction does not exist because the tropical "
            "semiring has no additive inverses"
        )

    def __eq__(self, other):
        try:
            other = self._coerce(other)
        except (TypeError, ValueError):
            return NotImplemented

        return self.value == other.value

    def __lt__(self, other):
        return self.value < self._coerce(other).value

    def __le__(self, other):
        return self.value <= self._coerce(other).value

    def __gt__(self, other):
        return self.value > self._coerce(other).value

    def __ge__(self, other):
        return self.value >= self._coerce(other).value

    @staticmethod
    def zero() -> TropicalNumber:
        """Return the tropical additive identity, positive infinity."""
        return TropicalNumber(math.inf)

    @staticmethod
    def one() -> TropicalNumber:
        """Return the tropical multiplicative identity, zero."""
        return TropicalNumber(0.0)

    def is_infinite(self) -> bool:
        return self.value == math.inf

    def __repr__(self):
        if self.is_infinite():
            return "T(∞)"

        value = int(self.value) if self.value.is_integer() else self.value
        return f"T({value})"

    def __hash__(self):
        return hash(self.value)