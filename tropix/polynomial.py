from __future__ import annotations

import math
from typing import Dict, Tuple

from .arithmetic import TropicalNumber

class TropicalPolynomial:
    def __init__(
        self,
        terms: Dict[Tuple[int, ...], float],
        nvars: int | None = None,
    ):
        if nvars is not None and nvars < 0:
            raise ValueError("nvars must be non-negative")

        clean_terms = {}

        for exp, coeff in terms.items():
            exp = tuple(exp)

            if not all(isinstance(e, int) and e >= 0 for e in exp):
                raise ValueError(
                    "Exponents must be tuples of non-negative integers"
                )

            coeff = float(coeff)

            if math.isnan(coeff):
                raise ValueError("Coefficients cannot be NaN")

            if coeff == -math.inf:
                raise ValueError(
                    "Negative infinity is not a valid coefficient "
                    "in the min-plus tropical semiring"
                )

            if coeff == math.inf:
                continue

            if exp in clean_terms:
                clean_terms[exp] = min(clean_terms[exp], coeff)
            else:
                clean_terms[exp] = coeff

        if clean_terms:
            inferred_nvars = len(next(iter(clean_terms)))

            if any(len(exp) != inferred_nvars for exp in clean_terms):
                raise ValueError(
                    "All exponent tuples must have the same length"
                )

            if nvars is not None and nvars != inferred_nvars:
                raise ValueError(
                    f"Expected exponent tuples of length {nvars}, "
                    f"got {inferred_nvars}"
                )

            self.nvars = inferred_nvars
        else:
            self.nvars = 0 if nvars is None else nvars

        self.terms = clean_terms

    @classmethod
    def zero(cls, nvars: int) -> TropicalPolynomial:
        return cls({}, nvars=nvars)

    @classmethod
    def constant(
        cls,
        value: float,
        nvars: int,
    ) -> TropicalPolynomial:
        return cls({(0,) * nvars: value}, nvars=nvars)

    @staticmethod
    def _coerce(
        other,
        nvars: int,
    ) -> TropicalPolynomial:
        if isinstance(other, TropicalPolynomial):
            return other

        if isinstance(other, (int, float)):
            return TropicalPolynomial.constant(other, nvars)

        raise TypeError(f"Unsupported operand type: {type(other).__name__}")

    def __call__(self, point: Tuple[float, ...]) -> TropicalNumber:
        if len(point) != self.nvars:
            raise ValueError(
                f"Expected a {self.nvars}-dimensional point, "
                f"got {len(point)}"
            )

        if not self.terms:
            return TropicalNumber.zero()

        minimum = min(
            coeff + sum(e * p for e, p in zip(exp, point))
            for exp, coeff in self.terms.items()
        )

        return TropicalNumber(minimum)

    def argmin_terms(
        self,
        point: Tuple[float, ...],
        tol: float = 1e-10,
    ) -> list[Tuple[int, ...]]:
        if len(point) != self.nvars:
            raise ValueError(
                f"Expected a {self.nvars}-dimensional point, "
                f"got {len(point)}"
            )

        if tol < 0:
            raise ValueError("tol must be non-negative")

        if not self.terms:
            return []

        values = {
            exp: coeff + sum(e * p for e, p in zip(exp, point))
            for exp, coeff in self.terms.items()
        }

        minimum = min(values.values())

        return [
            exp
            for exp, value in values.items()
            if abs(value - minimum) <= tol
        ]

    def __add__(self, other):
        other = self._coerce(other, self.nvars)

        if self.nvars != other.nvars:
            raise ValueError("Polynomial dimensions must match")

        result = dict(self.terms)

        for exp, coeff in other.terms.items():
            if exp in result:
                result[exp] = min(result[exp], coeff)
            else:
                result[exp] = coeff

        return TropicalPolynomial(result, nvars=self.nvars)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._coerce(other, self.nvars)

        if self.nvars != other.nvars:
            raise ValueError("Polynomial dimensions must match")

        if not self.terms or not other.terms:
            return TropicalPolynomial.zero(self.nvars)

        result = {}

        for exp1, coeff1 in self.terms.items():
            for exp2, coeff2 in other.terms.items():
                new_exp = tuple(
                    e1 + e2 for e1, e2 in zip(exp1, exp2)
                )
                new_coeff = coeff1 + coeff2

                if new_exp in result:
                    result[new_exp] = min(
                        result[new_exp],
                        new_coeff,
                    )
                else:
                    result[new_exp] = new_coeff

        return TropicalPolynomial(result, nvars=self.nvars)

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        if not isinstance(other, TropicalPolynomial):
            return NotImplemented

        return (
            self.nvars == other.nvars
            and self.terms == other.terms
        )

    def newton_polytope_points(self) -> list[Tuple[int, ...]]:
        return sorted(self.terms)

    def __repr__(self):
        if not self.terms:
            return "∞"

        terms = []

        for exp, coeff in sorted(self.terms.items()):
            factors = []

            for i, power in enumerate(exp, start=1):
                if power == 1:
                    factors.append(f"x{i}")
                elif power > 1:
                    factors.append(f"x{i}^{power}")

            monomial = "".join(factors)

            if monomial:
                terms.append(f"{coeff:g}⊙{monomial}")
            else:
                terms.append(f"{coeff:g}")

        return " ⊕ ".join(terms)