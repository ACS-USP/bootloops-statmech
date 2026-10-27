"""Displayed equations of `percolation_lattice` (catalog_percolation_lattice.py).  The sympy blocks are
checked against the percolation/ code: the self-dual cubic at the code's certified thresholds, the
trigonometric closed form, Cardy's Eq. (8) displayed in sympy at the float route's cross-ratio against
the certified arb value, and the exponents against their scaling relations.
"""
import sympy as sp

from .tex import Check, cited, sym, text

P, A, ETA = sp.symbols("p a eta", positive=True)
_GM = "Grimmett & Manolescu, arXiv:1105.5535, Eqs. (1.1)-(1.3), Theorem 1.1, Proposition 2.1"
_CARDY = "Cardy, arXiv:hep-th/9111026, Eq. (8) and p. 8"
_SW = "Smirnov & Werner, arXiv:math/0109120, Theorem 1, Eqs. (1), (2)"
_NZ = "Newman & Ziff, arXiv:cond-mat/0005264, Eq. (1)"
_BR = "Bollobas & Riordan, arXiv:math/0410359, Lemma 3, Corollaries 4, 5"


def _pc(lattice, dilution, key="p_c"):
    from .percolation.exact2d import threshold
    got = threshold(lattice, dilution)
    return (got[0] if key == "p_c" else got[1][key]).mid()


def _cardy(a):
    from .percolation.cardy import cardy_ball
    return cardy_ball(a).mid()


def _eta(a):
    from .percolation.cardy import eta_of
    return eta_of(a)


def _ex(name):
    from .percolation.exact2d import exponents
    q = exponents()[0][name]
    return sp.Rational(q.numerator, q.denominator).evalf(30)


def _exact_blocks():
    third = sp.Rational(1, 3)
    cardy = 3 * sp.gamma(2 * third) / sp.gamma(third) ** 2 * ETA ** third * sp.hyper([third, 2 * third],
                                                                                     [4 * third], ETA)
    a1, a4 = sp.Rational(5, 48), sp.Rational(5, 4)
    nu = 1 / (2 - a4)
    return [
        text("Exact route", "closed-form thresholds in arb balls, each with a second route or an exact "
             "identity; the 2D exponents as rationals; Cardy's crossing formula by two routes; the exact "
             "crossing probability of a small open lattice by enumeration", source=_GM),
        cited("Critical surfaces of inhomogeneous bond percolation",
              r"\kappa_\square = p_h + p_v - 1,\quad \kappa_\triangle = p_0 + p_1 + p_2 - p_0 p_1 p_2 - 1,"
              r"\quad \kappa_{hex}(p) = -\kappa_\triangle(1 - p)", _GM),
        sym("Self-dual cubic, at the code's certified thresholds (triangular, 1 - honeycomb, the bisected "
            "root)", sp.Eq(P ** 3 - 3 * P + 1, 0), source=_GM,
            check=Check((P,), lambda p: 0, (lambda: (_pc("triangular", "bond"),),
                                            lambda: (1 - _pc("honeycomb", "bond"),),
                                            lambda: (_pc("triangular", "bond", "p_c_cubic_root"),)), "lhs")),
        sym("Triangular bond threshold (derived here: x = 2 sin t gives x^3 - 3x = -2 sin 3t)",
            sp.Eq(sp.Symbol("p_c"), 2 * sp.sin(sp.pi / 18)),
            check=Check((), lambda: _pc("triangular", "bond"), ((),), "rhs")),
        cited("Star-triangle transformation (connection laws; polynomials derived here)",
              r"P_\triangle(ABC) - P_\star(ABC) = \kappa_\triangle,\quad P_\triangle(\emptyset) - "
              r"P_\star(\emptyset) = -\kappa_\triangle,\quad P_\triangle(AB) = P_\star(AB)", _GM),
        sym("Cardy's crossing formula, displayed at the float route's eta, against the arb value",
            sp.Eq(sp.Symbol("pi_h"), cardy), source=_CARDY,
            check=Check((A, ETA), lambda a, eta: _cardy(a), tuple(
                (lambda a=a: (a, _eta(a))) for a in (1, 2, 0.5)), "rhs")),
        cited("Rectangle of aspect ratio r (horizontal crossing: r = height / width)",
              r"r = \frac{K(1-k^2)}{2K(k^2)},\qquad \eta = \left(\frac{1-k}{1+k}\right)^2", _CARDY),
        sym("Correlation-length exponent from the four-arm exponent", sp.Eq(sp.Symbol("nu"), nu), source=_SW,
            check=Check((), lambda: _ex("nu"), ((),), "rhs")),
        sym("Order-parameter exponent", sp.Eq(sp.Symbol("beta"), nu * a1), source=_SW,
            check=Check((), lambda: _ex("beta"), ((),), "rhs")),
        sym("Mean-cluster-size exponent", sp.Eq(sp.Symbol("gamma"), nu * (2 - 2 * a1)), source=_SW,
            check=Check((), lambda: _ex("gamma"), ((),), "rhs")),
        cited("Exact crossing probability of the open W x H lattice (A_n crossing configurations with n "
              "occupied elements, enumerated)", r"R(p) = \sum_{n=0}^{M} A_n\, p^n (1-p)^{M-n},\qquad "
              r"R_h^{W\times H}(p) + R_h^{(H+1)\times(W-1)}(1-p) = 1 \ \text{(square bond)}", _BR)]


def _sample_blocks():
    return [
        text("Newman-Ziff sweeps", "the bonds or sites are occupied one at a time in a random order; "
             "union-find keeps the clusters, their sizes and whether one joins the left and right "
             "columns; one sweep gives every n = 0 .. M", source=_NZ),
        cited("Canonical from microcanonical (NZ Eq. (1))",
              r"Q(p) = \sum_{n=0}^{M} \binom{M}{n} p^n (1-p)^{M-n}\, Q_n,\qquad R_n = P(n^* \le n)", _NZ),
        cited("Finite-size scaling: crossings and collapse",
              r"R_L(p_\times) = R_{2L}(p_\times),\qquad (p_L(R^*) - p_c)\, L^{1/\nu} = f^{-1}(R^*)",
              "percolation/fss2d.py; " + _NZ)]


def percolation_lattice(route):
    return _sample_blocks() if route == "sample" else _exact_blocks()


__all__ = ["percolation_lattice"]
