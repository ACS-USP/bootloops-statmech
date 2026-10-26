"""Displayed equations of the bound entry (E10): sk_bounds (route bound).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against bounds/ (the certified ball's midpoint);
the displayed GaussE expectations are evaluated by mpmath quadrature (tex.evaluate), the code by arb
(acb.integral); the K-step recursion and the chain are cited strings.
"""
import sympy as sp

from .gauss import GaussE
from .tex import Check, cited, sym, text

B, HF, Q, Z = sp.Symbol(r"\beta", positive=True), sp.Symbol("h", real=True), \
    sp.Symbol("q", nonnegative=True), sp.Symbol("z", real=True)
Q0, Q1, M = sp.symbols("q_0 q_1 m", nonnegative=True)
_G = "Guerra, Comm. Math. Phys. 233, 1 (2003), arXiv:cond-mat/0205123"
_PTS = (("2", "0", "0.3"), ("1.5", "0.3", "0.42"), ("0.5", "0", "0.1"))


def _quad(q0, q1, m):
    from fractions import Fraction
    from .bounds.krsb import _quad_arb
    from .lattice.ball import ball_of, baller
    with baller().hygiene.ctx_guard(prec=200):
        return ball_of(_quad_arb((Fraction(q0), Fraction(q1)), (Fraction(m),))).mid()


def sk_bounds(route):
    from .bounds import guerra_rs_bound
    from .bounds.guerra import rs_bound_derivative
    from .bounds.krsb import quadratic_term
    xi = B * (sp.sqrt(Q) * Z + HF)
    rs = sp.log(2) + B**2 * (1 - Q)**2 / 4 + GaussE(sp.log(sp.cosh(xi)), Z)
    return [
        text("Bound route", "SK with J_ij ~ N(0, 1/N), field h, P = -beta f = lim (1/N) E ln Z_N. "
             "Guerra's Theorem 3: (1/N) E ln Z_N <= P_K(q, m) for every N and every admissible "
             "order parameter; each value below is a ball enclosure of P_K (python-flint arb through "
             "tools/baller)", source=f"{_G}, Theorem 3"),
        sym("Replica-symmetric bound (K = 0: Eq. (9) in Eq. (14))", sp.Eq(sp.Symbol("P_{RS}(q)"), rs),
            source=f"{_G}, Eqs. (9), (14)",
            check=Check((B, HF, Q), lambda b, h, q: guerra_rs_bound(b, h, q).mid(), _PTS, "rhs")),
        sym("Its q-derivative (Gaussian integration by parts): stationary iff q = E tanh^2",
            sp.Eq(sp.Symbol("P_{RS}'(q)"), B**2 / 2 * (Q - GaussE(sp.tanh(xi)**2, Z))),
            source=f"{_G}, Eq. (14)",
            check=Check((B, HF, Q), lambda b, h, q: rs_bound_derivative(b, h, q).mid(), _PTS, "rhs")),
        cited("K-step bound (Guerra's recursion (12) in the package's variables)",
              r"P_K = \ln 2 + \frac{\beta^2}{4}\Big[(1-q_K)^2 - \sum_{k=1}^{K} m_k (q_k^2 - q_{k-1}^2)\Big]"
              r" + \mathbb{E}_z \Phi_0(\beta h + \beta\sqrt{q_0}\,z),\quad \Phi_K = \ln\cosh,\ "
              r"\Phi_{k-1}(y) = \frac{1}{m_k}\ln \mathbb{E}_z "
              r"e^{m_k \Phi_k(y + \beta\sqrt{q_k - q_{k-1}}\,z)}",
              f"{_G}, Eqs. (12), (14), (26)"),
        sym("The bracket for K = 1 (identity (26)); families/sk.py's quadratic term",
            sp.Eq(sp.Symbol("B_1"), quadratic_term((Q0, Q1), (M,))), source=f"{_G}, Eq. (26)",
            check=Check((Q0, Q1, M), _quad, (("0.2", "0.6", "0.3"), ("0", "0.9", "1"),
                                             ("0.1", "0.4", "0.7")), "rhs")),
        cited("Chain at the package's optimisers (a consistency check, not a theorem)",
              r"P_K(q^{(K)}, m^{(K)}) \le \dots \le P_1(q^{(1)}, m^{(1)}) \le P_{RS}(q_{RS})",
              f"{_G}, Theorem 3; the ordering of the infima"),
    ]


__all__ = ["sk_bounds"]
