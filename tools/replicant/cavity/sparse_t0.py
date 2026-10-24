"""Exact T = 0 RS cavity solution for +-J couplings on the (k+1)-regular graph.

Re-exported by cavity/sparse.py.  Conventions as there (E = -sum J s s, J = +-1).
"""
import mpmath as mp
import sympy as sp

from ..core import Refusal
from ..families.common import CAVITY_CAVEAT

_MP02 = "Mezard & Parisi, The cavity method at zero temperature, arXiv:cond-mat/0207121"


def rs_zero_temperature_pm1(k):
    """At T = 0 with J = +-1, cavity fields are integers and u = J sign(h) in {-1, 0, 1}:
    with P(u = 0) = x, P(u = +-1) = (1 - x)/2, h = sum of k u's and the closure is
    x = P(h = 0).  The root x in (0, 1) (x = 1 is the paramagnet) gives the field law and
    e = f = -[(k+1)/2 E max_{s,s'}(J s s' + h1 s + h2 s') - k E |H|], H = sum of k+1 u's."""
    if not (isinstance(k, int) and k >= 2):
        raise Refusal("precondition:k", f"needs integer excess degree k >= 2, got {k!r}")
    x, z = sp.symbols("x z")
    law = lambda n: sp.Poly(sp.expand((x + (1 - x) / 2 * (z + 1 / z))**n * z**n), z)
    coeff = lambda n, r: law(n).coeff_monomial(z**(n + r))
    roots = [r for r in sp.Poly(coeff(k, 0) - x, x).all_roots() if r.is_real and 0 < r < 1]
    if len(roots) != 1:
        raise Refusal("no-solution:rs-t0", f"k = {k}: {len(roots)} roots of x = P(h=0) in (0,1)")
    x0 = roots[0]
    ph = {r: sp.simplify(coeff(k, r).subs(x, x0)) for r in range(-k, k + 1)}
    pH = {r: sp.simplify(coeff(k + 1, r).subs(x, x0)) for r in range(-k - 1, k + 2)}
    link = sum(sp.Rational(1, 2) * ph[a] * ph[c] * max(J + a + c, J - a - c, -J + a - c,
                                                       -J - a + c)
               for J in (-1, 1) for a in ph for c in ph)
    site = sum(p * abs(r) for r, p in pH.items())
    e = sp.simplify(-(sp.Rational(k + 1, 2) * link - k * site))
    with mp.workdps(30):
        q = sp.simplify(1 - pH[0])          # q = E tanh^2(beta H) -> P(H != 0) at T = 0
        return {"p0": mp.mpf(str(sp.N(x0, 40))), "e": mp.mpf(str(sp.N(e, 40))),
                "q": mp.mpf(str(sp.N(q, 40))), "q_exact": q,
                "p_exact": {r: ph[r] for r in range(0, k + 1)}, "e_exact": e,
                "cite": f"{_MP02}, Sec. III.B (recursion); computed here", "caveat": CAVITY_CAVEAT}
