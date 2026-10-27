"""Degree laws of the cavity / generating-function route (P1): the degree generating function G0, the
excess-degree generating function G1 = G0'/G0'(1), their derivatives in arb balls, exact moments, and
the Taylor coefficients the cluster-size distribution needs.  The finite-graph sampler
(percolation/sampling.py) parses its laws itself and imports nothing from here.

Conventions (Newman, Strogatz & Watts, PRE 64, 026118 (2001), arXiv:cond-mat/0007235 (NSW), Eqs. (2),
(9), (12); Callaway, Newman, Strogatz & Watts, arXiv:cond-mat/0007300v2 (CNSW)):
  ("bethe", k)     every vertex has degree k (coordination k >= 2): G0 = x^k, G1 = x^(k-1); the Bethe
                   lattice (an infinite k-regular tree) and the random k-regular graph.  NOTE: the
                   coordination, unlike cavity/graphs.py's ("regular", k), where k is the excess degree
  ("poisson", c)   Erdos-Renyi, mean degree c > 0 (rational): G0 = G1 = exp(c (x - 1)) (NSW Eq. (12))
  ("finite", {k: p_k})   any finite-support law with rational p_k summing to 1 (also the string
                   "k:p,k:p", e.g. "1:1/4,3:3/4"), degrees 0 .. 1000, mean degree > 0
  ("powerlaw", tau) refused: tau <= 3 has <k^2> = oo, G1'(1) = oo (unsupported:infinite-second-moment:
                   no finite threshold to certify); tau > 3 is an infinite series whose tail this
                   module does not bound (unsupported:degree-law: truncate it to a "finite" law)
Dilution: ("bond", t) keeps every edge with probability t, ("site", t) every vertex; t rational in
[0, 1] (`occupation`).

  parse_law(law) -> Law      Law.kind, .mean() <k>, .branching() G1'(1), .g1pp() G1''(1),
                             .molloy_reed_Q() sum k (k - 2) p_k, all exact Fractions
  Law.G0(x), .dG0(x), .G1(x), .dG1(x)      arb balls at arb x
  Law.taylor(a, n, which)    [coefficients of v^0 .. v^n] of dG0(a + v) or G1(a + v) (arb), a = 1 - t
  Law.g1_audit(x)            (G1(x), G0'(x) / G0'(1)) by two codings (the excess-degree coefficients,
                             and differentiating the degree coefficients): they must overlap
_HOOKS (planted errors, tests): "g1_as_g0" (G1 := G0, the Poisson identity applied to every law) and
"g0_as_g1" (G0 := G1).  Both are invisible on the Poisson law, where G0 = G1 (asserted in the tests),
and both are refused by Law.g1_audit.  "coordination_shift" (0): the Bethe law coded with coordination
k + shift in both G0 and G1, a consistent wrong law the audit cannot see (the closed forms do).
"""
from fractions import Fraction

from ..core import Refusal
from ..lattice.ball import arb_q, rational

KINDS = ("bethe", "poisson", "finite")
MAX_DEGREE = 1000
_HOOKS = {"g1_as_g0": False, "g0_as_g1": False, "coordination_shift": 0}


def occupation(dilution):
    """("bond" | "site", t) -> (kind, Fraction t in [0, 1])."""
    if not (isinstance(dilution, tuple) and len(dilution) == 2 and dilution[0] in ("bond", "site")):
        raise Refusal("precondition:dilution", f"{dilution!r}: need ('bond' | 'site', t)")
    t = rational(dilution[1], "occupation", positive=False)
    if not 0 <= t <= 1:
        raise Refusal("precondition:occupation", f"occupation = {dilution[1]!r}: need 0 <= t <= 1")
    return dilution[0], t


def _finite_pk(spec):
    if isinstance(spec, str):
        try:
            spec = dict(item.split(":") for item in spec.replace(" ", "").split(",") if item)
        except ValueError:
            raise Refusal("precondition:degree-law", f"{spec!r}: write the finite law as 'k:p,k:p'")
    if not isinstance(spec, dict) or not spec:
        raise Refusal("precondition:degree-law", f"{spec!r}: a finite law is a nonempty {{k: p_k}}")
    pk = {}
    for k, p in spec.items():
        try:
            kk = int(k)
        except (TypeError, ValueError):
            raise Refusal("precondition:degree-law", f"degree {k!r}: need an integer")
        if isinstance(k, bool) or str(kk) != str(k).strip() or not 0 <= kk <= MAX_DEGREE:
            raise Refusal("precondition:degree-law", f"degree {k!r}: need an int in 0 .. {MAX_DEGREE}")
        q = rational(p, "degree-law", positive=False)
        if q < 0:
            raise Refusal("precondition:degree-law", f"p_{kk} = {p!r} < 0")
        if q:
            pk[kk] = pk.get(kk, Fraction(0)) + q
    if sum(pk.values()) != 1:
        raise Refusal("precondition:degree-law", f"the p_k sum to {sum(pk.values())}, not 1 (exactly)")
    if not any(k for k in pk):
        raise Refusal("precondition:degree-law", "mean degree 0: no edges")
    return pk


class Law:
    def __init__(self, kind, k=None, c=None, pk=None):
        self.kind, self.k, self.c = kind, k, c
        kk = k + _HOOKS["coordination_shift"] if kind == "bethe" else None
        self._pk = pk if kind == "finite" else ({kk: Fraction(1)} if kind == "bethe" else None)

    def __repr__(self):
        return f"Law({self.kind}, {self.k if self.kind == 'bethe' else self.c if self.c else self._pk})"

    # ---- coefficients (exact) ------------------------------------------------------------
    def _true_excess(self):
        z = sum(k * p for k, p in self._pk.items())
        return {k - 1: k * p / z for k, p in self._pk.items() if k}

    def degree(self):
        """{k: p_k} as the generating function G0 uses it (planted g0_as_g1: the excess law)."""
        return dict(self._true_excess()) if _HOOKS["g0_as_g1"] else dict(self._pk)

    def excess(self):
        """{j: q_j} = coefficients of G1 (planted g1_as_g0: the degree law itself)."""
        return dict(self._pk) if _HOOKS["g1_as_g0"] else self._true_excess()

    # ---- exact moments -------------------------------------------------------------------
    def mean(self):
        if self.kind == "poisson":
            return self.c
        return sum(k * p for k, p in self.degree().items())

    def branching(self):
        """G1'(1) = <k (k - 1)> / <k> (NSW Eqs. (9), (32))."""
        if self.kind == "poisson":
            return self.c
        return sum(j * q for j, q in self.excess().items())

    def g1pp(self):
        if self.kind == "poisson":
            return self.c * self.c
        return sum(j * (j - 1) * q for j, q in self.excess().items())

    def molloy_reed_Q(self):
        """Q = sum_k k (k - 2) p_k (Molloy & Reed 1995, p. 163)."""
        if self.kind == "poisson":
            return self.c * self.c - self.c
        return sum(k * (k - 2) * p for k, p in self.degree().items())

    # ---- generating functions in arb -----------------------------------------------------
    @staticmethod
    def _poly(coeffs, x):
        out = arb_q(0)
        for j in range(max(coeffs), -1, -1):
            out = out * x + arb_q(coeffs.get(j, 0))
        return out

    @staticmethod
    def _dpoly(coeffs, x):
        d = {j - 1: j * c for j, c in coeffs.items() if j and c}
        return Law._poly(d, x) if d else 0 * x

    def G0(self, x):
        if self.kind == "poisson":
            return (arb_q(self.c) * (x - 1)).exp()
        return self._poly(self.degree(), x)

    def dG0(self, x):
        if self.kind == "poisson":
            return arb_q(self.c) * (arb_q(self.c) * (x - 1)).exp()
        return self._dpoly(self.degree(), x)

    def G1(self, x):
        if self.kind == "poisson":
            return (arb_q(self.c) * (x - 1)).exp()
        return self._poly(self.excess(), x)

    def dG1(self, x):
        if self.kind == "poisson":
            return arb_q(self.c) * (arb_q(self.c) * (x - 1)).exp()
        return self._dpoly(self.excess(), x)

    def g1_audit(self, x):
        """(G1(x), G0'(x) / G0'(1)): the excess coefficients against the differentiated degree
        coefficients (NSW Eq. (9)); a wrong G0 or G1 coding makes them differ."""
        if self.kind == "poisson":
            c = arb_q(self.c)
            return self.G1(x), c * (c * (x - 1)).exp() / c
        return self.G1(x), self._dpoly(self.degree(), x) / arb_q(self.mean())

    def taylor(self, a, n, which):
        """[v^0 .. v^n] of dG0(a + v) ("dG0") or G1(a + v) ("G1"); a an arb."""
        if self.kind == "poisson":
            c = arb_q(self.c)
            base = (c * (a - 1)).exp() * (c if which == "dG0" else 1)
            out, term = [], base
            for m in range(n + 1):
                out.append(term)
                term = term * c / (m + 1)
            return out
        coeffs = self.excess() if which == "G1" else {j - 1: j * p for j, p in self.degree().items() if j}
        out = []
        for m in range(n + 1):
            s = arb_q(0)
            for j, cj in coeffs.items():
                if j >= m and cj:
                    s += arb_q(cj) * _binom(j, m) * a ** (j - m)
            out.append(s)
        return out


def _binom(j, m):
    from math import comb
    return comb(j, m)


def parse_law(law):
    if isinstance(law, Law):
        return law
    if not (isinstance(law, tuple) and len(law) == 2):
        raise Refusal("unsupported:degree-law", f"{law!r}: supported ('bethe', k), ('poisson', c), "
                      "('finite', {k: p_k}); ('powerlaw', tau) is refused by name")
    kind, par = law
    if kind == "powerlaw":
        tau = rational(par, "tau")
        if tau <= 3:
            raise Refusal("unsupported:infinite-second-moment", f"power law p_k ~ k^-{tau}: <k^2> = oo, "
                          "so G1'(1) = oo and the threshold 1/G1'(1) = 0 (any occupation percolates, "
                          "CNSW Eq. (12)); there is no finite fixed point to certify")
        raise Refusal("unsupported:degree-law", f"power law tau = {tau} > 3: G1 is an infinite series "
                      "whose tail is not bounded here; truncate it to a ('finite', {k: p_k}) law")
    if kind == "bethe":
        if isinstance(par, bool) or not isinstance(par, int) or not 2 <= par <= MAX_DEGREE:
            raise Refusal("precondition:coordination", f"k = {par!r}: need an int in 2 .. {MAX_DEGREE}")
        return Law("bethe", k=par)
    if kind == "poisson":
        return Law("poisson", c=rational(par, "mean-degree"))
    if kind == "finite":
        return Law("finite", pk=_finite_pk(par))
    raise Refusal("unsupported:degree-law", f"{law!r}: kinds {KINDS} (and 'powerlaw', refused)")


__all__ = ["KINDS", "Law", "occupation", "parse_law"]
