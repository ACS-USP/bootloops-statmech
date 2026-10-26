"""Holonomic recognition of an exact series through tools/annihilator (its documented surface only:
pf_from_series, nullspace_mod_fast, rec_to_theta; GUIDE.md), plus the singular points of the result.

  recognize_holonomic(series, max_order, max_degree, *, fit_terms=None, primes=PRIMES)
      -> Recognition(operator, certificate, verified_terms, recurrence, fit_terms)
  singularities(operator) -> [Singularity(point, value, factor, multiplicity, kind, exponents)]

Naming.  An operator is L = sum c_ik x^i theta^k (theta = x d/dx): its ORDER is the largest k (the
annihilator's `s`, the n-degree of the recurrence), its DEGREE the largest i (the annihilator's `r`,
the recurrence order).  The recurrence sum_j c_j(n) a_{n+j} = 0 holds for n >= 0; rec_to_theta maps
it to L, and L f = P(x) with P the boundary terms (degree < r, from the indices n < 0):
Operator.rhs.  P = 0 means a homogeneous ODE.

Procedure.  (1) Fit window: the first fit_terms coefficients (default all).  It must be long enough
for the maximal ansatz (r, s) = (max_degree, max_order) to be tested with >= 5 spare equations:
fit_terms >= (r + 1)(s + 1) + 5 + r (else precondition:series-length).  (2) The ODE-order-first mod-p
scan (pf_from_series mode='modp') at two primes; they must agree (GUIDE: the 2-prime confirm is
mandatory).  (3) None found: the maximal ansatz matrix (columns a_{n+j} n^k, rows n < fit_terms - r)
has full column rank mod both primes (nullspace_mod_fast returns None), hence over Q, and so does
every smaller ansatz (its columns are a subset on a subset of rows): no operator of order <= s and
degree <= r annihilates these terms.  Raised as NoOperatorWithinBounds (reason
no-operator-within-bounds) carrying the Certificate.  This is a statement about the bounds, never a
proof that the series is not D-finite.  (4) Found: one exact Q-nullspace at the pinned (r, s)
(mode='auto'); the recurrence is then checked here, in exact arithmetic, on every row that reaches
past the fit window (the held-out terms, never shown to the annihilator): any failure is
no-solution:held-out.
"""
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

from ..core import Refusal

PRIMES = (2147483647, 2147483629)     # 2^31 - 1 and the next prime below it (both < 2^31)
OVERDET = 5                            # find_recurrence_fast's spare equations in pf_from_series


def _annihilator():
    try:
        import annihilator
    except ImportError as exc:                       # tools/ must be on the path
        raise Refusal("missing-tool:annihilator", f"tools/annihilator not importable ({exc})")
    return annihilator


@dataclass(frozen=True, eq=False)
class Operator:
    terms: dict                     # {(x power, theta power): Fraction}
    rhs: tuple = ()                 # P(x) ascending, L f = P

    @property
    def order(self):
        return max(k for _, k in self.terms)

    @property
    def degree(self):
        return max(i for i, _ in self.terms)

    def coefficient(self, k):
        """Coefficient polynomial of theta^k, ascending in x, trailing zeros stripped (() if 0)."""
        p = [Fraction(0)] * (self.degree + 1)
        for (i, kk), c in self.terms.items():
            if kk == k:
                p[i] += c
        while p and p[-1] == 0:
            p.pop()
        return tuple(p)

    def apply(self, series):
        """L applied to the truncated series: coefficients 0 .. len - 1 (all exact)."""
        N = len(series)
        out = [Fraction(0)] * N
        for (i, k), c in self.terms.items():
            for n in range(N - i):
                if series[n]:
                    out[n + i] += c * n**k * series[n]
        return out


@dataclass(frozen=True)
class Certificate:
    found: bool
    max_order: int
    max_degree: int
    n_terms: int                    # coefficients the annihilator saw (the fit window)
    equations: int                  # rows of the maximal ansatz
    unknowns: int                   # (max_degree + 1)(max_order + 1)
    primes: tuple
    detail: str


@dataclass(frozen=True)
class Recognition:
    operator: Operator
    certificate: Certificate
    verified_terms: int             # held-out coefficients checked here, exactly
    recurrence: tuple               # (r, s, {(j, k): c}): sum_j sum_k c n^k a_{n+j} = 0
    fit_terms: int


class NoOperatorWithinBounds(Refusal):
    def __init__(self, certificate):
        c = certificate
        super().__init__("no-operator-within-bounds",
                         f"no operator of theta-order <= {c.max_order} and degree <= {c.max_degree} "
                         f"annihilates the first {c.n_terms} terms: the maximal ansatz ({c.equations} "
                         f"equations, {c.unknowns} unknowns) has full column rank mod "
                         f"{', '.join(map(str, c.primes))} (hence over Q); a bound, not a proof of "
                         "non-D-finiteness")
        self.certificate = certificate


def _exact(series):
    out = []
    for x in series:
        if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
            raise Refusal("precondition:exact-series", f"coefficient {x!r}: need int or Fraction")
        out.append(Fraction(x))
    return out


def _bounds(max_order, max_degree):
    for name, v in (("max_order", max_order), ("max_degree", max_degree)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise Refusal("precondition:bounds", f"{name}={v!r}: need an int >= 1")


def _ansatz(a, r, s, p):
    rows = len(a) - r
    out = []
    for n in range(rows):
        out.append([a[n + j] * pow(n, k, p) % p for j in range(r + 1) for k in range(s + 1)])
    return out


_MEMO = {}


def recognize_holonomic(series, max_order, max_degree, *, fit_terms=None, primes=PRIMES):
    """Recognition of the exact `series` (see the module docstring); successes are memoised per
    (series, bounds, fit window, primes) in this process, refusals are recomputed."""
    _bounds(max_order, max_degree)
    a = _exact(series)
    key = (tuple(a), max_order, max_degree, fit_terms, tuple(primes))
    if key not in _MEMO:
        _MEMO[key] = _recognize(a, max_order, max_degree, fit_terms, primes)
    return _MEMO[key]


def _recognize(a, max_order, max_degree, fit_terms, primes):
    A = _annihilator()
    fit = len(a) if fit_terms is None else fit_terms
    if isinstance(fit, bool) or not isinstance(fit, int) or not 0 < fit <= len(a):
        raise Refusal("precondition:fit-terms", f"fit_terms={fit_terms!r}: need 1 .. {len(a)}")
    r, s = max_degree, max_order
    need = (r + 1) * (s + 1) + OVERDET + r
    if fit < need:
        raise Refusal("precondition:series-length", f"{fit} terms in the fit window; the maximal "
                      f"ansatz (degree {r}, order {s}) needs >= {need}")
    if len(set(primes)) < 2:
        raise Refusal("precondition:two-primes", f"primes={primes!r}: need two distinct primes")
    for p in primes:
        if any(x.denominator % p == 0 for x in a[:fit]):
            raise Refusal("precondition:prime", f"a denominator is divisible by {p}")
    head = a[:fit]
    hk = (tuple(head), r, s, tuple(primes))
    if hk not in _FIT:
        _FIT[hk] = _fit(A, head, r, s, primes)
    rr, ss, coeffs = _FIT[hk]
    return _verify(A, a, fit, r, s, primes, rr, ss, coeffs)


_FIT = {}


def _fit(A, head, r, s, primes):
    """(r, s, coeffs) from the fit window alone, or a refusal (memoised by the caller)."""
    fit = len(head)
    hits = [A.pf_from_series(head, rmax=r, smax=s, mode="modp", p=p) for p in primes]
    unknowns, equations = (r + 1) * (s + 1), fit - r
    if all(h is None for h in hits):
        for p in primes:
            mat = _ansatz([x.numerator % p * pow(x.denominator, -1, p) % p for x in head], r, s, p)
            if A.nullspace_mod_fast(mat, p) is not None:
                raise Refusal("no-solution:inconsistent-scan", f"the scan found nothing but the "
                              f"maximal ansatz has a kernel mod {p}")
        raise NoOperatorWithinBounds(Certificate(False, s, r, fit, equations, unknowns,
                                                 tuple(primes), "full column rank at the maximal ansatz"))
    pins = [None if h is None else h[:2] for h in hits]
    if len(set(pins)) != 1:
        raise Refusal("no-solution:prime-disagreement", f"(degree, order) per prime: {pins}")
    rr, ss = pins[0]
    got = A.pf_from_series(head, rmax=r, smax=s, mode="auto", p=primes[0])
    if got is None or got[2] is None or got[:2] != (rr, ss):
        raise Refusal("no-solution:exact-reconstruction", f"mod-p pin {(rr, ss)}, exact step {got and got[:2]}")
    return rr, ss, got[2]


def _verify(A, a, fit, r, s, primes, rr, ss, coeffs):
    for n in range(len(a) - rr):
        val = sum(c * n**k * a[n + j] for (j, k), c in coeffs.items())
        if val:
            raise Refusal("no-solution:held-out", f"the recurrence fitted on {fit} terms fails at "
                          f"row n = {n} (a_{n + rr}){' (inside the fit window)' if n + rr < fit else ''}")
    terms = {k: Fraction(v) for k, v in A.rec_to_theta(rr, ss, coeffs).items() if v}
    op = Operator(terms)
    res = op.apply(a)
    rhs = list(res[:rr])
    while rhs and rhs[-1] == 0:
        rhs.pop()
    if any(res[rr:]):
        raise Refusal("no-solution:held-out", "the theta-form operator leaves a residual at degree >= r")
    cert = Certificate(True, s, r, fit, fit - rr, (rr + 1) * (ss + 1), tuple(primes),
                       f"found at (degree {rr}, order {ss}) mod both primes, exact Q-nullspace; "
                       "equations / unknowns are those of the pinned ansatz")
    return Recognition(Operator(terms, tuple(rhs)), cert, len(a) - fit, (rr, ss, dict(coeffs)), fit)


# ---- singular points ---------------------------------------------------------------------
@dataclass(frozen=True)
class Singularity:
    point: object                   # QuadNumber (exact) or None (irreducible factor of degree > 2)
    value: object                   # numeric (mpf / mpc)
    factor: tuple                   # the irreducible factor of the leading coefficient, ascending
    multiplicity: int               # of the factor in the d/dx-form leading coefficient
    kind: str                       # "singular" | "apparent" | "irregular" | "not-analysed"
    exponents: object               # LocalExponents or None


_SING = {}


def singularities(op):
    """(memoised per operator)  The finite singular points: x = 0 (theta-form) and the roots of the theta^order
    coefficient, each classified by its local exponents (series/local.py): "apparent" only when
    every local solution is analytic (distinct non-negative integer exponents, no logarithm)."""
    key = (frozenset(op.terms.items()), tuple(op.rhs))
    if key not in _SING:
        _SING[key] = _singularities(op)
    return list(_SING[key])


def _singularities(op):
    import sympy as sp
    from .local import local_exponents, roots_of
    x = sp.Symbol("x")
    lead = sum(sp.Rational(c.numerator, c.denominator) * x**i
               for i, c in enumerate(op.coefficient(op.order)))
    facs = [([Fraction(0), Fraction(1)], op.order)]       # d/dx form: x^order times the lead
    for f, mult in sp.factor_list(lead)[1]:
        co = [Fraction(int(c.p), int(c.q)) for c in reversed(sp.Poly(f, x).all_coeffs())]
        if co == [0, 1]:
            facs[0] = (co, op.order + mult)
        else:
            facs.append((co, mult))
    out = []
    for co, mult in facs:
        if len(co) > 3:
            with mp.workdps(30):
                for z in mp.polyroots([mp.mpf(c.numerator) / c.denominator for c in reversed(co)]):
                    out.append(Singularity(None, z, tuple(co), mult, "not-analysed", None))
            continue
        for pt in roots_of(co):
            loc = local_exponents(op, pt)
            if not loc.regular:
                kind = "irregular"
            elif (not loc.other_exponents and not loc.log_exponents and
                  len(set(loc.integer_exponents)) == len(loc.integer_exponents) and
                  all(e >= 0 for e in loc.integer_exponents)):
                kind = "apparent"
            else:
                kind = "singular"
            with mp.workdps(30):
                out.append(Singularity(pt, pt.mpf(), tuple(co), mult, kind, loc))
    return out


__all__ = ["Certificate", "NoOperatorWithinBounds", "Operator", "PRIMES", "Recognition", "Singularity",
           "recognize_holonomic", "singularities"]
