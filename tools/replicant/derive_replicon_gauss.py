"""Level (a): replicon rules for a formal Gaussian-integral energy and for a site with no
conjugate field (used by derive_replicon.replicon).

  gauss_prod_replicon(coef, g)   -> GaussReplicon(e2, parts, note)
  conjugate_free_site(site)      -> (chi_R, s_R, note)
  replicon_vector(n)             -> {(a, b): eps_ab}, sum_b eps_ab = 0 for every a

Energy  coef * LnGaussProd(g, q_ab, n) = coef ln Z_n(Q),  Z_n = E_lambda prod_a g(lambda_a),
lambda ~ N(0, Q), Q_aa = 1, Q_ab = q_ab.  Its Hessian in the pair variables q_ab (a < b) at
the RS point, restricted to the replicon direction, is produced by these rules, each applied
by the code below (nothing typed in):
  1. gauss.price: d/dq_ab E F = E[d^2 F/dlambda_a dlambda_b]; twice for the second variation.
     On four distinguished replicas with F = g(l1) g(l2) g(l3) g(l4) sympy differentiates;
     each slot then holds g^(k)(l_i), k its multiplicity in the two pairs.
  2. RS split (gauss.rs_split): lambda_a = sqrt(q) t + sqrt(1-q) z_a; at fixed t the slots
     are independent, E_z g^(k)(lambda) = M_k(t) by gauss.slot_moments (derivative in the
     conditional mean h = sqrt(q) t of the closed form M_0 = E_z g).
  3. Replica measure, n -> 0: <prod_slots g^(k_i)> = E_t[M_0^n prod_i M_{k_i}/M_0] / E_t[M_0^n]
     (the other n - 4 replicas give M_0 each) -> E_t[prod_i R_{k_i}], R_k = M_k / M_0, at
     n -> 0 (the same continuation as gauss.replica_limit; physics-level).
  4. Hessian of ln Z: <d_ab d_cd> - <d_ab><d_cd> over the 6 pairs of the four replicas,
     applied to eps (the column H eps, with D_eps = sum_cd eps_cd d_cd).
  5. Replicon projection: eps = replicon_vector(4) is checked to be an eigenvector
     (H eps = lambda eps, symbolically) and lambda = eps.H.eps / eps.eps is read off;
     the four-replica matrix carries the RS classes (same pair, one shared index, disjoint)
     whose replicon eigenvalue does not depend on n.
  The linear combination of E_t terms is merged and factored (sympy.factor) before the R_k
  are substituted.  Result: E''_R = coef E_t[(R_2 - R_1^2)^2].
Site    a site with no conjugate (derive.SITES entry with conjugate=False, e.g. "spherical"):
  its registered trace is evaluated at EXPLICIT n: the pair Hessian column at n = 4 (eps
  checked to be an eigenvector, s_R its eigenvalue) and the second derivative along
  q_ab = q + delta eps_ab at n = 5, which must equal s_R (n-independent, so its n -> 0
  value; two values of n are evidence, not a proof, of n-independence).  The
  conjugate-free analogue of the Ising rule is chi_R = -1/s_R (stationarity in q of the
  site term replaces the elimination of qhat).
Honest label: rules 1, 2, 5 are exact identities; rule 3 and the use of the RS replicon as
the stability criterion are physics-level (replica n -> 0), as everywhere in level (a).
"""
import itertools
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from . import derive as D
from . import gauss
from .core import Refusal

_PAIRS4 = tuple(itertools.combinations(range(1, 5), 2))
_T = sp.Symbol("t", real=True)
_Z = sp.Symbol("z", real=True)
Et = sp.Function("E_t")           # formal n -> 0 replica-measure average over t
H_MEAN = sp.Symbol("h", real=True)   # display: conditional mean sqrt(q) t of a slot


@dataclass(frozen=True)
class GaussReplicon:
    e2: sp.Expr          # E''_R in the RS unknowns and parameters (GaussE over t)
    parts: tuple         # (title, sympy Eq) pairs for display (derive_tex)
    note: str


def replicon_vector(n):
    """eps_12 = eps_34 = 1, eps_13 = eps_24 = -1, every other pair 0 (needs n >= 4): each
    row sums to 0, so eps lies in the replicon sector (orthogonal to the longitudinal and
    anomalous ones); checked here."""
    if n < 4:
        raise Refusal("precondition:replicon-sector", f"n = {n}: the replicon needs n >= 4")
    eps = {p: 0 for p in itertools.combinations(range(1, n + 1), 2)}
    eps.update({(1, 2): 1, (3, 4): 1, (1, 3): -1, (2, 4): -1})
    for a in range(1, n + 1):
        assert sum(v for p, v in eps.items() if a in p) == 0
    return eps


def _project(v, eps, what):
    """lambda with v = H eps = lambda eps (checked symbolically); v the Hessian column
    along eps, {pair: (H eps)_pair}."""
    norm = sum(e**2 for e in eps.values())
    lam = sp.expand(sum(eps[p] * v[p] for p in eps) / norm)
    if any(sp.expand(sp.cancel(v[p] - lam * eps[p])) != 0 for p in eps):
        raise Refusal("precondition:replicon-eigenvector",
                      f"{what}: eps is not an eigenvector of the Hessian (not RS-invariant)")
    return lam


def _et(poly):
    """E_t of a polynomial in the R_k placeholders, by linearity over its monomials."""
    poly = sp.expand(poly)
    gens = sorted(poly.free_symbols, key=str)
    if not gens:
        return poly * Et(sp.S(1))
    return sp.Add(*(c * Et(sp.Mul(*(x**k for x, k in zip(gens, mon))))
                    for mon, c in sp.Poly(poly, *gens).terms()))


def _rational_or_same(e):
    c = sp.cancel(e)
    return c if c.is_rational_function(_T) else e


@lru_cache(maxsize=64)
def gauss_prod_replicon(coef, g, q=D.Q_RS):
    lam = sp.symbols("lambda_1:5", real=True)
    gf = sp.Function("g")
    F = sp.Mul(*(gf(x) for x in lam))
    R = sp.symbols("R_0:5", real=True)

    def slots(expr):          # rule 2 + 3: g^(k)(l_i) -> R_k, g(l_i) -> R_0 = 1
        expr = sp.expand(expr)
        expr = expr.xreplace({d: R[d.derivative_count] for d in expr.atoms(sp.Derivative)})
        return expr.xreplace({gf(x): 1 for x in lam})
    eps = replicon_vector(4)
    first = {p: _et(slots(gauss.price(F, lam, [p]))) for p in _PAIRS4}
    # (H eps)_p = <d_p D_eps F> - <d_p F> <D_eps F>, D_eps = sum_r eps_r d_r (rule 1 twice)
    along = sp.Add(*(eps[r] * gauss.price(F, lam, [r]) for r in _PAIRS4))
    mean_along = sp.Add(*(eps[r] * first[r] for r in _PAIRS4))
    col = {p: sp.expand(_et(slots(gauss.price(along, lam, [p]))) - first[p] * mean_along)
           for p in _PAIRS4}
    lam_e = _project(col, eps, "LnGaussProd energy")
    atoms = sorted(lam_e.atoms(Et), key=str)
    dum = [sp.Dummy() for _ in atoms]
    lin = sp.Poly(lam_e.xreplace(dict(zip(atoms, dum))), *dum)
    if lin.total_degree() > 1 or lin.coeff_monomial(1) != 0:
        raise Refusal("precondition:replicon-linear",
                      f"replicon of the energy is not linear in E_t terms: {lam_e}")
    body = sp.factor(sp.Add(*(lin.coeff_monomial(d) * a.args[0] for d, a in zip(dum, atoms))))
    notes = []
    M = gauss.slot_moments(g, gauss.rs_split(q, _T, _Z), _Z, 2, notes)
    sub = {R[k]: M[k] / M[0] for k in (1, 2)}
    # a factor of the factored integrand that cancels to a rational function of t is
    # kept cancelled (the Gaussian weight: the expectation then closes); others as they are
    integrand = sp.Mul(*(_rational_or_same(b.xreplace(sub))**x for b, x in
                         (f.as_base_exp() for f in sp.Mul.make_args(body))))
    e2 = coef * gauss.GaussE(integrand, _T)
    G0 = sp.Function("G_0")
    m0 = gauss.slot_moments(g, H_MEAN + sp.sqrt(1 - q) * _Z, _Z, 0)[0]
    parts = (("Conditional slot weight (RS split, closed form; h = sqrt(q) t)",
              sp.Eq(G0(H_MEAN), m0)),
             ("Slot ratios (derivatives in the conditional mean)",
              [sp.Eq(R[k], sp.Derivative(G0(H_MEAN), H_MEAN, k) / G0(H_MEAN)) for k in (1, 2)]),
             ("Energy replicon eigenvalue (covariance derivative, n -> 0, projection; "
              "E_t averages t ~ N(0, 1) with the R_k at h = sqrt(q) t)",
              sp.Eq(sp.Symbol("E''_R"), coef * Et(body))),
             ("Energy replicon eigenvalue with the slot ratios substituted",
              sp.Eq(sp.Symbol("E''_R"), e2)))
    note = ("energy coef ln E_lambda prod_a g(lambda_a): Price covariance derivative "
            "(gauss.price) on four replicas, RS split and slot moments "
            "(gauss.slot_moments), n -> 0 of the replica measure, eigenvector projection "
            f"-> E''_R = {coef} E_t[{body}], R_k = M_k/M_0"
            + "".join(f" ({x})" for x in notes))
    return GaussReplicon(e2, parts, note)


@lru_cache(maxsize=16)
def conjugate_free_site(site, q=D.Q_RS):
    """(chi_R, s_R, note) for a SITES entry without conjugate, from its explicit-n trace:
    the pair Hessian column H eps at n = 4 (eps checked to be an eigenvector), the Rayleigh
    quotient along eps at n = 4 and n = 5 (equal: n-independent)."""
    eps = replicon_vector(4)
    x = {p: sp.Dummy(f"q_{p[0]}{p[1]}") for p in eps}
    tr = site.trace(4, x, {p: sp.S(0) for p in eps}, 0)
    rs = {v: q for v in x.values()}
    along = sp.Add(*(eps[r] * sp.diff(tr, x[r]) for r in eps))
    col = {p: sp.cancel(sp.diff(along, x[p]).subs(rs)) for p in eps}
    s4 = sp.factor(_project(col, eps, "site trace at n = 4"))
    d = sp.Dummy("delta")
    eps5 = replicon_vector(5)
    tr5 = site.trace(5, {p: q + d * v for p, v in eps5.items()}, {p: sp.S(0) for p in eps5}, 0)
    s5 = sp.cancel(sp.diff(tr5, d, 2).subs(d, 0) / sum(v**2 for v in eps5.values()))
    if sp.cancel(s4 - s5) != 0:
        raise Refusal("precondition:replicon-n-independent",
                      f"site replicon {s4} (n = 4) != {s5} (n = 5)")
    if s4 == 0:
        raise Refusal("precondition:nondegenerate-site", "site replicon s_R = 0: chi_R = -1/s_R "
                      "undefined (the site does not fix q in the replicon direction)")
    return (sp.factor(-1 / s4), s4,
            f"conjugate-free site: explicit-n replicon of the registered trace, s_R = {s4} "
            "(eigenvector at n = 4, equal at n = 5), chi_R = -1/s_R")
