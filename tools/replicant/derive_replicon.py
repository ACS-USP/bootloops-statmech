"""Level (a): the replicon (de Almeida-Thouless) eigenvalue, where it is derivable.

  replicon(state)                     -> Replicon(expr | None, note, route=...)
  derived_replicon(spec, rs_solution) -> Replicon(expr, note, value, route)

`state` is the State of agree.derive_rs.  The n-replica functional is
  G = energy(q_ab) - sum_{a<b} qhat_ab q_ab + site(qhat_ab)
(derive.Functional).  Around the RS saddle, in the replicon sector
(delta q_ab with sum_b delta q_ab = 0) the second variation is
  E''_R dq^2 - 2 dq dqhat + chi_R dqhat^2,
with E''_R the replicon eigenvalue of the energy Hessian and chi_R that of the
site Hessian.  Eliminating dqhat (its own stationarity, dqhat = dq/chi_R) leaves
(E''_R - 1/chi_R) dq^2, so the RS saddle is replicon-stable iff
  lambda = 1 - E''_R chi_R > 0.
A site without conjugate (the spherical volume) enters as s_R dq^2 directly; with
chi_R = -1/s_R the same lambda results.  Rules, each applied only where its
precondition is checked:
  energy, term by term (E''_R is additive; terms free of q contribute 0):
    pair-separable  (q-free) + sum_{a<b} f(q_ab): the Hessian is diagonal, E''_R = f''(q).
                    (SK: f = beta^2 q^2/2, E''_R = beta^2.)
    Gaussian        coef LnGaussProd(g, q_ab, n), coef free of q (perceptron, Hopfield):
                    derive_replicon_gauss.gauss_prod_replicon (Price covariance
                    derivative, RS split, n -> 0 of the replica measure, eigenvector
                    projection) -> coef E_t[(R_2 - R_1^2)^2].
    anything else (a LogDet energy, a covariance other than q_ab, ...): not derived.
  site:
    Ising           one Ising site trace (derive.IsingSiteRS, recorded by rs_ansatz):
                    chi_R = P - 2Q + R of the connected 4-point function of s^a s^b in
                    the replica measure, P = 1 - q^2, Q = q - q^2, R = r - q^2 with r =
                    <s_a s_b s_c s_d>; at RS with Hubbard-Stratonovich (q = E tanh^2 u,
                    r = E tanh^4 u) this is E_z[(1 - tanh^2 u)^2] = E_z[((ln single)''(u))^2],
                    computed here by differentiating the recorded single-replica factor.
                    (SK: lambda = 1 - beta^2 E sech^4(beta(h + sqrt(q) z)), de
                    Almeida-Thouless 1978.)  This chi_R is a hand-written identity (the
                    4-point reduction is stated, the differentiation is the code's).
    no conjugate    derive_replicon_gauss.conjugate_free_site: explicit-n replicon of the
                    registered SITES trace (spherical: s_R = -1/(1-q)^2).
  Extra replica-indexed order parameters (Hopfield m_a) are not varied: at RS their
  couplings to q_ab, qhat_ab are vectors x_a (one value for a in the pair, one outside),
  whose contraction with a replicon eps vanishes (sum_b eps_ab = 0).  ARGUED, NOT
  MECHANISED; the m-mode itself is a separate (longitudinal) stability question.
None means UNKNOWN (not derived), never "stable".
Honest label: physics-level (replica n -> 0, RS); the sign convention of lambda is
checked against families.sk.sk_replicon (tests/test_solve_derived.py) and the
families' perceptron and Hopfield replicons (tests/test_replicon_derived.py).
"""
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from . import derive as D
from .core import Refusal
from .derive_replicon_gauss import conjugate_free_site, gauss_prod_replicon
from .gauss import GaussE

_NOT = "replicon not derived for this family: "


@dataclass(frozen=True)
class Replicon:
    expr: object          # sympy expression in the RS unknowns and parameters, or None
    note: str
    value: object = None  # derived_replicon: lambda at the solution (mpf), else None
    route: str = ""       # the rules used, e.g. "gauss-covariance+ising-site"
    parts: tuple = ()     # (title, sympy object) display pieces (derive_tex)


def _pair_term(term, F, Q):
    """f(Q) with term = sum_{a<b} f(q_ab), or (None, reason)."""
    c, rest = term.as_independent(sp.Sum, as_Add=False)
    if not (isinstance(rest, sp.Sum)
            and rest.limits == ((D._b, D._a + 1, F.n), (D._a, 1, F.n))):
        return None, f"energy term {term} is not a pair sum over a<b"
    body = (c * rest.function).xreplace({F.q[D._a, D._b]: Q})
    if body.has(F.q) or body.has(D._a) or body.has(D._b) or body.has(sp.Indexed):
        return None, f"pair-sum summand {rest.function} is not a function of q_ab alone"
    return body, ""


def _gauss_term(term, F):
    """(coef, g) with term = coef LnGaussProd(g, q_ab, n), or (None, reason)."""
    c, rest = term.as_independent(D.LnGaussProd, as_Add=False)
    if not (rest.func == D.LnGaussProd and not c.has(F.q)):
        return None, f"energy term {term} is not coef * LnGaussProd(...)"
    g, cov, nn = rest.args
    if cov != F.q[D._a, D._b] or nn != F.n:
        return None, (f"LnGaussProd covariance {cov} is not q_ab (no chain rule for a "
                      "covariance that is a function of q_ab)")
    return (c, g), ""


def _energy(F):
    """(E''_R, notes, parts, route) or (None, reason, (), "")."""
    Q = sp.Dummy("Q", positive=True)
    e2, notes, parts, route = sp.S(0), [], (), set()
    if F.energy.has(D.LogDet):
        return None, "energy holds a LogDet (no replicon rule for it)", (), ""
    for term in sp.Add.make_args(sp.expand(F.energy)):
        if not term.has(F.q):
            continue
        if term.has(D.LnGaussProd):
            cg, why = _gauss_term(term, F)
            if cg is None:
                return None, why, (), ""
            r = gauss_prod_replicon(*cg)
            e2, parts = e2 + r.e2, parts + r.parts
            notes.append(r.note)
            route.add("gauss-covariance")
            continue
        f, why = _pair_term(term, F, Q)
        if f is None:
            return None, why + " (only pair-separable or LnGaussProd energies have a rule)", (), ""
        d2 = sp.diff(f, Q, 2).xreplace({Q: D.Q_RS})
        e2 += d2
        notes.append(f"pair-separable energy term, E''_R += f''(q) = {d2}")
        route.add("pair-separable")
    return e2, notes, parts, "+".join(sorted(route))


def _site(F, P):
    """(chi_R, note, route) or (None, reason, "")."""
    if F.conjugate and len(P.sites) == 1:
        site = P.sites[0]
        U = sp.Dummy("U", real=True)
        d2 = sp.diff(sp.log(site.single(U)), U, 2).xreplace({U: site.u})
        return (GaussE(d2**2, site.z),
                f"Ising site chi_R = E_z[((ln single)''(u))^2], u = {site.u}", "ising-site")
    if not F.conjugate and not P.sites:
        try:
            chi, _, note = conjugate_free_site(D.SITES[F.spec.variables])
        except Refusal as r:
            return None, f"site {F.spec.variables}: {r.reason}: {r.detail}", ""
        return chi, note, "explicit-n-site"
    return None, ("needs exactly one Ising site trace with its conjugate qhat, or a site "
                  "without conjugate"), ""


def replicon(state) -> Replicon:
    F, P = state["functional"], state["rs_free"]
    chi, snote, sroute = _site(F, P)
    if chi is None:
        return Replicon(None, _NOT + snote)
    e2, enotes, parts, eroute = _energy(F)
    if e2 is None:
        return Replicon(None, _NOT + enotes)
    extra = ("; extra order parameters " + ", ".join(sorted(F.extra))
             + " not varied (decoupled from the replicon sector: argued, not mechanised)"
             if F.extra else "")
    note = ("replicon lambda = 1 - E''_R chi_R: " + "; ".join(enotes) + "; " + snote + extra)
    lam = sp.Symbol("lambda_R")
    chi_sym, e2_sym = sp.Symbol("chi_R"), sp.Symbol("E''_R")
    parts = parts + (("Site replicon susceptibility", sp.Eq(chi_sym, chi)),
                     ("Replicon eigenvalue of the RS saddle (positive: RS replicon-stable)",
                      sp.Eq(lam, 1 - e2_sym * chi_sym)))
    return Replicon(1 - e2 * chi, note, route=f"{eroute or 'q-free-energy'}+{sroute}",
                    parts=parts)


def _residual(num, saddle_tol, qtol):
    """|equation| at the point by the independent rule; only a failing value is re-decided
    by _derived_num.checked (so a coarse rule cannot refuse a true saddle)."""
    from ._derived_num import checked
    r = abs(num.audit([]))
    return r if r < mp.mpf(saddle_tol) else abs(checked(num, [], qtol)[0])


def derived_replicon(spec, rs_solution, *, dps=20, saddle_tol="1e-9") -> Replicon:
    """lambda_R of `spec` at an RS solution: a solve_derived.DerivedRSSolution, or a
    mapping {name: value} of the unknowns and parameters (by symbol name).  A mapping is
    first checked to be a root of the derived RS equations (independent quadrature,
    max |lhs - rhs| < saddle_tol, else Refusal precondition:rs-saddle); saddle_tol=None
    skips that check (a probe of the formula off the saddle).  Every number is
    _derived_num.checked: two quadrature rules agree, else fine-panel rules, else Refusal
    no-convergence:quadrature.  A family without a rule returns value None and the
    reason in `note`."""
    from ._derived_num import Num, checked, safe
    from .agree import derive_rs
    from .gauss import _sympy_point, linear, stein
    from .solve_derived import DerivedRSSolution, lift_numeric_beta
    audited = isinstance(rs_solution, DerivedRSSolution)
    vals = ({**rs_solution.params, **rs_solution.values} if audited else dict(rs_solution))
    spec, vals, _ = lift_numeric_beta(spec, vals)
    eqs, state, _ = derive_rs(spec)
    rep = replicon(state)
    if rep.expr is None:
        return rep
    syms = {s.name: s for e in [rep.expr] + [x.lhs - x.rhs for x in eqs]
            for s in e.free_symbols}
    missing = sorted(set(syms) - set(vals))
    if missing:
        raise Refusal("precondition:complete-point", f"no value for {missing}")
    qtol = max(mp.mpf("1e-12"), mp.mpf(10)**(2 - dps))
    with mp.workdps(dps + 3):
        point = _sympy_point({syms[k]: v for k, v in vals.items() if k in syms}, dps + 15)
        if saddle_tol is not None and not audited:
            res = max(_residual(Num(linear(stein(safe(e.lhs - e.rhs).subs(point))), [], dps),
                                saddle_tol, qtol) for e in eqs)
            if not res < mp.mpf(saddle_tol):
                raise Refusal("precondition:rs-saddle",
                              f"the point is not a root of the derived RS equations: "
                              f"residual {mp.nstr(res, 3)} >= {saddle_tol}")
        value, qerr, rules = checked(Num(safe(rep.expr).subs(point), [], dps), [], qtol)
    note = rep.note + (f"; value cross-checked by two quadrature rules, {rules} "
                       f"(|difference| {mp.nstr(qerr, 3)} <= {mp.nstr(qtol, 3)})")
    return Replicon(rep.expr, note, value, rep.route, rep.parts)
