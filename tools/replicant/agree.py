"""Level agreement: (a) derived RS equations against (b) the families' equations.

  derived_rs_equations(spec)        level (a): replicate -> average-disorder ->
                                    introduce-order-parameters -> rs-ansatz ->
                                    replica-limit -> rs-stationarity (symbolic n)
  derive_rs(spec, n=None)           the same, returning (equations, State, Trace)
  pipeline_rs_equations(family)     level (b): the family module's
                                    rs_equations_symbolic(), the equations its
                                    numeric solver solves; family "sk",
                                    "perceptron" (spherical), "perceptron_ising"
                                    (binary couplings) or "hopfield"
  compare_equations(a, b, route=)   Agreement(agree, route, detail)
  equations_agree(a, b, **kw)       compare_equations(...).agree
  is_identically_zero(expr)         the symbolic zero test used below

Canonical form.  Each side is a list of sympy.Eq.
 1. Free symbols are matched by NAME (assumptions dropped, all taken positive:
    the physical domain; an identity proved there extends by analyticity to
    parameters such as h or kappa where both sides are analytic); every bound
    Gaussian variable is renamed to one symbol.
 2. Each equation becomes lhs - rhs, with expectations split by linearity and one
    round of Gaussian integration by parts on bare z factors (gauss.stein).
 3. Conjugates: a symbol on one side only (qhat, r) is eliminated through an
    equation of that side that is linear in it and has it in no expectation; the
    detail string records the substitution.
 4. Pairing: an equation of (a) and one of (b) are compared through a common
    TARGET symbol that appears linearly outside every expectation in both (each
    E_z[...] masked as an atom): v = F_a and v = F_b.  The numerator of
    F_a - F_b, which must be linear in the expectations, is merged into one
    E_z[g(z)].
 5. Route "symbolic": g == 0 identically (is_identically_zero: exp/erfc/log/Abs
    as generators, radicals with their defining relations, polynomial reduction).
    Route "symbolic+stein": g = mu (z h - h') with mu free of z, for h one of the
    z-dependent factors of g (a Stein boundary term, E[z h - h'] = 0 when h phi
    -> 0 at +-infinity, checked at |z| = 12 at a random point).  This is an
    identity GIVEN Stein's boundary condition: the condition is satisfied at the
    sampled point, not mechanised (no proof of the decay is produced).
    Route "numeric" (backup, or forced): |F_a - F_b| <= tol max(1, |F_b|) at
    `npoints` (>= 3) random points, each parameter uniform in (0.2, 0.8) (so
    this route never samples kappa < 0, h < 0 or any value outside that box),
    expectations by solve.gauss_expect at 16 digits.  With route="auto" every
    pair first gets a one-point numeric screen: a pair that differs there is
    rejected without the (slower) symbolic attempt.
    A False verdict comes from the numeric route (the symbolic route proves
    equality or is inconclusive), except when no canonical form exists (no
    common target, a conjugate that cannot be eliminated, counts differ: route
    "none") or the route was forced to "symbolic" (False = not proven).
Honest label: the equations compared are physics-level (replica n -> 0 and the
RS ansatz are not proven); agreement says both levels encode the same RS
saddle, not that RS is correct.
"""
import itertools
import random
from dataclasses import dataclass
from math import lcm

import mpmath as mp
import sympy as sp

from . import derive as D
from .core import Refusal, State, run_pipeline
from .gauss import GaussE, linear, numeric, rename_bound, stein
from .spec import ModelSpec

_Z = sp.Symbol("_z", real=True)
_Z0 = sp.Symbol("_z0", real=True)
_DPS = 16


# ---- level (a) ----------------------------------------------------------------------

def default_spec(family):
    """The specs the agreement battery derives (parameters left symbolic)."""
    from .derive_families import ALPHA, KAPPA
    if family == "sk":
        return ModelSpec("sk", "ising", "gaussian", params={"beta": D.BETA, "h": D.H_FIELD})
    if family in ("perceptron", "perceptron_ising"):
        variables = "ising" if family == "perceptron_ising" else "spherical"
        return ModelSpec("perceptron", variables, "gaussian", constraint="margin kappa",
                         params={"alpha": ALPHA, "kappa": KAPPA})
    if family == "hopfield":
        return ModelSpec("hopfield", "ising", "pm_one", params={"alpha": ALPHA, "beta": D.BETA})
    raise Refusal("precondition:supported-family", f"no default spec for {family!r}")


def derive_rs(spec: ModelSpec, n=None):
    n = sp.Symbol("n", positive=True) if n is None else n
    steps = [D.replicate(spec, n), D.average_disorder(), D.introduce_order_parameters(),
             D.rs_ansatz(), D.replica_limit(), D.rs_stationarity()]
    state, trace = run_pipeline(steps, State())
    return state["rs_equations"], state, trace


def derived_rs_equations(spec: ModelSpec) -> list:
    return derive_rs(spec)[0]


# ---- level (b) ----------------------------------------------------------------------

def pipeline_rs_equations(family: str) -> list:
    if family == "sk":
        from .families.sk import rs_equations_symbolic
    elif family in ("perceptron", "perceptron_ising"):
        from .families.perceptron import rs_equations_symbolic as _p
        return _p("ising" if family == "perceptron_ising" else "spherical")
    elif family == "hopfield":
        from .families.hopfield import rs_equations_symbolic
    else:
        raise Refusal("precondition:supported-family",
                      f"{family!r} has no symbolic RS equations "
                      "(sk, perceptron, perceptron_ising, hopfield)")
    return rs_equations_symbolic()


# ---- the zero test ---------------------------------------------------------------

_HYP = (sp.tanh, sp.cosh, sp.sinh, sp.coth, sp.sech, sp.csch)


def is_identically_zero(expr) -> bool:
    """True only if expr == 0 identically.  tanh/cosh/... are rewritten as exp;
    every radical base^(k/2) (numerator and denominator of the base separately,
    each up to its numeric content) becomes a symbol R with R^2 = base, also
    inside function arguments; every exp is split into monomials, each monomial
    up to a rational factor one generator; erfc, erf, log, Abs are one generator
    per (expanded) argument; the numerator is then reduced modulo the radical
    relations (a Groebner basis: distinct leading variables).  False means
    'not shown', not 'nonzero'."""
    e = sp.sympify(expr).rewrite(_HYP, sp.exp)
    e = e.xreplace({sp.pi: sp.Dummy("pi", positive=True)})
    rads, rels = {}, []

    def radical(base, k):
        c, prim = sp.factor_terms(sp.expand(base)).as_content_primitive()
        if prim == 1:
            return sp.sqrt(c)**k
        if prim not in rads:
            rads[prim] = sp.Dummy("R")
            rels.append(rads[prim]**2 - prim)
        return sp.sqrt(c)**k * rads[prim]**k

    def grad(p):
        k = int(2 * p.exp)
        num, den = sp.fraction(sp.together(p.base))
        return radical(num, k) / radical(den, k)
    e = e.replace(lambda y: y.is_Pow and y.exp.is_Rational and y.exp.q == 2
                  and not y.base.is_number, grad)
    split = {}
    for x in e.atoms(sp.exp):
        for term in sp.Add.make_args(sp.expand(x.args[0])):
            c, rest = sp.factor(term).as_coeff_Mul()
            split.setdefault(rest, set()).add(c)
    L = {r: lcm(*(int(c.q) for c in cs)) for r, cs in split.items()}
    X = {r: sp.Dummy("X") for r in split}

    def gexp(arg):
        out = sp.S(1)
        for term in sp.Add.make_args(sp.expand(arg)):
            c, rest = sp.factor(term).as_coeff_Mul()
            out *= X[rest]**int(c * L[rest])
        return out
    e = e.replace(lambda y: isinstance(y, sp.exp), lambda y: gexp(y.args[0]))
    Y = {}
    e = e.replace(lambda y: isinstance(y, (sp.erfc, sp.erf, sp.log, sp.Abs)),
                  lambda y: Y.setdefault((y.func, sp.expand(y.args[0])), sp.Dummy("Y")))
    num = sp.expand(sp.numer(sp.together(e)))
    if num == 0:
        return True
    if not rels:
        return False
    gens = list(rads.values()) + sorted(num.free_symbols - set(rads.values()), key=str)
    try:
        _, r = sp.reduced(num, rels, *gens)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return False
    return sp.expand(r) == 0


# ---- canonicalisation ----------------------------------------------------------------

@dataclass(frozen=True)
class Agreement:
    agree: bool
    route: str          # "symbolic" | "symbolic+stein" | "numeric" | "none"
    detail: str


def _canon(eqs):
    names = {}
    for e in eqs:
        for s in (e.lhs - e.rhs).free_symbols:
            names.setdefault(s.name, set()).add(s)
    sub = {s: sp.Symbol(nm, positive=True) for nm, ss in names.items() for s in ss}
    out = []
    for e in eqs:
        x = rename_bound(e.lhs - e.rhs, _Z).xreplace(sub)
        out.append(linear(stein(x)))
    return out


def _masked(fn, e):
    atoms = sorted(e.atoms(GaussE), key=str)
    dum = [sp.Dummy("G") for _ in atoms]
    r = fn(e.xreplace(dict(zip(atoms, dum))), dum)
    return r.xreplace(dict(zip(dum, atoms))) if isinstance(r, sp.Basic) else r


def _solve_linear(e, v):
    """v = F if e (expectations masked) is c1 v + c0 with c1, c0 != 0, else None."""
    def go(m, dum):
        try:
            p = sp.Poly(sp.expand(m), v)
        except sp.PolynomialError:
            return None
        if p.degree() != 1:
            return None
        c1, c0 = p.coeff_monomial(v), p.coeff_monomial(1)
        if c1 == 0 or c0 == 0:
            return None
        return -c0 / c1
    return _masked(go, e)


def _outside(e):
    """Symbols that occur outside every expectation."""
    return _masked(lambda m, dum: m.free_symbols - set(dum), e)


def _eliminate(es, extra, tag):
    es, notes = list(es), []
    for x in sorted(extra, key=lambda s: s.name):
        for k, e in enumerate(es):
            if any(x in g.free_symbols for g in e.atoms(GaussE)) or x not in _outside(e):
                continue
            sol = _solve_linear(e, x)
            if sol is None or sol.has(x):
                continue
            notes.append(f"eliminated {x} = {sol} ({tag})")
            es = [linear(stein(o.subs(x, sol))) for j, o in enumerate(es) if j != k]
            break
        else:
            return None, f"cannot eliminate {x} ({tag}): no equation linear in it outside E[...]"
    return es, "; ".join(notes)


def _target(ea, eb):
    common = sorted(_outside(ea) & _outside(eb), key=lambda s: s.name)
    for v in common:
        fa, fb = _solve_linear(ea, v), _solve_linear(eb, v)
        if fa is not None and fb is not None:
            return v, fa, fb
    return None


def _merged_integrand(fa, fb):
    """g(_Z) with E[g] = numerator of fa - fb, or None if not linear in E[...]."""
    d = fa - fb
    atoms = sorted(d.atoms(GaussE), key=str)
    dum = [sp.Dummy("G") for _ in atoms]
    num = sp.expand(sp.numer(sp.together(d.xreplace(dict(zip(atoms, dum))))))
    if not dum:
        return num
    p = sp.Poly(num, *dum)
    if p.total_degree() > 1:
        return None
    g = p.coeff_monomial(1)
    for k, atom in enumerate(atoms):
        g += p.coeff_monomial(tuple(int(j == k) for j in range(len(atoms)))) * atom.f
    return g


def _candidates(g):
    out = []
    for term in sp.Add.make_args(sp.expand(g, deep=False)):
        _, rest = term.as_independent(_Z, as_Add=False)
        h = sp.Mul(*(f for f in sp.Mul.make_args(rest)
                     if not (f == _Z or (f.is_Pow and f.base == _Z and f.exp.is_Integer))))
        if h.has(_Z) and h not in out:
            out.append(h)
    return out[:6]


def _boundary_ok(h, rng):
    pt = {s: sp.Rational(rng.randint(200, 800), 1000) for s in h.free_symbols - {_Z}}
    with mp.workdps(_DPS):
        phi = mp.exp(-72) / mp.sqrt(2 * mp.pi)
        for zv in (12, -12):
            try:
                v = complex(sp.N(h.subs(pt).subs(_Z, zv), _DPS))
            except (TypeError, ValueError):
                return False
            if not abs(v) * phi <= 1e-20:      # also rejects nan
                return False
    return True


def _symbolic_pair(fa, fb, rng):
    g = _merged_integrand(fa, fb)
    if g is None:
        return None
    if is_identically_zero(g):
        return "symbolic"
    for h in _candidates(g):
        B = _Z * h - sp.diff(h, _Z)
        if is_identically_zero(B):
            continue
        if is_identically_zero(g * B.subs(_Z, _Z0) - g.subs(_Z, _Z0) * B) and \
                _boundary_ok(h, rng):
            return "symbolic+stein"
    return None


def _numeric_pair(fa, fb, syms, rng, npoints, tol):
    worst = 0
    syms = [s for s in syms if s in (fa - fb).free_symbols] or syms
    for _ in range(npoints):
        pt = {s: sp.Rational(rng.randint(200, 800), 1000) for s in syms}
        try:
            va, vb = numeric(fa, pt, _DPS), numeric(fb, pt, _DPS)
        except (Refusal, ValueError, TypeError, ZeroDivisionError):
            return False, "evaluation failed"
        if isinstance(va, mp.mpc) or isinstance(vb, mp.mpc):
            return False, "complex value"
        with mp.workdps(_DPS):
            d = abs(va - vb) / max(1, abs(vb))
        worst = max(worst, d)
        if not d <= tol:
            return False, f"relative residual {mp.nstr(d, 3)} at {pt}"
    return True, f"max relative residual {mp.nstr(worst, 3)} at {npoints} points"


def compare_equations(a, b, *, route="auto", seed=0, npoints=3, tol=1e-12) -> Agreement:
    if route not in ("auto", "symbolic", "numeric"):
        raise ValueError(f"route {route!r}: use auto, symbolic or numeric")
    if npoints < 3:
        raise ValueError("the numeric route uses at least 3 points")
    if not a or not b:
        return Agreement(False, "none", f"equation count {len(a)} vs {len(b)}")
    ea, eb = _canon(a), _canon(b)
    sa = set().union(*(e.free_symbols for e in ea))
    sb = set().union(*(e.free_symbols for e in eb))
    ea, na = _eliminate(ea, sa - sb, "a")
    if ea is None:
        return Agreement(False, "none", na)
    eb, nb = _eliminate(eb, sb - sa, "b")
    if eb is None:
        return Agreement(False, "none", nb)
    notes = "; ".join(x for x in (na, nb) if x)
    if len(ea) != len(eb):
        return Agreement(False, "none", f"equation count after elimination "
                                        f"{len(ea)} vs {len(eb)}; {notes}")
    syms = sorted(set().union(*(e.free_symbols for e in ea + eb)), key=lambda s: s.name)
    rng = random.Random(seed)
    last = "no pairing with a common target symbol"
    for perm in itertools.permutations(range(len(eb))):
        pairs = [_target(ea[i], eb[j]) for i, j in enumerate(perm)]
        if any(p is None for p in pairs):
            continue
        routes, info = [], []
        for (i, j), (v, fa, fb) in zip(enumerate(perm), pairs):
            if route == "auto":       # one-point screen: a pair that already
                ok, why = _numeric_pair(fa, fb, syms, rng, 1, tol)   # differs is not
                if not ok:            # sent to the symbolic prover
                    info.append(f"(a)[{i}]~(b)[{j}] on {v}: numeric screen, {why}")
                    last = "; ".join(x for x in [notes] + info if x)
                    break
            r = _symbolic_pair(fa, fb, rng) if route != "numeric" else None
            if r is None and route != "symbolic":
                ok, why = _numeric_pair(fa, fb, syms, rng, npoints, tol)
                r = "numeric" if ok else None
                info.append(f"(a)[{i}]~(b)[{j}] on {v}: numeric, {why}")
                if not ok:
                    last = "; ".join(x for x in [notes] + info if x)
                    break
            elif r is None:
                info.append(f"(a)[{i}]~(b)[{j}] on {v}: not proven symbolically")
                last = "; ".join(x for x in [notes] + info if x)
                break
            else:
                info.append(f"(a)[{i}]~(b)[{j}] on {v}: {r}")
            routes.append(r)
        else:
            overall = ("numeric" if "numeric" in routes else
                       "symbolic+stein" if "symbolic+stein" in routes else "symbolic")
            return Agreement(True, overall, "; ".join(x for x in [notes] + info if x))
    deciding = "symbolic" if route == "symbolic" else "numeric"
    if last.startswith("no pairing"):
        deciding = "none"
    return Agreement(False, deciding, last)


def equations_agree(a, b, **kw) -> bool:
    return compare_equations(a, b, **kw).agree
