"""Generic numerical solution of DERIVED RS equations (level (a) -> numbers).

  solve_derived_rs(spec, *, branch=None, seed="0.5", seeds=None, dps=20, **params)
      -> DerivedRSSolution(values, free_energy, residual, equations_used, trace,
                           stable, caveat, branch, entropy, method, note)

What it does, with no family name anywhere in this module (everything
model-specific arrives through agree.derive_rs(spec): the stationarity
equations d phi / d p = 0, the declared unknowns p = RSFree.params, and phi):
 1. Parameters.  Every free symbol of the equations that is not a declared
    unknown needs a value: from `params` (by symbol name; strings or numbers),
    else a Refusal.  Numbers given inside spec.params are already in the
    derivation.
 2. Canonical form.  lhs - rhs, expectations split by linearity, one round of
    Gaussian integration by parts on bare z factors (gauss.linear, gauss.stein),
    then merged into one expectation per equation (gauss._merge_linear), which
    also cancels the sqrt(qhat)/sqrt(qhat) that integration by parts leaves.
 3. Elimination.  Unknowns are tried in reverse declared order (conjugates and
    extras before the primary overlap): an unknown x is eliminated through an
    equation in which it occurs linearly and outside every expectation,
    x = S(others), when S does not contain x and no earlier S contains x
    (an acyclic evaluation order).  Eliminated values are computed numerically
    from the remaining unknowns, never substituted symbolically (no nested
    expectations).  SK: qhat = beta^2 q; Ising perceptron: qhat from the d/dq
    equation.
 4. Remaining unknowns u.  If each remaining equation can be rearranged as
    u_k = F_k(u) (u_k linear outside the expectations; any such rearrangement
    has the same roots, and the residual audit checks the original system):
      one unknown   solve.fixed_point(F, seed, branch=branch): branch-aware,
                    branches "trivial" (u = 0, F(0) = 0), "nontrivial", ...
      several       damped iteration u <- (u + F(u))/2 from `seeds`, polished
                    by mp.findroot (mdnewton); branch "trivial" if all u = 0.
    Otherwise (e.g. the spherical perceptron, q/(2(1-q)^2) + alpha E[...] = 0)
    mp.findroot on the residual from `seeds` (secant for one unknown).
    Default seeds: (0.5, 0.25, 0.75, 0.1, 0.9) for every unknown.
 5. Audit.  max |lhs - rhs| over ALL derived equations (eliminated ones too) at
    the solution must be < 1e-12, else Refusal("no-solution").
 6. Outputs.  free_energy = phi at the solution, phi = lim (1/(N n)) ln E Z^n
    (RS): -beta f for a thermal model, the entropy (log volume / log count per
    variable) for a constraint-satisfaction model such as the perceptron.
    entropy: phi - beta d phi/d beta when phi depends on a symbol named beta
    (stationarity makes the partial the total derivative), phi itself when no
    temperature enters, None when beta was fixed numerically inside spec.params.
    stable: False when the variables are discrete (spec.variables "ising": a
    counting measure, entropy >= 0) and the RS entropy is negative (RS is then
    wrong, as for the Ising perceptron beyond alpha ~ 0.83); otherwise None.
    The replicon (de Almeida-Thouless) eigenvalue is NOT derived at level (a):
    the RS steps reduce the functional on the RS manifold only, so the
    transverse Hessian (SK: 1 - beta^2 E sech^4) is not available here and no
    stability is claimed from it (the family modules compute it at level (b)).
    No capacity or threshold is computed: this solves the RS equations at the
    given parameters only.

Numerics: mpmath at `dps` (default 20) under mp.workdps (the caller's precision
is restored); every GaussE is lambdified once (cse) and integrated by
solve.gauss_expect (tanh-sinh on [-inf, -6, 0, 6, inf]); erfc is kept as
written (sympy would turn erfc(-x) into 2 - erfc(x), losing all digits in the
tail and dividing by 0 in phi/H).  The quadrature is not
validated for sharply varying integrands (e.g. the perceptron at q -> 1); the
audit and the tests against the family solvers are the check.
Honest label (CAVEAT): the equations are physics-level (replica n -> 0, RS
ansatz); a solution of them is an RS saddle, not a proof that RS is correct.
"""
from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

from . import agree
from .core import Refusal, Trace, TraceEntry
from .families.common import CAVEAT
from .gauss import GaussE, _merge_linear, _sympy_point, linear, stein
from .solve import fixed_point, gauss_expect
from .spec import ModelSpec

_DEFAULT_SEEDS = ("0.5", "0.25", "0.75", "0.1", "0.9")
_AUDIT = "1e-12"
_DISCRETE = ("ising",)          # spec.variables with a counting measure (entropy >= 0)


@dataclass(frozen=True)
class DerivedRSSolution:
    values: dict                 # unknown name -> mpf (eliminated ones included)
    free_energy: object          # phi at the solution (see module docstring), or None
    residual: object             # max |lhs - rhs| over all derived equations
    equations_used: list         # the derived sympy equations (symbolic parameters)
    trace: Trace                 # the derivation's trace plus a "solve-derived" entry
    stable: bool | None
    caveat: str = CAVEAT
    branch: str = ""
    entropy: object = None
    method: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)


# ---- compiled numeric evaluation ---------------------------------------------------

class _Erfc(sp.Function):
    """erfc kept as written: sympy rewrites erfc(-x) as 2 - erfc(x) (e.g. once
    kappa = 0 is substituted), which loses every digit where erfc(x) -> 0
    (H(x) -> 1 far in the Gaussian tail) and divides by 0 in phi/H.  Same
    derivative; evaluated by mpmath.erfc."""
    def fdiff(self, argindex=1):
        u = self.args[0]
        return -2 * sp.exp(-u**2) / sp.sqrt(sp.pi)


def _safe(expr):
    return sp.sympify(expr).replace(lambda e: isinstance(e, sp.erfc),
                                    lambda e: _Erfc(e.args[0]))


_MODULES = [{"_Erfc": mp.erfc}, "mpmath"]


class _Num:
    """expr(args) numerically: each GaussE atom by gauss_expect, the rest by an
    mpmath lambdify.  Returns mpf, or mpc when the value is not real."""

    def __init__(self, expr, args, dps):
        self.dps = dps
        atoms = sorted(expr.atoms(GaussE), key=str)
        if len(atoms) > 1:
            expr = _merge_linear(expr, atoms)
            atoms = sorted(expr.atoms(GaussE), key=str)
        if any(a.f.atoms(GaussE) for a in atoms):
            raise Refusal("precondition:no-nested-expectation", str(expr))
        dum = [sp.Dummy(f"G{k}") for k in range(len(atoms))]
        self.inner = [sp.lambdify([a.z] + list(args), a.f, _MODULES, cse=True)
                      for a in atoms]
        self.outer = sp.lambdify(dum + list(args), expr.xreplace(dict(zip(atoms, dum))),
                                 _MODULES)

    def __call__(self, vals):
        g = [gauss_expect(lambda z, f=f: f(z, *vals), dps=self.dps) for f in self.inner]
        v = self.outer(*g, *vals)
        if isinstance(v, mp.mpc) and v.imag == 0:
            v = v.real
        return v if isinstance(v, (mp.mpf, mp.mpc)) else mp.mpf(v)


def _masked(e, fn):
    atoms = sorted(e.atoms(GaussE), key=str)
    dum = [sp.Dummy("G") for _ in atoms]
    back = dict(zip(dum, atoms))
    r = fn(sp.expand(e.xreplace(dict(zip(atoms, dum)))), set(dum))
    return None if r is None else tuple(sp.sympify(x).xreplace(back) for x in r)


def _linear_outside(e, x):
    """(c1, c0) with e = c1 x + c0, x not in c1 and absent from every expectation
    of c1; c0 may hold x inside expectations.  None otherwise."""
    def go(m, dum):
        try:
            p = sp.Poly(m, x)
        except sp.PolynomialError:
            return None
        if p.degree() != 1:
            return None
        c1, c0 = p.coeff_monomial(x), p.coeff_monomial(1)
        return None if c1 == 0 or c1.has(x) else (c1, c0)
    return _masked(e, go)


def _in_expectation(e, x):
    return any(x in g.free_symbols for g in e.atoms(GaussE))


# ---- the solver ---------------------------------------------------------------------

def solve_derived_rs(spec: ModelSpec, *, branch=None, seed="0.5", seeds=None, dps=20,
                     **params) -> DerivedRSSolution:
    eqs, state, trace = agree.derive_rs(spec)
    phi = state["rs_free"].expr
    unknowns = list(state["rs_free"].params)
    with mp.workdps(dps + 5):
        return _solve(spec, eqs, phi, unknowns, trace, branch, seed,
                      tuple(seeds or _DEFAULT_SEEDS), dps, params)


def _solve(spec, eqs, phi, unknowns, trace, branch, seed, seeds, dps, params):
    unames = {u.name for u in unknowns}
    free = {s for e in eqs for s in (e.lhs - e.rhs).free_symbols} | phi.free_symbols
    by_name = {s.name: s for s in free if s.name not in unames}
    missing = sorted(set(by_name) - set(params))
    if missing:
        raise Refusal("precondition:complete-point",
                      f"no value for parameter(s) {missing} (unknowns: {sorted(unames)})")
    unused = sorted(set(params) - set(by_name))
    point = _sympy_point({by_name[k]: v for k, v in params.items() if k in by_name}, dps + 5)
    canon = [linear(stein(_safe(e.lhs - e.rhs).subs(point))) for e in eqs]

    # elimination of unknowns that occur linearly outside the expectations
    elim, rest, notes = [], list(range(len(canon))), []
    for x in reversed(unknowns):
        if len(elim) == len(unknowns) - 1:       # keep at least one unknown to solve for
            break
        for k in rest:
            e = canon[k]
            if _in_expectation(e, x):
                continue
            lin = _linear_outside(e, x)
            if lin is None:
                continue
            S = sp.expand(-lin[1] / lin[0])
            if S.has(x) or any(Sj.has(x) for _, Sj in elim):
                continue
            elim.append((x, S))
            rest.remove(k)
            notes.append(f"eliminated {x} = {S}")
            break
    left = [u for u in unknowns if u not in {x for x, _ in elim}]
    if len(left) != len(rest):
        raise Refusal("precondition:square-system",
                      f"{len(rest)} equation(s) for unknowns {left} after elimination")
    order = left + [x for x, _ in elim]          # argument order of every compiled function
    comp_elim = [(x, _Num(S, order, dps)) for x, S in elim]

    def full(u):
        vals = list(u) + [mp.mpf(0)] * len(elim)
        for j, (x, f) in enumerate(comp_elim):
            vals[len(left) + j] = f(vals)
        return vals

    # fixed-point form u_k = F_k(u), one remaining equation per remaining unknown
    forms, used = [], set()
    for k in rest:
        for x in left:
            if x in used:
                continue
            lin = _linear_outside(canon[k], x)
            if lin is not None:
                forms.append((x, sp.expand(-lin[1] / lin[0])))
                used.add(x)
                break
        else:
            forms = None
            break
    if forms is not None:
        forms.sort(key=lambda p: left.index(p[0]))
        Fs = [_Num(F, order, dps) for _, F in forms]
        u, branch_name, seen, method = _fixed_point(Fs, full, left, branch, seed, seeds, dps)
    else:
        Rs = [_Num(canon[k], order, dps) for k in rest]
        u, branch_name, seen, method = _roots(Rs, full, left, branch, seeds, dps)

    vals = full(u)
    res_fns = [_Num(e, order, dps) for e in canon]
    residual = max(abs(f(vals)) for f in res_fns)
    if not residual < mp.mpf(_AUDIT):
        raise Refusal("no-solution", f"audit: max |residual| = {mp.nstr(residual, 3)} "
                                     f">= {_AUDIT} at {dict(zip(map(str, order), vals))}")
    values = {str(x): v for x, v in zip(order, vals)}

    phi = _safe(phi)
    phi_n = phi.subs(point)
    fe = _real(_Num(phi_n, order, dps)(vals))
    betas = [s for s in phi.free_symbols if s.name == "beta"]
    if betas:
        b = betas[0]
        ent = _real(_Num((phi - b * sp.diff(phi, b)).subs(point), order, dps)(vals))
        ent_note = "entropy = phi - beta d phi/d beta"
    elif "beta" in spec.params:
        ent, ent_note = None, "entropy not computed: beta fixed numerically in spec.params"
    else:
        ent, ent_note = fe, "entropy = phi (no temperature: log count / volume per variable)"
    stable, stab_note = None, "replicon not derived at level (a): stable = None"
    if ent is not None and spec.variables in _DISCRETE and ent < 0:
        stable = False
        stab_note = (f"RS entropy {mp.nstr(ent, 6)} < 0 for discrete variables "
                     f"({spec.variables}): RS is wrong here; replicon not derived")
    note = "; ".join(notes + [method, f"branches seen {list(seen)}", ent_note, stab_note]
                     + ([f"unused parameters {unused}"] if unused else [])
                     + ["no capacity / threshold computed"])
    if branch_name == "trivial" and branch is None:
        note += ("; only the trivial branch (all unknowns 0, a fixed point of the map) "
                 "was found from the seeds")
    out = Trace()
    for e in trace.entries:
        out.add(e)
    out.add(TraceEntry("solve-derived", None,
                       f"branch {branch_name}; residual {mp.nstr(residual, 3)}; {note}"))
    return DerivedRSSolution(values, fe, residual, list(eqs), out, stable, CAVEAT,
                             branch_name, ent, method, note,
                             {k: params[k] for k in by_name})


def _real(v):
    if isinstance(v, mp.mpc):
        raise Refusal("no-solution", f"complex value {v} at the solution")
    return v


def _fixed_point(Fs, full, left, branch, seed, seeds, dps):
    if len(Fs) == 1:
        F = lambda x: Fs[0](full([x]))
        try:
            fp = fixed_point(F, mp.mpf(seed), dps=dps + 5, tol="1e-16", branch=branch)
        except Refusal as r:
            raise Refusal("no-solution", f"derived RS fixed point {left[0]} = F({left[0]}): "
                                         f"{r.detail}") from None
        return [fp.value], fp.branch, fp.branches_seen, (
            f"solve.fixed_point on {left[0]} = F({left[0]}) (seed {seed}), branch {fp.branch}")
    G = lambda *u: [f(full(list(u))) - x for f, x in zip(Fs, u)]
    found = []
    for s in seeds:
        u = [mp.mpf(s)] * len(Fs)
        try:
            for _ in range(400):
                Fu = [f(full(u)) for f in Fs]
                if any(isinstance(v, mp.mpc) for v in Fu):
                    raise ValueError("complex")
                if max(abs(a - b) for a, b in zip(Fu, u)) < mp.mpf("1e-4"):
                    break
                u = [(a + b) / 2 for a, b in zip(u, Fu)]
            else:
                continue
            r = mp.findroot(G, u, solver="mdnewton", tol=mp.mpf(10)**(-2 * dps), maxsteps=30,
                            verify=False)
            u = [_real(r[i]) for i in range(len(Fs))]
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, Refusal):
            continue
        name = "trivial" if all(abs(v) < mp.mpf("1e-10") for v in u) else "nontrivial"
        if not any(max(abs(a - b) for a, b in zip(u, v)) < mp.mpf("1e-8") for _, v in found):
            found.append((name, u))
    return _pick(found, branch, left, "damped vector iteration + mdnewton from seeds "
                 f"{list(seeds)}")


def _roots(Rs, full, left, branch, seeds, dps):
    found = []
    for s in seeds:
        u0 = [mp.mpf(s)] * len(Rs)
        try:
            if len(Rs) == 1:
                r = mp.findroot(lambda x: Rs[0](full([x])), u0[0], solver="secant",
                                tol=mp.mpf(10)**(-2 * dps), maxsteps=50, verify=False)
                u = [_real(r)]
            else:
                r = mp.findroot(lambda *x: [R(full(list(x))) for R in Rs], u0,
                                solver="mdnewton", tol=mp.mpf(10)**(-2 * dps), maxsteps=50,
                                verify=False)
                u = [_real(r[i]) for i in range(len(Rs))]
            if max(abs(R(full(u))) for R in Rs) > mp.mpf(_AUDIT):
                continue
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, Refusal):
            continue
        name = "trivial" if all(abs(v) < mp.mpf("1e-10") for v in u) else "nontrivial"
        if not any(max(abs(a - b) for a, b in zip(u, v)) < mp.mpf("1e-8") for _, v in found):
            found.append((name, u))
    solver = "secant" if len(Rs) == 1 else "mdnewton"
    return _pick(found, branch, left, f"mp.findroot ({solver}) on the residual from seeds "
                 f"{list(seeds)} (no fixed-point form)")


def _pick(found, branch, left, method):
    seen = tuple(n for n, _ in found)
    if not found:
        raise Refusal("no-solution", f"no root of the derived RS equations in {left} "
                                     f"from the seeds ({method})")
    if branch is None:
        pick = next((t for t in found if t[0] != "trivial"), found[0])
    else:
        pick = next((t for t in found if t[0] == branch), None)
        if pick is None:
            raise Refusal("no-solution", f"branch {branch!r} not found; seen {list(seen)} "
                                         f"({method})")
    return pick[1], pick[0], seen, f"{method}, branch {pick[0]}"
