"""Generic numerical solution of DERIVED RS equations (level (a) -> numbers).

  solve_derived_rs(spec, *, branch=None, quad="fast", dps=20, seed="0.5", seeds=None,
                   gh_nodes=80, audit_tol=None, derived=None, **params) -> DerivedRSSolution
  lift_numeric_beta(spec, params) -> (spec, params, notes)  (step 1 below)

No family name appears in this module: everything model-specific arrives
through agree.derive_rs(spec) (the stationarity equations d phi/d p = 0, the
declared unknowns p = RSFree.params, phi itself, the recorded site traces) and
solve_derived_stability.assess(state).
 1. Parameters.  Every free symbol of the equations or of phi that is not a
    declared unknown needs a value from `params` (by symbol name), else a
    Refusal.  A numeric spec.params["beta"] is lifted back to the symbol beta
    (and its value moved to `params`) so that the entropy can be derived.
 2. Canonical form.  lhs - rhs, expectations split by linearity, one round of
    Gaussian integration by parts (gauss.linear, gauss.stein), erfc kept as
    written (_derived_num.safe).
 3. Elimination.  Unknowns are tried in reverse declared order (conjugates and
    extras before the primary overlap): x is eliminated through an equation in
    which it occurs linearly and outside every expectation, x = S(others), when
    S does not contain x and no earlier S contains x (acyclic order).  Eliminated
    values are computed numerically (no nested expectations).  SK: qhat = beta^2 q.
 4. Candidates.  If each remaining equation can be rearranged as u_k = F_k(u)
    (u_k linear outside the expectations; any rearrangement has the same roots,
    and the audit checks the original system):
      one unknown   solve.fixed_point(F, seed) with the FULL seed sweep: every
                    branch it sees is a candidate;
      several       damped iteration u <- u + gamma (F(u) - u) from the seeds (each
                    seed value for all unknowns, and with one unknown set to 0;
                    gamma = 1/2, halved when the step grows); in the float search a
                    non-converged iterate is handed to mdnewton (float);
    otherwise, one unknown: sign changes of the residual between consecutive
    sorted seeds, refined by the Illinois method; several: mp.findroot mdnewton
    from the seeds.  Default seeds (0.5, 0.25, 0.75, 0.1, 0.9, 0.99).  Only
    attracting points of the damped map (or bracketed roots) are found: an
    absent branch means "not found from these seeds".
    quad="fast" (default): the search runs in float64 Gauss-Hermite
    (`gh_nodes`), each candidate is then polished by findroot in mpmath at `dps`
    with the main quadrature; if the float search finds nothing the mpmath
    search is run.  quad="mp": search and polish in mpmath.  quad="gh": float
    Gauss-Hermite only, no polish (for sweeps; the audit then measures its error).
 5. Audit, per candidate.  Every derived equation (eliminated ones too) is
    re-evaluated at the candidate with an INDEPENDENT quadrature
    (_derived_num.Num.audit: Gauss-Legendre, other split, dps + 6), and
    max |lhs - rhs| there must be < audit_tol (default max(1e-12, 10^(2-dps))).
    quad_error = max |main - independent| over the equations and phi.  A
    candidate that fails is dropped (named in the note); none left -> Refusal.
 6. Selection.  Distinct audited roots are returned in `roots` (name, values,
    phi, residual, quad_error).  "trivial" = all unknowns 0; the others are
    "nontrivial" if there is one, else "nontrivial-1", "nontrivial-2", ...
    ordered by decreasing phi.  branch=None: the unique nontrivial root, else
    the trivial one (noted); several nontrivial roots without a selector ->
    Refusal("ambiguous:several-saddles") listing them.  branch="<name>" picks
    by name ("nontrivial" refuses when there are several).
 7. Outputs.  free_energy = phi at the solution, phi = lim (1/(N n)) ln E Z^n
    (RS): -beta f for a thermal model, the entropy (log volume / log count per
    variable) for the perceptron.  entropy = phi - beta d phi/d beta when phi
    depends on beta, else phi.  Stability: solve_derived_stability.assess (the
    derived replicon, SK / perceptrons / Hopfield; for Hopfield also the mode of the
    extra order parameter m from the derived phi), each value _derived_num.checked;
    quad_error includes a fine-panel disagreement when that fallback was used.
    STABILITY CONTRACT: stable is True only when every mode is computed and > 0 and
    (discrete variables) the RS entropy is >= 0; False when a computed one fails;
    None means UNKNOWN: a composer must treat None as unknown and never present that
    RS answer as final.  Local stability only.  No capacity or threshold is computed.

Numerics: mpmath under mp.workdps (the caller's precision is restored).  The
main quadrature is not validated for sharply varying integrands (perceptron at
q -> 1, large beta); quad_error and the independent audit are the check.
Honest label (CAVEAT): the equations are physics-level (replica n -> 0, RS
ansatz); a solution of them is an RS saddle, not a proof that RS is correct.
"""
from dataclasses import dataclass, field, replace

import mpmath as mp
import sympy as sp

from . import agree
from . import derive as D
from ._derived_num import Num, checked, safe
from .core import Refusal, Trace, TraceEntry
from .families.common import CAVEAT
from .gauss import GaussE, _sympy_point, linear, stein
from .solve import fixed_point
from .solve_derived_stability import _DISCRETE, assess  # noqa: F401 (compose reads _DISCRETE)
from .spec import ModelSpec

_DEFAULT_SEEDS = ("0.5", "0.25", "0.75", "0.1", "0.9", "0.99")
_QUADS = ("fast", "mp", "gh")
_SAME = mp.mpf("1e-8")
_ZERO = mp.mpf("1e-10")


@dataclass(frozen=True)
class DerivedRSSolution:
    values: dict                 # unknown name -> mpf (eliminated ones included)
    free_energy: object          # phi at the solution (module docstring)
    residual: object             # max |lhs - rhs|, all equations, INDEPENDENT quadrature
    equations_used: list         # the derived sympy equations (symbolic parameters)
    trace: Trace                 # the derivation's trace plus a "solve-derived" entry
    stable: bool | None          # None = UNKNOWN (see the stability contract)
    caveat: str = CAVEAT
    branch: str = ""
    entropy: object = None
    method: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    quad_error: object = None    # max |main - independent quadrature| (equations, phi)
    replicon: object = None      # derived lambda_R at the solution, or None
    roots: list = field(default_factory=list)   # every audited root: dicts
    replicon_route: str = ""     # derive_replicon's rules ("" when not derived)
    mode: object = None          # extras' (Hopfield m) longitudinal mode, or None


def _masked(e, fn):
    atoms = sorted(e.atoms(GaussE), key=str)
    dum = [sp.Dummy("G") for _ in atoms]
    back = dict(zip(dum, atoms))
    r = fn(sp.expand(e.xreplace(dict(zip(atoms, dum)))))
    return None if r is None else tuple(sp.sympify(x).xreplace(back) for x in r)


def _linear_outside(e, x):
    """(c1, c0) with e = c1 x + c0 and x absent from c1 (expectations included);
    c0 may hold x inside expectations.  None otherwise."""
    def go(m):
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


def _distinct(cands, tol):
    out = []
    for u in cands:
        if not any(max(abs(mp.mpf(a) - mp.mpf(b)) for a, b in zip(u, v)) < tol for v in out):
            out.append(u)
    return out


def _is_number(v):
    return not (isinstance(v, sp.Basic) and v.free_symbols)


# ---- the solver ---------------------------------------------------------------------

def lift_numeric_beta(spec: ModelSpec, params: dict):
    """(spec, params, notes): a numeric spec.params["beta"] is replaced by the symbol
    beta and its value moved to `params` (so the entropy can be derived)."""
    notes = []
    b = spec.params.get("beta")
    if b is not None and _is_number(b):
        if "beta" in params and sp.sympify(params["beta"]) != sp.sympify(b):
            raise Refusal("precondition:consistent-parameters",
                          f"beta = {b} in spec.params and {params['beta']} in params")
        params = {**params, "beta": params.get("beta", b)}
        spec = replace(spec, params={**spec.params, "beta": D.BETA})
        notes.append("numeric spec.params beta lifted to the symbol beta (entropy)")
    return spec, params, notes


def solve_derived_rs(spec: ModelSpec, *, branch=None, quad="fast", dps=20, seed="0.5",
                     seeds=None, gh_nodes=80, audit_tol=None, derived=None,
                     **params) -> DerivedRSSolution:
    """`derived` = (equations, State, Trace) of a derivation already run for THIS spec
    (the composer runs its own level-(a) steps); None runs agree.derive_rs(spec)."""
    if quad not in _QUADS:
        raise Refusal("precondition:known-quadrature", f"quad = {quad!r}: use {_QUADS}")
    spec, params, notes = lift_numeric_beta(spec, params)
    if derived is not None and notes:
        raise Refusal("precondition:symbolic-beta",
                      "`derived` was given for a spec with numeric beta: lift it first "
                      "(lift_numeric_beta) and derive at symbolic beta")
    eqs, state, trace = derived if derived is not None else agree.derive_rs(spec)
    tol = mp.mpf(audit_tol) if audit_tol is not None else max(mp.mpf("1e-12"),
                                                                mp.mpf(10)**(2 - dps))
    with mp.workdps(dps + 3):
        return _Solver(spec, eqs, state, trace, branch, quad, dps, seed,
                       tuple(seeds or _DEFAULT_SEEDS), gh_nodes, tol, params, notes).run()


class _Solver:
    def __init__(self, spec, eqs, state, trace, branch, quad, dps, seed, seeds, gh, tol,
                 params, notes):
        self.spec, self.eqs, self.state, self.trace = spec, eqs, state, trace
        self.branch, self.quad, self.dps, self.seed, self.seeds = branch, quad, dps, seed, seeds
        self.gh, self.tol, self.params, self.notes = gh, tol, params, notes
        self.phi = safe(state["rs_free"].expr)
        self.unknowns = list(state["rs_free"].params)

    def num(self, expr):
        return Num(expr, self.order, self.dps, self.gh)

    # -- setup: parameters, canonical form, elimination, fixed-point forms --
    def setup(self):
        unames = {u.name for u in self.unknowns}
        free = {s for e in self.eqs for s in (e.lhs - e.rhs).free_symbols} | self.phi.free_symbols
        self.by_name = {s.name: s for s in free if s.name not in unames}
        missing = sorted(set(self.by_name) - set(self.params))
        if missing:
            raise Refusal("precondition:complete-point",
                          f"no value for parameter(s) {missing} (unknowns: {sorted(unames)})")
        unused = sorted(set(self.params) - set(self.by_name))
        if unused:
            self.notes.append(f"unused parameters {unused}")
        self.point = _sympy_point({self.by_name[k]: v for k, v in self.params.items()
                                   if k in self.by_name}, self.dps + 15)
        canon = [linear(stein(safe(e.lhs - e.rhs).subs(self.point))) for e in self.eqs]
        elim, rest = [], list(range(len(canon)))
        for x in reversed(self.unknowns):
            if len(elim) == len(self.unknowns) - 1:      # keep one unknown to solve for
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
                self.notes.append(f"eliminated {x} = {S}")
                break
        self.left = [u for u in self.unknowns if u not in {x for x, _ in elim}]
        if len(self.left) != len(rest):
            raise Refusal("precondition:square-system",
                          f"{len(rest)} equation(s) for unknowns {self.left} after elimination")
        self.order = self.left + [x for x, _ in elim]
        self.elim = [self.num(S) for _, S in elim]
        self.canon = [self.num(e) for e in canon]
        self.phi_num = self.num(self.phi.subs(self.point))
        forms, used = [], set()
        for k in rest:
            for x in self.left:
                lin = None if x in used else _linear_outside(canon[k], x)
                if lin is not None:
                    forms.append((x, sp.expand(-lin[1] / lin[0])))
                    used.add(x)
                    break
            else:
                forms = None
                break
        if forms is not None:
            forms.sort(key=lambda p: self.left.index(p[0]))
            self.F = [self.num(F) for _, F in forms]
        else:
            self.F = None
            self.R = [self.canon[k] for k in rest]

    # -- evaluation with a chosen quadrature --
    def full(self, u, how):
        vals = list(u) + [0] * len(self.elim)
        for j, f in enumerate(self.elim):
            vals[len(self.left) + j] = getattr(f, how)(vals)
        return vals

    def G(self, u, how):
        """Residual of the solved form at u: F(u) - u, or the raw residual."""
        vals = self.full(u, how)
        if self.F is not None:
            return [getattr(f, how)(vals) - x for f, x in zip(self.F, u)]
        return [getattr(r, how)(vals) for r in self.R]

    # -- candidate search --
    def _vector_seeds(self):
        k = len(self.left)
        out = [[s] * k for s in self.seeds]
        if k > 1:
            out += [[s if i != j else "0" for i in range(k)] for s in self.seeds[:2]
                    for j in range(k)]
        return out

    def search(self, how):
        """Candidate roots (lists of numbers) found with evaluator `how`."""
        cands = []
        to = (lambda v: mp.mpf(v)) if how == "fast" else (lambda v: v)
        if self.F is not None and len(self.left) == 1:
            F = lambda x: to(self.F[0].__getattribute__(how)(self.full([x], how)))
            dps, tol = (15, "1e-11") if how == "fast" else (self.dps + 3, mp.mpf(10)**-self.dps)
            try:
                fp = fixed_point(F, mp.mpf(self.seed), dps=dps, tol=tol)
            except Refusal:
                return []
            return [[v] for _, v in fp.branch_values]
        if self.F is None and len(self.left) == 1:
            return self._brackets(how, to)
        fast = how == "fast"
        stop, maxit = (mp.mpf("1e-12"), 5000) if fast else (mp.mpf("1e-4"), 400)
        newton = lambda u: mp.findroot(lambda *x: [to(g) for g in self.G(list(x), how)], u,
                                       solver="mdnewton", tol=mp.mpf("1e-20"), maxsteps=50,
                                       verify=False)
        for s in self._vector_seeds():
            u = [mp.mpf(x) for x in s]
            try:
                if self.F is not None:
                    u, ok = self._damped(u, how, to, stop, maxit)
                    if not ok and not fast:
                        continue
                    if not ok:          # float: Newton from the last iterate is cheap
                        with mp.workdps(15):
                            r = newton(u)
                        u = [r[i] for i in range(len(u))]
                        if max(abs(to(g)) for g in self.G(u, how)) > mp.mpf("1e-9"):
                            continue
                else:
                    with mp.workdps(15 if fast else self.dps + 3):
                        r = newton(u)
                    u = [r[i] for i in range(len(u))]
                if any(isinstance(v, mp.mpc) for v in u):
                    continue
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, Refusal):
                continue
            cands.append(u)
        return cands

    def _damped(self, u, how, to, stop, maxit):
        """u <- u + gamma (F(u) - u), gamma = 1/2 halved (down to 1/1024) whenever the
        step grows.  Returns (last iterate, converged)."""
        gamma, last = mp.mpf("0.5"), None
        for _ in range(maxit):
            g = [to(x) for x in self.G(u, how)]
            size = max(abs(x) for x in g)
            if size < stop:
                return u, True
            if last is not None and size > last and gamma > mp.mpf(1) / 1024:
                gamma /= 2
            last = size
            u = [x + gamma * d for x, d in zip(u, g)]
        return u, False

    def _brackets(self, how, to):
        """One unknown, no fixed-point form: sign changes of the residual between
        consecutive sorted seeds, each refined by the Illinois method."""
        pts, cands = sorted({mp.mpf(x) for x in self.seeds}), []
        vals = []
        for x in pts:
            try:
                v = to(self.G([x], how)[0])
                vals.append(v if isinstance(v, mp.mpf) else None)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                vals.append(None)
        with mp.workdps(15 if how == "fast" else self.dps + 3):
            for (a, fa), (b, fb) in zip(zip(pts, vals), zip(pts[1:], vals[1:])):
                if fa is None or fb is None or (fa > 0) == (fb > 0):
                    continue
                try:
                    r = mp.findroot(lambda x: to(self.G([x], how)[0]), (a, b),
                                    solver="illinois", tol=mp.mpf("1e-24"), maxsteps=100,
                                    verify=False)
                except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                    continue
                if isinstance(r, mp.mpf):
                    cands.append([r])
        return cands

    def polish(self, u):
        """findroot in mpmath (main quadrature) from candidate u; exact 0 kept."""
        u = [mp.mpf(v) for v in u]
        if all(abs(v) < _ZERO for v in u):
            z = [mp.mpf(0)] * len(u)
            try:
                if max(abs(g) for g in self.G(z, "mp")) < mp.mpf(10)**-self.dps:
                    return z
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        try:
            r = mp.findroot(lambda *x: self.G(list(x), "mp") if len(u) > 1 else
                            self.G(list(x), "mp")[0], u if len(u) > 1 else u[0],
                            solver="secant" if len(u) == 1 else "mdnewton",
                            tol=mp.mpf(10)**(-self.dps - 1), maxsteps=30, verify=False)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None
        u = [r] if len(u) == 1 else [r[i] for i in range(len(u))]
        if any(isinstance(v, mp.mpc) for v in u):
            return None
        try:
            if max(abs(g) for g in self.G(u, "mp")) > mp.mpf(10)**(2 - self.dps):
                return None
        except (ValueError, TypeError, ZeroDivisionError):
            return None
        return u

    def candidates(self):
        if self.quad in ("fast", "gh"):
            found = _distinct(self.search("fast"), mp.mpf("1e-6"))
            if self.quad == "gh":
                self.method = f"float Gauss-Hermite ({self.gh} nodes) only, no polish"
                return found, "fast"
            pol = [p for p in (self.polish(u) for u in found) if p is not None]
            if pol:
                self.method = (f"search in float Gauss-Hermite ({self.gh} nodes), "
                               f"polished by mp.findroot at dps {self.dps}")
                return pol, "mp"
            self.notes.append("float search found no polishable root: mpmath search")
        found = _distinct(self.search("mp"), _SAME)
        self.method = f"search and polish in mpmath at dps {self.dps}"
        if self.F is None or len(self.left) > 1:
            found = [p for p in (self.polish(u) for u in found) if p is not None]
        return found, "mp"

    # -- audit, selection, outputs --
    def audit(self, u, how):
        vals = self.full(u, how)
        vals_mp = [mp.mpf(v) for v in vals]
        fns = self.canon + [self.phi_num]
        main = [getattr(f, how)(vals) for f in fns]
        ind = [f.audit(vals_mp) for f in fns]
        if any(isinstance(v, (mp.mpc, complex)) for v in main + ind):
            return None
        res = max(abs(v) for v in ind[:-1])
        qerr = max(abs(mp.mpf(a) - b) for a, b in zip(main, ind))
        return {"values": dict(zip(map(str, self.order), vals_mp)), "phi": ind[-1],
                "phi_main": mp.mpf(main[-1]), "residual": res, "quad_error": qerr,
                "_u": [mp.mpf(v) for v in u]}

    def run(self):
        self.setup()
        found, how = self.candidates()
        roots, dropped = [], []
        for u in found:
            if any(max(abs(mp.mpf(a) - b) for a, b in zip(u, r["_u"])) < _SAME for r in roots):
                continue
            try:
                r = self.audit(u, how)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                r = None
            if r is None or not r["residual"] < self.tol:
                dropped.append((u, None if r is None else r["residual"]))
                continue
            roots.append(r)
        if dropped:
            self.notes.append("dropped by the independent audit: " + "; ".join(
                f"{[mp.nstr(mp.mpf(v), 8) for v in u]} (residual "
                f"{'n/a' if res is None else mp.nstr(res, 3)})" for u, res in dropped))
        if not roots:
            raise Refusal("no-solution", "audit: no candidate root of the derived RS "
                          f"equations in {self.left} passes the independent audit "
                          f"(tol {mp.nstr(self.tol, 3)}; {self.method}; "
                          + "; ".join(self.notes) + ")")
        triv = [r for r in roots if all(abs(v) < _ZERO for v in r["_u"])]
        non = sorted((r for r in roots if r not in triv), key=lambda r: -r["phi"])
        for r in triv:
            r["name"] = "trivial"
        for k, r in enumerate(non, 1):
            r["name"] = "nontrivial" if len(non) == 1 else f"nontrivial-{k}"
        table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in triv + non]
        listing = "; ".join(f"{r['name']}: " + ", ".join(
            f"{k}={mp.nstr(v, 10)}" for k, v in r["values"].items())
            + f", phi={mp.nstr(r['phi'], 10)}" for r in table)
        if self.branch is None:
            if len(non) > 1:
                raise Refusal("ambiguous:several-saddles",
                              f"{len(non)} nontrivial RS roots ({listing}); pass branch= "
                              "one of the names (ordered by decreasing phi)")
            pick = non[0] if non else triv[0]
            if not non:
                self.notes.append("only the trivial branch (all unknowns 0, a fixed point "
                                  "of the map) was found from the seeds")
        else:
            hits = [r for r in triv + non if r["name"] == self.branch]
            if not hits:
                reason = ("ambiguous:several-saddles" if self.branch == "nontrivial" and non
                          else "no-solution")
                raise Refusal(reason, f"branch {self.branch!r} not found; roots: {listing}"
                              + "".join(f"; {n}" for n in self.notes if n.startswith("dropped")))
            pick = hits[0]
        return self.outputs(pick, table, listing, how)

    def outputs(self, pick, table, listing, how):
        vals = [pick["values"][str(x)] for x in self.order]
        fe = pick["phi"]
        betas = [s for s in self.phi.free_symbols if s.name == "beta"]
        if betas:
            b = betas[0]
            ent = self.num((self.phi - b * sp.diff(self.phi, b)).subs(self.point)).mp(vals)
            ent_note = "entropy = phi - beta d phi/d beta"
        else:
            ent, ent_note = fe, "entropy = phi (no temperature: log count / volume per variable)"
        a = assess(self.state, self.spec.variables,
                   lambda e: checked(self.num(linear(stein(safe(e).subs(self.point)))), vals,
                                     self.tol), ent)
        stable, lam, stab = a.stable, a.replicon, a.note
        qerr = max(pick["quad_error"], a.quad_error)
        note = "; ".join(self.notes + [self.method, f"roots: {listing}", ent_note, stab,
                                       "no capacity / threshold computed"])
        out = Trace()
        for e in self.trace.entries:
            out.add(e)
        out.add(TraceEntry("solve-derived", None,
                           f"branch {pick['name']}; stable {stable}; independent residual "
                           f"{mp.nstr(pick['residual'], 3)}; quad_error "
                           f"{mp.nstr(qerr, 3)}; {note}"))
        return DerivedRSSolution(
            pick["values"], fe, pick["residual"], list(self.eqs), out, stable, CAVEAT,
            pick["name"], ent, self.method, note,
            {k: self.params[k] for k in self.by_name}, qerr, lam, table, a.route, a.mode)
