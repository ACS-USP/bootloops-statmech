"""Numerical solution of the DERIVED 1RSB functional (derive_rsb.py -> numbers).

  solve_derived_1rsb(spec, *, frozen=False, zero_temperature=False, branch=None, seeds=None,
                     fixed=None, solve_for=None, audit_tol="1e-9", derived=None, **params)
      -> DerivedRSBSolution

No family name appears here: everything model-specific arrives through
derive_rsb.derive_1rsb(spec, frozen, zero_temperature).
 1. Functional.  psi = phi with the conjugates eliminated by rules
    (derive_rsb_t0.eliminate_conjugates); the unknowns are psi's order parameters (q0, q1,
    x = Parisi m, extras such as the Hopfield m; frozen: q0, x; T = 0: q0, mu = beta x).
    `fixed` = {name: value} pins order parameters (their equations are still imposed);
    `solve_for` = a model parameter (e.g. "alpha") made an unknown instead: the frozen
    threshold, where d psi/d x = 0 (S_RS = 0) holds at an isolated parameter value (frozen
    with solve_for and no `fixed`: x is fixed to 1, noted).
 2. Equations.  d psi/d p = 0 for every order parameter p (fixed ones too), one round of
    Gaussian integration by parts (gauss.stein, which also cancels the 1/sqrt(q0),
    1/sqrt(q1 - q0) factors of the derivatives), erfc kept as written.  Every free symbol
    that is not an unknown needs a value in params (else Refusal precondition:complete-point).
 3. Roots.  Newton (central-difference Jacobian, steps halved to stay in the domain
    0 < q0 < q1 < 1, 0 < x < 1, the conditions the reduction ASSUMED > 0
    (RSBFunctional.conditions: qhat1 > qhat0, Gaussian integrability) and qhat > 0 for the
    eliminated conjugates; at most min(0.2, |v|/2) per coordinate
    and step) from every seed: `seeds` (a list of
    {name: value}) or the default table, in float64 Gauss-Hermite with FAST nodes; each
    converged point is polished by Newton with the MAIN rule (_rsb_num.py) to max |eq| <
    1e-10 (a point that does not polish is named in the note, never returned).  Only roots
    reached from the seeds are found: an absent root means "not found from these seeds".
 4. Audit.  Every equation is re-evaluated with the INDEPENDENT rule (composite
    Gauss-Legendre, _rsb_num "audit"); max |eq| must be < audit_tol.  quad_error = max
    |main - audit| over the equations and psi.  A failing candidate is dropped (named).
 5. Classification.  q1 - q0 < 1e-4, or x > 1 - 1e-4 with q1 an unknown: "rs-degenerate"
    (the 1RSB collapses to RS there; the degenerate set is flat, so Newton stops anywhere on
    it): counted in the note, not a 1RSB root.  "trivial": every q ~ 0.  The
    others are "nontrivial" if unique, else "nontrivial-1", ... by decreasing psi.
    Selection: branch=None -> the unique nontrivial root; several -> Refusal
    ambiguous:several-saddles listing them; none -> Refusal no-solution.  branch="<name>"
    by name.  branch="replica-max": the n -> 0 rule, the free energy f = -phi/beta is
    MAXIMAL over the order parameters, i.e. psi minimal: the nontrivial root of smallest psi,
    which must be a local minimum of psi (central-difference Hessian in q0, q1, x / mu, all
    eigenvalues > 0), else Refusal no-solution:not-a-local-maximum.
 6. Outputs.  free_energy = psi at the root (= phi: -beta f, or the entropy per variable for
    a counting problem; T = 0: lim phi/beta = -e0, and energy = -psi).  stable = None
    always: the 1RSB replicon (stability toward further RSB) is Task 5's (E4); a composer
    must not present a 1RSB value as final.  hessian = the eigenvalues of step 5.
Honest label (CAVEAT): physics-level (Parisi 1RSB ansatz, n -> 0); float64 quadrature, the
main-vs-independent difference is the error estimate (not a bound).
"""
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np
import sympy as sp

from . import derive_rsb as R
from ._derived_num import safe
from ._rsb_num import NestedNum
from ._rsb_num import top_atoms as _top
from .core import Refusal, Trace, TraceEntry
from .derive_rsb_t0 import eliminate_conjugates
from .families.common import CAVEAT
from .gauss import _sympy_point, linear, stein
from .solve_derived import lift_numeric_beta

_SEEDS = {"q0": (0.1, 0.2, 0.3, 0.05, 0.4, 0.15, 0.5), "q1": (0.5, 0.7, 0.8, 0.9, 0.9, 0.6, 0.95),
          "x": (0.2, 0.3, 0.4, 0.15, 0.6, 0.25, 0.5), "mu": (1.0, 2.0, 0.5, 1.5, 3.0, 0.8, 1.2),
          "qhat0": (1.0, 2.0, 0.5, 3.0, 0.2, 1.5, 4.0),
          "qhat1": (3.0, 5.0, 2.0, 8.0, 1.0, 4.0, 6.0),
          "m": (0.95, 0.5, 0.8, 0.99, 0.3, 0.9, 0.7)}
_OTHER = (0.5, 1.0, 0.8, 0.3, 0.9, 0.6, 1.2)
_SAME, _DEGENERATE, _ZERO, _POLISH = 1e-7, 1e-4, 1e-10, 1e-10


@dataclass(frozen=True)
class DerivedRSBSolution:
    values: dict                 # unknown / eliminated / fixed name -> mpf
    free_energy: object          # psi at the root (module docstring, 6)
    m: object                    # Parisi block parameter x (None at T = 0: mu = beta x)
    q0: object
    q1: object
    residual: object             # max |eq| with the INDEPENDENT rule
    roots: list                  # every audited root (dicts), rs-degenerate ones excluded
    stable: bool | None          # always None here: 1RSB stability is not computed
    trace: Trace
    caveat: str = CAVEAT
    branch: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    quad_error: object = None
    hessian: tuple = ()          # eigenvalues of the Hessian of psi (q0, q1, x / mu)
    energy: object = None        # T = 0: e0 = -psi
    equations_used: list = field(default_factory=list)


def _domain_ok(names, v):
    d = dict(zip(names, v))
    for k, y in d.items():
        if k.startswith("q") and not 0 < y < 1:
            return False
        if k in ("x", "mu") and not y > 0:
            return False
        if k == "x" and not y < 1:
            return False
    return not ("q0" in d and "q1" in d and not d["q0"] < d["q1"])


def _newton(G, v, ok, tol, maxit=60, eps=1e-6, max_step=0.2):
    v = np.array(v, float)
    for _ in range(maxit):
        try:
            g = np.array(G(v))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if np.abs(g).max() < tol:
            return v
        J = np.empty((len(v), len(v)))
        try:
            for k in range(len(v)):
                d = np.zeros(len(v))
                d[k] = eps * max(1.0, abs(v[k]))
                J[:, k] = (np.array(G(v + d)) - np.array(G(v - d))) / (2 * d[k])
            step = np.linalg.solve(J, -g)
            # a bounded step (Newton is only trusted locally): at most max_step and half the
            # coordinate's own size per coordinate
            cap = np.minimum(max_step, 0.5 * np.maximum(np.abs(v), 0.02))
            big = (np.abs(step) / cap).max()
            if big > 1:
                step /= big
        except (ValueError, ZeroDivisionError, OverflowError, np.linalg.LinAlgError):
            return None
        lam = 1.0
        while not ok(v + lam * step):
            lam /= 2
            if lam < 1e-8:
                return None
        v = v + lam * step
    return None


def solve_derived_1rsb(spec, *, frozen=False, zero_temperature=False, branch=None, seeds=None,
                       fixed=None, solve_for=None, audit_tol="1e-9", derived=None,
                       **params) -> DerivedRSBSolution:
    notes = []
    if not zero_temperature:
        spec, params, notes = lift_numeric_beta(spec, params)
    elif "beta" in params or "beta" in {k for k, v in spec.params.items()
                                         if not isinstance(v, sp.Basic) or not v.free_symbols}:
        raise Refusal("precondition:symbolic-beta", "zero_temperature=True takes no beta")
    eqs, state, trace = derived if derived is not None else \
        R.derive_1rsb(spec, frozen=frozen, zero_temperature=zero_temperature)
    P = state["rsb_t0" if zero_temperature else "rsb_free"]
    if frozen and solve_for is not None and fixed is None:
        fixed = {"x": 1}
        notes.append("frozen threshold: x fixed to 1 (x is degenerate there: phi = S_RS/x "
                      "with S_RS = 0)")
    return _Solver(spec, P, eqs, trace, branch, seeds, dict(fixed or {}), solve_for,
                   float(audit_tol), params, notes, zero_temperature).run()


class _Solver:
    def __init__(self, spec, P, eqs, trace, branch, seeds, fixed, solve_for, tol, params, notes,
                 t0):
        self.spec, self.P, self.eqs, self.trace = spec, P, eqs, trace
        self.branch, self.seeds, self.fixed, self.solve_for = branch, seeds, fixed, solve_for
        self.tol, self.params, self.notes, self.t0 = tol, params, list(notes), t0

    def setup(self):
        psi, rest, self.elim, n = eliminate_conjugates(self.P.phi, self.P.params)
        self.notes += n
        self.order_params = list(rest)
        by = {p.name: p for p in rest}
        bad = sorted(set(self.fixed) - set(by))
        if bad:
            raise Refusal("precondition:known-order-parameter",
                          f"fixed {bad}: the order parameters are {sorted(by)}")
        free = psi.free_symbols - set(rest)
        model = {s.name: s for s in free}
        if self.solve_for is not None and self.solve_for not in model:
            raise Refusal("precondition:known-parameter",
                          f"solve_for={self.solve_for!r}: psi's parameters are {sorted(model)}")
        need = set(model) - {self.solve_for}
        missing = sorted(need - set(self.params))
        if missing:
            raise Refusal("precondition:complete-point",
                          f"no value for parameter(s) {missing} (order parameters: {sorted(by)})")
        unused = sorted(set(self.params) - need)
        if unused:
            self.notes.append(f"unused parameters {unused}")
        point = _sympy_point({model[k]: v for k, v in self.params.items() if k in need}, 30)
        point.update({by[k]: sp.sympify(v) for k, v in self.fixed.items()})
        self.unknowns = [p for p in rest if p.name not in self.fixed] + \
            ([model[self.solve_for]] if self.solve_for else [])
        self.names = [u.name for u in self.unknowns]
        if len(self.unknowns) != len(rest):
            raise Refusal("precondition:square-system",
                          f"{len(rest)} equations for unknowns {self.names}")
        self.point = point
        # two-stage evaluation: the eliminated conjugates s_j = S_j(v) first, then phi and its
        # derivatives at (v, s): d psi/d p = phi_p + sum_j phi_{qhat_j} dS_j/dp (chain rule; no
        # expectation is substituted into another, so the nesting depth stays that of phi)
        X = [x for x, _ in self.elim]
        args = self.unknowns + X
        phi = safe(self.P.phi)
        nested = any(_top(a.f) for a in _top(phi))
        prep = (lambda e: linear(e)) if nested else (lambda e: linear(stein(e)))
        num = lambda e, a=args: NestedNum(prep(sp.sympify(e).subs(point)), a)
        self.S = [num(safe(S), self.unknowns) for _, S in self.elim]
        self.A = [num(sp.diff(phi, p)) for p in rest]
        self.B = [num(sp.diff(phi, x)) for x in X]
        self.C = [[num(sp.diff(safe(S), p), self.unknowns) for p in rest] for _, S in self.elim]
        self.phi_num = NestedNum(phi.subs(point), args)
        self.eq_exprs = [sp.diff(phi, p) for p in rest]
        # the reduction's ASSUMED positivity conditions (in q, qhat) and the eliminated
        # conjugates' sign (qhat > 0, Hubbard-Stratonovich) bound the searched domain
        self.conds = [NestedNum(safe(c).subs(point), args) for c in self.P.conditions]

    def conj(self, v, rule):
        return [S.value(v, rule) for S in self.S]

    def ev(self, v, rule):
        s = self.conj(v, rule)
        w = list(v) + s
        B = [b.value(w, rule) for b in self.B]
        return [a.value(w, rule) + sum(bj * self.C[j][k].value(v, rule) for j, bj in enumerate(B))
                for k, a in enumerate(self.A)]

    def psi_value(self, v, rule):
        return self.phi_num.value(list(v) + self.conj(v, rule), rule)

    def _seed_list(self):
        if self.seeds is not None:
            out = []
            for s in self.seeds:
                miss = sorted(set(self.names) - set(s))
                if miss:
                    raise Refusal("precondition:complete-seed", f"seed {s} lacks {miss}")
                out.append([float(s[k]) for k in self.names])
            return out
        k = max(len(_SEEDS.get(nm, _OTHER)) for nm in self.names)
        return [[_SEEDS.get(nm, _OTHER)[i % len(_SEEDS.get(nm, _OTHER))] for nm in self.names]
                for i in range(k)]

    def ok(self, v):
        if not _domain_ok(self.names, v):
            return False
        try:
            sv = self.conj(v, "fast")
            return all(y > 0 for y in sv) and \
                all(c.value(list(v) + sv, "fast") > 0 for c in self.conds)
        except (ValueError, ZeroDivisionError, OverflowError):
            return False

    def candidates(self):
        ok = self.ok
        found = []
        for s in self._seed_list():
            if not ok(s):
                continue
            v = _newton(lambda y: self.ev(y, "fast"), s, ok, 1e-7)
            if v is None:
                continue
            w = _newton(lambda y: self.ev(y, "main"), v, ok, _POLISH, maxit=12)
            if w is None:
                self.unpolished.append(v)
            elif not any(np.abs(w - u).max() < _SAME for u in found):
                found.append(w)
        return found

    def audit(self, v):
        main = self.ev(v, "main") + [self.psi_value(v, "main")]
        ind = self.ev(v, "audit") + [self.psi_value(v, "audit")]
        vals = {nm: mp.mpf(float(y)) for nm, y in zip(self.names, v)}
        for (x, _), y in zip(self.elim, self.conj(v, "main")):
            vals[x.name] = mp.mpf(float(y))
        for k, y in self.fixed.items():
            vals[k] = mp.mpf(str(y))
        for k, y in (self.P.fixed or {}).items():
            if y is not None:
                vals.setdefault(k.name, mp.mpf(str(y)))
        return {"values": vals, "phi": mp.mpf(float(ind[-1])),
                "residual": mp.mpf(float(max(abs(y) for y in ind[:-1]))),
                "quad_error": mp.mpf(float(max(abs(a - b) for a, b in zip(main, ind)))),
                "_v": v}

    def hessian(self, v, d=1e-4):
        idx = [i for i, nm in enumerate(self.names) if nm in ("q0", "q1", "x", "mu")]
        f = lambda y: self.psi_value(y, "main")
        H = np.empty((len(idx), len(idx)))
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                ei, ej = np.eye(len(v))[i] * d, np.eye(len(v))[j] * d
                H[a, b] = (f(v + ei + ej) - f(v + ei - ej) - f(v - ei + ej)
                           + f(v - ei - ej)) / (4 * d * d)
        return tuple(float(e) for e in np.linalg.eigvalsh((H + H.T) / 2))

    def run(self):
        self.setup()
        roots, dropped, degen = [], [], []
        self.unpolished = []
        for v in self.candidates():
            try:
                r = self.audit(v)
            except (ValueError, ZeroDivisionError, OverflowError):
                dropped.append((v, None))
                continue
            if not r["residual"] < self.tol:
                dropped.append((v, r["residual"]))
                continue
            vv = r["values"]
            if ("q0" in vv and "q1" in vv and abs(vv["q1"] - vv["q0"]) < _DEGENERATE) or \
                    ("q1" in self.names and "x" in vv and vv["x"] > 1 - _DEGENERATE):
                degen.append(r)
                continue
            roots.append(r)
        if dropped:
            self.notes.append("dropped by the independent audit: " + "; ".join(
                f"{[round(float(y), 8) for y in v]} (residual "
                f"{'n/a' if res is None else mp.nstr(res, 3)})" for v, res in dropped))
        if self.unpolished:
            self.notes.append("found by the search but not polished to max |eq| < "
                              f"{_POLISH} with the main rule: " + "; ".join(
                                  str([round(float(y), 8) for y in v]) for v in self.unpolished))
        if degen:
            self.notes.append(f"{len(degen)} rs-degenerate root(s) (q1 = q0: 1RSB collapses to "
                              "RS) not counted as 1RSB roots")
        return self.select(roots)

    def select(self, roots):
        if not roots:
            raise Refusal("no-solution", "no 1RSB root with q1 > q0 of the derived equations in "
                          f"{self.names} passes the independent audit (tol {self.tol}; "
                          + "; ".join(self.notes) + ")")
        triv = [r for r in roots if all(abs(r["values"][k]) < _ZERO
                                        for k in ("q0", "q1") if k in r["values"])]
        non = sorted((r for r in roots if r not in triv), key=lambda r: -r["phi"])
        for r in triv:
            r["name"] = "trivial"
        for k, r in enumerate(non, 1):
            r["name"] = "nontrivial" if len(non) == 1 else f"nontrivial-{k}"
        table = [{k: y for k, y in r.items() if not k.startswith("_")} for r in triv + non]
        listing = "; ".join(f"{r['name']}: " + ", ".join(
            f"{k}={mp.nstr(y, 10)}" for k, y in r["values"].items())
            + f", psi={mp.nstr(r['phi'], 12)}" for r in table)
        if self.branch is None:
            if len(non) > 1:
                raise Refusal("ambiguous:several-saddles",
                              f"{len(non)} nontrivial 1RSB roots ({listing}); pass branch= one "
                              "of the names or 'replica-max' (the n -> 0 maximum of f)")
            pick = non[0] if non else triv[0]
        elif self.branch == "replica-max":
            if not non:
                raise Refusal("no-solution", f"no nontrivial root to select ({listing})")
            pick = min(non, key=lambda r: r["phi"])
        else:
            hits = [r for r in triv + non if r["name"] == self.branch]
            if not hits:
                raise Refusal("no-solution", f"branch {self.branch!r} not found; roots: {listing}")
            pick = hits[0]
        hess = () if self.solve_for else self.hessian(pick["_v"])
        local = bool(hess) and all(e > 0 for e in hess)
        if self.branch == "replica-max" and not local:
            raise Refusal("no-solution:not-a-local-maximum",
                          f"root {pick['name']} (smallest psi) has Hessian eigenvalues {hess}: "
                          "not a local maximum of f = -psi/beta")
        if hess:
            self.notes.append(f"Hessian of psi at {pick['name']}: eigenvalues "
                              f"{[round(e, 8) for e in hess]} ("
                              + ("local minimum of psi = local maximum of f" if local else
                                 "NOT a local maximum of f") + ")")
        return self.outputs(pick, table, listing, hess)

    def outputs(self, pick, table, listing, hess):
        vals, psi = pick["values"], pick["phi"]
        note = "; ".join(self.notes + [f"roots: {listing}",
                                       "stable = None: 1RSB stability (replicon toward further "
                                       "RSB) is not computed here"])
        out = Trace()
        for e in self.trace.entries:
            out.add(e)
        out.add(TraceEntry("solve-derived-1rsb", None,
                           f"branch {pick['name']}; independent residual "
                           f"{mp.nstr(pick['residual'], 3)}; quad_error "
                           f"{mp.nstr(pick['quad_error'], 3)}; {note}"))
        return DerivedRSBSolution(
            vals, psi, vals.get("x"), vals.get("q0"), vals.get("q1"), pick["residual"], table,
            None, out, CAVEAT, pick["name"], note,
            {k: self.params[k] for k in self.params}, pick["quad_error"], hess,
            -psi if self.t0 else None, list(self.eqs))
