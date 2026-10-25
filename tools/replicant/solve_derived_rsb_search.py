"""The 1RSB root search (helper of solve_derived_rsb.py; steps 1-5 of its docstring).

  Solver(spec, P, eqs, trace, branch, seeds, fixed, solve_for, tol, params, notes, t0,
         audit).run() -> DerivedRSBSolution
"""
import mpmath as mp
import numpy as np
import sympy as sp

from ._derived_num import safe
from ._rsb_num import MAIN_LEVELS, NestedNum, QuadNum
from ._rsb_num import top_atoms as _top
from .core import Refusal
from .derive_rsb_t0 import eliminate_conjugates
from .gauss import _sympy_point, linear, stein
from .solve_derived_rsb import (_DEGENERATE, _EDGE, _OTHER, _POLISH, _RS_SEEDS, _SAME, _SEEDS,
                                _domain_ok, _newton)
from .solve_derived_rsb_post import independent_audit, select


class Solver:
    def __init__(self, spec, P, eqs, trace, branch, seeds, fixed, solve_for, tol, params, notes,
                 t0, audit="independent"):
        self.spec, self.P, self.eqs, self.trace = spec, P, eqs, trace
        self.branch, self.seeds, self.fixed, self.solve_for = branch, seeds, fixed, solve_for
        self.tol, self.params, self.notes, self.t0 = tol, params, list(notes), t0
        self.audit_mode, self.seed_origin, self.n_seeds = audit, "", 0
        self.edge_searched, self._ind = False, False

    # -- 1, 2: functional, parameters, equations --
    def setup(self):
        _, rest, self.elim, n = eliminate_conjugates(self.P.phi, self.P.params)
        self.notes += n
        by = {p.name: p for p in rest}
        bad = sorted(set(self.fixed) - set(by))
        if bad:
            raise Refusal("precondition:known-order-parameter",
                          f"fixed {bad}: the order parameters are {sorted(by)}")
        X = [x for x, _ in self.elim]
        model = {s.name: s for s in self.P.phi.free_symbols - set(self.P.params)}
        if self.solve_for is not None and self.solve_for not in model:
            raise Refusal("precondition:known-parameter",
                          f"solve_for={self.solve_for!r}: the functional's parameters are "
                          f"{sorted(model)}")
        need = set(model) - {self.solve_for}
        unknown = sorted(set(self.params) - need)
        if unknown:
            raise Refusal("precondition:known-parameter",
                          f"the derived 1RSB functional does not read {', '.join(unknown)}; "
                          f"accepted: {', '.join(sorted(need)) or 'none'}")
        missing = sorted(need - set(self.params))
        if missing:
            raise Refusal("precondition:complete-point",
                          f"no value for parameter(s) {missing} (order parameters: {sorted(by)})")
        point = _sympy_point({model[k]: v for k, v in self.params.items()}, 30)
        point.update({by[k]: sp.sympify(v) for k, v in self.fixed.items()})
        self.unknowns = [p for p in rest if p.name not in self.fixed] + \
            ([model[self.solve_for]] if self.solve_for else [])
        self.names = [u.name for u in self.unknowns]
        if len(self.unknowns) != len(rest):
            raise Refusal("precondition:square-system",
                          f"{len(rest)} equations for unknowns {self.names}")
        self.point, self.rest = point, rest
        args = self.unknowns + X
        phi = safe(self.P.phi)
        nested = any(_top(a.f) for a in _top(phi))
        prep = (lambda e: linear(e)) if nested else (lambda e: linear(stein(e)))
        self._exprs = {
            "S": [prep(safe(S).subs(point)) for _, S in self.elim],
            "A": [prep(sp.diff(phi, p).subs(point)) for p in rest],
            "B": [prep(sp.diff(phi, x).subs(point)) for x in X],
            "C": [[prep(sp.diff(safe(S), p).subs(point)) for p in rest] for _, S in self.elim],
            "phi": phi.subs(point)}
        self.args = args
        e = self._exprs
        self.S = [NestedNum(x, self.unknowns) for x in e["S"]]
        self.A = [NestedNum(x, args) for x in e["A"]]
        self.B = [NestedNum(x, args) for x in e["B"]]
        self.C = [[NestedNum(x, self.unknowns) for x in row] for row in e["C"]]
        self.phi_num = NestedNum(e["phi"], args)
        self.conds = [NestedNum(safe(c).subs(point), args) for c in self.P.conditions]

    def audit_setup(self):
        if not self._ind:
            e = self._exprs
            self.S_ind = [QuadNum(x, self.unknowns) for x in e["S"]]
            self.A_ind = [QuadNum(x, self.args) for x in e["A"]]
            self.B_ind = [QuadNum(x, self.args) for x in e["B"]]
            self.C_ind = [[QuadNum(x, self.unknowns) for x in row] for row in e["C"]]
            self.phi_ind = QuadNum(e["phi"], self.args)
            self._ind = True

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

    def ok(self, v):
        if not _domain_ok(self.names, v):
            return False
        try:
            sv = self.conj(v, "fast")
            return all(y > 0 for y in sv) and \
                all(c.value(list(v) + sv, "fast") > 0 for c in self.conds)
        except (ValueError, ZeroDivisionError, OverflowError):
            return False

    # -- 3: seeds and search --
    def _rs_seeds(self):
        """Seeds from the DERIVED RS solution(s) at the same point (full finite-T 1RSB)."""
        from .solve_derived import solve_derived_rs
        sols = []
        try:
            sols.append(solve_derived_rs(self.spec, **self.params))
        except Refusal as r:
            if r.reason.startswith("ambiguous"):        # several RS roots: seed from each
                for b in ("nontrivial-1", "nontrivial-2"):
                    try:
                        sols.append(solve_derived_rs(self.spec, branch=b, **self.params))
                    except Refusal:
                        break
        out = []
        for sol in sols:
            q = float(sol.values.get("q", 0))
            if not 0 < q < 1:
                continue
            for a, b, x in _RS_SEEDS:
                d = {"q0": a * q, "q1": min(b * q, 0.99), "x": x}
                d.update({k: float(v) for k, v in sol.values.items() if k in self.names})
                out.append([d.get(nm, _OTHER[0]) for nm in self.names])
        return out

    def seed_list(self):
        if self.seeds is not None:
            out = []
            for s in self.seeds:
                miss = sorted(set(self.names) - set(s))
                if miss:
                    raise Refusal("precondition:complete-seed", f"seed {s} lacks {miss}")
                out.append([float(s[k]) for k in self.names])
            self.seed_origin = "given seeds"
            return out
        k = max(len(_SEEDS.get(nm, _OTHER)) for nm in self.names)
        out = [[_SEEDS.get(nm, _OTHER)[i % len(_SEEDS.get(nm, _OTHER))] for nm in self.names]
               for i in range(k)]
        self.seed_origin = "the default table"
        if "q1" in self.names and not self.t0 and not self.solve_for:
            rs = self._rs_seeds()
            out += rs
            self.seed_origin += f" and {len(rs)} seeds from the derived RS solution"
        return out

    def contract(self):
        edge = ("the q0 = 0 edge was searched (q0 pinned at 1e-9)" if self.edge_searched else
                "the q0 = 0 edge was not searched (q0 is not an unknown here)")
        return (f"roots found from {self.n_seeds} seeds ({self.seed_origin}): an absent root "
                f"means not found from these seeds; {edge}; the q1 = q0 edge is the RS "
                "collapse (rs-degenerate); the x -> 0 and q1 -> 1 edges are not searched")

    def audit_label(self):
        return ("accepted roots audited through a separate evaluation path (QuadNum: scalar "
                "math-module code, adaptive QUADPACK); "
                "candidates by the trapezoid rule (quadrature-only check)"
                if self.audit_mode == "independent" else
                "audit='quadrature': the trapezoid rule only (the same compiled integrands, "
                "other nodes): a quadrature check, NOT an independent evaluation")

    def _reduced(self, pin, rule):
        """(free indices, full(y), G(y), ok(y)) of the system with unknown `pin` held at
        _EDGE and its equation dropped (pin None: the whole system)."""
        free = [i for i in range(len(self.names)) if i != pin]
        full = (lambda y: list(y)) if pin is None else \
            (lambda y: [(_EDGE if i == pin else y[free.index(i)]) for i in range(len(self.names))])
        G = lambda y: [g for i, g in enumerate(self.ev(full(y), rule)) if i != pin]
        return free, full, G, lambda y: self.ok(full(y))

    def _search(self, seeds, pin=None):
        """Fast-rule Newton from each seed: the distinct converged points (full vectors)."""
        free, full, G, ok = self._reduced(pin, "fast")
        out = []
        for s in seeds:
            s = [s[i] for i in free]
            if not ok(s):
                continue
            v = _newton(G, s, ok, 1e-7)
            if v is not None and not any(np.abs(np.array(full(v)) - u).max() < _SAME
                                         for u in out):
                out.append(np.array(full(v)))
        return out

    def _accept(self, v, pin):
        """Polish candidate v with the main rule and audit it.  The candidate is re-polished
        with a refined rule (_rsb_num.MAIN_LEVELS) only when the main rule itself disagrees
        with the trapezoid rule there (|main - trapezoid| > tol/10 on the equations): a point
        that fails with an accurate main rule is not a root.  Returns (kind, r): "root",
        "degenerate", "edge-none" (pinned edge: the other equations hold, the q0 equation
        fails: no edge root there) or "dropped" (a root may exist: r = (v, why))."""
        free = [i for i in range(len(self.names)) if i != pin]
        red, vf = [v[i] for i in free], v
        for rule in MAIN_LEVELS:
            _, full, G, ok = self._reduced(pin, rule)
            w = _newton(G, red, ok, _POLISH, maxit=12)
            if w is not None:
                red, vf = w, np.array(full(w))
            try:
                r = self.audit(vf, pin is not None, rule)
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                return "dropped", (vf, f"audit not evaluable ({type(e).__name__}, rule {rule})")
            if self._degenerate(r["values"]):
                return "degenerate", r
            if w is not None and r["residual"] < self.tol:
                r["_rule"] = rule
                return "root", r
            trap = r["_trap"]
            if w is not None and pin is not None and abs(trap[pin]) >= self.tol and \
                    max([abs(y) for i, y in enumerate(trap) if i != pin] or [0]) < self.tol:
                return "edge-none", r
            why = ((f"the polish did not converge with main rule {rule}" if w is None else
                    f"trapezoid residual {mp.nstr(r['residual'], 3)} > tol {self.tol} with "
                    f"main rule {rule}") + f" (|main - trapezoid| {mp.nstr(r['quad_error'], 3)})")
            if not r["quad_error"] > self.tol / 10:     # the main rule is accurate here: stop
                break
        return "dropped", (vf, why)

    # -- 4, 5: audits and classification --
    def audit(self, v, edge, rule="main"):
        main = self.ev(v, rule) + [self.psi_value(v, rule)]
        trap = self.ev(v, "audit") + [self.psi_value(v, "audit")]
        vals = {nm: mp.mpf(float(y)) for nm, y in zip(self.names, v)}
        for (x, _), y in zip(self.elim, self.conj(v, rule)):
            vals[x.name] = mp.mpf(float(y))
        for k, y in self.fixed.items():
            vals[k] = mp.mpf(str(y))
        for k, y in (self.P.fixed or {}).items():
            if y is not None:
                vals.setdefault(k.name, mp.mpf(str(y)))
        return {"values": vals, "phi": mp.mpf(float(trap[-1])),
                "residual": mp.mpf(float(max(abs(y) for y in trap[:-1]))),
                "quad_error": mp.mpf(float(max(abs(a - b) for a, b in zip(main, trap)))),
                "edge": edge, "_v": np.array(v, float), "_trap": trap[:-1], "_rule": rule}

    def _degenerate(self, vv):
        return ("q0" in vv and "q1" in vv and abs(vv["q1"] - vv["q0"]) < _DEGENERATE) or \
            ("q1" in self.names and "x" in vv and vv["x"] > 1 - _DEGENERATE)

    def run(self):
        self.setup()
        seeds = self.seed_list()
        self.n_seeds = len(seeds)
        cands = [(v, None) for v in self._search(seeds)]
        if "q0" in self.names:
            self.edge_searched = True
            q = self.names.index("q0")
            cands += [(v, q) for v in self._search(seeds, pin=q)]
        roots, dropped, degen, edge_none = [], [], 0, 0
        for v, pin in cands:
            kind, r = self._accept(v, pin)
            if kind == "root":
                # several fast-rule points can polish to one root: keep it once
                if not any(np.abs(r["_v"] - u["_v"]).max() < 1e-6 for u in roots):
                    roots.append(r)
            elif kind == "degenerate":
                degen += 1
            elif kind == "edge-none":
                edge_none += 1
            else:
                dropped.append((r[0], ("q0 = 0 edge candidate: " if pin is not None else "")
                                + r[1]))
        if self.audit_mode == "independent" and roots:
            self.audit_setup()
            for r in list(roots):
                try:
                    res, psi, dev = independent_audit(self, r["_v"], r["_rule"])
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    roots.remove(r)
                    dropped.append((r["_v"], f"independent audit not evaluable "
                                              f"({type(e).__name__})"))
                    continue
                r["residual"], r["phi"] = mp.mpf(res), mp.mpf(psi)
                r["quad_error"] = max(r["quad_error"], mp.mpf(dev))
                if not res < self.tol:
                    roots.remove(r)
                    dropped.append((r["_v"], f"independent (QuadNum) residual "
                                              f"{mp.nstr(res, 3)} > tol {self.tol}"))
        if edge_none:
            self.notes.append(f"q0 = 0 edge: {edge_none} pinned solution(s) satisfy the other "
                              "equations but not the q0 equation (no edge root there)")
        if degen:
            self.notes.append(f"{degen} rs-degenerate root(s) (q1 = q0 or x = 1: 1RSB "
                              "collapses to RS) not counted as 1RSB roots")
        refined = sorted({r["_rule"] for r in roots} - {"main"})
        if refined:
            self.notes.append(f"main rule refined to {', '.join(refined)} for some roots (the "
                              "audit disagreed with the base rule)")
        return select(self, roots, self._blocking(dropped, roots))

    def _blocking(self, dropped, roots):
        """[(v, why)]: candidates dropped by the polish or an audit (a root may exist there)
        that are neither rs-degenerate nor close to an accepted root (near q0 = 0: to an
        accepted edge root, q0 ignored)."""
        out = []
        for v, why in dropped:
            vv = dict(zip(self.names, v))
            if self._degenerate(vv):
                continue
            skip = [self.names.index("q0")] if "q0" in self.names and vv["q0"] < 1e-3 else []
            near = any(np.abs(np.delete(v - r["_v"], skip)).max() < 1e-3 for r in roots)
            if not near and not any(np.abs(v - u).max() < 1e-6 for u, _ in out):
                out.append((v, why))
        return out
