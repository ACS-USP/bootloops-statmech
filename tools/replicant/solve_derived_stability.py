"""Stability of a derived RS solution (helper of solve_derived.py; E1).

  assess(state, variables, evaluate, entropy) -> Assessment(stable, replicon, route,
                                                           mode, note, quad_error)
  derived_stability(spec, point, ...)         -> Assessment at an audited RS point (a
                                                 mapping or a DerivedRSSolution; see
                                                 derive_replicon.at_point)

`evaluate(expr)` returns (value, |disagreement|, rules) of a sympy expression at the
solution (solve_derived binds the point and _derived_num.checked).  Computed here:
  replicon   derive_replicon.replicon(state) (None: no rule for this family)
  mode       when the functional has extra replica-indexed order parameters (Hopfield
             m_a, derive.Functional.extra): the longitudinal mode of those extras,
             mode = -d^2 phi_red / dx^2 (smallest eigenvalue when there are several), with
             phi_red(x) = phi(x, y*(x)) the RS functional stationary in the other RS
             parameters y (q, qhat):  d^2 phi_red/dx^2 = phi_xx - phi_xy phi_yy^-1 phi_yx,
             the Hessian of the derived phi by sympy differentiation (expectations
             differentiated under E_z, one Stein round).  phi = -beta f is maximal in
             a physical order parameter, so mode > 0 is the stable sign (for Hopfield,
             mode / beta = d^2 f_red/dm^2, the family's mode_eig; a target in the tests).
STABILITY CONTRACT (solve_derived): stable = True only if the replicon is derived and
> 0, the extras' mode (if any) is computed and > 0, and for discrete variables the RS
entropy is >= 0; False if any computed one fails; else None (UNKNOWN).  The note names
which modes were computed.  Local stability only, physics-level.
"""
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from .core import Refusal
from .derive_replicon import replicon as derive_replicon

_DISCRETE = ("ising",)          # spec.variables with a counting measure (entropy >= 0)


@dataclass(frozen=True)
class Assessment:
    stable: bool | None
    replicon: object
    route: str
    mode: object                # extras' mode (None: no extras or not computed)
    note: str
    quad_error: object          # largest fine-panel disagreement used (0 if none)


def _mode(state, evaluate):
    """(mode, note, qerr) of the extra order parameters, or (None, reason, 0)."""
    F, P = state["functional"], state["rs_free"]
    names = set(F.extra or {})
    xs = [p for p in P.params if p.name in names]
    ys = [p for p in P.params if p.name not in names]
    phi, qerr = P.expr, mp.mpf(0)

    def h(a, b):
        nonlocal qerr
        v, d, rules = evaluate(sp.diff(phi, a, b))
        if rules.startswith("fine"):
            qerr = max(qerr, d)
        return v
    Hxx = mp.matrix([[h(a, b) for b in xs] for a in xs])
    Hxy = mp.matrix([[h(a, b) for b in ys] for a in xs])
    Hyy = mp.matrix([[h(a, b) for b in ys] for a in ys])
    try:
        S = Hxx - Hxy * mp.inverse(Hyy) * Hxy.T
    except ZeroDivisionError:
        return None, "extras' mode not computed: d^2 phi/d(q, qhat)^2 is singular", qerr
    ev = mp.eigsy(-S)[0] if len(xs) > 1 else [-S[0, 0]]
    mode = min(ev)
    return mode, (f"mode of the extra order parameters {sorted(names)}: -d^2 phi_red = "
                  f"{mp.nstr(mode, 8)} (phi stationary in {[str(y) for y in ys]})"), qerr


def assess(state, variables, evaluate, entropy) -> Assessment:
    rep = derive_replicon(state)
    qerr, notes = mp.mpf(0), []
    lam = None
    if rep.expr is not None:
        lam, d, rules = evaluate(rep.expr)
        notes.append(f"replicon {mp.nstr(lam, 8)} ({rep.note}; quadrature: {rules}, "
                     f"|difference| {mp.nstr(d, 3)})")
        if rules.startswith("fine"):
            qerr = d
    else:
        notes.append(rep.note)
    mode = None
    flags = [] if lam is None else [lam > 0]
    if state["functional"].extra:
        try:
            mode, mnote, mq = _mode(state, evaluate)
        except Refusal as r:
            mode, mnote, mq = None, f"extras' mode not computed: {r.reason}: {r.detail}", 0
        qerr = max(qerr, mq)
        notes.append(mnote)
        if mode is not None:
            flags.append(mode > 0)
    neg_ent = variables in _DISCRETE and entropy < 0
    complete = lam is not None and (not state["functional"].extra or mode is not None)
    if neg_ent or (flags and not all(flags)):
        stable = False
    else:
        stable = True if complete else None
    computed = ["replicon"] * (lam is not None) + ["extras' mode"] * (mode is not None)
    head = ("modes computed: " + (", ".join(computed) or "none")
            + ("" if stable is not None else "; stable=None means UNKNOWN"))
    if neg_ent:
        head = (f"RS entropy {mp.nstr(entropy, 6)} < 0 for discrete variables ({variables}): "
                "RS is wrong here; " + head)
    return Assessment(stable, lam, rep.route if lam is not None else "", mode,
                      "; ".join([head] + notes), qerr)


def derived_stability(spec, point, *, dps=20, saddle_tol="1e-9") -> Assessment:
    """assess() at an RS point that is first checked to be a root of the derived RS
    equations (the same entry the solver's label goes through: a root the seeded search
    does not reach, e.g. a non-attracting lower Hopfield retrieval root, gets the same
    label it would get from solve_derived_rs)."""
    from .derive_replicon import at_point
    state, evaluate, _ = at_point(spec, point, dps=dps, saddle_tol=saddle_tol)
    P = state["rs_free"]
    b = [s for s in P.expr.free_symbols if s.name == "beta"]
    ent = evaluate(P.expr - b[0] * sp.diff(P.expr, b[0]))[0] if b else evaluate(P.expr)[0]
    return assess(state, spec.variables, evaluate, ent)
