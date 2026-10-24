"""Numerical solution of the DERIVED 1RSB functional (derive_rsb.py -> numbers).

  solve_derived_1rsb(spec, *, frozen=False, zero_temperature=False, branch=None, seeds=None,
                     fixed=None, solve_for=None, audit="independent", audit_tol="1e-9",
                     derived=None, **params) -> DerivedRSBSolution

No family name appears here: everything model-specific arrives through
derive_rsb.derive_1rsb(spec, frozen, zero_temperature).
 1. Functional.  phi(q, qhat, x, extras); the conjugates qhat_j = S_j are eliminated through
    d phi/d q_j = 0 (derive_rsb_t0.eliminate_conjugates), which leaves the stationary points
    unchanged; the unknowns are the order parameters left (q0, q1, x = Parisi m, extras such as
    the Hopfield m; frozen: q0, x; T = 0: q0, mu = beta x).  `fixed` = {name: value} pins
    order parameters (their equations are still imposed); `solve_for` = a model parameter
    made an unknown (the frozen threshold, where d phi/d x = 0 forces S_RS = 0; frozen with
    solve_for and no `fixed`: x is fixed to 1, noted).  Every model parameter needs a value
    (else precondition:complete-point) and every given name must be one this functional reads
    (else precondition:known-parameter, naming the offending and the accepted names).
 2. Equations.  d psi/d p = phi_p + sum_j phi_{qhat_j} dS_j/dp at (v, S(v)) (two-stage: no
    expectation is substituted into another), erfc kept as written; one round of Gaussian
    integration by parts (gauss.stein) when no expectation is nested (T = 0, frozen).
 3. Search (_rsb_num rules: "fast" Gauss-Hermite for the search, "main" composite
    Gauss-Legendre for the polish).  Newton with a central-difference Jacobian, steps of at
    most min(0.2, |v|/2) per coordinate, halved to stay in the domain (0 < q0 < q1 < 1,
    0 < x < 1, mu > 0, the conditions the reduction ASSUMED > 0 and qhat > 0 for the
    eliminated conjugates), from every seed: `seeds` (a list of {name: value}) or the default
    table plus, for a full finite-T 1RSB, three seeds built from the DERIVED RS solution
    (q0 = a q_RS, q1 = b q_RS, extras from the RS root).  The q0 = 0 edge is searched too
    (q0 pinned at 1e-9, its own equation then audited there); the q1 = q0 edge is the RS
    collapse (5).  Each converged point is polished with the main rule to max |eq| < 1e-10.
    An absent root means "not found from these seeds": every result note and every refusal
    states the seed count and which edges were searched.
 4. Audits.  Every candidate: the equations with the trapezoid rule ("audit": same compiled
    float64 integrands, other nodes: a QUADRATURE check only), max |eq| < audit_tol.  Every
    accepted root, audit="independent" (default): the equations and psi through
    _rsb_num.QuadNum, a separately compiled scalar (math-module) evaluation with adaptive
    QUADPACK: an INDEPENDENT evaluation path; residual = its max |eq|, quad_error = max
    |main - trapezoid|, |main - QuadNum|.  audit="quadrature" skips it (sweeps; the result
    note then says the check is quadrature-only).
 5. Classification.  q1 - q0 < 1e-4, or x > 1 - 1e-4 with q1 an unknown: "rs-degenerate" (RS
    collapse; counted in the note, not a 1RSB root).  "trivial": every q ~ 0 off the edge.
    The others "nontrivial" if unique, else "nontrivial-1", ... by decreasing psi; never the
    unique name while a nontrivial candidate failed to polish.  branch=None: the unique
    nontrivial root, several (or one plus unpolished candidates) -> ambiguous:several-saddles;
    none -> no-solution.  branch="<name>" by name.  branch="max-f": the n -> 0 rule, f =
    -psi/beta maximal: the nontrivial root of smallest psi, which must be a local maximum of f
    (solve_derived_rsb_post.schur_hessian: the Schur complement of the derived phi's Hessian
    over the conjugates, positive definite), else no-solution:not-a-local-maximum.
 6. Outputs.  free_energy = psi (= phi: -beta f, or the entropy per variable; T = 0: lim
    phi/beta = -e0, energy = -psi); hessian = the reduced eigenvalues of 5 at the picked
    root; stable = None always (the 1RSB replicon is E4's): never final.
Honest label (CAVEAT): physics-level (Parisi 1RSB ansatz, n -> 0); the quadrature errors are
estimates, not bounds.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from . import derive_rsb as R
from .core import Refusal, Trace
from .families.common import CAVEAT
from .solve_derived import lift_numeric_beta

_SEEDS = {"q0": (0.1, 0.2, 0.3, 0.05, 0.4, 0.15, 0.5), "q1": (0.5, 0.7, 0.8, 0.9, 0.9, 0.6, 0.95),
          "x": (0.2, 0.3, 0.4, 0.15, 0.6, 0.25, 0.5), "mu": (1.0, 2.0, 0.5, 1.5, 3.0, 0.8, 1.2),
          "qhat0": (1.0, 2.0, 0.5, 3.0, 0.2, 1.5, 4.0),
          "qhat1": (3.0, 5.0, 2.0, 8.0, 1.0, 4.0, 6.0),
          "m": (0.95, 0.5, 0.8, 0.99, 0.3, 0.9, 0.7)}
_OTHER = (0.5, 1.0, 0.8, 0.3, 0.9, 0.6, 1.2)
_SAME, _DEGENERATE, _ZERO, _POLISH, _EDGE = 1e-7, 1e-4, 1e-10, 1e-10, 1e-9
_RS_SEEDS = ((0.5, 1.15, 0.5), (0.3, 1.2, 0.3), (0.8, 1.05, 0.7))


@dataclass(frozen=True)
class DerivedRSBSolution:
    values: dict                 # unknown / eliminated / fixed name -> mpf
    free_energy: object          # psi at the root (module docstring, 6)
    m: object                    # Parisi block parameter x (None at T = 0: mu = beta x)
    q0: object
    q1: object
    residual: object             # max |eq| on the independent path (module docstring, 4)
    roots: list                  # every audited root (dicts), rs-degenerate ones excluded
    stable: bool | None          # always None here: 1RSB stability is not computed
    trace: Trace
    caveat: str = CAVEAT
    branch: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    quad_error: object = None
    hessian: tuple = ()          # reduced-Hessian eigenvalues at the root (docstring, 5)
    energy: object = None        # T = 0: e0 = -psi
    equations_used: list = field(default_factory=list)


def _domain_ok(names, v):
    d = dict(zip(names, v))
    if any(k in d and not 0 < d[k] < 1 for k in ("q0", "q1", "x")):
        return False
    if any(k in d and not d[k] > 0 for k in ("mu", "qhat0", "qhat1")):
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
                       fixed=None, solve_for=None, audit="independent", audit_tol="1e-9",
                       derived=None, **params) -> DerivedRSBSolution:
    if audit not in ("independent", "quadrature"):
        raise Refusal("precondition:known-audit",
                      f"audit = {audit!r}: use 'independent' or 'quadrature'")
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
    from .solve_derived_rsb_search import Solver
    return Solver(spec, P, eqs, trace, branch, seeds, dict(fixed or {}), solve_for,
                  float(audit_tol), params, notes, zero_temperature, audit).run()

