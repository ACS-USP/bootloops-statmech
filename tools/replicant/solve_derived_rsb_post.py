"""Post-processing of the 1RSB root search (helper of solve_derived_rsb.py): the independent
audit, the local-maximum check, classification, selection and outputs.

  independent_audit(solver, v)
                             every equation and psi at root v through _rsb_num.QuadNum (scalar
                             math-module code, adaptive QUADPACK), not the numpy product rules
  schur_hessian(solver, v)   (eigenvalues of the reduced Hessian, eigenvalues of the
                             conjugate block): the Hessian of the derived phi in (q0, q1, x /
                             mu) and the eliminated conjugates qhat, by central differences
                             (main rule), reduced by the Schur complement H_qq - H_qh H_hh^-1 H_hq,
                             i.e. qhat eliminated through ITS OWN equation d phi/d qhat = 0
  eliminated_hessian(solver, v)
                             the Hessian of psi = phi(q, qhat(q)) with qhat from d phi/d q = 0
                             (the solver's elimination).  Same stationary points, but no
                             max/min meaning in general (1D: its sign is sign(phi_qq phi_hh)
                             times the Schur sign); kept as the positive control of the tests,
                             and the only one at T = 0, whose functional is built that way
Sign convention (as E1's m-mode, phi = -beta f): the n -> 0 saddle MAXIMISES f over the
overlaps and the block size, so a local maximum of f is a positive definite reduced Hessian
of phi; the conjugate block is then negative definite (phi maximal in qhat).  Extras (the
Hopfield m) are held at the root: their mode is E1's (solve_derived_stability), not this one.
"""
import mpmath as mp
import numpy as np

from .core import Refusal, Trace, TraceEntry
from .families.common import CAVEAT

_ORDER = ("q0", "q1", "x", "mu")
_ZERO = 1e-10


def _central(f, w, idx, d=1e-4):
    H = np.empty((len(idx), len(idx)))
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            ei, ej = np.zeros(len(w)), np.zeros(len(w))
            ei[i], ej[j] = d, d
            H[a, b] = (f(w + ei + ej) - f(w + ei - ej) - f(w - ei + ej)
                       + f(w - ei - ej)) / (4 * d * d)
    return (H + H.T) / 2


def schur_hessian(sv, v):
    v = list(map(float, v))
    w = np.array(v + sv.conj(v, "main"))
    qi = [i for i, nm in enumerate(sv.names) if nm in _ORDER]
    hi = list(range(len(sv.names), len(w)))
    H = _central(lambda y: sv.phi_num.value(y, "main"), w, qi + hi)
    k = len(qi)
    if not hi:
        return tuple(map(float, np.linalg.eigvalsh(H))), ()
    Hqq, Hqh, Hhh = H[:k, :k], H[:k, k:], H[k:, k:]
    S = Hqq - Hqh @ np.linalg.solve(Hhh, Hqh.T)
    return (tuple(map(float, np.linalg.eigvalsh((S + S.T) / 2))),
            tuple(map(float, np.linalg.eigvalsh(Hhh))))


def eliminated_hessian(sv, v):
    v = np.array(list(map(float, v)))
    qi = [i for i, nm in enumerate(sv.names) if nm in _ORDER]
    return tuple(map(float, np.linalg.eigvalsh(_central(lambda y: sv.psi_value(y, "main"), v,
                                                         qi))))


def independent_audit(sv, v):
    """(residual, psi, max |main - independent|) of root v through QuadNum."""
    v = list(map(float, v))
    s = [S.value(v) for S in sv.S_ind]
    w = v + s
    B = [b.value(w) for b in sv.B_ind]
    eqs = [a.value(w) + sum(bj * sv.C_ind[j][k].value(v) for j, bj in enumerate(B))
           for k, a in enumerate(sv.A_ind)]
    psi = sv.phi_ind.value(w)
    main = sv.ev(v, "main") + [sv.psi_value(v, "main")]
    return max(abs(e) for e in eqs), psi, max(abs(a - b) for a, b in zip(main, eqs + [psi]))


def _is_edge(r):
    return r.get("edge", False)


def select(sv, roots, blocking):
    """Name the roots, pick one by `sv.branch`, check the pick (module docstring of
    solve_derived_rsb, 5)."""
    contract = sv.contract()
    if not roots:
        raise Refusal("no-solution", "no 1RSB root with q1 > q0 of the derived equations in "
                      f"{sv.names} passes the audits (tol {sv.tol}); {contract}; "
                      + "; ".join(sv.notes))
    triv = [r for r in roots if all(abs(r["values"][k]) < _ZERO for k in ("q0", "q1")
                                    if k in r["values"]) and not _is_edge(r)]
    non = sorted((r for r in roots if r not in triv), key=lambda r: -r["phi"])
    for r in triv:
        r["name"] = "trivial"
    for k, r in enumerate(non, 1):
        r["name"] = "nontrivial" if len(non) == 1 and not blocking else f"nontrivial-{k}"
    table = [{k: y for k, y in r.items() if not k.startswith("_")} for r in triv + non]
    listing = "; ".join(f"{r['name']}" + (" (q0 = 0 edge)" if r.get("edge") else "") + ": "
                        + ", ".join(f"{k}={mp.nstr(y, 10)}" for k, y in r["values"].items())
                        + f", psi={mp.nstr(r['phi'], 12)}" for r in table)
    pend = (f"; {len(blocking)} nontrivial candidate(s) the polish did not converge on: "
            + "; ".join(str([round(float(y), 6) for y in c]) for c in blocking)) if blocking else ""
    if sv.branch is None:
        if len(non) > 1 or (non and blocking):
            raise Refusal("ambiguous:several-saddles",
                          f"{len(non)} nontrivial 1RSB root(s) ({listing}){pend}; {contract}; "
                          "pass branch= one of the names or 'max-f' (the n -> 0 rule: the "
                          "root of largest f = -psi/beta, which must be a local maximum)")
        pick = non[0] if non else triv[0]
    elif sv.branch == "max-f":
        if not non:
            raise Refusal("no-solution", f"no nontrivial root to select ({listing}); {contract}")
        pick = min(non, key=lambda r: r["phi"])
    else:
        hits = [r for r in triv + non if r["name"] == sv.branch]
        if not hits:
            raise Refusal("no-solution", f"branch {sv.branch!r} not found; roots: {listing}; "
                          f"{contract}")
        pick = hits[0]
    if sv.solve_for or pick.get("edge"):
        hess, conj, rule = (), (), ("not computed (" + ("a threshold: solve_for" if sv.solve_for
                                                        else "a q0 = 0 edge root") + ")")
    elif sv.t0:
        hess, conj = eliminated_hessian(sv, pick["_v"]), ()
        rule = ("Hessian of the T = 0 functional (its conjugates were eliminated by d phi/d q in "
                "the derivation; the sign agrees with the Schur complement for SK at finite "
                "beta, tested, not proven at T = 0)")
    else:
        hess, conj = schur_hessian(sv, pick["_v"])
        rule = "Schur complement of the derived phi's Hessian over the conjugates"
    local = bool(hess) and all(e > 0 for e in hess)
    if sv.branch == "max-f" and not local:
        raise Refusal("no-solution:not-a-local-maximum",
                      f"root {pick['name']} (largest f) has reduced-Hessian eigenvalues {hess} "
                      f"({rule}): not a local maximum of f")
    if hess:
        sv.notes.append(f"local-maximum check at {pick['name']}: {rule}, eigenvalues "
                        f"{[round(e, 8) for e in hess]}"
                        + (f", conjugate block {[round(e, 8) for e in conj]}" if conj else "")
                        + (" (local maximum of f)" if local else " (NOT a local maximum of f)"))
    else:
        sv.notes.append(f"local-maximum check: {rule}")
    if pend:
        sv.notes.append(pend[2:])
    return outputs(sv, pick, table, listing, hess, contract)


def outputs(sv, pick, table, listing, hess, contract):
    from .solve_derived_rsb import DerivedRSBSolution
    vals, psi = pick["values"], pick["phi"]
    note = "; ".join(sv.notes + [contract, f"roots: {listing}", sv.audit_label(),
                                 "stable = None: 1RSB stability (replicon toward further RSB) "
                                 "is not computed here"])
    out = Trace()
    for e in sv.trace.entries:
        out.add(e)
    out.add(TraceEntry("solve-derived-1rsb", None,
                       f"branch {pick['name']}; independent residual "
                       f"{mp.nstr(pick['residual'], 3)}; quad_error "
                       f"{mp.nstr(pick['quad_error'], 3)}; {note}"))
    return DerivedRSBSolution(
        vals, psi, vals.get("x"), vals.get("q0"), vals.get("q1"), pick["residual"], table,
        None, out, CAVEAT, pick["name"], note, dict(sv.params), pick["quad_error"], hess,
        -psi if sv.t0 else None, list(sv.eqs))
