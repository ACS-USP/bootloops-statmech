"""Fix round 2 of E2: edge candidates and the main-rule window of solve_derived_1rsb.

An edge candidate (q0 pinned at the q0 = 0 edge) that fails only on the polish or an audit is
NOT "no edge root": it blocks the unique label like any dropped nontrivial candidate.  When
the audit disagrees with the main rule the candidate is re-polished with refined rules
(_rsb_num.MAIN_LEVELS: 32, 64, 128 Gauss-Legendre panels): SK at beta = 4 needs 64.
Conventions as in test_derive_rsb.py (x = Parisi m, phi = -beta f).
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import _rsb_num as N
from replicant import derive as D
from replicant.core import Refusal
from replicant.solve_derived_rsb import solve_derived_1rsb
from replicant.solve_derived_rsb_post import select
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _sk():
    return ModelSpec("sk", "ising", "gaussian", params={"beta": D.BETA, "h": D.H_FIELD})


class _Fake:
    """Just what select() reads of a solver."""
    names, branch, tol, solve_for, t0 = ["q0", "q1", "x"], None, 1e-9, None, False

    def __init__(self):
        self.notes = []

    def contract(self):
        return "roots found from 1 seed"

    def audit_label(self):
        return "audit label"


def test_a_dropped_candidate_blocks_the_unique_label():
    root = {"values": {"q0": mp.mpf("0.3"), "q1": mp.mpf("0.6"), "x": mp.mpf("0.3")},
            "phi": mp.mpf(1), "residual": mp.mpf(0), "quad_error": mp.mpf(0), "edge": False,
            "_v": np.array([0.3, 0.6, 0.3]), "_rule": "main"}
    blocked = [(np.array([1e-9, 0.5, 0.2]), "q0 = 0 edge candidate: trapezoid residual 3e-9")]
    r = _refusal(lambda: select(_Fake(), [dict(root)], blocked))
    assert r.reason == "ambiguous:several-saddles", r
    assert "q0 = 0 edge candidate: trapezoid residual 3e-9" in r.detail, r
    r = _refusal(lambda: select(_Fake(), [], blocked))
    assert r.reason == "no-solution" and "a root may exist there" in r.detail, r


@slow
def test_edge_candidate_failing_only_on_audit_is_not_no_edge_root():
    """Degraded main rules (4 Gauss-Legendre panels of 4 nodes, all levels): the pinned edge
    candidate fails on the audit, so the refusal lists it as dropped (a root may exist) and
    never says "no edge root".  The registry is restored (identity checked)."""
    saved = dict(N.RULES)
    y, w = np.polynomial.legendre.leggauss(4)
    edges = np.linspace(-6, 6, 5)
    x = np.concatenate([(a + b) / 2 + (b - a) / 2 * y for a, b in zip(edges, edges[1:])])
    wt = np.concatenate([(b - a) / 2 * w for a, b in zip(edges, edges[1:])])
    bad = (x, wt * np.exp(-x * x / 2) / np.sqrt(2 * np.pi))
    try:
        for k in N.MAIN_LEVELS:
            N.RULES[k] = bad
        r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=2, h=0, audit="quadrature"))
    finally:
        N.RULES.clear()
        N.RULES.update(saved)
    assert all(N.RULES[k] is saved[k] for k in saved)
    assert "no edge root" not in r.detail, r.detail
    assert "q0 = 0 edge candidate" in r.detail and "a root may exist" in r.detail, r.detail


@slow
def test_sk_beta_4_lists_both_roots_with_a_refined_rule():
    """beta = 4, h = 0: the base main rule drifts by about 1e-8; the refined rule resolves
    both saddles, so the solver refuses ambiguity (no silent unique root) and max-f picks the
    q0 > 0 root, a local maximum of f.  No family target (families/sk.py refuses beta > 3)."""
    r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=4, h=0, audit="quadrature"))
    assert r.reason == "ambiguous:several-saddles" and "(q0 = 0 edge)" in r.detail, r
    sol = solve_derived_1rsb(_sk(), beta=4, h=0, branch="max-f")
    edge = [x for x in sol.roots if x["values"]["q0"] < 1e-6]
    assert len(edge) == 1 and abs(edge[0]["values"]["q1"] - mp.mpf("0.85558")) < 1e-4
    assert abs(sol.q0 - mp.mpf("0.33991")) < 1e-4 and -sol.free_energy / 4 > -edge[0]["phi"] / 4
    assert abs(-sol.free_energy / 4 - mp.mpf("-0.767198")) < 1e-6
    assert all(e > 0 for e in sol.hessian) and "main2" in sol.note, sol.note
    assert sol.residual < 1e-9


@slow
def test_sk_beta_3_unchanged():
    from replicant.families.sk import sk_1rsb
    r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=3, h=0, audit="quadrature"))
    assert r.reason == "ambiguous:several-saddles" and "(q0 = 0 edge)" in r.detail, r
    fam = sk_1rsb(3)
    sol = solve_derived_1rsb(_sk(), beta=3, h=0, branch="max-f")
    assert abs(-sol.free_energy / 3 - fam.f) < 1e-8, (-sol.free_energy / 3, fam.f)


@slow
def test_t0_sign_agreement_at_several_beta():
    """The T = 0 local-maximum label rests on the d phi/d q reduction; its sign agrees with
    the Schur complement at the SK max-f root for beta = 1.5, 2, 3 (evidence, not a proof
    at T = 0)."""
    from replicant.solve_derived import lift_numeric_beta
    from replicant import derive_rsb as R
    from replicant.solve_derived_rsb_post import eliminated_hessian, schur_hessian
    from replicant.solve_derived_rsb_search import Solver
    for beta in ("1.5", "2", "3"):
        sol = solve_derived_1rsb(_sk(), beta=beta, h=0, branch="max-f", audit="quadrature")
        spec, params, _ = lift_numeric_beta(_sk(), {"beta": beta, "h": 0})
        eqs, st, tr = R.derive_1rsb(spec)
        sv = Solver(spec, st["rsb_free"], eqs, tr, None, None, {}, None, 1e-9, params, [], False)
        sv.setup()
        v = [float(sol.values[k]) for k in sv.names]
        assert all(e > 0 for e in schur_hessian(sv, v)[0]), beta
        assert all(e > 0 for e in eliminated_hessian(sv, v)), beta
