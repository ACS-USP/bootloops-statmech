"""Solving the DERIVED 1RSB functional (solve_derived_rsb.py) against targets that the
derivation never reads: families/sk.py (hand-built 1RSB, float Gauss-Hermite), the banked
Krauth-Mezard capacity and the banked SK T = 0 1RSB energy (reference/*.json).

Conventions as in test_derive_rsb.py (x = Parisi m; phi = -beta f, or the entropy).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import default_spec
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.solve_derived import solve_derived_rs
from replicant.solve_derived_rsb import solve_derived_1rsb
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


def _ising_perceptron():
    s = default_spec("perceptron")
    return ModelSpec("perceptron", "ising", s.disorder, s.constraint, s.prior, dict(s.params))


def _frozen_threshold(spec=None, x=1, **kw):
    return solve_derived_1rsb(spec or _ising_perceptron(), frozen=True, fixed={"x": x},
                              solve_for="alpha", kappa=0, **kw)


def test_ising_perceptron_frozen_1rsb_zero_of_entropy():
    """The frozen derivation gives phi = S_RS(q0, x^2 qhat0)/x; its x-equation forces
    S_RS = 0, so solving it for alpha (x fixed to 1) is a root of the derived RS entropy.
    TARGET: the banked Krauth-Mezard capacity (Ding-Sun bracket), checked through refdata.
    (Default tier: trapezoid audit only; the independent QUADPACK audit of the same root is
    in test_frozen_threshold_is_independent_of_the_block_size, slow.)"""
    sol = _frozen_threshold(audit="quadrature")
    bank = load("perceptron")
    a = sol.values["alpha"]
    assert abs(a - mp.mpf("0.8330786")) < 1e-6, a
    assert check(bank["ising_1rsb_capacity"], mp.nstr(a, 15), label="frozen alpha_c").ok
    assert check(bank["ising_1rsb_q"], mp.nstr(sol.q0, 15), label="q0 at alpha_c").ok
    assert abs(sol.free_energy) < 1e-9 and sol.residual < 1e-9 and sol.stable is None
    # positive control: the banked check rejects the RS endpoint of the same entropy
    assert not check(bank["ising_1rsb_capacity"], "1.2732395447", label="4/pi").ok


@slow
def test_frozen_threshold_is_independent_of_the_block_size():
    """x is degenerate at the threshold: x = 1/2 gives the same alpha and q0, with
    x^2 qhat0 the same RS conjugate."""
    one, half = _frozen_threshold(), _frozen_threshold(x="0.5")
    assert "QUADPACK" in one.note and one.residual < 1e-9
    assert check(load("perceptron")["ising_1rsb_capacity"], mp.nstr(one.values["alpha"], 15),
                 label="frozen alpha_c, independent audit").ok
    assert abs(one.values["alpha"] - half.values["alpha"]) < 1e-9
    assert abs(one.q0 - half.q0) < 1e-9
    assert abs(one.values["qhat0"] - half.values["qhat0"] / 4) < 1e-8


def test_planted_error_in_1rsb_hook_is_caught():
    """A planted error in the perceptron energy hook (Reduction energy x 11/10) moves the
    derived frozen capacity off the bank; the registry is restored (identity checked)."""
    fam = D.FAMILIES["perceptron"]
    orig = fam.order_parameters

    def planted(avg, n, sum_pairs):
        r = orig(avg, n, sum_pairs)
        return D.Reduction(r.energy * 11 / 10, r.correction, r.missing, r.field, r.extra)
    D.FAMILIES["perceptron"] = D.Family(fam.formal, fam.annealed, fam.supports_disorder,
                                        fam.average, planted, fam.variables, fam.constraints,
                                        fam.priors)
    try:            # (the trapezoid audit suffices to see the shift; the default tier is tight)
        bad = _frozen_threshold(audit="quadrature")
    finally:
        D.FAMILIES["perceptron"] = fam
    assert D.FAMILIES["perceptron"] is fam and fam.order_parameters is orig
    bank = load("perceptron")["ising_1rsb_capacity"]
    assert not check(bank, mp.nstr(bad.values["alpha"], 15), label="planted").ok
    assert abs(bad.values["alpha"] - mp.mpf("0.8330786")) > 1e-2, bad.values


_HOP = {"alpha": "0.05", "beta": "3"}
_HOP_SEED = [{"q0": 0.3, "q1": 0.75, "x": 0.08, "m": 0.0}]


@slow
def test_sk_derived_1rsb_matches_sk_1rsb():
    """SK at h = 0 has two nontrivial 1RSB saddles at T = 0 and at beta = 2: the q0 > 0 one
    of the family and one on the q0 = 0 edge (searched since fix round 1).  Without a
    selector the solver refuses; branch="max-f" (the n -> 0 rule: f maximal) picks the
    family's root and checks it is a local maximum of f (Schur complement over the
    conjugates)."""
    from replicant.families.sk import _e0_1rsb, sk_1rsb, sk_ground_energy
    r = _refusal(lambda: solve_derived_1rsb(_sk(), zero_temperature=True, h=0,
                                            audit="quadrature"))
    assert r.reason == "ambiguous:several-saddles" and "q0 = 0 edge" in r.detail, r
    t0 = solve_derived_1rsb(_sk(), zero_temperature=True, h=0, branch="max-f")
    assert abs(t0.energy - sk_ground_energy("1rsb")) < 1e-7, t0.energy
    assert check(load("sk")["rsb1_ground_energy"], mp.nstr(t0.energy, 15), label="e0").ok
    assert all(e > 0 for e in t0.hessian), t0.hessian
    edge = [x for x in t0.roots if x["values"]["q0"] < 1e-6]
    assert len(edge) == 1, t0.roots
    mu = edge[0]["values"]["mu"]          # target: the family's own E(q0 = 0, mu)
    assert abs(-edge[0]["phi"] - _e0_1rsb(0.0, float(mu))) < 1e-9
    assert abs(-edge[0]["phi"] - mp.mpf("-0.76883")) < 1e-5 and -edge[0]["phi"] < t0.energy
    fam = sk_1rsb(2)
    r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=2, h=0, audit="quadrature"))
    assert r.reason == "ambiguous:several-saddles" and "found from" in r.detail, r
    sol = solve_derived_1rsb(_sk(), beta=2, h=0, branch="max-f")
    assert abs(-sol.free_energy / 2 - fam.f) < 1e-8, (-sol.free_energy / 2, fam.f)
    for got, want in ((sol.q0, fam.q0), (sol.q1, fam.q1), (sol.m, fam.m)):
        assert abs(got - want) < 1e-8, (got, want)
    assert sol.residual < 1e-9 and sol.stable is None and "QUADPACK" in sol.note
    assert all(e > 0 for e in sol.hessian), sol.hessian
    edge = [x for x in sol.roots if x["values"]["q0"] < 1e-6]
    assert len(edge) == 1 and -edge[0]["phi"] / 2 < fam.f, sol.roots
    assert abs(edge[0]["values"]["q1"] - mp.mpf("0.60132")) < 1e-4
    # a field: h = 0.3 at beta = 2 (RS AT-unstable there), default seeds (the table plus
    # seeds from the DERIVED RS solution; no family number enters)
    famh = sk_1rsb(2, 0.3)
    solh = solve_derived_1rsb(_sk(), beta=2, h="0.3")
    assert abs(-solh.free_energy / 2 - famh.f) < 1e-8, (-solh.free_energy / 2, famh.f)


@slow
def test_schur_and_conjugate_elimination_hessians_agree_for_sk():
    """The Hessian with qhat eliminated by d phi/d q (the T = 0 functional is built that
    way) and the Schur complement over qhat agree in sign for SK at beta = 2: support for
    the T = 0 local-maximum label (not a proof at T = 0)."""
    from replicant.solve_derived_rsb_post import eliminated_hessian, schur_hessian
    sv = _solver(_sk(), {"beta": 2, "h": 0})
    v = [0.2146310645736, 0.6192008853559, 0.2818426445636]
    assert all(e > 0 for e in schur_hessian(sv, v)[0])
    assert all(e > 0 for e in eliminated_hessian(sv, v))


def _solver(spec, params):
    from replicant import derive_rsb as R
    from replicant.solve_derived import lift_numeric_beta
    from replicant.solve_derived_rsb_search import Solver
    sp_, params, _ = lift_numeric_beta(spec, params)
    eqs, st, tr = R.derive_1rsb(sp_)
    sv = Solver(sp_, st["rsb_free"], eqs, tr, None, None, {}, None, 1e-9, params, [], False)
    sv.setup()
    return sv


@slow
def test_ising_perceptron_full_1rsb_collapses_to_rs_no_capacity_claimed():
    """Full 1RSB (q1 < 1 free) at alpha = 0.8, below the frozen capacity, with the default
    seeds (table and derived-RS seeds) and the q0 = 0 edge: every root reached is
    rs-degenerate (q1 = q0 or x = 1: it collapses to RS, AT-stable there).  So no full-1RSB
    capacity is claimed (no independent target exists for one); the 1RSB capacity this
    package reports is the frozen one, below the banked RS endpoint 4/pi."""
    r = _refusal(lambda: solve_derived_1rsb(_ising_perceptron(), alpha="0.8", kappa=0))
    assert r.reason == "no-solution" and "rs-degenerate" in r.detail, r
    assert "seeds" in r.detail and "edge" in r.detail, r
    a = _frozen_threshold().values["alpha"]
    assert a < mp.mpf(load("perceptron")["ising_rs_capacity"].value) - mp.mpf("0.4")


@slow
def test_hopfield_1rsb():
    """No banked Hopfield 1RSB target (the T = 0 1RSB limit for a Gaussian-integral energy
    is refused, so a published T = 0 value could not be checked by any code route).
    Checks: the x = 1 reduction (test_derive_rsb), the T = 0 refusal, and a finite-T
    spin-glass root at alpha = 0.05, beta = 3 (RS AT-unstable there: derived replicon -1.03
    in E1) that is a genuine 1RSB point: f_1RSB > f_RS, q0 < q_RS < q1, and a local maximum
    of f (Schur complement of the derived phi's Hessian over the conjugates positive
    definite, the conjugate block negative definite)."""
    spec = default_spec("hopfield")
    r = _refusal(lambda: solve_derived_1rsb(spec, zero_temperature=True, alpha="0.05"))
    assert r.reason == "precondition:polynomial-energy-for-t0", r
    sol = solve_derived_1rsb(spec, seeds=_HOP_SEED, branch="max-f", **_HOP)
    rs = solve_derived_rs(spec, branch="nontrivial-1", **_HOP)
    assert abs(rs.values["m"]) < 1e-12 and abs(sol.values["m"]) < 1e-12
    assert sol.q0 < rs.values["q"] < sol.q1 and 0 < sol.m < 1
    assert sol.free_energy < rs.free_energy - 1e-3        # f_1RSB > f_RS
    assert sol.residual < 1e-9 and sol.quad_error < 1e-9 and sol.stable is None
    assert all(e > 0 for e in sol.hessian), sol.hessian
    assert "no independent target" not in sol.note        # the composer says that, not here


@slow
def test_wrong_reduction_disagrees_with_the_schur_complement_at_the_hopfield_root():
    """Positive control for the local-maximum check: eliminating qhat through d phi/d q
    (the reduction used before fix round 1) gives a Hessian with a negative eigenvalue at
    the Hopfield root, the Schur complement over qhat (qhat through its own equation) is
    positive definite.  The check must use the latter."""
    from replicant.solve_derived_rsb_post import eliminated_hessian, schur_hessian
    sol = solve_derived_1rsb(default_spec("hopfield"), seeds=_HOP_SEED, **_HOP)
    sv = _solver(default_spec("hopfield"), dict(_HOP))
    v = [float(sol.values[k]) for k in sv.names]
    schur, conj = schur_hessian(sv, v)
    assert all(e > 0 for e in schur) and all(e < 0 for e in conj), (schur, conj)
    assert min(eliminated_hessian(sv, v)) < 0


@slow
def test_solver_refusals_are_named():
    r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=2))
    assert r.reason == "precondition:complete-point" and "h" in r.detail, r
    r = _refusal(lambda: solve_derived_1rsb(_ising_perceptron(), frozen=True, kappa=0,
                                            fixed={"x": 1, "q7": 0}, solve_for="alpha"))
    assert r.reason == "precondition:known-order-parameter", r


def test_unknown_parameters_are_refused():
    r = _refusal(lambda: _frozen_threshold(kapa="0.3"))
    assert r.reason == "precondition:known-parameter", r
    assert "kapa" in r.detail and "accepted: kappa" in r.detail, r
