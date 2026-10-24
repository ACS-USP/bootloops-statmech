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
    TARGET: the banked Krauth-Mezard capacity (Ding-Sun bracket), checked through refdata."""
    sol = _frozen_threshold()
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
    try:
        bad = _frozen_threshold()
    finally:
        D.FAMILIES["perceptron"] = fam
    assert D.FAMILIES["perceptron"] is fam and fam.order_parameters is orig
    bank = load("perceptron")["ising_1rsb_capacity"]
    assert not check(bank, mp.nstr(bad.values["alpha"], 15), label="planted").ok
    assert abs(bad.values["alpha"] - mp.mpf("0.8330786")) > 1e-2, bad.values


@slow
def test_sk_derived_1rsb_matches_sk_1rsb():
    from replicant.families.sk import sk_1rsb, sk_ground_energy
    t0 = solve_derived_1rsb(_sk(), zero_temperature=True, h=0)
    assert abs(t0.energy - sk_ground_energy("1rsb")) < 1e-7, t0.energy
    assert check(load("sk")["rsb1_ground_energy"], mp.nstr(t0.energy, 15), label="e0").ok
    assert all(e > 0 for e in t0.hessian), t0.hessian        # f = e0 maximal
    fam = sk_1rsb(2)
    sol = solve_derived_1rsb(_sk(), beta=2, h=0)
    assert abs(-sol.free_energy / 2 - fam.f) < 1e-8, (-sol.free_energy / 2, fam.f)
    for got, want in ((sol.q0, fam.q0), (sol.q1, fam.q1), (sol.m, fam.m)):
        assert abs(got - want) < 1e-8, (got, want)
    assert sol.residual < 1e-9 and sol.stable is None
    assert all(e > 0 for e in sol.hessian), sol.hessian
    pick = solve_derived_1rsb(_sk(), beta=2, h=0, branch="replica-max")
    assert abs(pick.free_energy - sol.free_energy) < 1e-12
    # a field: h = 0.3 at beta = 2 (RS AT-unstable there), again against the family
    # (the default seed table misses this root: seeds from the DERIVED RS q, the recipe
    # families/sk.py uses on its own RS q; no family number enters)
    famh = sk_1rsb(2, 0.3)
    q = float(solve_derived_rs(_sk(), beta=2, h="0.3").values["q"])
    seeds = [{"q0": a * q, "q1": min(b * q, 0.99), "x": x}
             for a, b, x in ((0.5, 1.15, 0.5), (0.3, 1.2, 0.3), (0.8, 1.05, 0.7))]
    solh = solve_derived_1rsb(_sk(), beta=2, h="0.3", seeds=seeds)
    assert abs(-solh.free_energy / 2 - famh.f) < 1e-8, (-solh.free_energy / 2, famh.f)


@slow
def test_ising_perceptron_full_1rsb_capacity_above_rs():
    """Full 1RSB (q1 < 1 free): below the frozen capacity every root the solver reaches is
    rs-degenerate (q1 = q0 or x = 1: it collapses to RS, which is AT-stable there), so the
    1RSB capacity is the frozen one, below the RS endpoint 4/pi (banked).  No independent
    target exists for a q1 < 1 1RSB capacity of this model: none is claimed."""
    r = _refusal(lambda: solve_derived_1rsb(_ising_perceptron(), alpha="0.8", kappa=0,
                                            seeds=[{"q0": 0.3, "q1": 0.8, "x": 0.4},
                                                   {"q0": 0.2, "q1": 0.9, "x": 0.2}]))
    assert r.reason == "no-solution" and "rs-degenerate" in r.detail, r
    a = _frozen_threshold().values["alpha"]
    assert a < mp.mpf(load("perceptron")["ising_rs_capacity"].value) - mp.mpf("0.4")


@slow
def test_hopfield_1rsb():
    """No banked Hopfield 1RSB target: Steffan & Kuehn (arXiv:cond-mat/9404036, Table 1)
    give the T = 0 1RSB retrieval boundary 0.138 186 489, but the derived T = 0 limit for a
    Gaussian-integral energy is not implemented (refused), so nothing is banked.  Checks:
    the x = 1 reduction (test_derive_rsb), the T = 0 refusal, and a finite-T spin-glass root
    at alpha = 0.05, beta = 3 (RS AT-unstable there, derived replicon -1.03 in E1) that is
    a genuine 1RSB point with f_1RSB > f_RS (the n -> 0 maximum) and q0 < q_RS < q1."""
    spec = default_spec("hopfield")
    r = _refusal(lambda: solve_derived_1rsb(spec, zero_temperature=True, alpha="0.05"))
    assert r.reason == "precondition:polynomial-energy-for-t0", r
    sol = solve_derived_1rsb(spec, alpha="0.05", beta="3",
                             seeds=[{"q0": 0.3, "q1": 0.75, "x": 0.08, "m": 0.0}])
    rs = solve_derived_rs(spec, alpha="0.05", beta="3", branch="nontrivial-1")
    assert abs(rs.values["m"]) < 1e-12 and abs(sol.values["m"]) < 1e-12
    assert sol.q0 < rs.values["q"] < sol.q1 and 0 < sol.m < 1
    assert sol.free_energy < rs.free_energy - 1e-3        # f_1RSB > f_RS
    assert sol.residual < 1e-9 and sol.quad_error < 1e-9 and sol.stable is None
    # the replica-max selector demands a local maximum of f; with the conjugates eliminated
    # the Hopfield psi is not one (a negative Hessian eigenvalue): refused, not hidden
    assert min(sol.hessian) < 0
    r = _refusal(lambda: solve_derived_1rsb(spec, alpha="0.05", beta="3", branch="replica-max",
                                            seeds=[{"q0": 0.3, "q1": 0.75, "x": 0.08,
                                                    "m": 0.0}]))
    assert r.reason == "no-solution:not-a-local-maximum", r


@slow
def test_solver_refusals_are_named():
    r = _refusal(lambda: solve_derived_1rsb(_sk(), beta=2))
    assert r.reason == "precondition:complete-point" and "h" in r.detail, r
    r = _refusal(lambda: solve_derived_1rsb(_ising_perceptron(), frozen=True, kappa=0,
                                            fixed={"x": 1, "q7": 0}, solve_for="alpha"))
    assert r.reason == "precondition:known-order-parameter", r
