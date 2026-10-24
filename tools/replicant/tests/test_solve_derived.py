"""solve_derived_rs: numbers from the DERIVED RS equations (level (a)).

The comparison targets (families/sk.py, families/perceptron.py,
families/hopfield.py) are used only as targets: nothing from them feeds the
solver.  Conventions: SK J_ij ~ N(0, 1/N) (or +-1/sqrt(N)), field h, phi =
-beta f; perceptron P = alpha N Gaussian patterns, margin kappa, phi = RS
entropy per coupling (families/perceptron.py); Hopfield +-1 patterns, the
derived H keeps the i = j terms (phi = -beta f_AGS + alpha beta/2).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import default_spec, derived_rs_equations
from replicant.core import Refusal
from replicant.solve_derived import solve_derived_rs
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected Refusal")


def test_solve_derived_refuses_incomplete_point():
    r = _refusal(lambda: solve_derived_rs(default_spec("sk"), beta="1.5"))
    assert r.reason == "precondition:complete-point" and "h" in r.detail, r


@slow
def test_solve_derived_sk_matches_sk_rs():
    from replicant.families.sk import sk_rs
    before = mp.mp.dps
    s = solve_derived_rs(default_spec("sk"), beta="1.5", h=0, branch="nontrivial")
    assert mp.mp.dps == before
    ref = sk_rs("1.5")                                      # comparison target only
    assert abs(s.values["q"] - ref.q) < 1e-10, (s.values, ref.q)
    assert abs(s.values["qhat"] - mp.mpf("2.25") * ref.q) < 1e-10
    assert abs(s.free_energy - (-mp.mpf("1.5") * ref.f)) < 1e-10      # phi = -beta f
    assert s.residual < 1e-12 and s.branch == "nontrivial"
    assert s.stable is None and "replicon not derived" in s.note       # no AT claim
    assert s.entropy > 0 and "physics-level" in s.caveat
    assert "eliminated qhat" in s.note and "solve.fixed_point" in s.method
    assert s.trace.to_json()["steps"][-1] == "solve-derived"


@slow
def test_solve_derived_sk_pm_one_equals_gaussian():
    g, pm = default_spec("sk"), ModelSpec("sk", "ising", "pm_one",
                                          params={"beta": D.BETA, "h": D.H_FIELD})
    assert derived_rs_equations(g) == derived_rs_equations(pm)    # leading order in N
    a = solve_derived_rs(g, beta="1.5", h="0.3", branch="nontrivial")
    b = solve_derived_rs(pm, beta="1.5", h="0.3", branch="nontrivial")
    assert abs(a.values["q"] - b.values["q"]) < 1e-10
    text = b.trace.render()
    assert "dropped fourth cumulant" in text and "O(1/N) per spin" in text, text
    assert "dropped" not in a.trace.render()


@slow
def test_solve_derived_refuses_when_no_solution():
    spec = default_spec("sk")
    r = _refusal(lambda: solve_derived_rs(spec, beta="0.5", h=0, branch="nontrivial"))
    assert r.reason == "no-solution" and "trivial" in r.detail, r
    s = solve_derived_rs(spec, beta="0.5", h=0)              # no branch requested
    assert s.branch == "trivial" and s.values["q"] == 0
    assert "only the trivial branch" in s.note
    assert abs(s.free_energy - (mp.log(2) + mp.mpf("0.25") / 4)) < 1e-15   # ln 2 + beta^2/4


@slow
def test_solve_derived_ising_perceptron_matches_gardner_rs():
    from replicant.families.perceptron import gardner_entropy, gardner_rs_point
    s = solve_derived_rs(default_spec("perceptron_ising"), alpha="0.5", kappa=0)
    assert s.residual < 1e-12 and s.branch == "nontrivial"
    # Task 8 parametrises the saddle by qhat: at the derived qhat it must give
    # alpha = 0.5 back, the same q and the same RS entropy
    pt = gardner_rs_point("ising", s.values["qhat"], 0)
    assert abs(pt.alpha - mp.mpf("0.5")) < 1e-9, pt.alpha
    assert abs(pt.q - s.values["q"]) < 1e-9
    assert abs(pt.S - s.free_energy) < 1e-9
    assert abs(gardner_entropy("0.5", "ising") - s.free_energy) < 1e-9   # its own solve
    assert s.entropy == s.free_energy > 0
    assert s.stable is None and "no capacity" in s.note


@slow
def test_solve_derived_ising_perceptron_negative_entropy_is_unstable():
    # between the frozen-1RSB capacity 0.833 and the RS endpoint 4/pi the RS
    # equations still have a solution, but its entropy is negative
    s = solve_derived_rs(default_spec("perceptron_ising"), alpha="1.0", kappa=0)
    assert s.entropy < 0 and s.stable is False and "RS is wrong" in s.note


@slow
def test_solve_derived_spherical_perceptron_matches_gardner():
    from replicant.families.perceptron import gardner_entropy, gardner_rs_point
    s = solve_derived_rs(default_spec("perceptron"), alpha=1, kappa=0)
    assert s.residual < 1e-12 and "findroot" in s.method
    pt = gardner_rs_point("spherical", s.values["q"], 0)
    assert abs(pt.alpha - 1) < 1e-9 and abs(pt.S - s.free_energy) < 1e-9
    assert abs(gardner_entropy(1, "spherical") - s.free_energy) < 1e-9
    assert s.stable is None              # continuous variables: no entropy-sign test


@slow
def test_solve_derived_hopfield_matches_hopfield_rs():
    from replicant.families.hopfield import hopfield_free_energy, hopfield_rs
    al, be = mp.mpf("0.05"), mp.mpf(2)
    s = solve_derived_rs(default_spec("hopfield"), alpha="0.05", beta=2, seeds=("0.9",))
    ref = hopfield_rs(al, be)
    assert abs(s.values["m"] - ref.m) < 1e-10 and abs(s.values["q"] - ref.q) < 1e-10
    assert abs(s.values["qhat"] - al * be**2 * ref.r) < 1e-10
    f = hopfield_free_energy(ref.m, ref.q, ref.r, al, be)
    assert abs(s.free_energy - al * be / 2 - (-be * f)) < 1e-10


@slow
def test_planted_error_in_hook_moves_the_derived_solution():
    from dataclasses import replace
    from replicant.families.sk import sk_rs
    good = D.FAMILIES["sk"]

    def flipped(avg, n, sum_pairs):           # pair energy doubled in the hook
        r = good.order_parameters(avg, n, sum_pairs)
        return D.Reduction(2 * r.energy, r.correction, r.missing, r.field, r.extra)
    D.FAMILIES["sk"] = replace(good, order_parameters=flipped)
    try:
        s = solve_derived_rs(default_spec("sk"), beta="1.5", h=0, branch="nontrivial")
    finally:
        D.FAMILIES["sk"] = good
    assert abs(s.values["q"] - sk_rs("1.5").q) > 1e-2, s.values
