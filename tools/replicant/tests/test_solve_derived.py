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
import sympy as sp

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


def test_solve_derived_sk_default_tier():
    # the whole numeric path at low precision: derive -> canonical form -> Erfc /
    # lambdify -> float search -> mp polish -> independent audit -> phi, entropy,
    # derived replicon; plus the incomplete-point refusal
    from replicant.families.sk import sk_replicon
    spec = default_spec("sk")
    r = _refusal(lambda: solve_derived_rs(spec, beta="1.5", dps=12))
    assert r.reason == "precondition:complete-point" and "h" in r.detail, r
    before = mp.mp.dps
    s = solve_derived_rs(spec, beta="1.5", h=0, dps=12)
    assert mp.mp.dps == before
    assert [x["name"] for x in s.roots] == ["trivial", "nontrivial"] and s.branch == "nontrivial"
    q = s.values["q"]
    assert abs(s.values["qhat"] - mp.mpf("2.25") * q) < 1e-12 and 0.3 < q < 0.4
    assert s.residual < 1e-10 and 0 <= s.quad_error < 1e-10
    # the replicon comes from the derivation (pair energy beta^2 q^2/2, Ising site)
    assert abs(s.replicon - sk_replicon("1.5", 0, q=q)) < 1e-10 and s.replicon < 0
    assert s.stable is False and s.entropy > 0
    assert s.trace.to_json()["steps"][-1] == "solve-derived"


def test_coarse_quadrature_fails_the_independent_audit():
    # positive control for the audit: 4-node Gauss-Hermite, no polish
    r = _refusal(lambda: solve_derived_rs(default_spec("sk"), beta="1.5", h=0, quad="gh",
                                          gh_nodes=4, dps=12, branch="nontrivial"))
    assert r.reason == "no-solution", r
    # the nontrivial root of the coarse rule was found and then dropped by the
    # independent quadrature, with its residual (~3e-2) named
    assert "dropped by the independent audit" in r.detail and "residual 0.0" in r.detail, r


def test_erfc_kept_as_written_in_the_tail():
    # H(x) = erfc(x/sqrt 2)/2 with x = (kappa - sqrt(q) t)/sqrt(1-q): substituting
    # kappa = 0 makes sympy write 1 - erfc(+...)/2, which cancels catastrophically
    # for t -> -inf; the solver's form keeps erfc(-...) and is exact there
    from replicant._derived_num import Erfc, Num, safe
    from replicant.gauss import H
    q, t, kappa = sp.Symbol("q", positive=True), sp.Symbol("t", real=True), sp.Symbol("kappa")
    Hx = H((kappa - sp.sqrt(q) * t) / sp.sqrt(1 - q))
    u = sp.sqrt(2) * sp.sqrt(q) * t / (2 * sp.sqrt(1 - q))
    assert Hx.subs(kappa, 0) == 1 - sp.erfc(u) / 2               # sympy's rewrite
    kept = safe(Hx).subs(kappa, 0)
    assert kept == Erfc(-u) / 2 and not kept.has(sp.erfc)
    with mp.workdps(30):
        want = mp.erfc(8 / mp.sqrt(2)) / 2                       # x = 8 at q = 1/2, t = -8
        got = Num(kept, [q, t], 20).mp([mp.mpf("0.5"), mp.mpf(-8)])
        assert abs(got / want - 1) < mp.mpf(10)**-25, (got, want)
    with mp.workdps(15):
        naive = sp.lambdify([q, t], Hx.subs(kappa, 0), "mpmath")(mp.mpf("0.5"), mp.mpf(-8))
        assert abs(naive / want - 1) > 1e-2                      # the loss the fix avoids


@slow
def test_solve_derived_sk_matches_sk_rs():
    from replicant.families.sk import sk_replicon, sk_rs
    before = mp.mp.dps
    s = solve_derived_rs(default_spec("sk"), beta="1.5", h=0, branch="nontrivial")
    assert mp.mp.dps == before
    ref = sk_rs("1.5")                                      # comparison target only
    assert abs(s.values["q"] - ref.q) < 1e-10, (s.values, ref.q)
    assert abs(s.values["qhat"] - mp.mpf("2.25") * ref.q) < 1e-10
    assert abs(s.free_energy - (-mp.mpf("1.5") * ref.f)) < 1e-10      # phi = -beta f
    assert s.residual < 1e-12 and s.branch == "nontrivial" and s.quad_error < 1e-12
    # AT-unstable at h = 0, beta > 1: the derived replicon says so (level (b) agrees)
    assert abs(s.replicon - sk_replicon("1.5")) < 1e-10 and s.replicon < 0
    assert s.stable is False and ref.stable is False
    assert s.entropy > 0 and "physics-level" in s.caveat
    assert "eliminated qhat" in s.note and "polished" in s.method
    # above the AT line (h = 1 at T = 2/3) the replicon is positive: stable True
    s1 = solve_derived_rs(default_spec("sk"), beta="1.5", h=1, branch="nontrivial")
    lam = sk_replicon("1.5", 1)
    assert lam > 0 and abs(s1.replicon - lam) < 1e-10 and s1.stable is True
    assert abs(s1.values["q"] - sk_rs("1.5", 1).q) < 1e-10


@slow
def test_numeric_beta_in_spec_still_gives_the_entropy():
    spec = ModelSpec("sk", "ising", "gaussian", params={"beta": 1.5, "h": D.H_FIELD})
    a = solve_derived_rs(spec, h=0, dps=12)
    b = solve_derived_rs(default_spec("sk"), beta="1.5", h=0, dps=12)
    assert a.entropy is not None and abs(a.entropy - b.entropy) < 1e-10
    assert "lifted" in a.note


@slow
def test_solve_derived_sk_pm_one_equals_gaussian():
    g, pm = default_spec("sk"), ModelSpec("sk", "ising", "pm_one",
                                          params={"beta": D.BETA, "h": D.H_FIELD})
    assert derived_rs_equations(g) == derived_rs_equations(pm)    # leading order in N
    # (equal equations already imply equal solutions; the solve below is kept as an
    # end-to-end check that the pm_one trace, with its dropped order, reaches the
    # solver's output)
    b = solve_derived_rs(pm, beta="1.5", h="0.3", branch="nontrivial", dps=12)
    a = solve_derived_rs(g, beta="1.5", h="0.3", branch="nontrivial", dps=12)
    assert abs(a.values["q"] - b.values["q"]) < 1e-10
    text = b.trace.render()
    assert "dropped fourth cumulant" in text and "O(1/N) per spin" in text, text
    assert "dropped fourth" not in a.trace.render()


@slow
def test_solve_derived_refuses_when_no_solution():
    spec = default_spec("sk")
    r = _refusal(lambda: solve_derived_rs(spec, beta="0.5", h=0, branch="nontrivial"))
    assert r.reason == "no-solution" and "trivial" in r.detail, r
    s = solve_derived_rs(spec, beta="0.5", h=0)              # no branch requested
    assert s.branch == "trivial" and s.values["q"] == 0
    assert "only the trivial branch" in s.note
    assert abs(s.free_energy - (mp.log(2) + mp.mpf("0.25") / 4)) < 1e-15   # ln 2 + beta^2/4
    assert abs(s.replicon - mp.mpf("0.75")) < 1e-15 and s.stable is True  # 1 - beta^2


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
    # E1: the replicon is derived for the LnGaussProd energy with the Ising site and
    # equals the family's (target); RS is replicon-stable and S > 0 here
    assert abs(s.replicon - pt.replicon) < 1e-9 and s.replicon > 0
    assert s.stable is True and "no capacity" in s.note
    assert s.replicon_route == "gauss-covariance+ising-site"
    assert "replicon" in s.trace.entries[-1].summary and "UNKNOWN" not in s.note


@slow
def test_solve_derived_ising_perceptron_negative_entropy_is_unstable():
    # between the frozen-1RSB capacity 0.833 and the RS endpoint 4/pi the RS
    # equations still have a solution, but its entropy is negative
    s = solve_derived_rs(default_spec("perceptron_ising"), alpha="1.0", kappa=0)
    assert s.entropy < 0 and s.stable is False and "RS is wrong" in s.note
    assert s.replicon > 0               # below the AT point: unstable by the entropy only


@slow
def test_solve_derived_spherical_perceptron_matches_gardner():
    from replicant.families.perceptron import gardner_entropy, gardner_rs_point
    s = solve_derived_rs(default_spec("perceptron"), alpha=1, kappa=0)
    assert s.residual < 1e-12 and "findroot" in s.method
    pt = gardner_rs_point("spherical", s.values["q"], 0)
    assert abs(pt.alpha - 1) < 1e-9 and abs(pt.S - s.free_energy) < 1e-9
    assert abs(gardner_entropy(1, "spherical") - s.free_energy) < 1e-9
    # E1: derived replicon (no entropy-sign test for continuous variables) = the family's
    assert abs(s.replicon - pt.replicon) < 1e-9 and s.stable is True
    assert s.replicon_route == "gauss-covariance+explicit-n-site"


@slow
def test_solve_derived_hopfield_matches_hopfield_rs():
    from replicant.families.hopfield import hopfield_free_energy, hopfield_rs
    al, be = mp.mpf("0.05"), mp.mpf(2)
    # retrieval and spin glass both exist here: retrieval has the lower phi
    s = solve_derived_rs(default_spec("hopfield"), alpha="0.05", beta=2, dps=15,
                         branch="nontrivial-2")
    ref = hopfield_rs(al, be)
    assert abs(s.values["m"] - ref.m) < 1e-10 and abs(s.values["q"] - ref.q) < 1e-10
    assert abs(s.values["qhat"] - al * be**2 * ref.r) < 1e-10
    f = hopfield_free_energy(ref.m, ref.q, ref.r, al, be)
    assert abs(s.free_energy - al * be / 2 - (-be * f)) < 1e-10
    assert abs(s.replicon - ref.replicon) < 1e-10 and s.stable is ref.stable


@slow
def test_several_saddles_refuse_without_selector():
    # Hopfield at alpha = 0.1, beta = 5: retrieval (m != 0) and spin glass (m = 0)
    from replicant.families.hopfield import hopfield_rs
    spec = default_spec("hopfield")
    r = _refusal(lambda: solve_derived_rs(spec, alpha="0.1", beta=5, dps=12))
    assert r.reason == "ambiguous:several-saddles", r
    assert "nontrivial-1" in r.detail and "nontrivial-2" in r.detail, r.detail
    s = solve_derived_rs(spec, alpha="0.1", beta=5, dps=12, branch="nontrivial-2")
    assert len(s.roots) >= 2 and s.roots[-1]["phi"] < s.roots[-2]["phi"]   # phi ordering
    ret, sg = hopfield_rs("0.1", 5, branch="retrieval"), hopfield_rs("0.1", 5, branch="spinglass")
    # by decreasing phi: the spin glass (lower f) is nontrivial-1, retrieval nontrivial-2
    assert abs(s.values["m"] - ret.m) < 1e-9 and abs(s.values["q"] - ret.q) < 1e-9
    one = next(x for x in s.roots if x["name"] == "nontrivial-1")
    assert abs(one["values"]["m"]) < 1e-9 and abs(one["values"]["q"] - sg.q) < 1e-9
    assert abs(one["values"]["qhat"] - mp.mpf("0.1") * 25 * sg.r) < 1e-8
    r = _refusal(lambda: solve_derived_rs(spec, alpha="0.1", beta=5, dps=12,
                                          branch="nontrivial"))
    assert r.reason == "ambiguous:several-saddles", r


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
