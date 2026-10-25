"""E5a: random K-XORSAT (csp/xorsat.py closed forms, csp/xorsat_finite.py finite N).

Conventions: N boolean variables, M = alpha N parity checks; each check draws K distinct
variables uniformly (checks independent) and a uniform right-hand side.  Variable degrees
are Poisson(K alpha) as N -> oo.  Leaf removal (= zero-temperature warning propagation):
    q_hat = q^(K-1),   q = 1 - G1(1 - q_hat; K alpha)      (G1 = excess-degree pgf),
the largest root q is the probability that a variable reached from a check survives;
core variables V = 1 - G0(1 - q_hat) - q_hat G0'(1 - q_hat), core checks C = alpha q^K,
complexity Sigma = ln 2 (V - C).  alpha_d: first alpha with a nontrivial root (2-core
appears, Molloy 2005); alpha_s: Sigma = 0 on that branch (Dubois-Mandler 2002 for K = 3,
Pittel-Sorkin 2016 for K >= 3).

Bank: reference/xorsat.json (Mezard, Ricci-Tersenghi & Zecchina 2003, table of Sec. III.D;
lambda* digits of their Secs. III.B, III.D).  The banked digits are targets; the code
computes every threshold from the recursion through the DEGREE_LAWS hook.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.csp import xorsat as X
from replicant.csp import xorsat_finite as XF
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("xorsat")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


# ---- closed forms against the bank ---------------------------------------------------------
def test_k3_thresholds_match_bank():
    dps = mp.mp.dps
    assert _ok("alpha_d_k3", X.xorsat_bp_threshold(3))
    assert _ok("alpha_s_k3", X.xorsat_sat_threshold(3))
    th = X.xorsat_thresholds(3)
    assert _ok("lambda_d_k3", th.lambda_d), th
    assert _ok("lambda_s_k3", th.lambda_s), th
    assert th.alpha_d < th.alpha_s < 1
    # positive control: the other threshold is not within the tolerance of either key
    assert not _ok("alpha_s_k3", th.alpha_d)
    assert mp.mp.dps == dps


def test_k4_to_k6_thresholds_match_bank():
    """alpha_d is banked for K = 4, 5 only (the MRZ table's K = 6 alpha_d is 1.1e-6 from the
    computed value and has no second source: left out of the bank)."""
    for k in (4, 5, 6):
        th = X.xorsat_thresholds(k)
        assert k == 6 or _ok(f"alpha_d_k{k}", th.alpha_d), (k, th)
        assert _ok(f"alpha_s_k{k}", th.alpha_s), (k, th)


def test_closed_forms_agree_with_independent_parametrisations():
    """Three published forms, written here without the package's recursion: Pittel-Sorkin
    Thm 16 (alpha_s = g_k(mu*), psi(mu*) = k; alpha_d = min g_k after Molloy), Dubois-Mandler
    Sec. 3 (1 - u = exp(-3cu^2), cu^3 = u - 3cu^2(1 - u)), Cocco et al. Eq. (2) and the
    cluster entropy s0 = b - 3cb^2 + 2cb^3 (base 2)."""
    with mp.workdps(30):
        for k in (3, 4, 5):
            g = lambda x: x / (k * (1 - mp.exp(-x))**(k - 1))
            psi = lambda x: x * (mp.exp(x) - 1) / (mp.exp(x) - 1 - x)
            mu = mp.findroot(lambda x: psi(x) - k, 2)
            assert abs(g(mu) - X.xorsat_sat_threshold(k)) < mp.mpf("1e-20"), k
            xm = mp.findroot(lambda x: mp.diff(g, x), 1.2)
            assert abs(g(xm) - X.xorsat_bp_threshold(k)) < mp.mpf("1e-20"), k
            assert abs(X.xorsat_thresholds(k).lambda_s - mu) < mp.mpf("1e-18"), k
        c, u = mp.findroot([lambda c, u: 1 - u - mp.exp(-3 * c * u**2),
                            lambda c, u: c * u**3 - u + 3 * c * u**2 * (1 - u)], (0.92, 0.9))
        assert abs(c - X.xorsat_sat_threshold(3)) < mp.mpf("1e-20")
        c = mp.mpf("0.88")
        core = X.xorsat_core_size(3, c)
        b = core.q
        assert abs(b - 1 + mp.exp(-3 * c * b**2)) < mp.mpf("1e-25")
        s0 = b - 3 * c * b**2 + 2 * c * b**3
        assert abs(core.complexity - mp.log(2) * s0) < mp.mpf("1e-25")
        lam = 3 * c * b**2      # MRZ Eq. (13): Sigma = ln 2 [1 - lam/3 - e^-lam (1 + 2 lam/3)]
        assert abs(core.complexity - mp.log(2) * (1 - lam / 3 - mp.exp(-lam) * (1 + 2 * lam / 3))) \
            < mp.mpf("1e-25")


def test_core_size_closed_form_jumps_at_alpha_d():
    with mp.workdps(30):    # the comparisons below at the module's precision
        _core_jump()


def _core_jump():
    ad = X.xorsat_bp_threshold(3)
    below = X.xorsat_core_size(3, ad - mp.mpf("1e-6"))
    above = X.xorsat_core_size(3, ad + mp.mpf("1e-6"))
    assert below.trivial and below.variables == 0 and below.checks == 0
    assert not above.trivial and above.variables > mp.mpf("0.35"), above
    lam = above.lam      # MRZ Eqs. (32)-(33): Nc/N = 1 - (1 + lam) e^-lam, Mc/N = (lam/K)(1 - e^-lam)
    assert abs(above.variables - (1 - (1 + lam) * mp.exp(-lam))) < mp.mpf("1e-20")
    assert abs(above.checks - lam / 3 * (1 - mp.exp(-lam))) < mp.mpf("1e-20")
    s = X.xorsat_core_size(3, X.xorsat_sat_threshold(3))
    assert abs(s.variables - s.checks) < mp.mpf("1e-20") and abs(s.complexity) < mp.mpf("1e-20")
    assert X.xorsat_core_size(3, "0.95").complexity < 0


# ---- positive control: a planted wrong degree law inside the recursion ---------------------
def test_positive_control_wrong_degree_distribution():
    """The recursion reads the variable-degree law from X.DEGREE_LAWS.  Planting a law of
    mean (K + 1) alpha (an off-by-one degree count) moves both thresholds out of the bank
    tolerance, and its core size disagrees with leaf removal on sampled hypergraphs (which
    never reads the hook); a 1% error in the mean (1.01 K alpha) still moves both (alpha_d by
    8e-3, alpha_s by 9e-5, against tolerances of 1.5e-6 and 5e-7)."""
    orig = X.DEGREE_LAWS["poisson"]
    try:
        X.DEGREE_LAWS["poisson"] = X.PoissonLaw(mean_factor=lambda k: k + 1)
        th = X.xorsat_thresholds(3)
        assert not _ok("alpha_d_k3", th.alpha_d), th
        assert not _ok("alpha_s_k3", th.alpha_s), th
        wrong = X.xorsat_core_size(3, "0.9").variables
        X.DEGREE_LAWS["poisson"] = X.PoissonLaw(mean_factor=lambda k: 1.01 * k)
        th = X.xorsat_thresholds(3)
        assert not _ok("alpha_d_k3", th.alpha_d) and not _ok("alpha_s_k3", th.alpha_s), th
    finally:
        X.DEGREE_LAWS["poisson"] = orig
    assert X.DEGREE_LAWS["poisson"] is orig
    assert _ok("alpha_s_k3", X.xorsat_sat_threshold(3))     # cache keyed on the law
    v, _ = XF.xorsat_sampled_core(3, 20000, "0.9", seed=1)
    right = X.xorsat_core_size(3, "0.9").variables
    assert abs(v - float(right)) < 0.02 < abs(v - float(wrong)), (v, right, wrong)


# ---- finite N: exact GF(2) elimination -------------------------------------------------------
def _brute_force_sat(idx, rhs, n):
    for x in range(1 << n):
        if all(((x >> idx[m, 0]) ^ (x >> idx[m, 1]) ^ (x >> idx[m, 2])) & 1 == rhs[m]
               for m in range(len(rhs))):
            return True
    return False


def test_gf2_elimination_matches_brute_force():
    seen = set()
    for seed in range(25):
        idx, rhs = XF.xorsat_instance(3, 12, "1.0", seed)
        sat, rank = XF.gf2_solve(idx, rhs, 12)
        assert sat == _brute_force_sat(idx, rhs, 12), seed
        assert rank == XF.gf2_rank_reference(idx, 12), seed
        seen.add(sat)
    assert seen == {True, False}           # both outcomes exercised
    idx, rhs = XF.xorsat_instance(3, 12, "1.0", 3)
    assert len(rhs) == 12 and all(len(set(r)) == 3 for r in idx.tolist())
    # an inconsistent pair of identical rows is caught; a consistent pair is not
    two = np.array([[0, 1, 2], [0, 1, 2]])
    assert XF.gf2_solve(two, np.array([0, 1], np.uint8), 5) == (False, 1)
    assert XF.gf2_solve(two, np.array([1, 1], np.uint8), 5) == (True, 1)


def _crossing(N, w, alpha_star, seeds):
    """True when the 99% Clopper-Pearson interval of P(sat) lies entirely above 1/2 at
    alpha_star - w and entirely below 1/2 at alpha_star + w (P(sat) is decreasing in alpha,
    so the 1/2 crossing is then inside [alpha_star - w, alpha_star + w])."""
    lo = XF.xorsat_sat_probability(3, N, alpha_star - w, seeds)
    hi = XF.xorsat_sat_probability(3, N, alpha_star + w, seeds)
    return lo.lo > 0.5 and hi.hi < 0.5, (lo, hi)


def test_gf2_crossing_small_n():
    """Default tier: N = 200, seeds 0-39, band +-0.05.  Measured (seeds 0-199): P(sat) = 0.93,
    0.58, 0.07 at alpha_s - 0.04, alpha_s, alpha_s + 0.04.  The shifted threshold alpha_s + 0.05
    is excluded (its lower edge alpha_s has P ~ 0.58, whose 99% interval reaches below 1/2),
    and so is alpha_s - 0.05 (its upper edge is alpha_s).  The width was chosen after measuring
    seeds 0-199, which include the test seeds: on fixed seeds this is a regression check."""
    a_s = X.xorsat_sat_threshold(3)
    ok, info = _crossing(200, mp.mpf("0.05"), a_s, range(40))
    assert ok, info
    bad, info = _crossing(200, mp.mpf("0.05"), a_s + mp.mpf("0.05"), range(40))
    assert not bad, info
    bad, info = _crossing(200, mp.mpf("0.05"), a_s - mp.mpf("0.05"), range(40))
    assert not bad, info


@slow
def test_gf2_crossing_near_alpha_s():
    """N = 400, 1000, 2000 (seeds 0-99): the crossing of 1/2 is inside alpha_s +- w(N) with
    w = 0.04, 0.03, 0.02 (measured, seeds 0-199: P(sat) at alpha_s = 0.52 (N = 400), 0.49
    (N = 1000); at alpha_s +- 0.02, N = 1000: 0.97 / 0.02).  The shifted thresholds alpha_s +
    0.05, alpha_s + 0.02 and the -0.02, -0.05 shifts all fail at N = 2000: shifts of 0.02 are
    resolved there.  The widths were chosen after measuring seeds 0-199, which include the test
    seeds: on fixed seeds this is a regression check."""
    a_s = X.xorsat_sat_threshold(3)
    for N, w in ((400, "0.04"), (1000, "0.03"), (2000, "0.02")):
        ok, info = _crossing(N, mp.mpf(w), a_s, range(100))
        assert ok, (N, info)
    for shift in ("0.05", "0.02", "-0.02", "-0.05"):
        bad, info = _crossing(2000, mp.mpf("0.02"), a_s + mp.mpf(shift), range(100))
        assert not bad, (shift, info)


def test_sat_probability_band_and_labels():
    r = XF.xorsat_sat_probability(3, 200, "0.5", range(10))
    assert r.n == 10 and r.n_sat == 10 and r.fraction == 1 and r.hi == 1 and 0.4 < r.lo < 1
    assert tuple(r.seeds) == tuple(range(10))
    r = XF.xorsat_sat_probability(3, 200, "1.2", range(10))
    assert r.n_sat == 0 and r.lo == 0


# ---- finite N: leaf removal on sampled hypergraphs ------------------------------------------
def test_leaf_removal_peels_a_tree_and_keeps_a_cycle():
    tree = np.array([[0, 1, 2], [2, 3, 4]])
    assert XF.leaf_removal_core(tree, 5) == (0, 0)
    # the four checks on six variables, each variable in exactly two checks: a 2-core
    core = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]])
    assert XF.leaf_removal_core(core, 6) == (6, 4)
    assert XF.leaf_removal_core(np.vstack([core, [[5, 6, 7]]]), 8) == (6, 4)


def test_core_size_sampled_small_n():
    """Default tier: N = 20000, seeds 0-5, at alpha_d -+ 0.05: empty core below, the closed
    form within a 99% Student-t band (+ 2e-3 finite-size allowance) above.  Measured band
    half-widths: 0.0099 (variables), 0.0115 (checks); the core of the planted 1% degree-law
    error (1.01 K alpha: V = 0.5689 against 0.5523) falls outside it."""
    ad = X.xorsat_bp_threshold(3)
    v, c = XF.xorsat_sampled_core(3, 20000, ad - mp.mpf("0.05"), seed=0)
    assert v == 0 and c == 0
    a = ad + mp.mpf("0.05")
    bv, _ = _band(3, 20000, a, range(6), 2e-3)
    orig = X.DEGREE_LAWS["poisson"]
    try:
        X.DEGREE_LAWS["poisson"] = X.PoissonLaw(mean_factor=lambda k: 1.01 * k)
        planted = X.xorsat_core_size(3, a).variables
    finally:
        X.DEGREE_LAWS["poisson"] = orig
    assert X.DEGREE_LAWS["poisson"] is orig
    assert abs(bv.mean - float(planted)) > bv.half_width + 2e-3, (bv, planted)


def _band(K, N, alpha, seeds, slack):
    from replicant.amp._stats import student_band
    vs = [XF.xorsat_sampled_core(K, N, alpha, seed=s) for s in seeds]
    want = X.xorsat_core_size(K, alpha)
    out = []
    for got, ref in ((np.array([v for v, _ in vs]), want.variables),
                     (np.array([c for _, c in vs]), want.checks)):
        b = student_band(got)
        assert abs(b.mean - float(ref)) <= b.half_width + slack, (b, ref)
        out.append(b)
    return out


@slow
def test_core_size_jumps_at_alpha_d():
    """N = 100000, seeds 0-4: at alpha_d - 0.02 every sampled core is empty; at alpha_d + 0.02
    the sampled core (variables and checks) matches the closed form in a 99% Student-t band
    plus 1e-3 (measured half-widths 0.0064, 0.0068), and is larger than the jump size
    V(alpha_d+) = 0.357."""
    ad = X.xorsat_bp_threshold(3)
    for s in range(5):
        assert XF.xorsat_sampled_core(3, 100000, ad - mp.mpf("0.02"), seed=s) == (0, 0)
    _band(3, 100000, ad + mp.mpf("0.02"), range(5), 1e-3)
    assert X.xorsat_core_size(3, ad + mp.mpf("0.02")).variables > X.xorsat_thresholds(3).core_d


# ---- refusals --------------------------------------------------------------------------------
def test_refusals_are_named():
    _refuses(lambda: X.xorsat_bp_threshold(2), "precondition:integer-k-at-least-3")
    _refuses(lambda: X.xorsat_sat_threshold(3.5), "precondition:integer-k-at-least-3")
    _refuses(lambda: X.xorsat_sat_threshold(True), "precondition:integer-k-at-least-3")
    _refuses(lambda: X.xorsat_core_size(3, 0), "precondition:positive-alpha")
    _refuses(lambda: X.xorsat_core_size(3, "-0.1"), "precondition:positive-alpha")
    _refuses(lambda: X.xorsat_core_size(3, "abc"), "precondition:positive-alpha")
    _refuses(lambda: XF.xorsat_gf2_satisfiable(3, 10, "0.9", 0), "precondition:enough-variables")
    _refuses(lambda: XF.xorsat_gf2_satisfiable(2, 100, "0.9", 0),
             "precondition:integer-k-at-least-3")
    _refuses(lambda: XF.xorsat_gf2_satisfiable(3, 100, 0, 0), "precondition:positive-alpha")
    _refuses(lambda: XF.xorsat_sat_probability(3, 100, "0.9", []), "precondition:named-seeds")
    _refuses(lambda: X.xorsat_bp_threshold(3, law="regular"), "unsupported:degree-law")


def test_malformed_seeds_are_refused_by_name():
    """Fix round 1 (Important 1): seeds must be non-negative integers, named in the detail."""
    for bad in (["x"], (-1,), "x", 3, (True,), (0, 1.5)):
        r = _refuses(lambda: XF.xorsat_sat_probability(3, 200, "0.9", bad), "precondition:named-seeds")
        assert "non-negative integers" in r.detail, r.detail
    r = _refuses(lambda: XF.xorsat_gf2_satisfiable(3, 200, "0.9", -2), "precondition:named-seeds")
    assert "-2" in r.detail
    _refuses(lambda: XF.xorsat_sampled_core(3, 200, "0.9", seed="a"), "precondition:named-seeds")
    e = catalog()["xorsat"]
    r = _refuses(lambda: compose(e.spec, "cavity").run(alpha="0.9", N=200, seeds="x"),
                 "precondition:named-seeds")
    assert "'x'" in r.detail, r.detail
    import io
    from replicant.cli import main
    err = io.StringIO()
    code = main(["run", "xorsat", "--route", "cavity", "--param", "alpha=0.9", "--param", "N=200",
                 "--param", "seeds=x"], io.StringIO(), err)
    assert code == 2 and "REFUSED: precondition:named-seeds" in err.getvalue(), err.getvalue()
    err = io.StringIO()
    code = main(["run", "xorsat", "--route", "cavity", "--param", "alpha=0.9", "--param", "N=200",
                 "--param", "seeds=-1,"], io.StringIO(), err)
    assert code == 2 and "REFUSED: precondition:named-seeds" in err.getvalue(), err.getvalue()


def test_sat_fraction_uncertainty_is_the_clopper_pearson_halfwidth():
    """Fix round 1 (Minor 1): when every seed agrees the binomial stderr is 0; the reported
    uncertainty is the larger distance from the fraction to its 99% Clopper-Pearson bounds."""
    r = XF.xorsat_sat_probability(3, 200, "0.5", range(10))
    assert r.stderr == 0 and r.halfwidth99 == max(r.fraction - r.lo, r.hi - r.fraction) > 0.3
    e = catalog()["xorsat"]
    res = compose(e.spec, "cavity").run(alpha="0.5", N=200, seeds=tuple(range(10)))
    assert res.errors["sat_fraction"] == r.halfwidth99
    assert any("Clopper-Pearson" in n for n in res.notes), res.notes


# ---- catalog entry ---------------------------------------------------------------------------
def test_catalog_entry_xorsat():
    e = catalog()["xorsat"]
    assert e.routes == {"cavity"} and e.default_route == "cavity"
    res = compose(e.spec, "cavity").run()
    v = res.values
    assert _ok("alpha_s_k3", v["alpha_s"]) and _ok("alpha_d_k3", v["alpha_d"])
    assert res.finality["alpha_s"].state == "exact-by-literature"
    assert "Dubois" in res.finality["alpha_s"].cite and "Pittel" in res.finality["alpha_s"].cite
    assert "Molloy" in res.finality["alpha_d"].cite
    assert res.final and all(c.ok for c in res.reference_checks())
    k4 = compose(e.spec, "cavity").run(K=4)
    assert "Dubois" not in k4.finality["alpha_s"].cite and "Pittel" in k4.finality["alpha_s"].cite
    pt = compose(e.spec, "cavity").run(alpha="0.95")
    assert pt.values["complexity"] < 0 and pt.finality["complexity"].state == "not-computed"
    assert pt.finality["core_variables"].state == "exact-by-literature"
    pt = compose(e.spec, "cavity").run(alpha="0.9")
    assert pt.finality["complexity"].state == "exact-by-literature"
    note = pt.finality["complexity"].note
    assert "core solutions per variable" in note and "Ibrahimi" in note and "abstract only" in note
    fin = compose(e.spec, "cavity").run(alpha="0.9", N=200, seeds=(0, 1, 2))
    assert "sat_fraction" in fin.statistical and not fin.final
    assert fin.finality["sat_fraction"].state == "not-computed"
    r = _refuses(lambda: compose(e.spec, "cavity").run(alfa="0.9"), "precondition:known-parameter")
    assert "alpha" in r.detail
    _refuses(lambda: compose(e.spec, "cavity").run(N=200), "precondition:state-point")
    r = _refuses(lambda: compose(e.spec, "cavity").run(alpha="0.9", seeds=(0, 1)),
                 "precondition:known-parameter")
    assert "need N" in r.detail
    _refuses(lambda: compose(e.spec, "replica"), "missing-step:route:replica")
    assert e.derivation("cavity")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `xorsat`" in guide and "precondition:integer-k-at-least-3" in guide
    assert all(check_ref(e, r).ok for r in e.refs if not r.slow)
