"""M1, default tier: cluster algorithms (montecarlo/).  The exact critical points by two routes; the exact small-torus
distributions against Kaufman's integer polynomial and torus_thermo; detailed balance (stationarity) of
Swendsen-Wang, Wolff and Metropolis by chi^2 against exact Boltzmann weights on named seeds (alpha = 0.01), with a
planted +-5% bond probability (measured power >= 0.95), a wrong Wolff growth rule, a wrong cluster flip, a wrong
Metropolis acceptance and non-ergodic updates for the ergodicity chi^2,
each restored with an identity assert; the Wolff cluster against the shared union-find; the vectorised SW against
the union-find SW trajectory-for-trajectory; the FK machinery at q = 1 against P2's exact crossing polynomial; the
catalog entry's labels and refusals.  Larger tori, tau_int and z fits: tests/test_mc_cluster_slow.py."""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.catalog import catalog
from replicant.compose import compose
from replicant.core import Refusal
from replicant.montecarlo import balance as BL
from replicant.montecarlo import cluster as CL
from replicant.montecarlo import cluster_fast as CF
from replicant.montecarlo import exact_small as XS
from replicant.montecarlo import fk as FK
from replicant.montecarlo import fk_percolation as FP
from replicant.montecarlo import metropolis as MT
from replicant.percolation import enumerate2d as EN
from replicant.percolation import lattice2d as LT
from replicant.percolation.unionfind import UnionFind

ALPHA = 0.01
SEED = 7                          # the named seed of every default-tier chi^2
SW_N, WOLFF_N = 5000, 3000         # samples of the cases whose +-5% power is measured


def _refused(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"not refused: expected {reason}")


def _planted(hooks, key, value, fn):
    saved = hooks[key]
    hooks[key] = value
    try:
        return fn()
    finally:
        hooks[key] = saved
        assert hooks[key] is saved


# ---- coupling, critical points, exact small tori --------------------------------------------------------
def test_critical_points_two_routes_and_the_ising_convention():
    before = mp.mp.dps
    from replicant.lattice.ball import baller
    from replicant.lattice.coupling import coupling
    with baller().hygiene.ctx_guard(prec=200):
        kc_text = coupling("beta_c").arb().mid().str(40, radius=False)
    with mp.workdps(40):
        for q in (3, 4, 7):
            assert abs(FK.critical_beta(q)[0] - mp.log(1 + mp.sqrt(q))) < mp.mpf(10) ** -35
        kc = mp.mpf(kc_text)
        assert abs(FK.critical_beta(2)[0] - kc) < mp.mpf(10) ** -35      # = the lattice package's K_c
        assert abs(FK.critical_beta(2)[0] - mp.log(1 + mp.sqrt(2))) > mp.mpf("0.4")   # control: not unhalved
    assert mp.mp.dps == before
    # at beta_c the Potts FK bond probability is the self-dual point sqrt q / (1 + sqrt q) (BDC Theorem 1)
    for q in (2, 3, 4):
        assert abs(FK.bond_probability(q, "beta_c") - math.sqrt(q) / (1 + math.sqrt(q))) < 1e-14
    assert abs(FK.bond_probability(2, "2/5") + math.expm1(-0.8)) < 1e-15
    wrong = _planted(FK._HOOKS, "coupling_factor", 0.5, lambda: FK.bond_probability(2, "beta_c"))
    assert abs(wrong - math.sqrt(2) / (1 + math.sqrt(2))) > 0.1
    _refused(lambda: FK.critical_beta(1), "precondition:potts-q")
    _refused(lambda: FK.beta_value(2, "-1"), "precondition:beta")


def test_enumeration_matches_kaufman_and_torus_thermo():
    from replicant.lattice.torus_thermo import torus_thermo_polynomial
    for L in (2, 3):
        holds, detail = XS.kaufman_histogram_identity(L)
        assert holds, detail
    with mp.workdps(30):
        for beta in ("2/5", "beta_c"):
            ex = XS.exact_distribution(3, 2, beta)
            u, c = torus_thermo_polynomial(3, 3, beta)
            assert abs(ex.e - float(u.mid())) < 1e-12 and abs(ex.C - float(c.mid())) < 1e-12, (beta, ex)
        assert abs(XS.exact_distribution(3, 2, "1/2").e - float(u.mid())) > 1e-3          # control: other beta
    ex = XS.exact_distribution(2, 3, "beta_c")
    assert abs(ex.probs.sum() - 1) < 1e-12 and 1 / 3 < ex.abs_m < 1 and abs(ex.class_probs.sum() - 1) < 1e-12
    _refused(lambda: XS.enumerate_torus(5, 2), "precondition:exact-size")


# ---- detailed balance by chi^2 -------------------------------------------------------------------------
def test_stationarity_chi2_of_every_sampler():
    """One update applied to independent exact Boltzmann draws must leave the class distribution exact.  The first
    two cases are the ones whose power is measured in test_a_five_percent_bond_probability_error_fails_the_chi2."""
    cases = [("swendsen_wang", 2, 3, "beta_c", SW_N), ("wolff", 2, 2, "beta_c", WOLFF_N),
             ("swendsen_wang", 3, 2, "beta_c", 1000), ("wolff", 3, 2, "beta_c", 1000), ("metropolis", 2, 2, "beta_c", 1000)]
    for alg, q, L, beta, n in cases:
        r = BL.stationarity_chi2(alg, q, L, beta, n, SEED, ALPHA)
        assert r.ok and r.alpha == ALPHA and r.df >= 3, (alg, q, L, beta, r.text)


def test_a_five_percent_bond_probability_error_fails_the_chi2():
    """Power, measured on named seeds 100-139 (alpha = 0.01, beta_c, q = 2): p x 1.05 / x 0.95 rejected 39/40 and
    38/40 by SW (L = 3, 5000 samples) and 38/40, 40/40 by Wolff (L = 2, 3000 samples), the unplanted p 0/40 by
    both; so the default chi^2 rejects a +-5% bond-probability error with power >= 0.95.  Here on seed 7."""
    for scale in (1.05, 0.95):
        for alg, L, n in (("swendsen_wang", 3, SW_N), ("wolff", 2, WOLFF_N)):
            r = _planted(FK._HOOKS, "p_scale", scale, lambda: BL.stationarity_chi2(alg, 2, L, "beta_c", n, SEED, ALPHA))
            assert not r.ok, (alg, scale, r.text)


def test_planted_errors_fail_the_chi2():
    run = lambda alg, L=3: BL.stationarity_chi2(alg, 2, L, "3/10", 1000, SEED, ALPHA)
    assert not _planted(CL._HOOKS, "flip", "site", lambda: run("swendsen_wang")).ok      # colour per site
    assert not _planted(CL._HOOKS, "growth", "any", lambda: run("wolff")).ok             # grows across colours
    assert not _planted(MT._HOOKS, "acceptance_scale", 0.5, lambda: run("metropolis", 2)).ok
    assert run("wolff").ok                                       # the same seeds pass with the hooks restored


def test_ergodicity_chi2_and_its_non_ergodic_updates():
    """From the all-0 colouring, 150 chains x 10 updates on the 2 x 2 torus at beta_c must reach the exact law for SW
    and Wolff.  Each of these must fail: SW that never recolours (planted: the identity update, stationary, so the
    stationarity chi^2 passes it, but not ergodic); Wolff at p = 1 (planted: it alternates between the two ground
    states); and, unplanted, the checkerboard Metropolis on the 2 x 2 multigraph torus, which is genuinely not ergodic
    there (from all-up, a B-site sees h = 0 and the two B-sites flip together with probability 1, so the striped
    states are never reached; found by this test, so run.check_params refuses metropolis below L = 4; L = 4 passes
    the slow-tier ergodicity chi^2)."""
    erg = lambda alg: BL.ergodicity_chi2(alg, 2, 2, "beta_c", 150, 10, SEED, ALPHA)
    for alg in ("swendsen_wang", "wolff"):
        assert erg(alg).ok, (alg, erg(alg).text)
    assert not BL.ergodicity_chi2("metropolis", 2, 2, "beta_c", 1000, 10, SEED, ALPHA).ok   # vectorised: 1000 chains
    assert not _planted(CL._HOOKS, "flip", "none", lambda: erg("swendsen_wang")).ok
    assert _planted(CL._HOOKS, "flip", "none",
                    lambda: BL.stationarity_chi2("swendsen_wang", 2, 2, "beta_c", 1000, SEED, ALPHA)).ok
    assert not _planted(FK._HOOKS, "p_scale", 10, lambda: erg("wolff")).ok


def test_wolff_cluster_is_the_unionfind_component():
    rng = np.random.default_rng(SEED)
    L, q = 4, 3
    p = FK.bond_probability(q, "beta_c")
    nb = CL.neighbours(L)
    u, v = CL.bond_lists(L)
    s = rng.integers(q, size=L * L).tolist()
    for _ in range(40):
        before, rec = list(s), []
        size, d_sat, _, _ = CL.wolff_step(s, q, p, nb, rng, rec)
        CL.wolff_unionfind_check(before, q, s, rec, L)
        assert size == sum(a != b for a, b in zip(before, s))
        assert d_sat == sum((s[a] == s[b]) - (before[a] == before[b]) for a, b in zip(u, v))

    def planted():
        t, rec = list(before), []
        CL.wolff_step(t, 2, 0.9, CL.neighbours(L), np.random.default_rng(1), rec)
        CL.wolff_unionfind_check(before, 2, t, rec, L)
    before = [0, 1] * 8
    _refused(lambda: _planted(CL._HOOKS, "growth", "any", planted), "no-solution:sampler-invariant")


def test_vectorised_sw_reproduces_the_unionfind_trajectory():
    """Ruling A-R2: cluster_fast.py (csgraph) against cluster.py (UnionFind), sweep by sweep, same named seed."""
    for L, q, sweeps in ((4, 2, 12), (5, 3, 8), (CF.FAST_FROM, 2, 3)):
        p = FK.bond_probability(q, "beta_c")
        u, v = CL.bond_lists(L)
        ua, va = XS.torus_bonds(L)
        ra, rb = np.random.default_rng(SEED), np.random.default_rng(SEED)
        s = ra.integers(q, size=L * L).tolist()
        t = rb.integers(q, size=L * L)
        uf = UnionFind(L * L)
        for _ in range(sweeps):
            a = CL.sw_sweep(s, q, p, u, v, ra, uf)
            b = CF.sw_sweep_fast(t, q, p, ua, va, rb)
            ref = CF.canonical_labels(np.array(uf.labels()))
            assert a == b[:3] and s == t.tolist() and np.array_equal(ref, b[3]), (L, q)
    # control: the comparison can fail (the vectorised sweep at a bond probability off by 0.05 diverges)
    p = FK.bond_probability(2, "beta_c")
    u, v = CL.bond_lists(6)
    ua, va = XS.torus_bonds(6)
    ra, rb = np.random.default_rng(SEED), np.random.default_rng(SEED)
    s, t = ra.integers(2, size=36).tolist(), rb.integers(2, size=36)
    uf = UnionFind(36)
    differ = False
    for _ in range(6):
        CL.sw_sweep(s, 2, p, u, v, ra, uf)
        CF.sw_sweep_fast(t, 2, p + 0.05, ua, va, rb)
        differ = differ or s != t.tolist()
    assert differ


# ---- FK at q = 1 is P2's bond percolation -----------------------------------------------------------------
def test_fk_at_q1_reproduces_p2_crossing_polynomial():
    for W, H in ((3, 2), (2, 3), (4, 2)):
        holds, detail = FP.p2_tie(W, H)
        assert holds, detail
    want = EN.crossing_probability(EN.crossing_counts("square", "bond", 3, 3), Fraction(1, 2))   # 43/64
    hits, n = FP.fk_crossing_sampled(3, 3, 0.5, 1000, SEED)        # p_sd(1) = 1/2, the self-dual point
    sd = math.sqrt(float(want) * (1 - float(want)) / n)
    assert abs(hits / n - float(want)) < 2.576 * sd, (hits, n, want)    # binomial 99% band
    assert abs(hits / n - 0.5) > 2.576 * sd                              # control: not R = 1/2
    wrapped = _planted(LT._HOOKS, "wrap_crossing", True, lambda: FP.p2_tie(3, 2))   # periodic bonds: differs
    assert wrapped[0] is False


# ---- the catalog entry -------------------------------------------------------------------------------------
def _run(**p):
    e = catalog()["cluster_mc"]
    return compose(e.spec, "sample").run(**p)


def test_entry_labels_finality_and_refusals():
    r = _run(L=3, sweeps=60, burn_in=10, seeds=(0, 1, 2))
    assert r.rigor == "statistical" and not r.final
    assert r.values["mc_status"] == "undetermined"                 # 60 sweeps < 100 tau_int: labelled, not refused
    assert "UNDETERMINED" in r.finality["energy_per_site_sampled"].note
    assert r.finality["beta_c"].final and "energy_per_site_sampled" in r.errors
    ex = r.values["exact_targets"]
    assert abs(ex["energy_exact"] - ex["energy_enumerated"]) < 1e-12       # Kaufman = enumeration at L = 3
    w = _run(algorithm="wolff", q=3, L=2, sweeps=40, burn_in=10, seeds=(3, 4))
    assert w.values["wolff_flips_per_sweep"] and "m2_fk_estimator_sampled" not in w.values
    for bad, reason in (({"algorithm": "heatbath"}, "precondition:algorithm"),
                        ({"algorithm": "metropolis", "q": 3}, "precondition:algorithm"),
                        ({"algorithm": "metropolis", "L": 3}, "precondition:algorithm"),
                        ({"algorithm": "metropolis", "L": 2}, "precondition:algorithm"),
                        ({"L": 1}, "precondition:size"), ({"sweeps": 2}, "precondition:sweeps"),
                        ({"seeds": (1,)}, "precondition:named-seeds"), ({"q": 1}, "precondition:potts-q"),
                        ({"beta": "0"}, "precondition:beta"), ({"J": 2}, "precondition:known-parameter")):
        _refused(lambda: _run(**bad), reason)
