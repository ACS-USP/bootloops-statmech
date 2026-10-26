"""E6a: regular (l, r) LDPC ensembles on the binary erasure channel BEC(eps).

Conventions (inference/ldpc.py): edge-perspective degree polynomials lambda(x) = x^(l-1),
rho(x) = x^(r-1) (the probability that an edge sees a variable / check of each degree);
node perspective Lambda(y) = y^l, Gamma(z) = z^r; design rate 1 - l/r.  Density evolution
(Mezard-Montanari Eq. (15.33)):
    x_{t+1} = eps lambda(1 - rho(1 - x_t)),   x_0 = eps,
x = erasure probability of a variable-to-check message; the bit erasure probability is
eps Lambda(1 - rho(1 - x)).  eps_BP = min over x in (0, 1] of x / lambda(1 - rho(1 - x));
eps_MAP = eps(x*) at the root x* in (x_BP, 1) of the trial entropy (Measson-Montanari-
Urbanke 2008, Lemma 3, Definition 4).  Finite N (inference/ldpc_finite.py, independent of
ldpc.py): configuration-model graphs and the peeling decoder.

Bank: reference/ldpc.json (MMU 2008, Kudekar-Richardson-Urbanke 2010/2012, Pfister slides as
a named secondary source).  The banked digits are targets; the code computes every threshold
from the DEGREE_PAIRS hook.
"""
import io
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref, compare
from replicant.cavity._stats import t_quantile
from replicant.compose import compose
from replicant.core import Refusal
from replicant.inference import ldpc as L
from replicant.inference import ldpc_finite as LF
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("ldpc")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


# ---- hook-free published characterisations (KRU 2010, Lemma 4) ----------------------------
def _p_bp_root(l, r):
    """Root in (0, 1) of p_BP(x) = ((l-1)(r-1) - 1)(1-x)^(r-2) - sum_{i=0}^{r-3} (1-x)^i."""
    f = lambda x: ((l - 1) * (r - 1) - 1) * (1 - x)**(r - 2) - sum((1 - x)**i for i in range(r - 2))
    return mp.findroot(f, (mp.mpf("1e-6"), 1 - mp.mpf("1e-9")), solver="bisect")


def _p_map_root(l, r, lo):
    f = lambda x: x + (1 - x)**(r - 1) * (l + l * (r - 1) * x - r * x) / r - mp.mpf(l) / r
    return mp.findroot(f, (lo, 1 - mp.mpf("1e-9")), solver="bisect")


def _eps_of_x(l, r, x):
    return x / (1 - (1 - x)**(r - 1))**(l - 1)


# ---- thresholds against the bank ------------------------------------------------------------
def test_bp_threshold_36_matches_bank():
    dps = mp.mp.dps
    for l, r in ((3, 6), (4, 8), (5, 10), (6, 12), (7, 14)):
        e = L.bec_bp_threshold(l, r)
        assert _ok(f"eps_bp_l{l}_r{r}", e), (l, r, e)
        with mp.workdps(30):
            x = _p_bp_root(l, r)
            assert abs(_eps_of_x(l, r, x) - e) < mp.mpf("1e-20"), (l, r)
            assert abs(L.bec_thresholds(l, r).x_bp - x) < mp.mpf("1e-15"), (l, r)
    # positive control: the MAP threshold is not within the BP tolerance
    assert not _ok("eps_bp_l3_r6", L.bec_map_threshold(3, 6))
    assert mp.mp.dps == dps


def test_map_threshold_above_bp_threshold_and_matches_bank():
    dps = mp.mp.dps
    pairs = ((3, 6), (4, 8), (5, 10), (6, 12), (7, 14), (3, 4), (3, 5), (4, 6), (3, 7))
    for l, r in pairs:
        th = L.bec_thresholds(l, r)
        assert _ok(f"eps_map_l{l}_r{r}", th.eps_map), (l, r, th)
        assert th.eps_bp < th.eps_map < mp.mpf(l) / r, (l, r, th)      # below Shannon l/r
        assert th.x_bp < th.x_map < 1
        with mp.workdps(30):
            x = _p_map_root(l, r, th.x_bp)
            assert abs(_eps_of_x(l, r, x) - th.eps_map) < mp.mpf("1e-20"), (l, r)
    for l, r in ((3, 6), (4, 8), (3, 4)):       # the area theorem route (quadrature)
        a = L.bec_map_threshold_area(l, r)
        assert abs(a - L.bec_map_threshold(l, r)) < mp.mpf("1e-12"), (l, r, a)
        assert abs(L.trial_entropy(l, r, 1) - (1 - mp.mpf(l) / r)) < mp.mpf("1e-25")
    assert mp.mp.dps == dps


def test_density_evolution_fixed_points():
    eb = float(L.bec_bp_threshold(3, 6))
    lo = L.bec_density_evolution(3, 6, eb - 1e-3)
    assert lo.converged and lo.erasure_prob < 1e-12 and lo.edge_erasure < 1e-12, lo
    hi = L.bec_density_evolution(3, 6, eb + 1e-3)
    assert hi.converged and hi.edge_erasure > 0.2, hi
    for eps in (0.45, 0.5, 0.6):
        d = L.bec_density_evolution(3, 6, eps)
        x = d.edge_erasure
        assert d.converged and abs(eps * (1 - (1 - x)**5)**2 - x) < 1e-13, (eps, d)
        assert abs(d.erasure_prob - eps * (1 - (1 - x)**5)**3) < 1e-14
        with mp.workdps(30):        # the largest root of x = f(x) in (0, eps], by bisection
            g = lambda z: eps * (1 - (1 - z)**5)**2 - z
            root = mp.findroot(g, (mp.mpf(x) - mp.mpf("0.02"), mp.mpf(eps)), solver="bisect")
        assert abs(root - x) < 1e-10, (eps, root, x)
    assert L.bec_density_evolution(3, 6, 0).erasure_prob == 0
    assert L.bec_density_evolution(3, 6, 1).erasure_prob == 1
    two = L.bec_density_evolution(2, 4, 0.2)          # l = 2: stability limit 1/(r - 1)
    assert two.converged and two.erasure_prob < 1e-12


def test_positive_control_wrong_edge_polynomials():
    """The thresholds and DE read lambda, rho from L.DEGREE_PAIRS.  Planting the node-
    perspective polynomials (x^l, x^r) in place of the edge-perspective ones, or only a check
    side off by one (x^(l-1), x^r), moves both thresholds out of the bank tolerance; the
    finite-N peeling decoder never reads the hook, so its success at the planted BP
    threshold is not 1/2-like (checked in the slow crossing test)."""
    orig = L.DEGREE_PAIRS["regular"]
    try:
        L.DEGREE_PAIRS["regular"] = L.RegularPair(var_shift=1, check_shift=1)
        th = L.bec_thresholds(3, 6)
        assert not _ok("eps_bp_l3_r6", th.eps_bp) and not _ok("eps_map_l3_r6", th.eps_map), th
        L.DEGREE_PAIRS["regular"] = L.RegularPair(check_shift=1)
        th = L.bec_thresholds(3, 6)
        assert not _ok("eps_bp_l3_r6", th.eps_bp) and not _ok("eps_map_l3_r6", th.eps_map), th
        assert not _ok("eps_map_l4_r8", L.bec_map_threshold(4, 8))
        de = L.bec_density_evolution(3, 6, 0.5).erasure_prob
        e = catalog()["ldpc_bec"]          # the catalog's own reference checks catch it too
        vals = e.execute("cavity")[0]["values"]
        bad = [compare(e, ref, vals, {}).ok for ref in e.refs if ref.params == {}]
        assert len(bad) == 3 and not any(bad), bad
    finally:
        L.DEGREE_PAIRS["regular"] = orig
    assert L.DEGREE_PAIRS["regular"] is orig
    assert _ok("eps_bp_l3_r6", L.bec_bp_threshold(3, 6))    # cache keyed on the pair object
    assert abs(de - L.bec_density_evolution(3, 6, 0.5).erasure_prob) > 0.01


# ---- finite N --------------------------------------------------------------------------------
def test_ensemble_graph_sockets_and_multi_edges():
    g = LF.ldpc_ensemble_graph(3, 6, 600, seed=4)
    assert g.var_checks.shape == (600, 3) and g.M == 300
    assert np.all(np.bincount(g.var_checks.ravel(), minlength=300) == 6)
    dup = sum(len(set(row)) < 3 for row in g.var_checks.tolist())
    assert g.multi_edges == sum(3 - len(set(row)) for row in g.var_checks.tolist()) and dup <= 10
    again = LF.ldpc_ensemble_graph(3, 6, 600, seed=4)
    assert np.array_equal(g.var_checks, again.var_checks)
    assert not np.array_equal(g.var_checks, LF.ldpc_ensemble_graph(3, 6, 600, seed=5).var_checks)


def _peel_reference(var_checks, M, erased):
    """Parallel-rounds peeling (a different algorithm): each round resolves every check
    with exactly one erased socket at once."""
    er = erased.copy()
    while True:
        cnt = np.bincount(var_checks[er].ravel(), minlength=M)
        one = cnt == 1
        hit = er & one[var_checks].any(axis=1)
        if not hit.any():
            return er
        er &= ~hit


def test_peeling_residual_is_the_maximal_stopping_set():
    for seed in range(8):
        for eps in (0.3, 0.45, 0.6):
            res = LF.ldpc_peeling_decode(3, 6, 120, eps, seed)
            g = LF.ldpc_ensemble_graph(3, 6, 120, seed)
            cnt = np.bincount(g.var_checks[res.residual_mask].ravel(), minlength=g.M)
            assert not np.any(cnt == 1), (seed, eps)                  # a stopping set
            assert not np.any(res.residual_mask & ~res.erased_mask)
            ref = _peel_reference(g.var_checks, g.M, res.erased_mask)
            assert np.array_equal(ref, res.residual_mask), (seed, eps)
            assert res.success == (res.residual == 0)
            assert res.residual_fraction == res.residual / 120


def _residual_band(N, eps, seeds):
    xs = [LF.ldpc_peeling_decode(3, 6, N, eps, s).residual_fraction for s in seeds]
    m = math.fsum(xs) / len(xs)
    se = math.sqrt(math.fsum((x - m)**2 for x in xs) / (len(xs) - 1) / len(xs))
    return m, t_quantile(0.995, len(xs) - 1) * se


def test_density_evolution_vs_peeling_residual_erasure():
    """Default tier, N = 2000, seeds 0-19.  Below threshold (eps = 0.36) every instance
    decodes; at eps = 0.55 the mean residual erasure fraction lies in the 99% Student-t band
    of the DE bit erasure probability (measured: 0.5103 +- 0.0030, band 0.0086, DE 0.5103);
    the DE values at eps -+ 0.01 (0.4955, 0.5247) fall outside the band."""
    for s in range(20):
        r = LF.ldpc_peeling_decode(3, 6, 2000, 0.36, s)
        assert r.success and r.residual == 0, (s, r.residual)
    m, band = _residual_band(2000, 0.55, range(20))
    de = lambda e: L.bec_density_evolution(3, 6, e).erasure_prob
    assert abs(m - de(0.55)) <= band, (m, band, de(0.55))
    assert abs(m - de(0.54)) > band and abs(m - de(0.56)) > band, (m, band)


@slow
def test_density_evolution_vs_peeling_residual_erasure_n20000():
    """N = 20000, seeds 0-29: eps = 0.40 decodes always; eps = 0.50 mean residual in the 99%
    t band of DE (measured: 0.4291 +- 0.0015, band 0.0042, DE 0.4293); DE at eps -+ 0.005
    (0.4200, 0.4383) outside.  Closer to eps_BP the finite-N mean is biased low (measured at
    eps = 0.45, 60 seeds: 0.3118 +- 0.0016 vs DE 0.3159), so 0.50 is used."""
    for s in range(10):
        assert LF.ldpc_peeling_decode(3, 6, 20000, 0.40, s).success, s
    m, band = _residual_band(20000, 0.50, range(30))
    de = lambda e: L.bec_density_evolution(3, 6, e).erasure_prob
    assert abs(m - de(0.50)) <= band, (m, band, de(0.50))
    assert abs(m - de(0.495)) > band and abs(m - de(0.505)) > band, (m, band)


def _brackets_half(k, stats):
    """Crossing check centred on grid point k (eps = eps_BP + 0.005 k): the success CP interval
    lies above 1/2 one step below and below 1/2 one step above."""
    return stats[k - 1].lo > 0.5 and stats[k + 1].hi < 0.5


@slow
def test_peeling_finite_N_crossing():
    """N = 20000, (3, 6), seeds 0-59 at eps_BP + {-0.015, -0.01, -0.005, 0, 0.005, 0.01,
    0.015}.  The 99% Clopper-Pearson interval of the success fraction lies above 1/2 at
    eps_BP - 0.005 and below 1/2 at eps_BP + 0.005, so the crossing is within +-0.005 of
    eps_BP (measured: 54/60 [0.759, 0.974] at -0.005, 28/60 at 0, 6/60 [0.026, 0.241] at
    +0.005).  Controls: the same criterion centred on eps_BP -+ 0.01 fails; the BP thresholds
    computed with planted polynomials ((x^2, x^6): 0.3664; (x^3, x^6): 0.4366, 0.007 above
    eps_BP) are not crossings: the success interval there excludes 1/2."""
    eb = float(L.bec_bp_threshold(3, 6))
    stats = {k: LF.ldpc_peeling_success(3, 6, 20000, eb + 0.005 * k, range(60))
             for k in range(-3, 4)}
    assert _brackets_half(0, stats), {k: (s.n_success, s.lo, s.hi) for k, s in stats.items()}
    for shifted in (-2, 2):                     # thresholds at eps_BP -+ 0.01
        assert not _brackets_half(shifted, stats), shifted
    orig = L.DEGREE_PAIRS["regular"]
    planted = []
    try:
        for wrong in (L.RegularPair(check_shift=1), L.RegularPair(var_shift=1, check_shift=1)):
            L.DEGREE_PAIRS["regular"] = wrong
            planted.append(float(L.bec_bp_threshold(3, 6)))
    finally:
        L.DEGREE_PAIRS["regular"] = orig
    assert L.DEGREE_PAIRS["regular"] is orig
    assert all(abs(e - eb) > 0.005 for e in planted), planted
    for e in planted:
        p = LF.ldpc_peeling_success(3, 6, 20000, e, range(60))
        assert p.hi < 0.5 or p.lo > 0.5, (e, p)         # a planted threshold is not a crossing


# ---- refusals --------------------------------------------------------------------------------
def test_refusals_are_named():
    for bad in ((1, 4), (3, 3), (4, 3), (3.0, 6), (True, 6), (3, "6")):
        _refuses(lambda: L.bec_density_evolution(*bad, 0.4), "precondition:degree-pair")
        _refuses(lambda: LF.ldpc_peeling_decode(*bad, 600, 0.4, 0), "precondition:degree-pair")
    r = _refuses(lambda: L.bec_bp_threshold(2, 4), "precondition:variable-degree-at-least-3")
    assert "1/(r - 1)" in r.detail
    _refuses(lambda: L.bec_map_threshold(2, 6), "precondition:variable-degree-at-least-3")
    for eps in (-0.1, 1.5, "x", float("nan"), None):
        _refuses(lambda: L.bec_density_evolution(3, 6, eps), "precondition:erasure-probability")
        _refuses(lambda: LF.ldpc_peeling_decode(3, 6, 600, eps, 0),
                 "precondition:erasure-probability")
    _refuses(lambda: LF.ldpc_peeling_decode(3, 6, 50, 0.4, 0), "precondition:enough-variables")
    _refuses(lambda: LF.ldpc_peeling_decode(3, 6, 600.0, 0.4, 0), "precondition:enough-variables")
    r = _refuses(lambda: LF.ldpc_ensemble_graph(3, 5, 601, 0), "precondition:socket-count")
    assert "N l" in r.detail
    _refuses(lambda: L.bec_thresholds(3, 6, pair="irregular"), "unsupported:degree-pair")


def test_malformed_seeds_are_refused_by_name():
    for bad in (-1, "a", 1.5, True):
        r = _refuses(lambda: LF.ldpc_peeling_decode(3, 6, 600, 0.4, bad), "precondition:named-seeds")
        assert "non-negative integers" in r.detail
    for bad in ([], "x", 3, (0, -2)):
        _refuses(lambda: LF.ldpc_peeling_success(3, 6, 600, 0.4, bad), "precondition:named-seeds")
    e = catalog()["ldpc_bec"]
    _refuses(lambda: compose(e.spec, "cavity").run(eps="0.4", N=600, seeds="x"),
             "precondition:named-seeds")
    from replicant.cli import main
    err = io.StringIO()
    code = main(["run", "ldpc_bec", "--route", "cavity", "--param", "eps=0.4", "--param", "N=600",
                 "--param", "seeds=-1,"], io.StringIO(), err)
    assert code == 2 and "REFUSED: precondition:named-seeds" in err.getvalue(), err.getvalue()


# ---- catalog entry ---------------------------------------------------------------------------
def test_catalog_entry_ldpc_bec():
    dps = mp.mp.dps
    e = catalog()["ldpc_bec"]
    assert e.routes == {"cavity"} and e.default_route == "cavity"
    res = compose(e.spec, "cavity").run()
    v = res.values
    assert _ok("eps_bp_l3_r6", v["eps_bp"]) and _ok("eps_map_l3_r6", v["eps_map"])
    assert abs(v["eps_map_area"] - v["eps_map"]) < 1e-12
    assert res.finality["eps_bp"].state == "exact-by-literature"
    assert "Luby" in res.finality["eps_bp"].cite
    assert "Measson" in res.finality["eps_map"].cite and "Kudekar" in res.finality["eps_map"].cite
    assert res.final and all(c.ok for c in res.reference_checks())
    pt = compose(e.spec, "cavity").run(eps="0.45")
    assert pt.finality["de_bit_erasure"].state == "exact-by-literature" and pt.values["de_converged"]
    fin = compose(e.spec, "cavity").run(eps="0.36", N=600, seeds=(0, 1, 2))
    assert "peel_success_fraction" in fin.statistical and not fin.final
    assert fin.finality["peel_success_fraction"].state == "not-computed"
    assert fin.values["peel_success_fraction"] == 1.0
    r = _refuses(lambda: compose(e.spec, "cavity").run(epsilon="0.4"), "precondition:known-parameter")
    assert "eps" in r.detail
    _refuses(lambda: compose(e.spec, "cavity").run(N=600), "precondition:state-point")
    r = _refuses(lambda: compose(e.spec, "cavity").run(eps="0.4", seeds=(0, 1)),
                 "precondition:known-parameter")
    assert "need N" in r.detail
    _refuses(lambda: compose(e.spec, "replica"), "missing-step:route:replica")
    assert e.derivation("cavity")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `ldpc_bec`" in guide and "precondition:variable-degree-at-least-3" in guide
    assert all(check_ref(e, r).ok for r in e.refs if not r.slow)
    assert mp.mp.dps == dps


def test_map_cite_and_de_note_name_what_was_read():
    """T15 (E6a deferred minors): MMU's Example 6 is cited for (3, 6) only (it verifies the
    MAP conditions for that pair); other pairs cite KRU Lemma 4 without it.  The DE label
    says the concentration theorem (Richardson-Urbanke 2001) was not fetched.  l = 2 is
    refused by the catalog on every call (the threshold step always runs), while density
    evolution itself accepts l = 2 through the Python API."""
    e = catalog()["ldpc_bec"]
    c36 = compose(e.spec, "cavity").run().finality["eps_map"].cite
    c48 = compose(e.spec, "cavity").run(l=4, r=8).finality["eps_map"].cite
    assert "Ex. 6" in c36 and "Ex. 6" not in c48 and "Kudekar" in c48, (c36, c48)
    de = compose(e.spec, "cavity").run(eps="0.45").finality["de_bit_erasure"]
    assert "Richardson & Urbanke 2001, was not fetched" in de.note, de.note
    _refuses(lambda: compose(e.spec, "cavity").run(l=2, r=4, eps="0.3"),
             "precondition:variable-degree-at-least-3")
    assert L.bec_density_evolution(2, 4, 0.3).converged
