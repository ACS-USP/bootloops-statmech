"""E6b (b): the symmetric two-group stochastic block model, the Kesten-Stigum line and BP.

Convention (inference/sbm.py; Mossel-Neeman-Sly / Massoulie): labels sigma_i = +-1 i.i.d.
uniform; each pair is an edge with probability a/N if sigma_i = sigma_j, b/N otherwise
(DKMZ c_in = a, c_out = b); mean degree c = (a + b)/2.  Kesten-Stigum (DKMZ Eq. (44) at q = 2):
|a - b| > 2 sqrt(c), i.e. tau = (a - b)^2 / (2 (a + b)) > 1.  BP: DKMZ Eqs. (26)-(28) with the
external field (27); overlap Q = |(1/N) sum_i sigma_i sigmahat_i| (DKMZ Eq. (5) with n_a = 1/2,
maximised over the global flip).  Bank: reference/sbm.json (DKMZ Fig. 1 caption).
"""
import io
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.inference import sbm as B
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("sbm")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _ab(c, tau):
    """(a, b) with mean degree c and (a - b)^2 / (2 (a + b)) = tau, a > b."""
    d = math.sqrt(4 * c * tau)
    return c + d / 2, c - d / 2


# ---- the Kesten-Stigum line ----------------------------------------------------------------
def test_kesten_stigum_rule_and_bank():
    ks = B.sbm_kesten_stigum(5.5, 0.5)
    with mp.workdps(30):
        assert ks.detectable and abs(ks.margin - mp.mpf(25) / 12) < mp.mpf("1e-28") and ks.mean_degree == 3
    assert not B.sbm_kesten_stigum(4, 2).detectable
    assert not B.sbm_kesten_stigum(6, 2).detectable                  # tau = 1 exactly: impossible (MNS)
    assert B.sbm_kesten_stigum(6, 2).impossibility_proved and not B.sbm_kesten_stigum(1, 0.5).impossibility_proved
    assert B.sbm_kesten_stigum(0.5, 5.5).detectable                  # disassortative: same rule
    line = B.sbm_ks_line(3)
    assert check(BANK["inverse_eps_c_q2_c3"], line.ratio, label="inv").ok, line
    with mp.workdps(30):
        assert abs(line.a - (3 + mp.sqrt(3))) < mp.mpf("1e-25") and abs(line.b - (3 - mp.sqrt(3))) < 1e-25
    # positive control: a shifted rule tau' = (a - b)^2 / (a + b) gives the wrong line
    shifted = (3 + math.sqrt(3) / math.sqrt(2)) / (3 - math.sqrt(3) / math.sqrt(2))
    assert not check(BANK["inverse_eps_c_q2_c3"], shifted, label="shifted").ok
    _refuses(lambda: B.sbm_ks_line(1), "no-transition:ks-line")


# ---- the sampler ------------------------------------------------------------------------------
def test_sbm_sample_degrees_and_edge_classes():
    """Mean degree (a + b)/2 and in-group edge fraction a/(a + b), each within 4 standard errors
    of the binomial counts (N = 4000, seed 0; measured: see report)."""
    a, b, N = 5.0, 1.0, 4000
    g = B.sbm_sample(N, a, b, 0)
    M = len(g.edges)
    assert np.all(g.edges[:, 0] < g.edges[:, 1]) and len({tuple(e) for e in g.edges}) == M
    same = int(np.sum(g.labels[g.edges[:, 0]] == g.labels[g.edges[:, 1]]))
    n_plus = int(np.sum(g.labels == 1))
    pin = n_plus * (n_plus - 1) / 2 + (N - n_plus) * (N - n_plus - 1) / 2
    pout = n_plus * (N - n_plus)
    assert abs(same - pin * a / N) < 4 * math.sqrt(pin * a / N), (same, pin * a / N)
    assert abs((M - same) - pout * b / N) < 4 * math.sqrt(pout * b / N)
    assert abs(2 * M / N - (a + b) / 2) < 4 * math.sqrt(2 * (a + b) / 2 / N) * 2
    assert np.array_equal(B.sbm_sample(N, a, b, 0).edges, g.edges)
    assert not np.array_equal(B.sbm_sample(N, a, b, 1).edges, g.edges)


def test_triangular_index_map_is_a_bijection():
    for n in (2, 3, 7, 50):
        P = n * (n - 1) // 2
        i, j = B._tri_pairs(np.arange(P), n)
        assert np.all(i < j) and len(set(zip(i.tolist(), j.tolist()))) == P and j.max() == n - 1


# ---- BP --------------------------------------------------------------------------------------
def test_bp_fixed_point_satisfies_dkmz_equations():
    """Independent check of the vectorised BP: at its converged messages, DKMZ Eq. (26) with the
    field (27), evaluated edge by edge in plain Python, is satisfied.  Planted error in the engine
    (the external field dropped, _field_scale = 0): the per-edge residual is large."""
    g = B.sbm_sample(1500, 5.5, 0.5, 3)
    r = B.sbm_bp(g, 5.5, 0.5, iters=400, seed=0)
    assert r.converged and not r.factorized and r.overlap > 0.5, r
    assert B.dkmz_residual(g, 5.5, 0.5, r.messages) < 1e-6
    saved = B._field_scale
    try:
        B._field_scale = 0.0
        bad = B.sbm_bp(g, 5.5, 0.5, iters=400, seed=0)
    finally:
        B._field_scale = saved
    assert B._field_scale == saved
    assert B.dkmz_residual(g, 5.5, 0.5, bad.messages) > 1e-4


def test_bp_small_graphs_far_from_the_line():
    """Default tier: N = 2000, seeds 0-3, tau = 0.25 and tau = 2.5 (far from the KS line).  (At
    tau = 3, c = 3 the out-degree b is 0: the groups are disconnected, only the field ties their
    labels, and seed 1 at N = 5000 converges to a fixed point with 88% of the nodes in one group.)"""
    lo = B.sbm_bp_ensemble(2000, *_ab(3, 0.25), seeds=(0, 1, 2, 3), iters=300)
    hi = B.sbm_bp_ensemble(2000, *_ab(3, 2.5), seeds=(0, 1, 2, 3), iters=300)
    assert all(lo.factorized) and lo.band.mean < 0.05, lo.band
    assert not any(hi.factorized) and hi.band.mean - hi.band.half_width > 0.5, hi.band


@slow
def test_sbm_kesten_stigum_threshold():
    """N = 5000, mean degree 3, seeds 0-9.  Below the line (tau = 0.33, 0.75) BP converges to
    the factorized fixed point and the overlap is that of a random guess, E|Q| = sqrt(2/(pi N))
    = 0.0113 (its band contains it); above (tau = 1.5, 2.08) the band's lower edge is > 0.1.
    Shifted-threshold controls: the rule tau' = 2 tau (line (a - b)^2 = a + b) predicts
    detection at tau = 0.75, and tau' = tau / 2 (line (a - b)^2 = 4 (a + b)) predicts none at
    tau = 1.5; BP contradicts both, and agrees with sbm_kesten_stigum at all four points."""
    seeds = tuple(range(10))
    guess = math.sqrt(2 / (math.pi * 5000))
    rules = {"code": lambda a, b: B.sbm_kesten_stigum(a, b).detectable,
             "x2": lambda a, b: 2 * (a - b)**2 / (2 * (a + b)) > 1,
             "half": lambda a, b: (a - b)**2 / (4 * (a + b)) > 1}
    seen = {}
    for tau in (1 / 3, 0.75, 1.5, 25 / 12):
        a, b = _ab(3, tau)
        ens = B.sbm_bp_ensemble(5000, a, b, seeds=seeds, iters=2000)
        if tau < 1:
            assert abs(ens.band.mean - guess) <= ens.band.half_width, (tau, ens.band)
            detected = False
        else:
            assert not any(ens.factorized) and ens.band.mean - ens.band.half_width > 0.1, (tau, ens.band)
            detected = True
        seen[tau] = {k: f(a, b) == detected for k, f in rules.items()}
    assert all(v["code"] for v in seen.values()), seen
    assert not seen[0.75]["x2"] and not seen[1.5]["half"], seen


# ---- refusals -----------------------------------------------------------------------------------
def test_sbm_refusals_are_named():
    for bad in ((-1, 2), (2, "x"), (float("nan"), 1), (0, 0), (True, 1), (None, 1)):
        _refuses(lambda: B.sbm_kesten_stigum(*bad), "precondition:degree-parameters")
        _refuses(lambda: B.sbm_sample(500, *bad, 0), "precondition:degree-parameters")
    r = _refuses(lambda: B.sbm_kesten_stigum(5, 1, q=3), "unsupported:q")
    assert "q = 2" in r.detail
    for bad in (5, 500.0, True):
        _refuses(lambda: B.sbm_sample(bad, 5, 1, 0), "precondition:size")
    _refuses(lambda: B.sbm_sample(20, 30, 1, 0), "precondition:edge-probability")
    for bad in (-1, "a", 1.5):
        _refuses(lambda: B.sbm_sample(500, 5, 1, bad), "precondition:named-seeds")
    for bad in ([], "x", (0, -2)):
        _refuses(lambda: B.sbm_bp_ensemble(500, 5, 1, seeds=bad, iters=10), "precondition:named-seeds")
    g = B.sbm_sample(500, 5, 1, 0)
    _refuses(lambda: B.sbm_bp(g, 5, 1, iters=0, seed=0), "precondition:positive-iters")
    _refuses(lambda: B.sbm_bp(g, 5, 1, iters=5, seed=0, damping=1), "precondition:damping")
    _refuses(lambda: B.sbm_bp("graph", 5, 1, iters=5, seed=0), "precondition:sbm-graph")


# ---- catalog entry --------------------------------------------------------------------------------
def test_catalog_entry_sbm():
    dps = mp.mp.dps
    e = catalog()["sbm"]
    assert e.routes == {"cavity"} and e.default_route == "cavity"
    res = compose(e.spec, "cavity").run(c=3)
    assert check(BANK["inverse_eps_c_q2_c3"], res.values["inverse_eps_ks"], label="x").ok
    assert res.finality["inverse_eps_ks"].state == "exact-by-literature" and res.final
    assert "Massoulie" in res.finality["inverse_eps_ks"].cite
    pt = compose(e.spec, "cavity").run(a="5.5", b="0.5")
    assert pt.values["ks_detectable"] and pt.final
    assert "Mossel" in pt.finality["ks_margin"].cite
    low = compose(e.spec, "cavity").run(a="1", b="0.5")       # a + b <= 2: MNS Thm 2.1 silent
    assert low.finality["ks_margin"].state == "not-computed" and not low.final
    fin = compose(e.spec, "cavity").run(a="5.5", b="0.5", N=1000, seeds=(0, 1, 2), iters=200)
    assert "bp_overlap" in fin.statistical and not fin.final and fin.errors["bp_overlap"] > 0
    assert fin.finality["bp_overlap"].state == "not-computed"
    _refuses(lambda: compose(e.spec, "cavity").run(), "precondition:state-point")
    _refuses(lambda: compose(e.spec, "cavity").run(a="5"), "precondition:state-point")
    r = _refuses(lambda: compose(e.spec, "cavity").run(c_in="5", b="1"), "precondition:known-parameter")
    assert "c_in" in r.detail
    _refuses(lambda: compose(e.spec, "cavity").run(a="5", b="1", seeds=(0, 1)),
             "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "replica"), "missing-step:route:replica")
    assert e.derivation("cavity")
    from replicant.cli import main
    err = io.StringIO()
    code = main(["run", "sbm", "--route", "cavity", "--param", "a=5", "--param", "b=1",
                 "--param", "N=500", "--param", "seeds=-1,"], io.StringIO(), err)
    assert code == 2 and "REFUSED: precondition:named-seeds" in err.getvalue(), err.getvalue()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `sbm`" in guide and "precondition:degree-parameters" in guide
    assert all(check_ref(e, r).ok for r in e.refs if not r.slow)
    assert mp.mp.dps == dps
