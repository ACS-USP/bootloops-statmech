"""P3 slow tier: large sampled graphs (N = 100000, 16 named seeds) against the certified k-core and
bootstrap values in a 99% Student-t band, the jump seen on both sides of the ER 3-core threshold, and
a planted wrong k leaving the band."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp

from replicant.percolation import kcore_sample as KS
from replicant.percolation.kcore_dgm import dgm_threshold
from replicant.percolation.kcore_threshold import occupation_threshold
from replicant.percolation.kcore import kcore_fixed_points
from replicant.percolation.kcore_sample import sample_bootstrap, sample_kcore
from replicant.testing import slow

F = Fraction
N, SEEDS = 100000, tuple(range(16))


def _band(certified, r, key):
    return abs(r.mean[key] - float(certified)) <= r.halfwidth99[key]


@slow
def test_large_sampled_cores_match_the_certified_values():
    for args in ((("poisson", 4), 2, 3, ("site", 1)), (("poisson", F(27, 10)), 3, 2, ("site", 1)),
                 (("bethe", 5), 2, 3, ("bond", F(9, 10))), (("finite", "1:1/4,3:1/4,5:1/2"), 2, 3, ("site", 1))):
        R, r = kcore_fixed_points(*args), sample_kcore(*args, N, SEEDS)
        assert _band(R.core.mid(), r, "core_fraction") and _band(R.edges.mid(), r, "core_edges"), (args, r.mean)
    B = kcore_fixed_points(("bethe", 5), 2, 4, ("site", F(49, 50)))
    r = sample_bootstrap(5, 2, "1/50", N, SEEDS)
    assert _band(1 - B.core.mid(), r, "final_infected"), r.mean
    args = (("poisson", 4), 2, 3, ("site", 1))
    saved = KS._HOOKS["peel_k_shift"]
    KS._HOOKS["peel_k_shift"] = 1
    try:
        r = sample_kcore(*args, N, SEEDS[:4])
    finally:
        KS._HOOKS["peel_k_shift"] = saved
        assert KS._HOOKS["peel_k_shift"] is saved
    assert not _band(kcore_fixed_points(*args).core.mid(), r, "core_fraction")


@slow
def test_the_er_three_core_jumps_at_c_3():
    above = (("poisson", F(345, 100)), 2, 3, ("site", 1))           # c_3 = 3.3509...
    R, r = kcore_fixed_points(*above), sample_kcore(*above, N, SEEDS)
    assert float(R.core.mid()) > 0.3 and _band(R.core.mid(), r, "core_fraction"), r.mean
    below = (("poisson", F(325, 100)), 2, 3, ("site", 1))
    assert kcore_fixed_points(*below).phase == "no core"
    r = sample_kcore(*below, N, SEEDS)
    assert r.mean["core_fraction"] < 1e-3, r.mean


def _near(ball, v, tol):
    with mp.workdps(50):
        return abs(ball.mid(45) - v) < mp.mpf(tol)


@slow
def test_dgm_route_reproduces_thresholds_of_several_laws():
    for law, K, k, kind in ((("finite", "1:1/4,3:1/4,5:1/2"), 2, 3, "site"), (("bethe", 6), 2, 4, "bond"),
                            (("poisson", 3), 3, 3, "site"), (("poisson", 3), 4, 2, "bond")):
        T = occupation_threshold(law, K, k, kind)
        dlaw = ("poisson", F(law[1])) if law[0] == "poisson" else (
            "finite", {6: F(1)} if law[0] == "bethe" else {1: F(1, 4), 3: F(1, 4), 5: F(1, 2)})
        g = dgm_threshold(dlaw, K, k, kind)
        assert _near(T.value, g["threshold_t"], "1e-15") and _near(T.core, g["core_at_jump"], "1e-14")


@slow
def test_second_routes_never_block_far_out_and_at_the_jump():
    from replicant.catalog import catalog
    from replicant.compose import compose
    for name, params in (("percolation_kcore", {"k": 15, "c": 30}), ("percolation_kcore", {"k": 40, "c": 60}),
                         ("percolation_bootstrap", {"d": 50, "ell": 25}),
                         ("percolation_kcore", {"K": 3, "k": 2, "alpha": "818469160761376/1000000000000000"})):
        res = compose(catalog()[name].spec, "cavity").run(**params)
        notes = [n for n in res.notes if n.startswith(("DGM", "csp/"))]
        assert notes and all(": agrees" in n or ": SKIPPED" in n for n in notes), (params, notes)
        assert res.rigor == "certified"
