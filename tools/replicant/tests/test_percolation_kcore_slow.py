"""P3 slow tier: large sampled graphs (N = 100000, 16 named seeds) against the certified k-core and
bootstrap values in a 99% Student-t band, the jump seen on both sides of the ER 3-core threshold, and
a planted wrong k leaving the band."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.percolation import kcore_sample as KS
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
