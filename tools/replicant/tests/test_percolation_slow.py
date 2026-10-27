"""P1, slow tier: sampling bands with enough seeds to be meaningful, and the finite-size scaling of
the Erdos-Renyi critical window (percolation/window.py).  Many comparisons are made here, so each
band is the 99.9% Student-t half-width (Bonferroni-style margin over about 40 comparisons)."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.percolation import sampling as SM
from replicant.percolation.cavity import percolation_giant
from replicant.percolation.clusters import cluster_distribution
from replicant.percolation.sampling import sample_percolation
from replicant.percolation.window import window_scaling
from replicant.testing import slow

CASES = ((("bethe", 3), ("bond", "3/4")), (("bethe", 3), ("site", "3/4")), (("bethe", 4), ("site", "2/5")),
         (("bethe", 3), ("bond", "2/5")), (("bethe", 3), ("bond", "1/4")), (("poisson", 2), ("bond", 1)),
         (("poisson", 4), ("site", "1/2")), (("poisson", "3/2"), ("bond", "1/2")),
         (("finite", "1:1/4,3:3/4"), ("site", "2/3")))


@slow
def test_sampled_graphs_match_the_certified_values_with_many_seeds():
    seeds = tuple(range(100, 132))
    rows = []
    for law, dil in CASES:
        d = (dil[0], Fraction(dil[1]))
        g = percolation_giant(law, d)
        ps = cluster_distribution(law, d, 3).lagrange
        r = sample_percolation(law, dil, 20000, seeds, 3)
        t = SM._t(0.9995, len(seeds) - 1)
        cert = {"mean_finite": g.mean_finite, **{f"P_s{s}": ps[s - 1] for s in (1, 2, 3)}}
        if g.phase == "above threshold":
            cert["P_inf"] = g.P_inf
        for key, c in cert.items():
            off, band = abs(r.mean[key] - float(c.mid())), t * r.stderr[key]
            rows.append((law, dil, key, off, band))
            assert off <= band, (law, dil, key, r.mean[key], float(c.mid()), band)
    assert sum(1 for *_, off, band in rows if off > band / 3) >= 1     # not all trivially inside


@slow
def test_er_critical_window_scales_as_n_to_the_two_thirds():
    out = window_scaling([2000, 8000, 32000, 128000, 512000], range(60))
    print("ER window: slope", out["slope"], "band99", out["band99"], "systematic", out["systematic"],
          "excludes 1/2:", out["excludes_1_2"], "excludes 3/4:", out["excludes_3_4"])
    assert out["contains_2_3"], out
    assert out["np_theorem_1"]["fraction_above"] == 0 and out["np_theorem_2"]["cp99_lower"] <= 0.946
    # planted: a supercritical "critical" graph (edge probability x 1.5) leaves the window
    saved = SM._HOOKS["edge_prob_scale"]
    SM._HOOKS["edge_prob_scale"] = 1.5
    try:
        try:
            bad = window_scaling([2000, 8000, 32000], range(20))
        except Refusal as r:
            assert r.reason == "no-solution:np-theorem-1", r.reason
        else:
            raise AssertionError(f"planted supercritical window not refused: {bad['slope']}")
    finally:
        SM._HOOKS["edge_prob_scale"] = saved
        assert SM._HOOKS["edge_prob_scale"] is saved


@slow
def test_er_window_width_is_n_to_the_minus_one_third():
    """At fixed lambda = +-1 the scaled size E|C1|/N^(2/3) agrees between N = 4000 and 256000 (band plus
    the stated systematic); a fixed offset c = 1 +- 0.05 and a planted N^(-1/4) window drift."""
    from replicant.percolation import window as W
    Ns, seeds = [4000, 256000], range(60)
    up, down = W.window_scaling(Ns, seeds, lam=1), W.window_scaling(Ns, seeds, lam=-1)
    assert up["scaled_agreement"]["agrees"] and down["scaled_agreement"]["agrees"], (up, down)
    assert up["rows"][-1]["scaled_mean"] > down["rows"][-1]["scaled_mean"]
    for delta in (0.05, -0.05):
        fixed = W.window_scaling(Ns, seeds, offset=delta)
        print("fixed offset", delta, fixed["scaled_agreement"])
        assert fixed["scaled_agreement"]["drifts"], fixed["scaled_agreement"]
    saved = W._HOOKS["window_exponent"]
    W._HOOKS["window_exponent"] = -5 / 4
    try:
        wide = W.window_scaling(Ns, seeds, lam=1)
    finally:
        W._HOOKS["window_exponent"] = saved
        assert W._HOOKS["window_exponent"] is saved
    assert wide["scaled_agreement"]["drifts"], wide["scaled_agreement"]
