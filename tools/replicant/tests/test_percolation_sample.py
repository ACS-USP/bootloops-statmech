"""P1: the sample route (percolation/sampling.py) against the certified cavity values: a small band in
the default tier, and the sampler's planted errors (a wrong edge probability, a wrong occupation).
The bands with many seeds and the critical window are in tests/test_percolation_slow.py."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.core import Refusal
from replicant.percolation import sampling as SM
from replicant.percolation.cavity import percolation_giant
from replicant.percolation.sampling import sample_percolation

SEEDS = tuple(range(8))


def _refused(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"not refused: expected {reason}")


def _planted(key, value, fn):
    saved = SM._HOOKS[key]
    SM._HOOKS[key] = value
    try:
        return fn()
    finally:
        SM._HOOKS[key] = saved
        assert SM._HOOKS[key] is saved


def _within(r, key, cert):
    """|sampled mean - certified| <= the 99% Student-t half-width over the seeds."""
    return abs(r.mean[key] - float(cert.mid())) <= r.halfwidth99[key]


def test_sampled_giant_matches_the_certified_value_in_a_small_band():
    for law, dil in ((("bethe", 3), ("site", "3/4")), (("poisson", 2), ("bond", 1))):
        r = sample_percolation(law, dil, 2000, SEEDS)
        g = percolation_giant(law, (dil[0], Fraction(dil[1])))
        assert _within(r, "P_inf", g.P_inf), (law, r.mean["P_inf"], r.halfwidth99["P_inf"])
        assert _within(r, "mean_finite", g.mean_finite), law
        assert r.halfwidth99["P_inf"] < 0.03                      # the band is not vacuous
    # the band can fail: a different state point is excluded
    g = percolation_giant(("poisson", 2), ("bond", Fraction(9, 10)))
    assert not _within(sample_percolation(("poisson", 2), ("bond", 1), 2000, SEEDS), "P_inf", g.P_inf)


def test_planted_wrong_edge_probability_is_refused_and_moves_the_giant():
    r = _planted("edge_prob_scale", 1.1, lambda: _refused(
        lambda: sample_percolation(("poisson", 2), ("bond", 1), 2000, SEEDS), "no-solution:sampler-degree"))
    assert "mean degree" in r.detail
    # with the self-check bypassed (instances directly), the giant leaves the certified band
    g = percolation_giant(("poisson", 2), ("bond", 1))
    runs = _planted("edge_prob_scale", 1.1, lambda: [
        SM._instance(SM._float_law(("poisson", 2)), ("bond", 1.0), 2000, 3, s)["P_inf"] for s in SEEDS])
    m, se = np.mean(runs), np.std(runs, ddof=1) / np.sqrt(len(runs))
    assert abs(m - float(g.P_inf.mid())) > SM._t(0.995, len(runs) - 1) * se


def test_planted_wrong_occupation_is_refused():
    for law in (("bethe", 3), ("finite", "1:1/4,3:3/4")):
        for kind in ("bond", "site"):
            _planted("occupation_scale", 1.1, lambda: _refused(
                lambda: sample_percolation(law, (kind, "1/2"), 2000, SEEDS), "no-solution:sampler-occupation"))


def test_sampler_preconditions_and_checks():
    r = sample_percolation(("bethe", 4), ("bond", "1/2"), 1000, (5, 9))
    assert r.checks["mean_degree"]["sampled"] == 4.0 and r.seeds == (5, 9)
    _refused(lambda: sample_percolation(("bethe", 3), ("bond", 1), 1001, SEEDS), "precondition:n")
    _refused(lambda: sample_percolation(("bethe", 3), ("bond", 1), 2000, (1,)), "precondition:named-seeds")
    _refused(lambda: sample_percolation(("bethe", 3), ("bond", 1), 2000, (1, 1)), "precondition:named-seeds")
    _refused(lambda: sample_percolation(("bethe", 3), ("bond", 1), 2000, (-1, 2)), "precondition:named-seeds")
    _refused(lambda: sample_percolation(("powerlaw", 2), ("bond", 1), 2000, SEEDS),
             "unsupported:infinite-second-moment")
    _refused(lambda: sample_percolation(("finite", "1:1/2,3:1/2"), ("bond", 1), 51, SEEDS), "precondition:n")


def test_sample_route_is_statistical_and_never_final():
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()["percolation_graph"]
    res = compose(e.spec, "sample").run(law="finite", pk="1:1/4,3:3/4", p="2/3", dilution="site")
    assert res.rigor == "statistical" and not res.final
    assert set(res.statistical) == {k for k in res.values if k not in e.meta}
    assert all(k in res.errors for k in res.statistical)
    assert all(not st.final for st in res.finality.values())
