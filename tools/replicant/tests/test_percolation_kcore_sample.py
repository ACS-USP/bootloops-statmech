"""P3 sample route: exact peeling and bootstrap spreading on sampled graphs (percolation/kcore_sample.py)
against brute-force cores, the exact identities checked per instance, and the certified values in a
99% Student-t band over named seeds.  Planted errors are restored with an identity assert."""
import os
import sys
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.core import Refusal
from replicant.percolation import kcore_sample as KS
from replicant.percolation.kcore import kcore_fixed_points
from replicant.percolation.kcore_sample import peel, sample_bootstrap, sample_kcore, spread

F = Fraction


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


def _brute_core(edges, n, k):
    """The union of every vertex set S in which each vertex lies in >= k hyperedges inside S."""
    core = set()
    for r in range(n, 0, -1):
        for S in combinations(range(n), r):
            s = set(S)
            inside = [e for e in edges if set(e) <= s]
            if all(sum(e.count(v) for e in inside) >= k for v in s):
                core |= s
    return [v in core for v in range(n)]


def test_peeling_equals_the_brute_force_core_and_a_planted_k_does_not():
    rng = np.random.default_rng(3)
    cases = []
    for K, n, m in ((2, 8, 14), (3, 8, 9), (2, 7, 12)):
        for _ in range(2):
            cases.append((rng.integers(0, n, size=(m, K)).tolist(), n))
    for edges, n in cases:
        for k in (2, 3):
            assert peel(edges, [True] * n, k)[0] == _brute_core(edges, n, k), (edges, k)
    differs = sum(_planted(KS._HOOKS, "peel_k_shift", 1, lambda: peel(e, [True] * n, 2)[0]) !=
                  _brute_core(e, n, 2) for e, n in cases)
    assert differs >= len(cases) // 2


def test_spreading_and_peeling_are_one_identity_and_planted_rules_break_it():
    r = sample_bootstrap(4, 2, "1/20", 400, (0, 1))
    assert abs(r.mean["final_infected"] + r.mean["uninfected_core"] - 1) < 1e-12
    _planted(KS._HOOKS, "spread_shift", 1, lambda: _refused(lambda: sample_bootstrap(4, 2, "1/20", 400, (0, 1)),
                                                            "no-solution:sampler-invariant"))
    _planted(KS._HOOKS, "peel_k_shift", 1, lambda: _refused(lambda: sample_bootstrap(4, 2, "1/20", 400, (0, 1)),
                                                            "no-solution:sampler-invariant"))
    # K = 3, k = 2: the peeled 2-core is csp/xorsat_finite.leaf_removal_core's on the same hypergraph
    sample_kcore(("poisson", F(27, 10)), 3, 2, ("site", 1), 600, (0, 1))
    _planted(KS._HOOKS, "peel_k_shift", -1, lambda: _refused(
        lambda: sample_kcore(("poisson", F(27, 10)), 3, 2, ("site", 1), 600, (0, 1)), "no-solution:sampler-invariant"))


def _band(certified, r, key):
    return abs(r.mean[key] - float(certified)) <= r.halfwidth99[key]


def test_sampled_cores_match_the_certified_values_and_a_wrong_k_leaves_the_band():
    seeds, N = tuple(range(8)), 1000          # more laws, N = 100000: tests/test_percolation_kcore_slow.py
    args = (("poisson", 4), 2, 3, ("site", 1))
    R = kcore_fixed_points(*args)
    r = sample_kcore(*args, N, seeds)
    assert _band(R.core.mid(), r, "core_fraction") and _band(R.edges.mid(), r, "core_edges"), r.mean
    r = _planted(KS._HOOKS, "peel_k_shift", 1, lambda: sample_kcore(*args, N, seeds[:4]))
    assert not _band(R.core.mid(), r, "core_fraction")
    B = kcore_fixed_points(("bethe", 4), 2, 3, ("site", F(19, 20)))
    r = sample_bootstrap(4, 2, "1/20", N, seeds)
    assert _band(1 - B.core.mid(), r, "final_infected"), r.mean


def test_sample_preconditions_and_the_statistical_route():
    _refused(lambda: sample_kcore(("poisson", 4), 2, 3, ("site", 1), 10, (0, 1)), "precondition:n")
    _refused(lambda: sample_kcore(("poisson", 4), 2, 3, ("site", 1), 500, (1,)), "precondition:named-seeds")
    _refused(lambda: sample_kcore(("finite", "3:1"), 3, 2, ("site", 1), 500, (0, 1)), "unsupported:hypergraph-law")
    _refused(lambda: sample_kcore(("poisson", 4), 2, 3, ("site", 2), 500, (0, 1)), "precondition:occupation")
    _refused(lambda: sample_bootstrap(3, 2, "1/20", 501, (0, 1)), "precondition:n")
    from replicant.catalog import catalog
    from replicant.compose import compose
    for name in ("percolation_kcore", "percolation_bootstrap"):
        res = compose(catalog()[name].spec, "sample").run(N=500, seeds=(0, 1, 2))
        assert res.rigor == "statistical" and not res.final
        assert all(st.state == "not-computed" for st in res.finality.values())
