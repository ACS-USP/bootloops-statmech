"""P1: percolation on trees and random graphs, the certified cavity route (percolation/laws.py,
cavity.py, clusters.py, closed_forms.py), the union-find module, and the catalog entries
percolation_bethe / percolation_graph.  Planted errors are restored with an identity assert."""
import os
import sys
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.core import Refusal
from replicant.percolation import cavity as CV
from replicant.percolation import closed_forms as CF
from replicant.percolation import laws as LW
from replicant.percolation.cavity import percolation_fixed_points, percolation_giant
from replicant.percolation.clusters import cluster_distribution
from replicant.percolation.unionfind import UnionFind

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


# ---- union-find --------------------------------------------------------------------------------
def _bfs_components(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, sizes = [False] * n, []
    for v in range(n):
        if seen[v]:
            continue
        seen[v], stack, size = True, [v], 0
        while stack:
            x = stack.pop()
            size += 1
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        sizes.append(size)
    return sorted(sizes)


def test_union_find_matches_breadth_first_search_and_keeps_its_invariants():
    rng = np.random.default_rng(7)
    for n, m in ((1, 0), (10, 4), (300, 280), (500, 900)):
        edges = rng.integers(0, n, size=(m, 2)).tolist()
        uf = UnionFind(n)
        merges = uf.add_edges(edges)
        assert sorted(uf.set_sizes()) == _bfs_components(n, edges)
        assert uf.n_sets == len(uf.set_sizes()) == n - merges
        assert uf.largest == max(uf.set_sizes()) and uf.sum_sq == sum(x * x for x in uf.set_sizes())
        lab = uf.labels()
        assert all(uf.parent[x] == lab[x] for x in range(n))          # fully compressed
        assert all(uf.connected(a, b) for a, b in edges)
        assert sum(uf.size(r) for r in set(lab)) == n
    uf.reset()
    assert uf.n_sets == 500 and uf.largest == 1 and uf.union(3, 3) == 3
    r = uf.union(1, 2)
    assert r in (1, 2) and uf.size(1) == 2 and uf.union(2, 1) == r      # a no-op returns the root
    _refused(lambda: uf.find(500), "precondition:union-find-element")
    _refused(lambda: uf.find(-1), "precondition:union-find-element")   # no silent wrap-around
    _refused(lambda: UnionFind(-1), "precondition:union-find-element")


def test_a_broken_union_find_fails_the_breadth_first_comparison():
    class NoTies(UnionFind):                 # planted: equal-size sets are never merged
        def union(self, x, y):
            if self.size(x) == self.size(y):
                return self.find(x)
            return UnionFind.union(self, x, y)
    edges = np.random.default_rng(7).integers(0, 300, size=(280, 2)).tolist()
    uf = NoTies(300)
    uf.add_edges(edges)
    assert sorted(uf.set_sizes()) != _bfs_components(300, edges)


# ---- fixed points --------------------------------------------------------------------------------
def test_bethe_fixed_points_switch_at_p_c_and_refuse_it():
    below = percolation_fixed_points(("bethe", 3), ("bond", F(49, 100)))
    above = percolation_fixed_points(("bethe", 3), ("bond", F(51, 100)))
    assert [f.branch for f in below] == ["trivial"] and below[0].stable
    assert [f.branch for f in above] == ["nontrivial", "trivial"]
    assert above[0].stable and not above[1].stable
    assert above[0].u.contains(F(49, 51))                              # (1 - p)/p, MS Eq. (3.10)
    r = _refused(lambda: percolation_fixed_points(("bethe", 3), ("site", F(1, 2))),
                 "ambiguous:critical-point")
    assert "marginal" in r.detail
    # NSW Eq. (24)'s law: the switch at q_c = 1148/2271 = 0.50550 (bank nsw_eq24_site_threshold)
    law = ("finite", "0:86/1000,1:150/1000,2:363/1000,3:238/1000,4:109/1000,5:54/1000")
    assert len(percolation_fixed_points(law, ("site", "0.505"))) == 1
    assert len(percolation_fixed_points(law, ("site", "0.506"))) == 2
    _refused(lambda: percolation_fixed_points(law, ("site", F(1148, 2271))), "ambiguous:critical-point")


def test_near_critical_is_certified_until_it_cannot_be_separated():
    eps = F(1, 10 ** 30)
    g = percolation_giant(("bethe", 3), ("bond", F(1, 2) + eps))
    assert g.selected.branch == "nontrivial" and g.selected.stable
    # P_inf = (slope) eps + O(eps^2): the slope 12 of the bond Bethe k = 3 law
    with mp.workdps(60):
        assert abs(g.P_inf.mid() / mp.mpf(eps.numerator) * eps.denominator - 12) < mp.mpf("1e-28")
    assert not g.bracket["resolved"]                     # the auxiliary bracket reports, never refuses
    _refused(lambda: percolation_giant(("bethe", 3), ("bond", F(1, 2) + F(1, 10 ** 1300))),
             "undetermined:near-critical")


def test_extreme_parameters_are_certified():
    g = percolation_giant(("bethe", 3), ("bond", 0))                    # p = 0: isolated vertices
    assert g.P_inf.contains(0) and g.mean_finite.contains(1)
    g = percolation_giant(("bethe", 3), ("bond", F(1, 10 ** 30)))
    assert g.mean_finite.contains(1 + 3 * F(1, 10 ** 30) / (1 - 2 * F(1, 10 ** 30)))
    g = percolation_giant(("bethe", 3), ("bond", 1))                    # p = 1: u = 0 exactly
    assert g.selected.u.contains(0) and g.selected.u.rad() == 0 and g.P_inf.contains(1)
    assert g.mean_finite is None                                        # no finite cluster at all
    g = percolation_giant(("bethe", 3), ("bond", 1 - F(1, 10 ** 30)))   # p -> 1
    assert g.selected.u.digits >= 40 and g.selected.u.mid() > 0
    g = percolation_giant(("poisson", 1000), ("bond", 1))               # large mean degree
    with mp.workdps(50):
        assert abs(g.selected.u.mid() / mp.exp(-1000) - 1) < mp.mpf("1e-30")   # u = e^-c (1 + O(c e^-c))
    g = percolation_giant(("bethe", 1000), ("site", F(1, 2)))
    assert g.selected.stable and g.P_inf.mid() > mp.mpf("0.49")
    _refused(lambda: percolation_giant(("powerlaw", "5/2"), ("bond", 1)), "unsupported:infinite-second-moment")
    _refused(lambda: percolation_giant(("powerlaw", 3), ("bond", 1)), "unsupported:infinite-second-moment")
    _refused(lambda: percolation_giant(("powerlaw", 4), ("bond", 1)), "unsupported:degree-law")
    _refused(lambda: percolation_giant(("bethe", 3), ("bond", "1.5")), "precondition:occupation")
    _refused(lambda: percolation_giant(("finite", "1:1/2,3:1/3"), ("bond", 1)), "precondition:degree-law")
    r = _refused(lambda: percolation_giant(("bethe", 2), ("bond", 1)), "ambiguous:critical-point")  # cycles
    assert "every u in [0, 1]" in r.detail
    r = _refused(lambda: percolation_giant(("bethe", 3), ("bond", F(1, 2))), "ambiguous:critical-point")
    assert "double, marginal" in r.detail


def test_closed_forms_agree_with_the_certified_roots():
    with CF.prec_guard(256):
        for k, t in ((3, F(3, 4)), (3, F(51, 100)), (4, F(1, 2)), (4, F(9, 10))):
            u = percolation_fixed_points(("bethe", k), ("bond", t))[0].u
            assert u.overlaps(CF.bethe_u(k, t)), (k, t)
        for lam in (F(11, 10), F(2), F(7)):
            for kind, c, t in (("bond", lam, 1), ("bond", 2 * lam, F(1, 2)), ("site", 2 * lam, F(1, 2))):
                g = percolation_giant(("poisson", c), (kind, t))
                S = CF.er_giant(lam)
                assert g.P_inf.overlaps(S if kind == "bond" else CF.arb_q(t) * S), (lam, kind)
                assert g.mean_finite.overlaps(CF.er_mean(lam, S)), (lam, kind)
    for p in (F(1, 10), F(1, 4), F(2, 5)):
        assert percolation_giant(("bethe", 3), ("site", p)).mean_finite.contains(CF.bethe_mean_below(3, p))


def test_threshold_amplitudes_and_molloy_reed():
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = rng.integers(0, 5, size=6)
        if w[1:].sum() == 0:
            continue
        law = LW.parse_law(("finite", {k: F(int(x), int(w.sum())) for k, x in enumerate(w)}))
        # Q = sum k (k - 2) p_k = <k> (G1'(1) - 1): the Molloy-Reed sign is the threshold at t = 1
        assert law.molloy_reed_Q() == law.mean() * (law.branching() - 1)
    # the amplitudes against finite differences of the certified values
    pc, h = F(1, 2), F(1, 10 ** 12)
    for kind, slope in (("bond", 12), ("site", 6)):
        g = percolation_giant(("bethe", 3), (kind, pc + h))
        assert g.p_inf_slope.contains(slope)
        assert abs(g.P_inf.mid() / mp.mpf(h.numerator) * h.denominator - slope) < 1e-9
    g = percolation_giant(("poisson", 2), ("bond", pc - h))
    assert g.chi_amplitude.contains(F(1, 2))                            # <k> t_c^2 = 2/4
    assert abs(g.mean_finite.mid() * float(h) - F(1, 2)) < 1e-9


# ---- cluster sizes ------------------------------------------------------------------------------
def _subtrees_with_root(k, s):
    """Connected s-vertex subsets containing the root of the k-regular tree, counted directly:
    grow from the root by adding a boundary vertex, deduplicated as sets (vertices are paths)."""
    frontier = {frozenset([()])}
    for _ in range(s - 1):
        nxt = set()
        for sub in frontier:
            for v in sub:
                kids = range(k) if v == () else range(k - 1)
                for c in kids:
                    w = v + (c,)
                    if w not in sub:
                        nxt.add(sub | {w})
        frontier = nxt
    return len(frontier)


def test_bethe_animals_counted_directly():
    for k in (3, 4):
        for s in range(1, 6 if k == 3 else 5):
            # P_s = s g_s t^(s-1)(1-t)^... : s g_s is the number of rooted subtrees
            assert s * CF.animals(k, s) == _subtrees_with_root(k, s), (k, s)


def test_cluster_distribution_three_routes_and_the_sum_rule():
    for law, t in ((("bethe", 3), F(3, 4)), (("bethe", 5), F(1, 10)), (("poisson", 2), F(1, 2)),
                   (("poisson", 3), F(1))):
        d = cluster_distribution(law, ("bond", t), 8)
        with CF.prec_guard(256):
            for s, b in enumerate(d.lagrange, 1):
                c = (CF.arb_q(CF.bethe_ps(law[1], t, s)) if law[0] == "bethe"
                     else CF.borel_ps(F(law[1]) * t, s))
                assert b.arb.overlaps(c) and d.series[s - 1].arb.overlaps(c), (law, t, s)
    d = cluster_distribution(("finite", "1:1/4,3:3/4"), ("site", F(2, 3)), 20)   # no closed form
    assert all(a.overlaps(b) for a, b in zip(d.series, d.lagrange))
    g = percolation_giant(("bethe", 3), ("bond", F(3, 4)))
    cluster_distribution(("bethe", 3), ("bond", F(3, 4)), 10, g.S_occupied)       # partial sums ok
    _refused(lambda: cluster_distribution(("bethe", 3), ("bond", F(3, 4)), 10, percolation_giant(
        ("bethe", 3), ("bond", F(99, 100))).S_occupied), "no-solution:sum-rule")
    _refused(lambda: cluster_distribution(("bethe", 3), ("bond", F(3, 4)), 61), "precondition:s-max")


# ---- planted errors ------------------------------------------------------------------------------
def test_planted_wrong_g0_or_g1_is_refused():
    for key in ("g1_as_g0", "g0_as_g1"):
        _planted(LW._HOOKS, key, True, lambda: _refused(
            lambda: percolation_giant(("bethe", 3), ("bond", F(3, 4))), "no-solution:generating-function-audit"))
        _planted(LW._HOOKS, key, True, lambda: _refused(
            lambda: percolation_giant(("finite", "1:1/4,3:3/4"), ("site", 1)),
            "no-solution:generating-function-audit"))
        # invisible on the Poisson law, where G0 = G1 (the plant cannot be seen there, by design)
        ok = _planted(LW._HOOKS, key, True, lambda: percolation_giant(("poisson", 2), ("bond", 1)))
        assert ok.P_inf.overlaps(percolation_giant(("poisson", 2), ("bond", 1)).P_inf)


def test_planted_consistent_wrong_law_is_caught_by_the_closed_forms():
    from replicant.catalog import catalog
    e = catalog()["percolation_bethe"]
    _planted(LW._HOOKS, "coordination_shift", 1, lambda: _refused(
        lambda: e.execute("cavity"), "no-solution:route-disagreement"))
    # the audit cannot see a law wrong in G0 and G1 alike: the fixed point itself runs
    g = _planted(LW._HOOKS, "coordination_shift", 1, lambda: percolation_giant(("bethe", 3), ("bond", F(3, 4))))
    assert not g.selected.u.overlaps(percolation_giant(("bethe", 3), ("bond", F(3, 4))).selected.u)
    # the cluster-size step alone: the generating functions against s g_s t^(s-1) (1-t)^((k-2)s+2)
    from replicant.catalog_percolation import _bethe_law, _clusters
    _planted(LW._HOOKS, "coordination_shift", 1, lambda: _refused(
        lambda: _clusters(_bethe_law)({"p": "3/4"}), "no-solution:route-disagreement"))


def test_planted_shifted_root_is_refused_by_the_mean_size_audit():
    _planted(CV._HOOKS, "root_shift", F(1, 10 ** 20), lambda: _refused(
        lambda: percolation_giant(("poisson", 2), ("bond", 1)), "no-solution:audit"))


def test_planted_short_series_iteration_is_refused():
    from replicant.percolation import clusters as CL
    _planted(CL._HOOKS, "extra_iterations", -1, lambda: _refused(
        lambda: cluster_distribution(("bethe", 3), ("bond", F(3, 4)), 6), "no-solution:route-disagreement"))
    cluster_distribution(("bethe", 3), ("bond", F(3, 4)), 6)          # restored: NSW's n + 1 pass


def test_planted_dropped_root_is_refused():
    for drop in ("nontrivial", "trivial"):
        _planted(CV._HOOKS, "drop_root", drop, lambda: _refused(
            lambda: percolation_giant(("bethe", 3), ("bond", F(3, 4))), "no-solution:dropped-root"))
    # below threshold there is no nontrivial root to drop: the run is unchanged
    g = _planted(CV._HOOKS, "drop_root", "nontrivial", lambda: percolation_giant(("bethe", 3), ("bond", F(1, 4))))
    assert g.mean_finite.contains(F(5, 2))


def test_monotone_bracket_is_a_second_route_that_can_fail():
    g = percolation_giant(("poisson", 2), ("bond", 1))
    assert g.bracket["resolved"] and g.bracket["iterations"] > 10
    wrong = percolation_giant(("poisson", 3), ("bond", 1)).selected.u
    _refused(lambda: CV._bracket(g.law, F(1), wrong), "no-solution:route-disagreement")


# ---- catalog ------------------------------------------------------------------------------------
def test_catalog_entries_are_certified_with_per_value_finality():
    from replicant.catalog import catalog
    from replicant.compose import compose
    from flint import ctx
    cat = catalog()
    before, prec = mp.mp.dps, ctx.prec
    for name, params in (("percolation_bethe", {}), ("percolation_bethe", {"p": "1/4", "dilution": "site"}),
                         ("percolation_graph", {}), ("percolation_graph", {"law": "finite", "pk": "1:1/4,3:3/4"})):
        e = cat[name]
        res = compose(e.spec, "cavity").run(**params)
        assert res.rigor == "certified"
        phys = [k for k in res.values if k not in e.meta]
        assert phys and all(k in res.errors for k in phys)
        fp, fc = res.finality["P_inf"], res.finality["p_c"]
        assert fp.final and res.finality["P_s1"].state == "not-computed"
        if name == "percolation_bethe":       # a BGW theorem (Abraham & Delmas Cor. 2.5, Lemma 2.6)
            assert fp.state == fc.state == "exact-by-literature" and "1506.05571" in fp.cite
            assert res.finality["mean_finite_cluster_size"].state == "stable"
        else:                                 # the tree-like fixed point; Janson & Luczak for the graph
            assert fp.state == "stable" and "0707.1786" in fp.note and not fc.final
    assert mp.mp.dps == before and ctx.prec == prec
    r = _refused(lambda: compose(cat["percolation_bethe"].spec, "cavity").run(p="1/2"), "ambiguous:critical-point")
    _refused(lambda: compose(cat["percolation_graph"].spec, "cavity").run(law="powerlaw", tau=2),
             "unsupported:infinite-second-moment")
    _refused(lambda: compose(cat["percolation_graph"].spec, "cavity").run(law="finite"),
             "precondition:degree-law")
    _refused(lambda: compose(cat["percolation_bethe"].spec, "cavity").run(N=100),
             "precondition:known-parameter")
    assert r.detail
