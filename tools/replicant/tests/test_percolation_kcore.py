"""P3: k-core and bootstrap percolation, the certified cavity route (percolation/kcore.py,
kcore_threshold.py), its second routes (kcore_dgm.py, csp/xorsat.py, Balogh-Pittel's closed forms) and
the catalog entries percolation_kcore / percolation_bootstrap.  Planted errors are restored with an
identity assert."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp

from replicant.core import Refusal
from replicant.percolation import kcore as KC
from replicant.percolation import kcore_threshold as KT
from replicant.percolation.kcore import kcore_fixed_points
from replicant.percolation.kcore_dgm import dgm_fixed, dgm_threshold
from replicant.percolation.kcore_threshold import (bootstrap_closed_form, degree_threshold,
                                                   occupation_threshold)

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


def _near(ball, v, tol, scale=1):
    with mp.workdps(50):
        return abs(ball.mid(45) / scale - v) < mp.mpf(tol)


_PLAIN = {}


def _plain(name, **params):
    """An unplanted run, shared between tests (every hook at its default: asserted)."""
    assert KC._HOOKS == {"k_shift": 0, "drop_hyperedge": False, "drop_root": None}
    assert KT._HOOKS == {"threshold_offset": 0, "edges_exponent_shift": 0}
    key = (name, tuple(sorted(params.items())))
    if key not in _PLAIN:
        _PLAIN[key] = _run(name, **params)
    return _PLAIN[key]


def _run(name, **params):
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()[name]
    return compose(e.spec, "cavity").run(**params)


def test_roots_thresholds_and_the_jump_against_independent_routes():
    # Erdos-Renyi 3-core at c = 4: three roots, the largest selected, DGM's own form agrees
    r = kcore_fixed_points(("poisson", 4), 2, 3, ("site", 1))
    assert len(r.roots) == 3 and r.selected is r.roots[-1] and r.selected.stable
    assert not r.roots[1].stable and r.bracket["resolved"]
    g = dgm_fixed(("poisson", F(4)), 2, 3, "site", 1)
    assert _near(r.core, g["core"], "1e-15") and _near(r.selected.x, g["x"], "1e-15")
    # the 3-core of the 4-regular graph: t_c = 8/9 at y* = 3/4 (y / (3 y^2 - 2 y^3), exact)
    T = occupation_threshold(("bethe", 4), 2, 3, "site")
    assert T.value.contains(F(8, 9)) and T.s_star.contains(F(3, 4)) and not T.endpoint
    # bootstrap closed forms (Balogh-Pittel as stated by Bartha-Pete) for ell = 2, d = 4 .. 7
    for d in range(4, 8):
        qc, ys = bootstrap_closed_form(d, 2)
        T = occupation_threshold(("bethe", d), 2, d - 1, "site")
        assert T.value.contains(1 - qc) and T.s_star.contains(ys), d
    # occupation and mean-degree thresholds are one minimum: t_c(c = 4) = c_3 / 4 (site, K = 2)
    c3, t4 = degree_threshold(2, 3, "site", 1), occupation_threshold(("poisson", 4), 2, 3, "site")
    with mp.workdps(50):
        assert abs(c3.value.mid(45) / 4 - t4.value.mid(45)) < mp.mpf("1e-35")
    # bond and site share t_c at K = 2 (Janson Thms 4.5, 4.6), not the core at the jump
    tb = occupation_threshold(("poisson", 4), 2, 3, "bond")
    assert tb.value.overlaps(t4.value) and not tb.core.overlaps(t4.core)
    # K = 3, k = 2 is E5's XORSAT 2-core: alpha_d, lambda_d and (V, C, q) at alpha = 0.9
    from replicant.csp.xorsat import xorsat_core_size, xorsat_thresholds
    X, D = xorsat_thresholds(3), degree_threshold(3, 2, "site", 1)
    assert _near(D.value, X.alpha_d, "1e-25", 3) and _near(D.s_star, X.lambda_d, "1e-25")
    with mp.workdps(40):                     # csp/xorsat.py parses alpha at the caller's precision
        r, Y = kcore_fixed_points(("poisson", F(27, 10)), 3, 2, ("site", 1)), xorsat_core_size(3, "0.9")
    assert _near(r.core, Y.variables, "1e-25") and _near(r.edges, Y.checks, "1e-25")
    # the continuous 2-core (K = 2) goes through P1's cavity equation: t_c = 1/G1'(1) exactly
    T2 = occupation_threshold(("poisson", 4), 2, 2, "bond")
    assert T2.continuous and T2.value.contains(F(1, 4))


def test_planted_wrong_core_order_is_refused():
    for name, params in (("percolation_kcore", {}), ("percolation_kcore", {"law": "regular", "d": 5}),
                         ("percolation_bootstrap", {})):
        _planted(KC._HOOKS, "k_shift", 1, lambda: _refused(lambda: _run(name, **params),
                                                         "no-solution:route-disagreement"))


def test_planted_missing_hyperedge_term_is_invisible_on_graphs_and_refused_on_hypergraphs():
    base = kcore_fixed_points(("poisson", 4), 2, 3, ("site", 1))
    same = _planted(KC._HOOKS, "drop_hyperedge", True, lambda: kcore_fixed_points(("poisson", 4), 2, 3,
                                                                                 ("site", 1)))
    assert same.core.render(30) == base.core.render(30) and same.edges.render(30) == base.edges.render(30)
    for params in ({"K": 3, "k": 2, "alpha": 1}, {"K": 3, "k": 3, "c": 6}):
        _plain("percolation_kcore", **params)
        _planted(KC._HOOKS, "drop_hyperedge", True, lambda: _refused(
            lambda: _run("percolation_kcore", **params), "no-solution:route-disagreement"))


def test_planted_dropped_root_and_threshold_offset_are_refused():
    for args in ((("poisson", 4), 2, 3, ("site", 1)), (("poisson", 4), 2, 2, ("site", 1)),
                 (("poisson", 3), 3, 2, ("site", 1))):
        _planted(KC._HOOKS, "drop_root", "largest", lambda: _refused(lambda: kcore_fixed_points(*args),
                                                                    "no-solution:dropped-root"))
    _planted(KT._HOOKS, "threshold_offset", F(1, 10 ** 6), lambda: _refused(
        lambda: _run("percolation_kcore"), "no-solution:route-disagreement"))


def test_at_and_near_the_jump():
    # t = 8/9 exactly, inside the certified t_c ball: ambiguous:critical-point (P1's convention); a
    # double root the ball cannot place at t_c stays undetermined:near-critical
    _refused(lambda: kcore_fixed_points(("bethe", 4), 2, 3, ("site", F(8, 9))), "ambiguous:critical-point")
    _refused(lambda: _run("percolation_bootstrap", q="1/9"), "ambiguous:critical-point")
    _refused(lambda: kcore_fixed_points(("bethe", 4), 2, 3, ("site", F(8, 9) + F(1, 10 ** 70))),
             "undetermined:near-critical")
    r = kcore_fixed_points(("bethe", 4), 2, 3, ("site", F(8, 9) + F(1, 10 ** 16)))
    assert len(r.roots) == 3 and r.core.digits >= 30          # the two nontrivial roots 1e-8 apart
    assert kcore_fixed_points(("bethe", 4), 2, 3, ("site", F(8, 9) - F(1, 10 ** 16))).phase == "no core"
    # the 3-regular graph is its own 3-core only at t = 1: the root x = 1 is marginal there
    _refused(lambda: kcore_fixed_points(("bethe", 3), 2, 3, ("site", 1)), "ambiguous:critical-point")
    assert kcore_fixed_points(("bethe", 4), 2, 3, ("site", 1)).core.contains(1)
    _refused(lambda: kcore_fixed_points(("bethe", 3), 2, 5, ("site", 1)), "precondition:core-order")
    _refused(lambda: kcore_fixed_points(("poisson", 4), 2, 1, ("site", 1)), "precondition:core-order")
    _refused(lambda: kcore_fixed_points(("poisson", 4), 13, 2, ("site", 1)), "precondition:hyperedge-size")


def test_catalog_entries_are_certified_with_per_value_finality():
    from flint import ctx
    from replicant.catalog import catalog
    before, prec = mp.mp.dps, ctx.prec
    res = _plain("percolation_kcore")
    assert res.rigor == "certified"
    e = catalog()["percolation_kcore"]
    phys = [k for k in res.values if k not in e.meta]
    assert phys and all(k in res.errors and mp.mpf(res.errors[k]) < mp.mpf("1e-25") for k in phys)
    fin = res.finality
    assert fin["c_k"].state == fin["core_fraction"].state == "exact-by-literature" and "0804.1656" in fin["p_c"].cite
    assert fin["core_at_p_c"].state == "not-computed" and fin["jacobian"].state == "stable"
    hyper = _plain("percolation_kcore", K=3, k=2, alpha=1).finality
    assert hyper["alpha_k"].state == "not-computed" and hyper["core_fraction"].state == "stable"
    boot = _plain("percolation_bootstrap")
    assert boot.finality["q_c"].state == "exact-by-literature" and "1509.08454" in boot.finality["q_c"].cite
    with mp.workdps(50):
        assert boot.values["closed_form"] and abs(mp.mpf(boot.values["q_c"]) - mp.mpf(1) / 9) < 1e-30
    assert mp.mp.dps == before and ctx.prec == prec
    _refused(lambda: _run("percolation_kcore", c=4, alpha=1), "precondition:mean-degree")
    _refused(lambda: _run("percolation_kcore", law="finite"), "precondition:degree-law")
    _refused(lambda: _run("percolation_bootstrap", ell=4), "precondition:bootstrap-threshold")
    _refused(lambda: _run("percolation_bootstrap", d=2), "precondition:degree")
    _refused(lambda: _run("percolation_bootstrap", q=2), "precondition:infection")
    _refused(lambda: _run("percolation_kcore", N=100), "precondition:known-parameter")


def test_second_routes_never_block_a_certified_run():
    # far outside the DGM grid's comfort (k = 15, 40; a 50-regular 26-core): computed or skipped with a
    # note, never a crash; within ~1e-13 of the jump the conditioning-scaled tolerance (or the 40-digit
    # rerun) keeps a correct certified value, and csp/xorsat.py's own audit refusal only skips it
    # (more points, k = 15 at c = 30, a 50-regular 26-core, alpha_d + 4e-17: tests/test_percolation_kcore_slow.py)
    for name, params in (("percolation_kcore", {"k": 15, "c": 4}),
                         ("percolation_kcore", {"c": "33509188715117/10000000000000"})):
        res = _plain(name, **params)
        notes = [n for n in res.notes if n.startswith(("DGM", "csp/"))]
        assert notes and all(": agrees" in n or ": SKIPPED" in n for n in notes), (params, notes)
    import replicant.catalog_percolation_kcore as C       # a route that raises is skipped, not fatal
    good = C.dgm_threshold

    def broken(*a, **k):
        raise ZeroDivisionError("planted")
    C.dgm_threshold = broken
    try:
        res = _run("percolation_kcore")
        assert any(n.startswith("DGM Eqs. (5)-(8)") and "SKIPPED" in n and "planted" in n for n in res.notes)
        assert res.finality["c_k"].state == "exact-by-literature"
    finally:
        C.dgm_threshold = good
        assert C.dgm_threshold is good
    near = _plain("percolation_kcore", c="33509188715117/10000000000000")
    with mp.workdps(30):                      # an 80-digit bisection (review): 0.53516138657116076
        assert abs(mp.mpf(near.values["x"]) - mp.mpf("0.53516138657116076")) < mp.mpf("1e-16")
    assert any("first appearance" in n for n in near.notes)


def test_edge_counts_are_checked_and_planted_edges_are_refused():
    from replicant.percolation.kcore import Model
    good = Model.edges

    def no_b(self, x):                        # the bond edge count without the factor t
        return self._q(self.z) / self.K * x ** self.K
    Model.edges = no_b
    try:
        _refused(lambda: _run("percolation_kcore", dilution="bond", p="9/10"), "no-solution:route-disagreement")
    finally:
        Model.edges = good
        assert Model.edges is good
    _plain("percolation_kcore", dilution="bond", p="9/10")   # the unplanted bond run must pass the edge check
    _planted(KT._HOOKS, "edges_exponent_shift", -1, lambda: _refused(
        lambda: _run("percolation_kcore"), "no-solution:route-disagreement"))
    v = _plain("percolation_kcore").values               # JL07 Thm 1.1: e/n -> mu^2 / (2 lambda) at the birth
    with mp.workdps(40):
        mu, ck = mp.mpf(v["mu_at_c_k"]), mp.mpf(v["c_k"])
        assert abs(mp.mpf(v["edges_at_c_k"]) - mu ** 2 / (2 * ck)) < mp.mpf("1e-35")
