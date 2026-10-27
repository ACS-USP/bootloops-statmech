"""P2, default tier: 2D lattice percolation.  Exact thresholds and their second routes and identities,
the exponents' scaling relations, Cardy's two routes and its rectangle convention (a tiny Monte Carlo),
the exhaustive enumeration and its duality identities, the exact cross-check of the Newman-Ziff sweep
against the enumeration, edge inputs, and a planted control for every check (each hook restored with
an identity assert).  The finite-size scaling is in tests/test_percolation_lattice_slow.py."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.catalog import catalog
from replicant.compose import compose
from replicant.core import Refusal
from replicant.percolation import cardy as CY
from replicant.percolation import enumerate2d as EN
from replicant.percolation import exact2d as X
from replicant.percolation import fss2d as F
from replicant.percolation import lattice2d as LT

SMALL = (("square", "bond", 3, 3), ("square", "site", 3, 3), ("triangular", "site", 3, 3),
         ("triangular", "bond", 3, 2), ("honeycomb", "bond", 4, 3), ("honeycomb", "site", 4, 3))


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


def _run(route, **p):
    return compose(catalog()["percolation_lattice"].spec, route).run(**p)


# ---- thresholds, identities, exponents ------------------------------------------------------
def test_thresholds_two_routes_and_duality():
    tri, sec, _ = X.threshold("triangular", "bond")
    hexb, hsec, _ = X.threshold("honeycomb", "bond")
    assert tri.overlaps(sec["p_c_cubic_root"]) and hexb.overlaps(hsec["p_c_cubic_root"])
    assert tri.rad() < mp.mpf("1e-60") and sec["p_c_cubic_root"].rad() < mp.mpf("1e-55")
    with mp.workdps(60):
        assert abs(tri.mid() + hexb.mid() - 1) < mp.mpf("1e-55")           # duality
        t = tri.mid()
        assert abs(t ** 3 - 3 * t + 1) < mp.mpf("1e-55")
    assert X.threshold("square", "site") is None and X.threshold("honeycomb", "site") is None
    # planted: a wrong closed form disagrees with the bisected root of the cubic
    saved = X.trig_ball
    X.trig_ball = lambda: X.ball_of(X._arb()("0.3473"))
    try:
        _refused(lambda: X.threshold("triangular", "bond"), "no-solution:route-disagreement")
    finally:
        X.trig_ball = saved
        assert X.trig_ball is saved


def test_star_triangle_polynomials_and_planted_star():
    rows = X.star_triangle_identity()
    assert sorted(n for n, _ in rows) == [0, 1, 1, 1, 3]
    _refused(lambda: _planted(X._HOOKS, "star_prob", "same", X.star_triangle_identity), "no-solution:star-triangle")
    # homogeneous kappa_tri(p, p, p) = 3p - p^3 - 1, so p^3 - 3p + 1 = 0 is the self-dual point
    assert all(X.kappa_tri_hom(Fraction(k, 7)) == -(Fraction(k, 7) ** 3 - 3 * Fraction(k, 7) + 1) for k in range(8))


def test_exponents_from_arm_exponents_and_planted_exponent():
    ex, rels = X.exponents()
    assert (ex["beta"], ex["nu"], ex["gamma"], ex["fractal_dimension"]) == (
        Fraction(5, 36), Fraction(4, 3), Fraction(43, 18), Fraction(91, 48)) and len(rels) == 5
    for name, val in (("nu", 1), ("gamma", Fraction(43, 17)), ("beta", Fraction(5, 32))):
        _refused(lambda: _planted(X._HOOKS, "exponent", (name, val), X.exponents), "no-solution:scaling-relation")


# ---- Cardy ---------------------------------------------------------------------------------------
def test_cardy_two_routes_symmetry_and_limits():
    for a in ("1/64", "1/3", "1", "3/2", "5", "64"):
        b, f = CY.cardy_ball(a), CY.cardy_float(a)
        with mp.workdps(40):
            assert abs(b.mid() - f) <= mp.mpf("1e-25") * max(f, mp.mpf("1e-30")), (a, b, f)
            assert abs(b.mid() + CY.cardy_ball(1 / Fraction(a)).mid() - 1) < mp.mpf("1e-60")
    assert CY.cardy_ball(1).contains(Fraction(1, 2))
    assert 0.17 < float(CY.cardy_ball(2).mid()) < 0.18          # a wide rectangle is rarely crossed
    for bad in (0, "-1", "1/65", 65, "x", None, True):
        _refused(lambda: CY.aspect_ratio(bad), "precondition:aspect-ratio")


def _tri_rect(aspect, H):
    W = max(2, round(float(aspect) * H * 3 ** 0.5 / 2))
    return W, H, W / (H * 3 ** 0.5 / 2)


def test_cardy_rectangle_convention_against_a_small_monte_carlo():
    """Triangular site at p = 1/2 (Smirnov's theorem): width/height 2 is crossed horizontally with
    probability ~0.18, height/width 2 with ~0.82; the planted inversion r = aspect swaps them."""
    rng = np.random.default_rng(7)
    for aspect in ("2", "1/2"):
        W, H, a = _tri_rect(Fraction(aspect), 8)
        stars = [LT.sweep("triangular", "site", W, H, rng, full=False)["n_star"] for _ in range(60)]
        mc = float(LT.binomial_weights(W * H, 0.5) @ LT.crossing_cdf(stars, W * H))
        se = (mc * (1 - mc) / 60) ** 0.5
        want = float(CY.cardy_ball(Fraction(a).limit_denominator(1000)).mid())
        assert abs(mc - want) < 4 * se + 0.08, (aspect, mc, want)     # 0.08: an O(1/L) allowance
        wrong = _planted(CY._HOOKS, "invert_aspect", True,
                         lambda: float(CY.cardy_ball(Fraction(a).limit_denominator(1000)).mid()))
        assert abs(mc - wrong) > 4 * se + 0.08, (aspect, mc, wrong)


# ---- enumeration, duality, the Newman-Ziff cross-check -----------------------------------------
def test_enumeration_duality_identities_and_planted_dual():
    for W, H in ((2, 1), (3, 2), (3, 3), (4, 2), (2, 4)):
        assert EN.duality_identity("square", "bond", W, H)[0], (W, H)
    for W, H in ((2, 2), (3, 3), (4, 3), (3, 4)):
        assert EN.duality_identity("triangular", "site", W, H)[0], (W, H)
    assert EN.crossing_probability(EN.crossing_counts("square", "bond", 3, 2), Fraction(1, 2)) == Fraction(1, 2)
    assert EN.crossing_probability(EN.crossing_counts("triangular", "site", 3, 3), Fraction(1, 2)) != Fraction(1, 2)
    assert not _planted(EN._HOOKS, "dual_shape", (0, -1), lambda: EN.duality_identity("square", "bond", 3, 3))[0]
    _refused(lambda: _planted(EN._HOOKS, "dual_shape", (0, -1), lambda: _run("exact")),
             "no-solution:duality-identity")
    _refused(lambda: EN.crossing_counts("triangular", "bond", 4, 4), "precondition:exact-size")
    # exact polynomials at the end points and on a 1-wide strip
    c = EN.crossing_counts("square", "site", 1, 3)
    assert EN.crossing_probability(c, 0) == 0 and EN.crossing_probability(c, 1) == 1
    assert EN.crossing_probability(c, Fraction(1, 2)) == Fraction(7, 8)


def _orders(lattice, dilution, W, H, n, seed):
    rng = np.random.default_rng(seed)
    u, v = LT.bonds(lattice, W, H)
    M = len(u) if dilution == "bond" else W * H
    orders = [rng.permutation(M) for _ in range(n)]
    stars = [LT.sweep(lattice, dilution, W, H, None, order=o)["n_star"] for o in orders]
    return orders, stars, (u, v)


def test_newman_ziff_matches_the_enumeration_exactly_and_planted_lattices_fail():
    for lattice, dilution, W, H in SMALL:
        EN.nz_cross_check(lattice, dilution, W, H, *_orders(lattice, dilution, W, H, 40, 3))
    for hook, val in (("neighbourhood", "diagonal"), ("wrap_crossing", True)):
        for lattice, dilution, W, H in SMALL:
            _refused(lambda: _planted(LT._HOOKS, hook, val, lambda: EN.nz_cross_check(
                lattice, dilution, W, H, *_orders(lattice, dilution, W, H, 40, 3))), "no-solution:nz-enumeration")


def test_sampled_crossing_matches_the_exact_polynomial_and_planted_bc_leaves_the_band():
    """Statistical (fixed named seeds): NZ's canonical R(p) against the exact polynomial."""
    for lattice, dilution, W, H, p in (("square", "bond", 3, 3, 0.5), ("triangular", "bond", 3, 2, 0.35),
                                       ("honeycomb", "site", 4, 3, 0.7)):
        exact = float(EN.crossing_probability(EN.crossing_counts(lattice, dilution, W, H), Fraction(p)))
        r = LT.sample_lattice(lattice, dilution, p, W, H, tuple(range(8)), 40)
        assert abs(r.mean["R"] - exact) <= r.halfwidth99["R"], (lattice, r.mean["R"], exact)
        assert r.halfwidth99["R"] < 0.08
    exact = float(EN.crossing_probability(EN.crossing_counts("square", "site", 4, 3), Fraction(3, 5)))
    bad = _planted(LT._HOOKS, "wrap_crossing", True,
                   lambda: LT.sample_lattice("square", "site", 0.6, 4, 3, tuple(range(8)), 40))
    assert abs(bad.mean["R"] - exact) > bad.halfwidth99["R"], (bad.mean["R"], exact)


def test_sampler_invariants_refuse_a_disconnected_lattice():
    saved = LT.bonds
    LT.bonds = lambda lattice, W, H: tuple(a[: W - 1] for a in saved(lattice, W, H))   # one row only
    try:
        _refused(lambda: LT.sweep("square", "bond", 3, 3, np.random.default_rng(0)), "no-solution:sampler-invariant")
    finally:
        LT.bonds = saved
        assert LT.bonds is saved


# ---- the catalog entry: edge inputs -----------------------------------------------------------
def test_entry_edge_inputs_give_values_or_named_refusals():
    r = _run("exact", lattice="square", dilution="bond", Lx=3, Ly=2)
    assert r.values["crossing_exact"].startswith("0.5000") and r.finality["crossing_exact"].final
    assert "p_c" not in _run("exact", lattice="square", dilution="site").values
    assert float(_run("exact", p=0).values["crossing_exact"]) == 0
    assert float(_run("exact", p=1).values["crossing_exact"]) == 1
    assert float(_run("exact", Lx=1, Ly=1).values["crossing_exact"]) == 1
    s = _run("sample", Lx=4, Ly=2, sweeps=4, p="0")
    assert s.values["crossing_probability_sampled"] == 0 and s.rigor == "statistical" and not s.final
    assert _run("sample", Lx=1, Ly=3, sweeps=2, dilution="site", p=1).values["crossing_probability_sampled"] == 1
    for route in ("exact", "sample"):
        for bad, reason in (({"lattice": "kagome"}, "precondition:lattice"), ({"lattice": 3}, "precondition:lattice"),
                            ({"dilution": "mixed"}, "precondition:dilution"), ({"p": "3/2"}, "precondition:occupation"),
                            ({"p": "x"}, "precondition:occupation"), ({"Lx": 0}, "precondition:lattice-size"),
                            ({"Lx": 2.5}, "precondition:lattice-size")):
            _refused(lambda: _run(route, **bad), reason)
    _refused(lambda: _run("sample", Ls=(4, 8)), "precondition:fss-sizes")
    _refused(lambda: _run("sample", seeds=[1]), "precondition:named-seeds")
    _refused(lambda: _run("sample", sweeps=0), "precondition:sweeps")
    _refused(lambda: _run("exact", aspect="0"), "precondition:aspect-ratio")


def test_finality_labels_follow_the_lattice():
    tri = _run("exact", lattice="triangular", dilution="site")
    assert tri.finality["nu"].state == tri.finality["cardy_crossing_h"].state == "exact-by-literature"
    assert tri.finality["fractal_dimension"].state == "not-computed" and not tri.final
    sq = _run("exact", lattice="square", dilution="bond")
    assert sq.finality["p_c"].state == "exact-by-literature" and sq.finality["nu"].state == "not-computed"
    assert sq.finality["cardy_crossing_h"].state == "not-computed" and not sq.final


# ---- FSS machinery on synthetic curves (no sweeps) ----------------------------------------------
def test_fss_crossing_level_and_contrast_on_synthetic_curves():
    grid = np.linspace(0.4, 0.6, 201)
    a, b = 1 / (1 + np.exp(-(grid - 0.5) * 40)), 1 / (1 + np.exp(-(grid - 0.5) * 80))
    assert abs(F._cross(grid, a, b) - 0.5) < 1e-3 and abs(F._level(grid, a, 0.5) - 0.5) < 1e-3
    assert abs(F._golden(lambda y: (y - 0.75) ** 2, 0.3, 1.5) - 0.75) < 1e-3
    c = F.contrast((0.59, 0.585, 0.595, 0.001), 0.59274621, 0.59274605)
    assert c["a_inside"] and c["b_inside"] and not c["resolves"]
    assert F.contrast((0.75, 0.7, 0.8), 0.75, 1.0)["resolves"]
    _refused(lambda: F.fss("square", "bond", (8, 4, 16), (0, 1), 2), "precondition:fss-sizes")
