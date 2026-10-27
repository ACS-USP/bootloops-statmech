"""I3: the anisotropic square lattice, the triangular and honeycomb lattices, the star-triangle and
duality relations (lattice/anisotropic.py, lattice/tri_honey.py, lattice/geom_tori.py,
catalog_lattice_aniso.py, catalog_lattice_geom.py).

Routes.  Free energies: the Kac-Ward integral (Athanasopoulos & Ueltschi) against Baxter's
diagonal-form square integral (11.7.15), his three-square-lattice form of the triangular (11.7.13),
Pierre et al.'s isotropic integrals, and Onsager (lattice/onsager.py) at J1 = J2; the honeycomb by
Baxter's star-triangle (6.4.7) and duality (6.3.7) maps.  Finite lattices: enumeration of all
configurations, the transfer matrix in balls and in exact integers (three algorithms sharing only
the unit cell), and the existing square-torus polynomial (lattice/transfer.py, independent code);
cylinders towards the bulk (consistency, no bound).  Exact identities: star-triangle on matched tori,
four-sector duality as integer polynomials.  Planted errors go through each module's _HOOKS
(restored in try/finally, identity asserted).
"""
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp

from replicant.core import Refusal
from replicant.refdata import load
from replicant.testing import slow

WRONG_DUAL = (-1, 1, 1, 2)          # e^{-2K*} = (1 - y)/(2 + y): a wrong duality map


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _bank(key):
    return load("lattice")[key]


def _near(ball, key):
    rec = _bank(key)
    with mp.workdps(45):
        return abs(ball.mid() - mp.mpf(rec.value)) <= mp.mpf(rec.tol)


def _run(name, **kw):
    from replicant.catalog import catalog
    return catalog()[name].run(**kw)


def _cached(name, **kw):
    """The process-wide memoised run (shared with the catalog's banked checks at these params)."""
    from replicant.catalog import catalog, run_cached
    return run_cached(catalog()[name], "exact", kw)[0]


def _arbs(*ks):
    from replicant.lattice.ball import arb_q
    return [arb_q(F(k)) for k in ks]


# ---- the anisotropic square lattice ------------------------------------------------------------
def test_anisotropic_free_energy_two_routes_and_the_isotropic_limit():
    from replicant.lattice import anisotropic as A
    from replicant.lattice.onsager import onsager_free_energy
    for beta, J1, J2 in (("0.2", 1, 2), ("0.5", 1, 3), ("0.1", 2, 1), ("0.7", "0.25", 1)):
        a, b = A.free_energy(beta, J1, J2), A.free_energy_baxter(beta, J1, J2)
        assert a.overlaps(b) and a.rad() < 1e-39 and b.rad() < 1e-39, (beta, J1, J2)
        assert a.overlaps(A.free_energy(beta, J2, J1))                    # K1 <-> K2
    for beta in ("0.3", "0.6"):
        assert A.free_energy(beta, 1, 1).overlaps(onsager_free_energy(beta))
    assert _near(A.free_energy("0.2", 1, 2), "aniso_square_minus_beta_f_beta0_2_j1_1_j2_2")
    assert _refusal(lambda: A.free_energy("0.2", 1, -1)).reason == "precondition:J2"


def test_anisotropic_critical_line_and_duality():
    from replicant.lattice import anisotropic as A
    from replicant.lattice.onsager import critical_beta
    bc = A.critical_beta(1, 2)
    assert _near(bc, "aniso_square_beta_c_j1_1_j2_2") and bc.rad() < 1e-44
    assert A.critical_beta(1, 1).overlaps(critical_beta())
    k2 = A.critical_coupling("0.3")
    with mp.workdps(45):
        assert abs(mp.sinh(mp.mpf("0.6")) * mp.sinh(2 * k2.mid()) - 1) < mp.mpf("1e-40")
    # (K1, K2) and its dual satisfy the free-energy duality; the dual of the dual is the identity
    for beta, J1, J2 in (("0.2", 1, 2), ("0.9", 1, "0.5")):
        lhs, rhs = A.duality_relation(beta, J1, J2)
        assert lhs.overlaps(rhs), (beta, J1, J2)
    from replicant.lattice.ball import certify
    back = certify(lambda: A.dual_coupling(A.dual_coupling(_arbs("0.37")[0])))
    assert back.contains(F(37, 100))
    # sinh 2K1 sinh 2K2 on either side of the line, and its phase label
    assert A.criticality("0.2", 1, 2).mid() < 1 < A.criticality("0.4", 1, 2).mid()


def test_planted_cross_coupling_is_seen_only_off_the_isotropic_point():
    from replicant.lattice import anisotropic as A
    a = A.free_energy_baxter("0.2", 1, 2)
    wrong = _planted(A, "cross_coupling", True, lambda: A.free_energy("0.2", 1, 2))
    assert not wrong.overlaps(a) and abs(wrong.mid() - a.mid()) > 1e-3
    iso = _planted(A, "cross_coupling", True, lambda: A.free_energy("0.3", 1, 1))
    assert iso.overlaps(A.free_energy("0.3", 1, 1))
    r = _planted(A, "cross_coupling", True, lambda: _refusal(lambda: _run("ising2d_anisotropic")))
    assert r.reason == "no-solution:route-disagreement", r


def test_planted_wrong_duality_map_is_refused_everywhere():
    """One plant (e^{-2K*} = (1 - y)/(2 + y)) through anisotropic._HOOKS: the square's critical line
    is no longer fixed, the triangular K_c and honeycomb L_c are no longer fixed points of star-
    triangle o duality, the honeycomb's duality route disagrees with its star-triangle route, and
    the four-sector polynomial identity fails."""
    from replicant.lattice import anisotropic as A, geom_tori as G, tri_honey as T

    def body():
        assert not G.duality_identity(3, 3).exact
        lhs, rhs = A.duality_relation("0.2", 1, 2)
        assert not lhs.overlaps(rhs)
        assert not T.honeycomb_free_energy_duality("0.4").overlaps(T.honeycomb_free_energy("0.4"))
        return [_refusal(lambda n=n: _run(n)).reason for n in
                ("ising2d_anisotropic", "ising_triangular", "ising_honeycomb")]
    reasons = _planted(A, "dual_mobius", WRONG_DUAL, body)
    assert reasons == ["no-solution:duality-fixed-point"] * 3, reasons
    assert G.duality_identity(3, 3).exact                                  # restored


def test_anisotropic_entry_at_large_beta_and_defaults():
    v = _cached("ising2d_anisotropic")
    assert v["phase"] == "T > T_c" and "torus_high_t_identity" in v
    assert float(v["cylinder_sequence"]["offset"]) < 1e-3
    v = _run("ising2d_anisotropic", beta=40, J1=1, J2=2)        # no overflow; ground state 3 beta
    assert abs(mp.mpf(v["minus_beta_f"]) - 120) < 1e-30 and v["phase"] == "T < T_c"
    assert _refusal(lambda: _run("ising2d_anisotropic", L=7)).reason == "precondition:L"


# ---- triangular and honeycomb: the bulk ---------------------------------------------------------
def test_triangular_free_energy_three_routes():
    from replicant.lattice import anisotropic as A, tri_honey as T
    for beta, Js in (("0.6", (1, "0.5", 2)),):          # 0.2 isotropic: the catalog run; more: slow
        f = T.triangular_free_energy(beta, *Js)
        assert f.overlaps(T.triangular_free_energy_square(beta, *Js)), (beta, Js)
        if Js == (1, 1, 1):
            assert f.overlaps(T.triangular_free_energy_pierre(beta))
        assert f.overlaps(T.triangular_free_energy(beta, Js[2], Js[0], Js[1]))   # symmetric
    # J3 -> 0 is the anisotropic square lattice: the same integral, so compare with Baxter's
    assert T.triangular_free_energy("0.3", 1, 2, "1e-60").overlaps(A.free_energy_baxter("0.3", 1, 2))
    assert _near(T.triangular_free_energy("0.2"), "triangular_minus_beta_f_beta0_2")


def test_honeycomb_free_energy_three_routes():
    from replicant.lattice import tri_honey as T
    for beta, Js in (("0.3", (1, 2, 3)),):           # 0.4: the catalog run; 0.66: the slow grid
        f = T.honeycomb_free_energy(beta, *Js)
        assert f.overlaps(T.honeycomb_free_energy_duality(beta, *Js)), (beta, Js)
        if Js == (1, 1, 1):
            assert f.overlaps(T.honeycomb_free_energy_pierre(beta))
    assert _near(T.honeycomb_free_energy("0.4"), "honeycomb_minus_beta_f_beta0_4")


def test_planted_triangular_weight_is_refused():
    """The S1 S2 S3 term of the triangular integrand dropped (a wrong lattice weight): it disagrees with Baxter's
    three square lattices and with Pierre et al.; the catalog refuses route-disagreement."""
    from replicant.lattice import anisotropic as A, tri_honey as T

    def body():
        f = T.triangular_free_energy("0.2")
        assert not f.overlaps(T.triangular_free_energy_pierre("0.2"))
        assert not f.overlaps(T.triangular_free_energy_square("0.2"))
        return _refusal(lambda: _run("ising_triangular")).reason
    assert _planted(A, "triple_sign", 0, body) == "no-solution:route-disagreement"


def test_critical_points_closed_forms_and_anisotropic_roots():
    from replicant.lattice import tri_honey as T
    from replicant.lattice.ball import arb_q, baller
    assert _near(T.triangular_critical_beta(), "triangular_beta_c")
    assert _near(T.honeycomb_critical_beta(), "honeycomb_beta_c")
    assert T.triangular_critical_beta(2, 2, 2).overlaps(_half(T.triangular_critical_beta()))
    bc = T.triangular_critical_beta(1, 2, 3)
    hc = T.honeycomb_critical_beta(1, 2, 3)
    with baller().hygiene.ctx_guard(prec=400):
        K = [bc.arb * j for j in (1, 2, 3)]
        S = [(2 * k).sinh() for k in K]
        m = (2 * K[0]).cosh() * (2 * K[1]).cosh() * (2 * K[2]).cosh() + S[0] * S[1] * S[2] - sum(S)
        assert m.contains(0)                         # the Kac-Ward integrand's minimum vanishes
        assert T._kappa(*K).contains(1)
        L = [hc.arb * j for j in (1, 2, 3)]
        assert T._l(*L).contains(1) and T._kappa(*T.star_triangle(*L)[:3]).contains(1)
        assert all(a.overlaps(b) for a, b in zip(T.honeycomb_self_dual(L), L))
        assert all(a.overlaps(b) for a, b in zip(T.triangular_self_dual(K), K))
        # off the critical point the minimum is strictly positive (the check can fail)
        K2 = [(bc.arb + arb_q(F(1, 10 ** 12))) * j for j in (1, 2, 3)]
        S2 = [(2 * k).sinh() for k in K2]
        m2 = (2 * K2[0]).cosh() * (2 * K2[1]).cosh() * (2 * K2[2]).cosh() + S2[0] * S2[1] * S2[2] \
            - sum(S2)
        assert m2 > 0
    # Baxter (6.5.2): star-triangle(dual(K)) has sinh 2K* = k sinh 2K
    from replicant.lattice.ball import certify
    for i in range(3):
        a = certify(lambda i=i: T.triangular_self_dual(_arbs("0.2", "0.35", "0.5"))[i])
        b = certify(lambda i=i: (T._kappa(*_arbs("0.2", "0.35", "0.5"))
                                 * (2 * _arbs(("0.2", "0.35", "0.5")[i])[0]).sinh()).asinh() / 2)
        assert a.overlaps(b), i
    assert _refusal(lambda: T.honeycomb_critical_beta(1, -1, 1)).reason == "precondition:J2"


def _half(b):
    from replicant.lattice.ball import certify
    return certify(lambda: b.arb / 2)


# ---- finite lattices ----------------------------------------------------------------------------
def test_three_finite_lattice_algorithms_agree():
    from replicant.lattice import geom_tori as G, transfer
    from replicant.lattice.ball import certify
    for Lx, Ly in ((3, 4), (4, 4), (4, 3)):
        assert G.univariate(G.counts("square", Lx, Ly), 2 * Lx * Ly) == \
            transfer.torus_partition_polynomial(Lx, Ly)                   # independent code
    for geom, secs in (("triangular", ((0, 0), (1, 0), (0, 1), (1, 1))), ("honeycomb", ((1, 1),)),
                       ("honeycomb_dual", ((0, 0),))):
        E = G.class_sizes(geom, 3, 3)
        for sec in secs:
            c = G.counts(geom, 3, 3, sec)
            assert sum(c.values()) == 2 ** (G.GEOM[geom][0] * 9)
            if geom == "triangular":           # the honeycomb's integer rows: slow tier
                assert G.univariate(c, sum(E)) == G.kronecker(geom, 3, 3, sec), (geom, sec)
            ks = ("0.2", "0.3", "-0.5")
            a = certify(lambda: G.transfer_z(geom, 3, 3, _arbs(*ks), sec))
            b = certify(lambda: G.z_of_counts(c, E, _arbs(*ks)))
            assert a.overlaps(b), (geom, sec)
    # the catalog's 4 x 4 tori (the run path uses the transfer matrix only)
    for geom, ks in (("triangular", ("0.2", "-0.3", "0.25")),):        # the square 4 x 4: above
        E, c = G.class_sizes(geom, 4, 4), G.counts(geom, 4, 4)
        a = certify(lambda: G.transfer_z(geom, 4, 4, _arbs(*ks)))
        assert a.overlaps(certify(lambda: G.z_of_counts(c, E, _arbs(*ks)))), geom


def test_star_triangle_is_an_exact_identity_on_matched_tori():
    """Z_H(L1, L2, L3) = R^N Z_T(h = K3, v = K1, d = K2) on matched tori, anisotropic couplings;
    planted: K on an adjacent edge (st_opposite), a wrong edge weight (edge_scale).  On an L x L
    torus every permutation of the three directions is a symmetry, so st_opposite passes there (a
    check that cannot fail); 4 x 3 and 3 x 4 break it."""
    from replicant.lattice import geom_tori as G, tri_honey as T
    Ls = (F(1, 5), F(3, 10), F(1, 2))
    zh, zt = G.star_triangle_check(4, 3, Ls)
    assert zh.overlaps(zt) and zh.rad() < 1e-30 * zh.mid()
    zh, zt = _planted(G, "edge_scale", (2, 1, 1), lambda: G.star_triangle_check(4, 3, Ls))
    assert not zh.overlaps(zt)
    zh, zt = _planted(T, "st_opposite", False, lambda: G.star_triangle_check(4, 3, Ls))
    assert not zh.overlaps(zt) and abs(zh.mid() / zt.mid() - 1) > 1e-3
    # the bulk free energy is blind to st_opposite (the triangular f is symmetric in K1, K2, K3);
    # the anisotropic critical point is not (dual o star-triangle no longer fixes L_c), and the
    # catalog's 4 x 3 identity step sees it when the fixed-point step is bypassed
    f = _planted(T, "st_opposite", False, lambda: T.honeycomb_free_energy("0.3", 1, 2, 3))
    assert f.overlaps(T.honeycomb_free_energy("0.3", 1, 2, 3))
    r = _planted(T, "st_opposite", False,
                 lambda: _refusal(lambda: _run("ising_honeycomb", J1=1, J2=2, J3=3)))
    assert r.reason == "no-solution:duality-fixed-point", r
    from replicant import catalog_lattice_geom as CG
    r = _planted(T, "st_opposite", False, lambda: _refusal(
        lambda: CG._hc_st({"beta": "0.3", "J1": 1, "J2": 2, "J3": 3})))
    assert r.reason == "no-solution:star-triangle-identity", r


def test_duality_is_an_exact_polynomial_identity_with_four_sectors():
    from replicant.lattice import geom_tori as G
    d = G.duality_identity(3, 3)                     # three couplings: 76 nonzero monomials
    assert d.exact and d.terms == 76 and d.periodic_only_mismatch > 0


def test_planted_edge_weight_is_seen_by_the_torus_identities():
    """A wrong Boltzmann weight on one edge class, shared by every finite-lattice route (so the
    enumeration = transfer check cannot see it): the torus = bulk identity at beta = 1e-8 (square,
    triangular, honeycomb; a 2 % and a 0.1 % weight), the star-triangle identity (honeycomb); the
    cylinders alone see the 2 % weight only when the identity is bypassed.  Steps called directly."""
    from replicant import catalog_lattice_aniso as CA, catalog_lattice_geom as CG
    from replicant.lattice import geom_tori as G
    for w, steps in (((F(51, 50), 1, 1), (CA._tor, CG._hc_torus)),
                     ((1, F(1001, 1000), 1), (CG._tri_torus,))):
        for step in steps:
            r = _planted(G, "edge_scale", w, lambda f=step: _refusal(lambda: f({})))
            assert r.reason == "no-solution:torus-weight", (w, step, r)
    r = _planted(G, "edge_scale", (F(51, 50), 1, 1), lambda: _refusal(lambda: CG._hc_st({})))
    assert r.reason == "no-solution:star-triangle-identity", r
    # unplanted: the identity holds far inside its tolerance (finite size first enters at K^L)
    for step in (CA._tor, CG._tri_torus, CG._hc_torus):
        ht = step({}).values["torus_high_t_identity"]
        assert float(ht["difference"]) < 1e-3 * float(ht["tolerance"]), ht
    # the 2 % weight with the identity bypassed: the cylinder band is all that is left
    from replicant.lattice import tri_honey as T
    bulk = T.triangular_free_energy("0.2")
    r = _planted(G, "edge_scale", (2, 1, 1), lambda: _refusal(
        lambda: CG.cylinder_sequence("triangular", (3, 4, 5), [F(1, 5)] * 3, bulk)))
    assert r.reason == "no-solution:torus-limit", r


def test_cylinders_at_low_temperature():
    from replicant.lattice import geom_tori as G
    low = G.cylinder_free_energy("triangular", 4, [F(40)] * 3)       # T -> 0: exact to 1e-40
    assert abs(low.mid() - 120) < 1e-9 and low.rad() < 1e-9     # enclosed to 1e-12 relative


@slow
def test_slow_cylinders_approach_the_bulk():
    from replicant.lattice import geom_tori as G, tri_honey as T
    bulk = T.triangular_free_energy("0.2").mid()
    offs = [abs(G.cylinder_free_energy("triangular", m, [F(1, 5)] * 3).mid() - bulk)
            for m in (3, 4, 5, 6)]
    assert all(b < a for a, b in zip(offs, offs[1:])) and offs[-1] < 1e-3, offs


# ---- the catalog entries ------------------------------------------------------------------------
def test_triangular_and_honeycomb_entries():
    before = mp.mp.dps
    v = _cached("ising_triangular")
    assert v["phase"] == "T > T_c" and v["duality_identity"]["exact"]
    assert {"minus_beta_f_square_route", "minus_beta_f_pierre"} <= set(v)
    v = _run("ising_triangular", beta=40)
    assert abs(mp.mpf(v["minus_beta_f"]) - 120) < 1e-30 and v["phase"] == "T < T_c"
    v = _cached("ising_honeycomb")
    assert v["star_triangle_identity"]["overlap"] and v["phase"] == "T > T_c"
    assert _refusal(lambda: _run("ising_honeycomb", J1=0)).reason == "precondition:J1"
    assert _refusal(lambda: _run("ising_triangular", J2=0)).reason == "precondition:J2"
    assert mp.mp.dps == before


def test_triangular_antiferromagnet_ground_state_entropy():
    from replicant.lattice import tri_honey as T
    s0 = T.wannier_entropy()
    assert _near(s0, "triangular_af_ground_state_entropy") and s0.rad() < 1e-40
    f1 = T.triangular_free_energy(1, -1, -1, -1)                  # -beta f - beta |e0| -> s0
    assert 0 <= f1.mid() - 1 - s0.mid() < 0.03                     # 0.0218 (beta = 3: slow tier)
    v = _cached("ising_triangular", J1=-1, J2=-1, J3=-1)
    assert "beta_c" not in v and v["ground_state_counts"]["3 x 3"]["frustrated_bonds"] == 9
    assert "minus_beta_f_pierre" in v                  # the second route at v < 0 (Sec. 5.2)
    for key in ("triangular_af_ground_state_entropy_zukovic",
                "triangular_af_ground_state_entropy_pierre_integral"):
        assert _near(s0, key), key


def test_gauge_equivalent_and_frustrated_sign_patterns():
    """J1 J2 J3 > 0 is gauge-equivalent to |J| (its values, critical point and routes); J1 J2 J3 < 0 is
    frustrated and computed with its signs, refused only by the integrator's own stall."""
    v = _run("ising_triangular", beta=5, J1=-1, J2=-1, J3=1)
    w = _run("ising_triangular", beta=5)
    assert v["gauge"].startswith("J = ") and v["minus_beta_f"] == w["minus_beta_f"]
    assert v["beta_c"] == w["beta_c"] and "minus_beta_f_square_route" in v
    v = _run("ising_triangular", beta="0.5", J1=-1, J2=2, J3=1)          # frustrated (beta = 3: slow)
    assert "beta_c" not in v and "gauge" not in v and "minus_beta_f" in v
    v = _run("ising_honeycomb", J1=-1)                                   # bipartite: |J|
    assert v["gauge"] and v["minus_beta_f"] == _cached("ising_honeycomb")["minus_beta_f"]
    v = _run("ising2d_anisotropic", J2=-2)
    assert v["gauge"] and v["minus_beta_f"] == _cached("ising2d_anisotropic")["minus_beta_f"]


def test_planted_swapped_dual_couplings_are_refused():
    from replicant.lattice import anisotropic as A
    r = _planted(A, "dual_swapped", True, lambda: _refusal(lambda: _run("ising2d_anisotropic")))
    assert r.reason == "no-solution:duality-fixed-point", r


def test_memo_is_keyed_on_the_hooks_and_can_be_cleared():
    from replicant.lattice import anisotropic as A
    a = A.free_energy("0.2", 1, 2)
    assert A.free_energy("0.2", 1, 2) is a                               # memoised
    b = _planted(A, "cross_coupling", True, lambda: A.free_energy("0.2", 1, 2))
    assert b is not a and not b.overlaps(a)                              # a plant is a new key
    A.clear_memo()
    c = A.free_energy("0.2", 1, 2)
    assert c is not a and c.overlaps(a)
