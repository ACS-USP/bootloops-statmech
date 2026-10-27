"""I3 slow tier (run with selftest.py --full): larger identities and cylinders, all sectors, the
antiferromagnet's larger tori and its 2D entropy integral, the frustrated runs at beta = 3 and 5, a
grid of route agreements.  Helpers from tests/test_lattice_geom.py.
"""
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp

from replicant.testing import slow
from test_lattice_geom import _planted, _refusal, _run

@slow
def test_slow_larger_identities_and_cylinders():
    from replicant.lattice import geom_tori as G, tri_honey as T
    Ls = (F(1, 5), F(3, 10), F(1, 2))                 # on L x L the permutation symmetry hides
    zh, zt = _planted(T, "st_opposite", False, lambda: G.star_triangle_check(3, 3, Ls))
    assert zh.overlaps(zt)                            # a K on the wrong edge (cannot fail there)
    for Lx in (3, 4, 5, 6, 8):                        # one coupling, integer transfer matrices
        u = G.duality_identity_univariate(Lx, 3)
        assert u.exact and u.periodic_only_mismatch > 0, Lx
    for size in ((3, 4), (5, 3)):
        Ls = (F(1, 5), F(3, 10), F(1, 2))
        zh, zt = G.star_triangle_check(*size, Ls)
        assert zh.overlaps(zt), size
        zh, zt = _planted(T, "st_opposite", False, lambda s=size: G.star_triangle_check(*s, Ls))
        assert not zh.overlaps(zt), size
    zh, zt = G.star_triangle_check(4, 4, (F(2, 5), F(1, 5), F(7, 10)))       # 32 + 16 sites
    assert zh.overlaps(zt)
    bulk = T.honeycomb_free_energy("0.4").mid()
    offs = [abs(G.cylinder_free_energy("honeycomb", m, [F(2, 5)] * 3).mid() - bulk) for m in (2, 3, 4)]
    assert all(b < a for a, b in zip(offs, offs[1:])) and offs[-1] < 1e-3, offs
    v = _run("ising_honeycomb", L=4)
    assert v["torus_size"] == "4 x 4"


@slow
def test_slow_all_sectors_and_the_antiferromagnet_at_beta_3():
    from replicant.lattice import geom_tori as G, tri_honey as T
    for geom in ("honeycomb", "honeycomb_dual"):
        for sec in ((0, 0), (1, 0), (0, 1), (1, 1)):
            c = G.counts(geom, 3, 3, sec)
            assert G.univariate(c, 27) == G.kronecker(geom, 3, 3, sec), (geom, sec)
    f3, s0 = T.triangular_free_energy(3, -1, -1, -1), T.wannier_entropy()
    assert 0 <= f3.mid() - 3 - s0.mid() < 1e-4


@slow
def test_slow_frustrated_at_beta_3_certifies():
    v = _run("ising_triangular", beta=3, J1=-1, J2=2, J3=1)        # was refused at max |K| > 3
    assert "beta_c" not in v and "gauge" not in v and float(v["minus_beta_f"]) > 6


@slow
def test_slow_frustrated_soft_mode_is_refused_by_name():
    r = _refusal(lambda: _run("ising_triangular", beta=5, J1=-1, J2=-1, J3=-1))
    assert r.reason == "no-solution:frustrated-soft-mode", r


@slow
def test_slow_entropy_reduction_against_the_2d_integral():
    """ln 2 + (1/2) int d^2k/(4 pi^2) ln(9/4 - xi(k)) (Pierre et al.'s X_k(-1), 2D) by nested mpmath
    quadrature, split at the inner minimum and at k1 = +-2pi/3, against the certified 1D reduction."""
    from replicant.lattice import tri_honey as T
    with mp.workdps(20):
        def inner(k1):
            # the zeros are isolated points (k = +-(2pi/3)(1, 1)); a node landing on one within the
            # working precision is clamped (measure zero)
            g = lambda k2: mp.log(max(abs(mp.mpf(9) / 4 - mp.sin(k1 / 2) ** 2 - mp.sin(k2 / 2) ** 2
                                          - mp.sin((k1 + k2) / 2) ** 2), mp.mpf("1e-60")))
            # the inner minimum (a near-zero for k1 near +-2pi/3) sits at k2 = -k1/2 (mod pi)
            z = [-k1 / 2 + j * mp.pi for j in (-1, 0, 1)]
            pts = sorted({-mp.pi, mp.pi, *[x for x in z if -mp.pi < x < mp.pi]})
            return mp.quad(g, pts) / (2 * mp.pi)
        I = mp.quad(inner, [-mp.pi, -2 * mp.pi / 3, 0, 2 * mp.pi / 3, mp.pi]) / (2 * mp.pi)
        two_d = mp.log(2) + I / 2
    assert abs(two_d - T.wannier_entropy().mid()) < 1e-8, two_d
    assert abs(I + mp.mpf("0.74016")) < 5e-6                             # Pierre et al., Sec. 5.2


@slow
def test_slow_antiferromagnet_ground_states_on_larger_tori():
    from replicant.lattice import geom_tori as G
    for Lx, Ly in ((6, 6), (6, 3), (9, 3)):
        c = G.kronecker("triangular", Lx, Ly)
        j = max(i for i, m in enumerate(c) if m)
        assert 3 * Lx * Ly - j == Lx * Ly, (Lx, Ly)       # one frustrated bond per triangle
        assert c[j] > 0


@slow
def test_slow_grid_of_route_agreements():
    from replicant.lattice import anisotropic as A, tri_honey as T
    for beta in ("0.05", "0.1", "0.27", "0.28", "0.5", "0.66", "2", "40"):
        assert T.triangular_free_energy(beta).overlaps(T.triangular_free_energy_pierre(beta)), beta
        assert T.triangular_free_energy(beta, 1, 2, 3).overlaps(
            T.triangular_free_energy_square(beta, 1, 2, 3)), beta
        assert T.honeycomb_free_energy(beta, 1, 2, 3).overlaps(
            T.honeycomb_free_energy_duality(beta, 1, 2, 3)), beta
        assert T.honeycomb_free_energy(beta).overlaps(T.honeycomb_free_energy_pierre(beta)), beta
        assert A.free_energy(beta, 1, 3).overlaps(A.free_energy_baxter(beta, 1, 3)), beta
    v = _run("ising_triangular", beta="0.3", J1=1, J2=-1, J3=2)           # frustrated, mixed signs
    assert "beta_c" not in v and "minus_beta_f_square_route" not in v
    v = _run("ising_honeycomb", beta=40)                        # large beta: no overflow
    assert abs(mp.mpf(v["minus_beta_f"]) - 60) < 1e-30
    for b in ("0.05", "0.3", "0.66", "1"):
        _run("ising_honeycomb", beta=b)
        _run("ising_triangular", beta=b, J1=1, J2=2, J3=3)
        _run("ising2d_anisotropic", beta=b, J1=3, J2=1)


@slow
def test_slow_signed_tori_on_the_square_and_honeycomb():
    v = _run("ising2d_anisotropic", J2=-2, L=3)                  # odd L: not gauge-closed
    w = _run("ising2d_anisotropic", L=3)
    assert abs(mp.mpf(v["torus_minus_beta_f"]) - mp.mpf(w["torus_minus_beta_f"])) > 0.01
    assert _run("ising2d_anisotropic", J2=-2, L=4)["torus_minus_beta_f"] == \
        _run("ising2d_anisotropic", L=4)["torus_minus_beta_f"]
    v = _run("ising_honeycomb", J1=-1, J2=-1, J3=-1)             # one sublattice flip: closed
    assert v["torus_minus_beta_f"] == _run("ising_honeycomb")["torus_minus_beta_f"]
    v = _run("ising_honeycomb", J1=-1)                           # one class negative: not on 3 x 3
    assert v["torus_minus_beta_f"] != _run("ising_honeycomb")["torus_minus_beta_f"]
