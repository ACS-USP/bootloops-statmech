"""Triangular and honeycomb Ising models in zero field: critical points, free energies, the
star-triangle and duality maps, and the triangular antiferromagnet's ground-state entropy,
certified in ball arithmetic.  Couplings K_j = beta J_j (j = 1, 2, 3, one per edge direction).

Sources read.
  Baxter, Exactly Solved Models (1982), ANU PDF, page images of pp. 78-87 and 294-297:
    (6.3.6)-(6.3.8) duality: tanh K_j = e^{-2L_j} gives Z^H_2N{L} = (2 s1 s2 s3)^{N/2} Z^T_N{K},
        s_j = sinh 2L_j = 1/sinh 2K_j (honeycomb of 2N sites, triangular of N);
    (6.4.4)-(6.4.10), (6.4.14) star-triangle: 2 cosh(L1 s_i + L2 s_j + L3 s_k) = R exp(K1 s_j s_k +
        K2 s_k s_i + K3 s_i s_j) (K_j on the triangle edge opposite the star leg L_j), e^{4K1} = c c1
        /(c2 c3), c = cosh(L1 + L2 + L3), c_i = cosh(-L_i + L_j + L_k); R^2 = 2k sinh 2L1 sinh 2L2
        sinh 2L3; (6.4.7) Z^H_2N{L} = R^N Z^T_N{K};
    (6.4.13), (6.4.16) sinh 2K_j sinh 2L_j = k^-1 (the same for j = 1, 2, 3), k = (1 - v1^2)(1 -
        v2^2)(1 - v3^2) / (4 [(1 + v1 v2 v3)(v1 + v2 v3)(v2 + v3 v1)(v3 + v1 v2)]^(1/2)), v = tanh K;
    (6.5.1), (6.5.2), (6.5.5)-(6.5.7) triangular self-duality sinh 2K*_j = k sinh 2K_j; criticality
        k = 1; isotropic sinh 2K_c = 3^(-1/2) (K_c = 0.27465307...), sinh 2L_c = 3^(1/2) (L_c =
        0.6584789...);
    (11.7.6) the honeycomb's l^2 = 16 (1 + z1 z2 z3)(z1 + z2 z3)(z2 + z3 z1)(z3 + z1 z2) / prod (1 -
        z_j^2)^2, z = e^{-2L} (l = the k of its star-triangle partner; critical at l = 1);
    (11.7.13) psi_T(K') = (1/2)[ln R + sum_j psi_SQ(K_j, K'_j)] with psi = beta f (so -beta f_T =
        (1/2)[-ln R + sum_j (-beta f_SQ(L_j, K_j))], L the honeycomb partner); (11.7.15) psi_SQ.
  Athanasopoulos & Ueltschi, arXiv:2407.05303v2 (read), Theorem 2.1(b): the triangular free energy
    (lattice/anisotropic.py), any signs of J_j.
  Pierre, Bernu & Messio, SciPost Phys. 19, 025 (2025), arXiv:2501.18569v2 (PDF read), Eqs. (16),
    (17), (19), (20), Tables 1, 2, 5, Sec. 5.2, isotropic coupling, v = tanh K:
    -beta m f = m ln 2 - (n_l/2) ln(1 - v^2) + n_v ln(1 + v) + (1/2) int_BZ d^2k/(4 pi^2) ln X_k,
    X_k = a0^2 + a1 xi(k), xi = sin^2(k1/2) + sin^2(k2/2) + sin^2((k1 + k2)/2):
      triangular (A333333): m = 1, n_l = 3, n_v = 1, a0 = 1 - 4v + v^2, a1 = 4v(1 - v)^2;
      honeycomb (A666): m = 2, n_l = 3, n_v = 0, a0 = 1 - 3v^2, a1 = 4v^2(1 - v^2);
    v_c = 2 - sqrt 3 and 1/sqrt 3 (Table 1); the antiferromagnet's X_k(-1) = 2^4 (9/4 - xi(k)) and
    s0 = 0.3231 (Table 5, Wannier, Phys. Rev. 79, 357 (1950), not fetched).
Derived here.  The inner angle of Pierre et al.'s integral in closed form (xi = (3 - cos k1)/2 -
|cos(k1/2)| cos(k2 + phi)): with s = sin(k1/2), P = a0^2 + a1 (1 + s^2), P^2 - Q^2 = a0^4 + 2 a0^2 a1
(1 + s^2) + a1^2 (3 s^2 + s^4),  (1/2) int d^2k/(4pi^2) ln X = (1/(2pi)) int_0^pi ln[(P + sqrt(P^2 -
Q^2))/2] dk1.  The antiferromagnet's s0 = -beta f + beta e0 at K -> -oo (e0 = J per site) reduces
to ln 2 + (1/(2pi)) int_0^pi 2 ln max(|cos(k/2)|, 1/2) dk = (2/pi) int_0^{pi/3} ln(2 cos w) dw.

  triangular_free_energy(beta, J1, J2, J3) -> Ball   Kac-Ward (any nonzero signs)
  triangular_free_energy_square(beta, J1, J2, J3)    Baxter (11.7.13): three square lattices
  triangular_free_energy_pierre(beta, J)             Pierre et al., isotropic, J > 0
  honeycomb_free_energy(beta, J1, J2, J3)            star-triangle (6.4.7) from the triangular
  honeycomb_free_energy_duality(beta, J1, J2, J3)    duality (6.3.7) from the triangular
  honeycomb_free_energy_pierre(beta, J)              Pierre et al., isotropic
  triangular_critical_beta(J1, J2, J3), honeycomb_critical_beta(J1, J2, J3) -> Ball
      J's equal: Baxter's closed forms (1/2) asinh(3^-1/2) / J, (1/2) asinh(3^1/2) / J; otherwise
      the certified root of k = 1 (triangular, (6.4.16): k decreases in beta) or l = 1 (honeycomb,
      (11.7.6): l decreases in beta)
  triangular_kappa(beta, J...), honeycomb_l(beta, J...) -> Ball   k, l (> 1 above T_c)
  star_triangle(L1, L2, L3) -> (K1, K2, K3, R)  arbs;  triangle_to_star(K1, K2, K3) -> (L, R, k)
  triangular_self_dual(K) -> K*: the star-triangle image of the dual honeycomb (Baxter Sec. 6.5)
  honeycomb_self_dual(L) -> L*: the dual of the star-triangle triangle
  wannier_entropy() -> Ball   s0 of the triangular antiferromagnet
_HOOKS: "st_opposite" (True: K_j on the edge opposite L_j; planted False puts it on an adjacent
edge, a wrong lattice weight of the decimated star).  The duality map and the triangular weight
are planted through lattice/anisotropic._HOOKS ("dual_mobius", "triple_sign").
"""
from ..core import Refusal
from . import anisotropic as A
from .ball import Ball, arb_q, certify

_HOOKS = {"st_opposite": True}


def _ks(beta, Js, positive=True):
    if len(Js) != 3:
        raise Refusal("precondition:J", "need three couplings J1, J2, J3")
    return A.couplings(beta, Js, positive)


def _arbs(ks):
    return [arb_q(k) for k in ks]


# ---- the maps --------------------------------------------------------------------------------
def star_triangle(L1, L2, L3):
    """(K1, K2, K3, R) of Baxter (6.4.8)-(6.4.10), (6.4.14): e^{4K_i} = c c_i / (c_j c_k)."""
    c = (L1 + L2 + L3).cosh()
    ci = [(-L1 + L2 + L3).cosh(), (L1 - L2 + L3).cosh(), (L1 + L2 - L3).cosh()]
    K = [(c * ci[i] / (ci[(i + 1) % 3] * ci[(i + 2) % 3])).log() / 4 for i in range(3)]
    if not _HOOKS["st_opposite"]:                      # planted: K on an adjacent edge
        K = [K[1], K[2], K[0]]
    R = (16 * c * ci[0] * ci[1] * ci[2]).root(4)
    return K[0], K[1], K[2], R


def _kappa(K1, K2, K3):
    v1, v2, v3 = K1.tanh(), K2.tanh(), K3.tanh()
    num = (1 - v1 * v1) * (1 - v2 * v2) * (1 - v3 * v3)
    return num / (4 * ((1 + v1 * v2 * v3) * (v1 + v2 * v3) * (v2 + v3 * v1) * (v3 + v1 * v2)).sqrt())


def _l(L1, L2, L3):
    z1, z2, z3 = (-2 * L1).exp(), (-2 * L2).exp(), (-2 * L3).exp()
    num = 16 * (1 + z1 * z2 * z3) * (z1 + z2 * z3) * (z2 + z3 * z1) * (z3 + z1 * z2)
    return (num / ((1 - z1 * z1) * (1 - z2 * z2) * (1 - z3 * z3)) ** 2).sqrt()


def triangle_to_star(K1, K2, K3):
    """(L1, L2, L3, R, k): sinh 2L_j = 1/(k sinh 2K_j) (6.4.13), k by (6.4.16), R^2 = 2/(k^2 sinh 2K1
    sinh 2K2 sinh 2K3) (6.4.14).  Ferromagnetic K only (k > 0)."""
    k = _kappa(K1, K2, K3)
    L = [(1 / (k * (2 * Kj).sinh())).asinh() / 2 for Kj in (K1, K2, K3)]
    R = (2 / (k * k * (2 * K1).sinh() * (2 * K2).sinh() * (2 * K3).sinh())).sqrt()
    return L[0], L[1], L[2], R, k


def triangular_self_dual(K):
    """K* = star-triangle(dual(K)): Baxter Sec. 6.5 (equal to sinh 2K*_j = k sinh 2K_j, (6.5.2))."""
    L = [A.dual_coupling(Kj) for Kj in K]
    return star_triangle(*L)[:3]


def honeycomb_self_dual(L):
    """L* = dual(star-triangle(L)): fixed at the honeycomb's critical point."""
    return tuple(A.dual_coupling(Kj) for Kj in star_triangle(*L)[:3])


# ---- free energies ---------------------------------------------------------------------------
@A.memo
def triangular_free_energy(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3), positive=False)
    return certify(lambda: A.integral3(*_arbs(ks)))


@A.memo
def triangular_free_energy_square(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3))

    def thunk():
        K = _arbs(ks)
        L1, L2, L3, R, _ = triangle_to_star(*K)
        sq = sum((A.baxter_square(Lj, Kj) for Lj, Kj in zip((L1, L2, L3), K)), 0 * R)
        return (sq - R.log()) / 2
    return certify(thunk)


def _pierre(v, lattice):
    """Pierre et al. (19)-(20) per site, the inner angle in closed form (arb v = tanh K)."""
    from flint import acb, arb
    if lattice == "triangular":
        a0, a1, m, nl, nv = 1 - 4 * v + v * v, 4 * v * (1 - v) ** 2, 1, 3, 1
    else:
        a0, a1, m, nl, nv = 1 - 3 * v * v, 4 * v * v * (1 - v * v), 2, 3, 0
    a02 = a0 * a0

    def f(t, analytic):
        s = (t / 2).sin()
        s2 = s * s
        P = a02 + a1 * (1 + s2)
        d = a02 * a02 + 2 * a02 * a1 * (1 + s2) + a1 * a1 * s2 * (3 + s2)
        return ((P + d.sqrt(analytic=analytic)) / 2).log(analytic=analytic)
    integral = acb.integral(f, 0, arb.pi()).real / (2 * arb.pi())
    one_m = 1 - v * v
    return (m * arb(2).log() - arb(nl) / 2 * one_m.log() + nv * (1 + v).log() + integral) / m


@A.memo
def triangular_free_energy_pierre(beta, J=1) -> Ball:
    (k,) = A.couplings(beta, (J,))
    return certify(lambda: _pierre(arb_q(k).tanh(), "triangular"))


@A.memo
def honeycomb_free_energy_pierre(beta, J=1) -> Ball:
    (k,) = A.couplings(beta, (J,))
    return certify(lambda: _pierre(arb_q(k).tanh(), "honeycomb"))


@A.memo
def honeycomb_free_energy(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3))

    def thunk():
        K1, K2, K3, R = star_triangle(*_arbs(ks))
        return (R.log() + A.integral3(K1, K2, K3)) / 2
    return certify(thunk)


@A.memo
def honeycomb_free_energy_duality(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3))

    def thunk():
        L = _arbs(ks)
        Kd = [A.dual_coupling(Lj) for Lj in L]
        s = (2 * L[0]).sinh() * (2 * L[1]).sinh() * (2 * L[2]).sinh()
        return ((2 * s).log() / 2 + A.integral3(*Kd)) / 2
    return certify(thunk)


# ---- critical points ---------------------------------------------------------------------------
@A.memo
def triangular_kappa(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3))
    return certify(lambda: _kappa(*_arbs(ks)))


@A.memo
def honeycomb_l(beta, J1=1, J2=1, J3=1) -> Ball:
    ks = _ks(beta, (J1, J2, J3))
    return certify(lambda: _l(*_arbs(ks)))


def _critical(Js, closed, cond):
    js = _ks(1, Js)
    if js[0] == js[1] == js[2]:
        return certify(lambda: closed() / arb_q(js[0]))
    from fractions import Fraction
    g = lambda b: 1 - cond(*[b * arb_q(j) for j in js])     # increasing in beta
    lo = Fraction(1, 1 << 20)
    hi = Fraction(1)
    while A._sign(g, hi) < 0:
        hi *= 2
    return A.certified_root(g, lo, hi)


@A.memo
def triangular_critical_beta(J1=1, J2=1, J3=1) -> Ball:
    return _critical((J1, J2, J3), lambda: (1 / arb_q(3).sqrt()).asinh() / 2, _kappa)


@A.memo
def honeycomb_critical_beta(J1=1, J2=1, J3=1) -> Ball:
    return _critical((J1, J2, J3), lambda: arb_q(3).sqrt().asinh() / 2, _l)


# ---- the triangular antiferromagnet ------------------------------------------------------------
@A.memo
def wannier_entropy() -> Ball:
    def thunk():
        from flint import acb, arb
        f = lambda w, analytic: (2 * w.cos()).log(analytic=analytic)
        return 2 * acb.integral(f, 0, arb.pi() / 3).real / arb.pi()
    return certify(thunk)


__all__ = ["honeycomb_critical_beta", "honeycomb_free_energy", "honeycomb_free_energy_duality",
           "honeycomb_free_energy_pierre", "honeycomb_l", "honeycomb_self_dual", "star_triangle",
           "triangle_to_star", "triangular_critical_beta", "triangular_free_energy",
           "triangular_free_energy_pierre", "triangular_free_energy_square", "triangular_kappa",
           "triangular_self_dual", "wannier_entropy"]
