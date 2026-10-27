"""Displayed equations of the I3 entries ising2d_anisotropic, ising_triangular and ising_honeycomb
(catalog_lattice_aniso.py, catalog_lattice_geom.py).  Each sympy block carries a tex.Check against
lattice/ (the certified ball's midpoint) at three points; the integrals are cited strings whose
checks are hand-coded mpmath evaluations of the displayed formula (no arb, no lattice/ code).
"""
import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym, text

K1, K2, K3, L = sp.symbols("K_1 K_2 K_3 L", positive=True)
L1, L2, L3 = sp.symbols("L_1 L_2 L_3", positive=True)
_BAX = "Baxter, Exactly Solved Models (1982)"
_AU = "Athanasopoulos & Ueltschi, arXiv:2407.05303, Theorem 2.1(b) (Houtappel 1950)"
_PBM = "Pierre, Bernu & Messio, arXiv:2501.18569, Eqs. (16)-(20), Tables 2, 5"


def _kw_mp(k1, k2, k3):
    """The displayed Kac-Ward integral, inner angle in closed form, by mpmath quadrature."""
    with mp.workdps(30):
        k = [mp.mpf(str(x)) for x in (k1, k2, k3)]
        C = mp.cosh(2 * k[0]) * mp.cosh(2 * k[1]) * mp.cosh(2 * k[2])
        S = [mp.sinh(2 * x) for x in k]

        def g(t):
            a = C + S[0] * S[1] * S[2] - S[0] * mp.cos(t)
            return mp.log((a + mp.sqrt(a * a - S[1] ** 2 - S[2] ** 2 - 2 * S[1] * S[2] * mp.cos(t))) / 2)
        return mp.log(2) + mp.quad(g, [0, mp.pi / 2, mp.pi]) / (2 * mp.pi)


def _baxter_mp(k, kp):
    with mp.workdps(30):
        k, kp = mp.mpf(str(k)), mp.mpf(str(kp))
        li = mp.sinh(2 * k) * mp.sinh(2 * kp)
        g = lambda t: mp.log(2 * (mp.cosh(2 * kp) * mp.cosh(2 * k)
                                  + mp.sqrt(1 + li ** 2 - 2 * li * mp.cos(2 * t))))
        return mp.quad(g, [0, mp.pi / 2, mp.pi]) / (2 * mp.pi)


def _pierre_mp(k, lattice):
    with mp.workdps(30):
        v = mp.tanh(mp.mpf(str(k)))
        if lattice == "triangular":
            a0, a1, m, nl, nv = 1 - 4 * v + v * v, 4 * v * (1 - v) ** 2, 1, 3, 1
        else:
            a0, a1, m, nl, nv = 1 - 3 * v * v, 4 * v * v * (1 - v * v), 2, 3, 0
        X = lambda a, b: mp.log(a0 ** 2 + a1 * (mp.sin(a / 2) ** 2 + mp.sin(b / 2) ** 2
                                                + mp.sin((a + b) / 2) ** 2))
        n = 48                     # periodic analytic integrand: the trapezoid rule, both angles
        nodes = [2 * mp.pi * (i + mp.mpf(1) / 2) / n for i in range(n)]
        I = mp.fsum(X(a, b) for a in nodes for b in nodes) / (n * n)
        return (m * mp.log(2) - mp.mpf(nl) / 2 * mp.log(1 - v * v) + nv * mp.log(1 + v) + I / 2) / m


def ising2d_anisotropic(route):
    from .lattice import anisotropic as A
    kc2 = sp.asinh(1 / sp.sinh(2 * K1)) / 2
    return [
        text("Exact route", "Square lattice, K1 = beta J1 (horizontal), K2 = beta J2 (vertical); "
             "-beta f = lim ln Z / N; each value is a ball enclosure", source=_AU),
        cited("Free energy (Kac-Ward, J3 = 0)",
              r"-\beta f = \ln 2 + \frac{1}{8\pi^2}\int_{[-\pi,\pi]^2}\ln\left[\cosh 2K_1\cosh 2K_2"
              r" - \sinh 2K_1\cos k_1 - \sinh 2K_2\cos k_2\right]dk_1\,dk_2", _AU,
              check=Check((K1, K2), lambda a, b: A.free_energy(1, a, b).mid(),
                          (("0.2", "0.4"), ("0.5", "0.1"), ("0.6", "0.9")),
                          display=lambda a, b: _kw_mp(a, b, 0))),
        cited("Baxter's diagonal form (l^{-1} = sinh 2K sinh 2K')",
              r"-\beta f = \frac{1}{2\pi}\int_0^\pi \ln\left\{2\left[\cosh 2K\cosh 2K' + (1 + l^{-2} "
              r"- 2 l^{-1}\cos 2\theta)^{1/2}\right]\right\}d\theta", f"{_BAX}, Eqs. (11.7.15), (11.7.16)",
              check=Check((K1, K2), lambda a, b: A.free_energy_baxter(1, a, b).mid(),
                          (("0.2", "0.4"), ("0.5", "0.1"), ("0.6", "0.9")), display=_baxter_mp)),
        sym("The critical line sinh 2K1 sinh 2K2 = 1: K2 given K1", sp.Eq(sp.Symbol("K_{2,c}"), kc2),
            source=f"{_BAX}, Eq. (6.2.17)",
            check=Check((K1,), lambda k: A.critical_coupling(k).mid(), (("0.2",), ("0.44",), ("1",)),
                        "rhs")),
        sym("Dual couplings (tanh K1* = e^{-2K2})", sp.Eq(sp.Symbol("K_1^*"),
                                                          sp.atanh(sp.exp(-2 * K2))),
            source=f"{_BAX}, Eq. (6.2.14a)",
            check=Check((K2,), lambda k: A.dual_couplings(1, 1, k)[0].mid(),
                        (("0.2",), ("0.5",), ("1",)), "rhs")),
        cited("Duality of the free energy",
              r"-\beta f(K_1^*, K_2^*) = -\beta f(K_1, K_2) + \ln(2\cosh K_1^*\cosh K_2^*) - K_1 - K_2",
              f"{_BAX}, Eqs. (6.2.11)-(6.2.15)"),
    ]


def ising_triangular(route):
    from .lattice import tri_honey as T
    v = [sp.tanh(k) for k in (K1, K2, K3)]
    kap = (1 - v[0]**2) * (1 - v[1]**2) * (1 - v[2]**2) / (4 * sp.sqrt(
        (1 + v[0] * v[1] * v[2]) * (v[0] + v[1] * v[2]) * (v[1] + v[2] * v[0]) * (v[2] + v[0] * v[1])))
    pts = (("0.1", "0.2", "0.3"), ("0.3", "0.3", "0.3"), ("0.5", "0.2", "0.7"))
    return [
        text("Exact route", "Triangular lattice, couplings K1, K2, K3 (horizontal, vertical, "
             "oblique); each value is a ball enclosure", source=_AU),
        cited("Free energy (Kac-Ward / Houtappel), k3 = k1 + k2",
              r"-\beta f = \ln 2 + \frac{1}{8\pi^2}\int_{[-\pi,\pi]^2}\ln\Big[\prod_i\cosh 2K_i + "
              r"\prod_i\sinh 2K_i - \sum_i \sinh 2K_i\cos k_i\Big]dk_1\,dk_2", _AU,
              check=Check((K1, K2, K3), lambda a, b, c: T.triangular_free_energy(1, a, b, c).mid(),
                          pts, display=_kw_mp)),
        sym("Baxter's k (critical at k = 1; v_j = tanh K_j)", sp.Eq(sp.Symbol("k"), kap),
            source=f"{_BAX}, Eqs. (6.4.16), (6.5.5)",
            check=Check((K1, K2, K3), lambda a, b, c: T.triangular_kappa(1, a, b, c).mid(), pts, "rhs")),
        sym("Isotropic critical point", sp.Eq(sp.Symbol("K_c"), sp.asinh(1 / sp.sqrt(3)) / 2),
            source=f"{_BAX}, Eq. (6.5.6)",
            check=Check((), lambda: T.triangular_critical_beta().mid(), ((),), "rhs")),
        cited("Three square lattices (L_j the star-triangle partners; psi_SQ of Eq. (11.7.15))",
              r"-\beta f_T(K) = \tfrac12\Big[-\ln R + \sum_{j=1}^3 \big(-\beta f_{SQ}(L_j, K_j)\big)\Big]",
              f"{_BAX}, Eq. (11.7.13)"),
        cited("Isotropic (Pierre et al.; v = tanh K, xi = sum of sin^2 of k1/2, k2/2, (k1+k2)/2)",
              r"-\beta f = \ln 2 - \tfrac32\ln(1 - v^2) + \ln(1 + v) + \frac12\int\frac{d^2k}{4\pi^2}"
              r"\ln\big[(1 - 4v + v^2)^2 + 4v(1 - v)^2\,\xi(k)\big]", _PBM,
              check=Check((K1,), lambda k: T.triangular_free_energy_pierre(k).mid(),
                          (("0.1",), ("0.2",), ("0.4",)), display=lambda k: _pierre_mp(k, "triangular"),
                          tol=mp.mpf("1e-12"))),
        cited("Antiferromagnet: ground-state entropy per site",
              r"s_0 = \frac{2}{\pi}\int_0^{\pi/3}\ln(2\cos\omega)\,d\omega", f"{_PBM}, Sec. 5.2",
              check=Check((), lambda: T.wannier_entropy().mid(), ((),), display=lambda: 2 / mp.pi
                          * mp.quad(lambda w: mp.log(2 * mp.cos(w)), [0, mp.pi / 3]))),
    ]


def ising_honeycomb(route):
    from .lattice import tri_honey as T
    c = sp.cosh(L1 + L2 + L3)
    c1, c2, c3 = sp.cosh(-L1 + L2 + L3), sp.cosh(L1 - L2 + L3), sp.cosh(L1 + L2 - L3)
    pts = (("0.3", "0.4", "0.5"), ("0.6", "0.6", "0.6"), ("1", "0.2", "0.4"))

    def e4k1(a, b, cc):
        from .lattice.ball import arb_q, certify
        return certify(lambda: (4 * T.star_triangle(arb_q(a), arb_q(b), arb_q(cc))[0]).exp()).mid()
    return [
        text("Exact route", "Honeycomb lattice, couplings L1, L2, L3 by edge direction; the values "
             "come from the triangular lattice through Baxter's maps", source=f"{_BAX}, Ch. 6"),
        sym("Star-triangle (K1 on the triangle edge opposite the leg L1)",
            sp.Eq(sp.exp(4 * sp.Symbol("K_1")), c * c1 / (c2 * c3)),
            source=f"{_BAX}, Eqs. (6.4.8)-(6.4.10)",
            check=Check((L1, L2, L3), e4k1, pts, "rhs")),
        cited("Star-triangle and duality relations of the partition functions",
              r"Z^H_{2N}\{L\} = R^N Z^T_N\{K\},\qquad Z^H_{2N}\{L\} = (2 s_1 s_2 s_3)^{N/2} "
              r"Z^T_N\{K^*\},\ \tanh K^*_j = e^{-2L_j},\ s_j = \sinh 2L_j",
              f"{_BAX}, Eqs. (6.3.6)-(6.3.8), (6.4.7)"),
        sym("Isotropic critical point", sp.Eq(sp.Symbol("L_c"), sp.asinh(sp.sqrt(3)) / 2),
            source=f"{_BAX}, Eq. (6.5.7)",
            check=Check((), lambda: T.honeycomb_critical_beta().mid(), ((),), "rhs")),
        cited("Isotropic (Pierre et al.), per site",
              r"-\beta f = \ln 2 - \tfrac34\ln(1 - v^2) + \frac14\int\frac{d^2k}{4\pi^2}"
              r"\ln\big[(1 - 3v^2)^2 + 4v^2(1 - v^2)\,\xi(k)\big]", _PBM,
              check=Check((L,), lambda k: T.honeycomb_free_energy_pierre(k).mid(),
                          (("0.3",), ("0.5",), ("0.9",)), display=lambda k: _pierre_mp(k, "honeycomb"),
                          tol=mp.mpf("1e-12"))),
    ]


__all__ = ["ising2d_anisotropic", "ising_honeycomb", "ising_triangular"]
