"""Displayed equations of the exact observables of `ising1d` and `ising2d` (catalog_lattice_obs.py),
appended to latex_lattice.ising1d / ising2d.  Each sympy block carries a tex.Check against lattice/
(the certified ball's midpoint) at three points; the near-critical form and the torus derivatives
are cited strings.
"""
from fractions import Fraction

import sympy as sp

from .tex import Check, cited, sym

K, H, R = sp.symbols("K H r", real=True)
_ONS = "Onsager, Phys. Rev. 65, 117 (1944)"
_BAX = "Baxter, Exactly Solved Models in Statistical Mechanics (1982), Ch. 2"
_KAUF = ("Izmailian & Hu, Phys. Rev. E 65 (2002), arXiv:cond-mat/0009024, Eqs. (10)-(16) (Kaufman, "
         "Phys. Rev. 76, 1232 (1949))")


def ising2d_blocks():
    from .lattice import onsager_thermo as OT
    C, S = sp.cosh(2 * K), sp.sinh(2 * K)
    m1 = (2 * S / C**2) ** 2
    kpp = 2 * sp.tanh(2 * K) ** 2 - 1
    K1, E1 = sp.elliptic_k(m1), sp.elliptic_e(m1)
    u = -sp.coth(2 * K) * (1 + 2 / sp.pi * kpp * K1)
    c = (K * sp.coth(2 * K)) ** 2 * 2 / sp.pi * (2 * K1 - 2 * E1 - (1 - kpp) * (sp.pi / 2 + kpp * K1))
    pts = (("0.3",), ("0.5",), ("0.8",))
    return [
        sym("Onsager's energy per site (J = 1; K(m), E(m) of parameter m = k_1^2, k_1 = 2 sinh 2K / "
            "cosh^2 2K; -sqrt 2 at beta_c)", sp.Eq(sp.Symbol("u"), u), source=f"{_ONS}, Eqs. (114), "
            "(116), (121)", check=Check((K,), lambda k: OT.onsager_energy(k).mid(), pts, "rhs")),
        sym("Onsager's specific heat per site (off beta_c)", sp.Eq(sp.Symbol("c"), c),
            source=f"{_ONS}, Eq. (117)",
            check=Check((K,), lambda k: OT.onsager_specific_heat(k).mid(), pts, "rhs")),
        cited("The logarithmic divergence at beta_c",
              r"K_1 \sim \ln\frac{4}{k_1'} \sim \ln\frac{\sqrt 2}{|K - K_c|},\quad "
              r"c \sim \frac{2}{\pi}\left(\ln\cot\frac{\pi}{8}\right)^2\left(K_1 - 1 - \frac{\pi}{4}\right)",
              f"{_ONS}, Eq. (120)"),
        sym("Its amplitude: c = -A ln|beta - beta_c| + O(1)",
            sp.Eq(sp.Symbol("A"), 2 / sp.pi * sp.log(1 + sp.sqrt(2)) ** 2), source=f"{_ONS}, Eq. (120)",
            check=Check((), lambda: OT.specific_heat_amplitude().mid(), ((),), "rhs")),
        cited("Finite torus (N = L_x L_y sites): derivatives of the exact Z",
              r"u_L = -\frac{1}{N}\frac{\partial \ln Z}{\partial K},\quad "
              r"c_L = \frac{K^2}{N}\frac{\partial^2 \ln Z}{\partial K^2}", _KAUF),
    ]


def ising1d_blocks():
    from .lattice import observables1d as O1
    D = sp.exp(2 * K) * sp.sinh(H) ** 2 + sp.exp(-2 * K)
    lp = sp.exp(K) * sp.cosh(H) + sp.sqrt(D)
    lm = sp.exp(K) * sp.cosh(H) - sp.sqrt(D)
    sin2 = 1 / (1 + (sp.exp(2 * K) * sp.sinh(H)) ** 2)
    h = lambda k, x: Fraction(x) / Fraction(k)
    pts = (("0.7", "0.21"), ("1", "0"), ("2", "-0.5"))
    return [
        sym("Magnetization (J = 1, K = beta, H = beta h)",
            sp.Eq(sp.Symbol("m"), sp.exp(K) * sp.sinh(H) / sp.sqrt(D)), source=f"{_BAX}, Eq. (2.1.18)",
            check=Check((K, H), lambda k, x: O1.ising1d_magnetization(k, h(k, x)).mid(), pts, "rhs")),
        sym("Susceptibility chi = dm/dh (the derivative of Eq. (2.1.18))",
            sp.Eq(sp.Symbol(r"\chi"), K * sp.exp(-K) * sp.cosh(H) / D ** sp.Rational(3, 2)),
            source=f"{_BAX}, Eqs. (1.7.17), (2.1.18)",
            check=Check((K, H), lambda k, x: O1.ising1d_susceptibility(k, h(k, x)).mid(), pts, "rhs")),
        sym("Connected correlation (sin^2 2 phi from cot 2 phi = e^{2K} sinh H)",
            sp.Eq(sp.Symbol("g(r)"), sin2 * (lm / lp) ** R), source=f"{_BAX}, Eqs. (2.2.9), (2.2.14)",
            check=Check((K, H, R), lambda k, x, r: O1.ising1d_correlation(k, h(k, x), int(r)).mid(),
                        (("0.7", "0.21", 1), ("1", "0", 3), ("2", "-0.5", 2)), "rhs")),
        sym("Correlation length", sp.Eq(sp.Symbol(r"\xi"), 1 / sp.log(lp / lm)),
            source=f"{_BAX}, Eq. (2.2.15)",
            check=Check((K, H), lambda k, x: O1.ising1d_correlation_length(k, h(k, x)).mid(), pts,
                        "rhs")),
        cited("Finite ring: exact derivatives of ln Z_N and the N -> oo gap (q = (lambda_2/lambda_1)^N)",
              r"m_N - m = \frac{q\,\ell'}{1 + q},\quad \frac{\chi_N - \chi}{\beta} = "
              r"\frac{N q\,\ell'^2}{(1 + q)^2} + \frac{q\,\ell''}{1 + q},\quad "
              r"\ell = \ln\frac{\lambda_2}{\lambda_1}", f"{_BAX}, Eqs. (2.1.15), (2.2.7)"),
    ]


__all__ = ["ising1d_blocks", "ising2d_blocks"]
