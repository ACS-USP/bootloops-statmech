"""Displayed equations of the `pspin` (spherical) and `ising_pspin` catalog entries: the
1RSB functionals, stationarity, transitions and the 1RSB replicon (E4).  Split out of
latex_entries.py; same rule: each sympy block carries a tex.Check against the family
code at three points (tests/test_latex.py), a cited block quotes its paper.
"""
import mpmath as mp
import sympy as sp

from .gauss import GaussE
from .latex_entries import B, M, Q, Z
from .tex import Check, cited, sym, text

_CS = "Crisanti & Sommers, Z. Phys. B 87, 341 (1992)"
_CLN = "Cugliandolo, Lozano & Nessi, J. Stat. Mech. (2017) 083301, Eqs. (8)-(10)"
_CL = "Crisanti & Leuzzi, PRB 73, 014412 (2006), arXiv:cond-mat/0505748"
_MRT = "Montanari & Ricci-Tersenghi, EPJB 33, 339 (2003), arXiv:cond-mat/0301591"
_YM = "Yeo & Moore, PRE 101, 032127 (2020)"
_RIZ = "Rizzo, PRE 88, 032135 (2013)"
LAMR = sp.Symbol("Lambda_R")


def pspin(route):
    from .families import gardner as G
    from .families import pspin as F
    p, y = sp.Symbol("p", positive=True), sp.Symbol("y", positive=True)
    fun = F.pspin_functional(3)
    betas = ("2", "3", "5")
    sols = lambda: tuple((s.q0, s.q1, s.m, b) for b in betas
                         for s in [F.pspin_spherical_1rsb(3, b)])
    bq = lambda: tuple((mp.mpf(b), F.pspin_spherical_1rsb(3, b).q1) for b in betas)
    ys = lambda: tuple((k, F._y(k)) for k in (3, 4, 5))
    return [
        text("1RSB functional", "q0 between blocks, q inside blocks of size m; (1/n) sum_ab "
             "Q_ab^p by ansatz.sum_offdiag_power and lim (1/n) ln det Q from the 1RSB "
             "eigenvalues (families/pspin.py); the q-independent sphere entropy is omitted"),
        sym("-beta f (1RSB, p = 3)", sp.Eq(-F._B * sp.Symbol("f"), fun),
            Check((F._Q0, F._Q, F._M, F._B), lambda q0, q, m, b: -mp.mpf(b) *
                  F.pspin_spherical_1rsb(3, b).f, sols, "rhs", slow=True)),
        cited("Reduced stationarity (zero field, q0 = 0)",
              r"\frac{1}{p} = -\frac{y(1 - y + \ln y)}{(1-y)^2},\quad "
              r"\frac{p}{2}\, q^{p-2}(1-q)^2 = y\, T^2,\quad m = \frac{(1-y)(1-q)}{y\, q}",
              f"{_CS}; {_CLN}"),
        sym("Static transition (m = 1, q_s = 1 - y)",
            sp.Eq(sp.Symbol("T_s"), sp.sqrt(p * y / 2) * (1 - y)**(p / 2 - 1)),
            Check((p, y), lambda k, yy: F.pspin_critical_temperature(int(k)), ys, "rhs")),
        sym("Dynamic (mode-coupling) transition",
            sp.Eq(sp.Symbol("T_d"), sp.sqrt(p * (p - 2)**(p - 2) / (2 * (p - 1)**(p - 1)))),
            Check((p,), lambda k: F.pspin_critical_temperature(int(k), kind="dynamic"),
                  ((3,), (4,), (5,)), "rhs")),
        sym("Energy per spin (p = 3)", sp.Eq(sp.Symbol("e"),
                                              -(F._B / 2) * (1 + (F._M - 1) * F._Q**3)),
            Check((F._Q0, F._Q, F._M, F._B), lambda q0, q, m, b:
                  F.pspin_spherical_1rsb(3, b).e, sols, "rhs", slow=True)),
        sym("1RSB replicon, innermost block (p = 3): Lambda_1 (1 - q)^2, Lambda_1 = "
            "1/(1-q)^2 - Sigma(q), Sigma = beta^2 p (p-1) q^(p-2)/2",
            sp.Eq(LAMR, 1 - 3 * B**2 * Q * (1 - Q)**2), source=f"{_CL}, Eq. (31)",
            check=Check((B, Q), lambda b, q: G.onersb_replicon("pspin", p=3, beta=b), bq,
                        "rhs")),
        sym("On the static line (no Gardner transition: positive for every T < T_s), "
            "evaluated at T = 0.3",
            sp.Eq(LAMR, 1 - (p - 1) * y), source=f"{_CL}, App. C; {_MRT}, Sec. 1",
            check=Check((p, y), lambda k, yy: G.onersb_replicon("pspin", p=int(k),
                                                                beta=1 / mp.mpf("0.3")),
                        ys, "rhs")),
    ]


def ising_pspin(route):
    from .families import gardner as G
    from .families import ising_pspin as F
    a = B * sp.sqrt(3 * Q**2 / 2)
    w = sp.cosh(a * Z)**M
    Ew = lambda g: GaussE(w * g, Z) / GaussE(w, Z)
    pts = (("2", "0.9", "0.5"), ("3", "0.95", "0.3"), ("1.6", "0.85", "0.9"))
    sols = lambda: tuple((mp.mpf(b), s.q1, s.m) for b in ("2", "2.5", "3.5")
                         for s in [F.ising_pspin_1rsb(3, b)])
    return [
        sym("-beta f (1RSB, q0 = 0, p = 3), a^2 = beta^2 p q^(p-1)/2",
            sp.Eq(-B * sp.Symbol("f"), B**2 / 4 * (1 + 2 * (1 - M) * Q**3 - 3 * Q**2)
                  + sp.log(2) + sp.log(GaussE(w, Z)) / M), source=f"{_MRT}, Eq. (2.2)",
            check=Check((B, Q, M), lambda b, q, m: F.minus_beta_f(3, b, q, m), pts, "rhs")),
        sym("Stationarity in q", sp.Eq(Q, Ew(sp.tanh(a * Z)**2)), source=f"{_MRT}, Eq. (2.3)",
            check=Check((B, Q, M), lambda b, q, m: G.ising_block_average(
                b, 3, q, m, lambda t: t * t), pts, "rhs", slow=True)),
        sym("Stationarity in m", sp.Eq(sp.Integer(0), B**2 * 2 * Q**3 * M**2 / 4
                                        + sp.log(GaussE(w, Z))
                                        - M * Ew(sp.log(sp.cosh(a * Z)))),
            source=f"{_YM}, Eq. (16)",
            check=Check((B, Q, M), lambda b, q, m: F._gm(float(m), 3, float(b), float(q)),
                        pts, "rhs", slow=True)),
        sym("Energy per spin", sp.Eq(sp.Symbol("e"), -(B / 2) * (1 - (1 - M) * Q**3)),
            source=f"{_MRT}, Eq. (2.5)",
            check=Check((B, Q, M), lambda b, q, m: F.ising_pspin_1rsb(3, b).e, sols, "rhs",
                        slow=True)),
        sym("1RSB replicon, innermost block (Gardner): stable where positive",
            sp.Eq(LAMR, 1 - 3 * B**2 * Q * Ew(sp.sech(a * Z)**4)),
            source=f"{_MRT}, Eq. (2.6); {_YM}, Eq. (17)",
            check=Check((B, Q, M), lambda b, q, m: G.replicon_formula("ising", 3, b, q, m),
                        pts, "rhs")),
        text("Transitions", "T_s: m_s = 1 on the larger-q branch; T_d: the m = 1 nonzero q "
             "roots appear (fold); T_G: the static-line replicon Lambda_R vanishes "
             f"(families/ising_pspin.py, families/gardner.py; values banked from {_RIZ})"),
    ]
