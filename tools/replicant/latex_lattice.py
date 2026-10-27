"""Displayed equations of the lattice catalog entries (E9): ising2d, ising1d (route exact) and
bethe_ising (route cavity).  Same rule as latex_entries.py: each sympy block carries a tex.Check
against lattice/ (the certified ball's midpoint) at three points (tests/test_latex.py); the double
integral, Kaufman's formula and the transfer matrix are cited strings, the single integral a cited
block with a hand-coded mpmath display (lattice/onsager.onsager_free_energy_mp, no arb).
"""
from fractions import Fraction

import mpmath as mp
import sympy as sp

from .latex_lattice_obs import ising1d_blocks, ising2d_blocks
from .tex import Check, cited, sym, text

K, H, B, X, KK = sp.symbols("K H B x k", real=True)
_DC = "Duminil-Copin, arXiv:2208.00864 (2022)"
_ONS = f"{_DC}, Eq. (3.6) (Onsager, Phys. Rev. 65, 117 (1944))"
_YANG = f"{_DC}, Eq. (3.7) (Yang, Phys. Rev. 85, 808 (1952))"
_OEIS = "OEIS A242743"
_KAUF = ("Izmailian & Hu, Phys. Rev. E 65 (2002), arXiv:cond-mat/0009024, Eqs. (10)-(16); Kastner, "
         "arXiv:cond-mat/0605568, Eq. (8b) (Kaufman, Phys. Rev. 76, 1232 (1949))")
_ONE_D = "Wang, Diaz-Mendez & Capdevila, arXiv:1907.11701, Eqs. (7), (9), (12)"
_DM = "Dembo & Montanari, Ann. Appl. Probab. 20, 565 (2010), arXiv:0804.4726"


def ising2d(route):
    from .lattice import BETA_C, onsager_free_energy, yang_magnetization
    from .lattice.onsager import onsager_free_energy_mp
    crit = sp.log(2) / 2 + 2 * sp.Catalan / sp.pi
    yang = (1 - sp.sinh(2 * K)**-4) ** sp.Rational(1, 8)
    return [
        text("Exact route", "Square-lattice Ising model, zero field, K = beta J; -beta f = ln Z / N. "
             "Each value is a ball enclosure (python-flint arb through tools/baller)", source=_DC),
        cited("Onsager's free energy",
              r"-\beta f = \ln 2 + \frac{1}{8\pi^2}\int_0^{2\pi}\!\!\int_0^{2\pi}"
              r"\ln\left[\cosh^2 2K - \sinh 2K\,(\cos\theta_1 + \cos\theta_2)\right]"
              r"d\theta_1\, d\theta_2",
              _ONS),
        cited("The inner angle in closed form (certified by acb.integral off beta_c)",
              r"-\beta f = \ln 2 + \frac{1}{2\pi}\int_0^{\pi}\ln\frac{a + \sqrt{a^2 - S^2}}{2}\,d\theta,"
              r"\quad a = C^2 - S\cos\theta,\ C = \cosh 2K,\ S = \sinh 2K", _ONS,
              check=Check((K,), lambda k: onsager_free_energy(k).mid(), (("0.3",), ("0.5",), ("0.8",)),
                          display=lambda k: onsager_free_energy_mp(k, dps=25))),
        sym("At beta_c = (1/2) ln(1 + sqrt 2) (G: Catalan's constant)",
            sp.Eq(sp.Symbol(r"-\beta f_c"), crit), source=_OEIS,
            check=Check((), lambda: onsager_free_energy(BETA_C).mid(), ((),), "rhs")),
        sym("Yang's spontaneous magnetization (beta > beta_c)", sp.Eq(sp.Symbol("m^*"), yang),
            source=_YANG, check=Check((K,), lambda k: yang_magnetization(k).mid(),
                                      (("0.5",), ("0.7",), ("1",)), "rhs")),
        cited("Kaufman's torus (M = Lx, N = Ly; g_0 signed)",
              r"Z = \tfrac12 (2\sinh 2K)^{MN/2}\sum_{i=1}^4 Z_i,\ Z_{1,2} = \prod_{r=0}^{N-1}"
              r"2\{\cosh,\sinh\}\tfrac{M\gamma_{2r+1}}{2},\ Z_{3,4} = \prod_{r=0}^{N-1}2\{\cosh,\sinh\}"
              r"\tfrac{M\gamma_{2r}}{2},\ \cosh\gamma_r = \frac{\cosh^2 2K}{\sinh 2K}"
              r" - \cos\frac{r\pi}{N},"
              r"\ \gamma_0 = 2K + \ln\tanh K", _KAUF),
        cited("Transfer matrix (exact integers, x = e^{-2K})",
              r"Z = e^{2NK}\,\mathrm{Tr}\,T^{L_x},\quad T_{s s'} = x^{u(s) + |s \oplus s'|}",
              _DC + ", Frame 2"),
        *ising2d_blocks(),
    ]


def ising1d(route):
    from .lattice import ising1d_free_energy
    lam = sp.exp(K) * sp.cosh(H) + sp.sqrt(sp.exp(2 * K) * sp.sinh(H)**2 + sp.exp(-2 * K))
    return [
        text("Exact route", "1D Ising ring, J = 1, K = beta, H = beta h; Z_N = Tr T^N = lambda_+^N + "
             "lambda_-^N", source=_ONE_D),
        sym("Free energy (largest transfer-matrix eigenvalue)",
            sp.Eq(sp.Symbol(r"-\beta f"), sp.log(lam)), source=_ONE_D,
            check=Check((K, H), lambda k, h: ising1d_free_energy(k, Fraction(h) / Fraction(k)).mid(),
                        (("0.7", "0.21"), ("1", "0"), ("2", "-0.5")), "rhs")),
        *ising1d_blocks(),
    ]


def _bethe_phi(form):
    """impl for the Bethe blocks: phi (or phi_dm) of the code's fixed point nearest x."""
    def impl(k, beta, Bf, x):
        from .lattice import bethe_ising_fixed_points
        fps = bethe_ising_fixed_points(int(k), beta, Fraction(Bf) / Fraction(beta))
        f = min(fps, key=lambda f: abs(f.cavity_field.mid() - mp.mpf(x)))
        return getattr(f, form).mid()
    return impl


def _bethe_point(k, beta, h, branch):
    def p():
        from .lattice import bethe_ising_fixed_points
        f = next(f for f in bethe_ising_fixed_points(k, beta, h) if f.branch == branch)
        return (k, beta, str(Fraction(beta) * Fraction(h)), f.cavity_field.mid())
    return p


def bethe_ising(route):
    beta = sp.Symbol(r"\beta", positive=True)
    t = sp.tanh(beta)
    xi = sp.atanh(t * sp.tanh(X))
    Xf = B + KK * xi
    phi = KK / 2 * sp.log(2 * sp.exp(beta) * sp.cosh(2 * X) + 2 * sp.exp(-beta)) \
        - (KK - 1) * sp.log(2 * sp.cosh(Xf))
    dm = KK / 2 * sp.log(sp.cosh(beta)) - KK / 2 * sp.log(1 + t * sp.tanh(X)**2) + sp.log(
        sp.exp(B) * (1 + t * sp.tanh(X))**KK + sp.exp(-B) * (1 - t * sp.tanh(X))**KK)
    pts = (_bethe_point(3, "1", "0.02", "upper"), _bethe_point(3, "1", "0.02", "lower"),
           _bethe_point(4, "0.5", "0.1", "upper"))
    return [
        text("Cavity route on the Bethe lattice",
             "coordination k, J = 1, B = beta h; x the cavity field "
             "(times beta); every fixed point is enclosed by interval Newton", source=_DM),
        cited("Cavity (BP) fixed point and its stability",
              r"x = B + (k-1)\,\xi(x),\ \ \xi(x) = \operatorname{atanh}(\tanh\beta\,\tanh x),\qquad"
              r"\text{stable iff } (k-1)\,\xi'(x) < 1", f"{_DM}, Eqs. (2.6)-(2.7)"),
        cited("Critical point", r"(k-1)\tanh\beta_c = 1", f"{_DM}, Eq. (2.5)"),
        sym("Bethe free energy (link and site terms)", sp.Eq(sp.Symbol(r"-\beta f"), phi), source=_DM,
            check=Check((KK, beta, B, X), _bethe_phi("phi"), pts, "rhs")),
        sym("Dembo-Montanari form (regular graph)", sp.Eq(sp.Symbol(r"\phi"), dm),
            source=f"{_DM}, Eq. (2.9)", check=Check((KK, beta, B, X), _bethe_phi("phi_dm"), pts, "rhs")),
    ]


__all__ = ["bethe_ising", "ising1d", "ising2d"]
