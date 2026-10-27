"""Displayed equations of the I2 values of `ising2d` (catalog_lattice_xi.py), appended to
latex_lattice.ising2d: the correlation length on both sides of T_c (sympy, tex.Check against
lattice/correlation2d.py at three points), Onsager's strip gaps, Cardy's gap amplitude and the
critical diagonal correlation (cited, the last two with hand-coded mpmath displays).
"""
import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym

K = sp.symbols("K", positive=True)
_ONS = "Onsager, Phys. Rev. 65, 117 (1944), Eqs. (17), (89a), (95)-(102)"
_XI = f"{_ONS}; Lyberg & McCoy, arXiv:math-ph/0612051; Dohm, arXiv:1907.00859, Eqs. (30), (46)"
_DIK = "Deift, Its & Krasovsky, arXiv:1207.4990, Eqs. (103)-(107)"
_PERK = "Perk & Au-Yang, arXiv:0901.1931, Eqs. (31), (32)"


def _perk(n):
    n = int(n)
    with mp.workdps(30):
        v = mp.mpf(2) / mp.pi
        out = v ** n
        for k in range(1, n):
            out /= (1 - mp.mpf(1) / (4 * k * k)) ** (n - k)
        return +out


def ising2d_xi_blocks():
    from .lattice import correlation2d as C
    from .lattice import critical2d as D
    kstar = -sp.log(sp.tanh(K)) / 2
    xi = sp.Symbol(r"\xi")
    # amplitudes: K = K_c/(1 + t); K* - K = lin t + O(t^2) (the constant vanishes at K_c, sinh 2K_c
    # = 1), lin = d(K* - K)/dt at t = 0 by sympy.diff.  (sympy.limit of t xi returns 0 here: it
    # does not see that -ln tanh(K_c)/2 - K_c = 0; the derivative form avoids it.)
    t, c = sp.symbols("t K_c", positive=True)
    lin = sp.diff((kstar - K).subs(K, c / (1 + t)), t).subs(t, 0)
    kc = sp.log(1 + sp.sqrt(2)) / 2
    amp = lambda i: C.xi_amplitudes()[i].mid()
    blocks_amp = [
        sym("Amplitude above T_c: xi = xi_0+ t^-1, t = K_c/K - 1 (lin = d(K* - K)/dt at t = 0)",
            sp.Eq(sp.Symbol(r"\xi_{0+}"), (1 / (2 * lin)).subs(c, kc)), source=_XI,
            check=Check((), lambda: amp(0), ((),), "rhs")),
        sym("Amplitude below T_c (1/xi = 4(K - K*))", sp.Eq(sp.Symbol(r"\xi_{0-}"),
                                                         (1 / (4 * lin)).subs(c, kc)), source=_XI,
            check=Check((), lambda: amp(1), ((),), "rhs")),
    ]
    return blocks_amp + [
        sym("Correlation length above T_c (exponential, along an axis; tanh K* = e^{-2K})",
            sp.Eq(xi, 1 / (2 * (kstar - K))), source=_XI,
            check=Check((K,), lambda k: C.correlation_length(k).mid(), (("0.2",), ("0.3",), ("0.4",)),
                        "rhs")),
        sym("Correlation length below T_c (the two-particle gap)", sp.Eq(xi, 1 / (4 * (K - kstar))),
            source=_XI,
            check=Check((K,), lambda k: C.correlation_length(k).mid(), (("0.5",), ("0.6",), ("0.8",)),
                        "rhs")),
        cited("Strip of width n (periodic): the gaps of Onsager's spectrum",
              r"\ln\frac{\lambda_0}{\lambda_1} = \tfrac12\Big(\sum_{r\ \mathrm{odd}}\gamma_r - "
              r"\sum_{r\ \mathrm{even} > 0}\gamma_r\Big) - (K - K^*),\quad \ln\frac{\lambda_0}"
              r"{\lambda_2} = 2\gamma_1,\quad \cosh\gamma_r = \frac{\cosh^2 2K}{\sinh 2K} - "
              r"\cos\frac{r\pi}{n}", _ONS),
        cited("Critical strip gap (Cardy): 2 pi x_sigma = pi eta",
              r"\lim_{n\to\infty} n \ln\frac{\lambda_0}{\lambda_1}\Big|_{\beta_c} = 2\pi x_\sigma = "
              r"\pi\eta = \frac{\pi}{4}",
              "Cardy, arXiv:0807.3472, Eq. (19); " + _DIK.replace("(103)-(107)", "(101)"),
              check=Check((), lambda: C.critical_gap_scaled(2048).mid(), ((),),
                          display=lambda: mp.pi / 4, tol=mp.mpf("1e-6"))),
        cited("Diagonal correlation at T_c (product form; = the Toeplitz determinant of (2/pi)/(2k+1))",
              r"\langle s_{0,0} s_{n,n}\rangle = \Big(\frac{2}{\pi}\Big)^n \prod_{k=1}^{n-1}"
              r"\Big(1 - \frac{1}{4k^2}\Big)^{-(n-k)}", f"{_PERK}; {_DIK}",
              check=Check((sp.Symbol("n"),), lambda n: D.diagonal_correlation(int(n)).mid(),
                          ((1,), (2,), (7,)), display=_perk)),
        cited("Its asymptote (A: Glaisher's constant)",
              r"\langle s_{0,0} s_{n,n}\rangle = e^{1/4} 2^{1/12} A^{-3} n^{-1/4}"
              r"\Big(1 - \frac{1}{64 n^2} + \dots\Big)", _DIK,
              check=Check((), lambda: D.diagonal_amplitude().mid(), ((),),
                          display=lambda: mp.exp(mp.mpf(1) / 4) * mp.mpf(2) ** (mp.mpf(1) / 12)
                          / mp.glaisher ** 3)),
    ]


__all__ = ["ising2d_xi_blocks"]
