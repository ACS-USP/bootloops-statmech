"""Displayed equations of the `online_perceptron` catalog entry (route online, O1).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against online/ at three points
(tests/test_latex.py); the update rule, the rate equations and the papers' asymptotes are cited.
"""
import math

import sympy as sp

from .tex import Check, cited, sym, text

RHO, LAM, ETA, AL, R0, Q0, D = sp.symbols("rho lambda eta alpha R_0 Q_0 Delta", real=True)
_BR = "Biehl & Riegler, Europhys. Lett. 28, 525 (1994)"
_VKC = "Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), arXiv:cond-mat/9801297"
_CE = ("Copelli, Eichhorn, Kinouchi, Biehl, Simonetti, Riegler & Caticha, Europhys. Lett. 37, 427 "
       "(1997)")
_PTS = (("0.3", "0"), ("0.7", "0.1"), ("0.95", "0.3"))


def _f(*xs):
    return [float(x) for x in xs]


def _closed(rule, field):
    from .online import fields as F
    return lambda rho, lam: getattr(F.closed(rule, math.acos(float(rho)), float(lam)), field)


def online_perceptron(route):
    from .online import odes as O
    from .online import optimal as OPT
    s = sp.sqrt(1 - RHO**2)
    th = sp.acos(RHO)
    c0 = 1 / sp.sqrt(2 * sp.pi)
    a = (1 - 2 * LAM) * sp.sqrt(2 / sp.pi)
    u = RHO * D / s
    phi = sp.exp(-u**2 / 2) / sp.sqrt(2 * sp.pi)
    Phi = sp.erfc(-u / sp.sqrt(2)) / 2
    wstar = s / RHO * (1 - 2 * LAM) * phi / (LAM + (1 - 2 * LAM) * Phi)

    def const_display(lam):
        import mpmath as mp
        with mp.workdps(20):
            lam = mp.mpf(lam)
            H = lambda t: mp.erfc(t / mp.sqrt(2)) / 2
            dt = lambda t: mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi)
            i = (1 - 2 * lam)**2 * mp.quad(lambda t: dt(t) * mp.exp(-t * t / 2) / (lam + (1 - 2 * lam) * H(t)),
                                           [-mp.inf, -1, 1, mp.inf])
            return 2 / i
    return [
        text("Online route", "Teacher |B| = 1, inputs xi with i.i.d. N(0, 1) components, observed label b "
             "= sign(B . xi) flipped with probability lambda (output noise), one fresh example per update, "
             "alpha = examples / N; R = J . B, Q = J . J, rho = R / sqrt(Q), eps = arccos(rho) / pi against "
             "the true teacher. The N -> oo limit is self-averaging by assumption (physics-level)",
             source=_BR),
        cited("Update rule and rate equations (BR Eqs. (2), (4)-(5) with a learning rate; x = h / sqrt Q, "
              "y = B . xi)",
              r"J \leftarrow J + \frac{\eta}{N} F(h, b, Q)\, b\, \xi,\quad \frac{dR}{d\alpha} = \eta"
              r"\langle F b y\rangle,\quad \frac{dQ}{d\alpha} = 2\eta \langle F b h\rangle + \eta^2"
              r"\langle F^2\rangle", _BR),
        sym("Hebb closed form (F = 1): R = R0 + eta a alpha, Q - R^2 = Q0 - R0^2 + eta^2 alpha, "
            "a = (1 - 2 lambda) sqrt(2/pi)",
            sp.Eq(sp.Symbol("epsilon"), sp.atan2(sp.sqrt(Q0 - R0**2 + ETA**2 * AL), R0 + ETA * a * AL) / sp.pi),
            source=_BR, check=Check((AL, ETA, LAM, R0, Q0),
                                    lambda al, eta, lam, r0, q0: O.hebb_closed_form(
                                        *_f(al, eta, lam), init=_f(r0, q0))[2],
                                    (("1", "1", "0", "0", "1"), ("7", "0.5", "0.2", "0.3", "2"),
                                     ("50", "2", "0.4", "0.1", "0.5")), "rhs")),
        sym("Perceptron rule F = Theta(-Delta), Delta = b x: <F b (y - rho x)> (quadrant moments)",
            sp.Eq(sp.Symbol("D_p"), (1 - 2 * LAM) * (1 - RHO**2) * c0), source=_BR,
            check=Check((RHO, LAM), _closed("perceptron", "drive"), _PTS, "rhs")),
        sym("Perceptron rule: <F b x>", sp.Eq(sp.Symbol("A_x"), -(1 - (1 - 2 * LAM) * RHO) * c0),
            source=_BR, check=Check((RHO, LAM), _closed("perceptron", "ax"), _PTS, "rhs")),
        sym("Perceptron rule: <F^2>, the training-error rate",
            sp.Eq(sp.Symbol("A_2"), (1 - LAM) * th / sp.pi + LAM * (1 - th / sp.pi)), source=_BR,
            check=Check((RHO, LAM), _closed("perceptron", "a2"), _PTS, "rhs")),
        sym("AdaTron F = -Delta Theta(-Delta) (per unit sqrt Q): <F b x> = -<F^2>",
            sp.Eq(sp.Symbol("A_x"), -((1 - LAM) * (th - RHO * s) + LAM * (sp.pi - th + RHO * s)) / sp.pi),
            source=_BR, check=Check((RHO, LAM), _closed("adatron", "ax"), _PTS, "rhs")),
        sym("AdaTron: <F b (y - rho x)>", sp.Eq(sp.Symbol("D_a"), (1 - 2 * LAM) * s**3 / sp.pi),
            source=_BR, check=Check((RHO, LAM), _closed("adatron", "drive"), _PTS, "rhs")),
        sym("Optimal modulation per unit |J|, u = rho Delta / sqrt(1 - rho^2) (VKC Eq. (25), chi = 2 lambda; "
            "BR Eq. (10) at lambda = 0): optimal among single-example modulated Hebbian rules that know rho",
            sp.Eq(sp.Symbol("W^*"), wstar), source=_VKC,
            check=Check((D, RHO, LAM), lambda d, rho, lam: float(OPT.w_star(
                float(d), math.acos(float(rho)), float(lam))),
                (("-0.5", "0.3", "0"), ("0.2", "0.8", "0.1"), ("-2", "0.95", "0.3")), "rhs")),
        cited("Asymptote of the optimal rule with the noise known, eps -> C / alpha (derived here; Copelli et "
              "al. Eq. (8) print Dt e^{-t^2}, which gives C(0) = 1.325, not the 0.88 of VKC Tables I-II)",
              r"C(\lambda) = \frac{2}{I(\lambda)},\quad I(\lambda) = (1-2\lambda)^2 \int Dt\, "
              r"\frac{e^{-t^2/2}}{\lambda + (1-2\lambda) H(t)}", _CE,
              check=Check((LAM,), lambda lam: OPT.asymptotic_constant(float(lam)),
                          (("0",), ("0.1",), ("0.3",)), display=const_display, tol="1e-10")),
        sym("No-learning line in the noise estimate Lambda (Copelli et al. p. 5, Fig. 1a dashed)",
            sp.Eq(sp.Symbol("Lambda_d"), 2 * LAM - sp.Rational(1, 2)), source=_CE,
            check=Check((LAM,), lambda lam: OPT.noise_boundaries(float(lam)).no_learning,
                        (("0.3",), ("0.35",), ("0.4",)), "rhs", tol="1e-9")),
        cited("Asymptotes (BR Eqs. (7), (8), (15); VKC Tables I-II for the optimal rule)",
              r"\epsilon_{Hebb} \simeq \frac{\alpha^{-1/2}}{\sqrt{2\pi}},\quad \epsilon_{perc} \simeq "
              r"\frac{1}{\pi}\left(\frac{2}{3}\right)^{1/3}\alpha^{-1/3},\quad \epsilon_{AdaTron} \simeq "
              r"\frac{3}{2\alpha},\quad \epsilon_{opt} \simeq \frac{0.88}{\alpha}", _BR),
        text("Finite-N checks", "online learning simulated at finite N over named seeds (online/simulate.py, "
             "no code shared with the ODEs), 99% Student-t bands; a planted ODE without the eta^2 term "
             "falls outside them (statistical)"),
    ]


__all__ = ["online_perceptron"]
