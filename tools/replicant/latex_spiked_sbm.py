"""Displayed equations of the `spiked_wigner` (amp route) and `sbm` (cavity route) catalog entries
(E6b).  Same rule as latex_entries.py: each sympy block carries a tex.Check against
inference/spiked.py or inference/sbm.py at three points (tests/test_latex.py); the algorithm and
theorem statements are cited strings.
"""
import mpmath as mp
import numpy as np
import sympy as sp

from .tex import Check, cited, sym, text

LAM, G, P = sp.symbols("lambda gamma p", positive=True)
A, B_, C = sp.symbols("a b c", positive=True)
_MV = "Montanari & Venkataramanan, Ann. Statist. 49, 321 (2021), arXiv:1711.01682"
_LM = "Lelarge & Miolane, Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888"
_DAM = "Deshpande, Abbe & Montanari, arXiv:1507.08685 (2015)"
_DKMZ = "Decelle, Krzakala, Moore & Zdeborova, Phys. Rev. E 84, 066106 (2011), arXiv:1109.3041"
_MNS = ("Mossel, Neeman & Sly, arXiv:1202.1499 (2015) and arXiv:1311.4115 (2018); Massoulie, "
        "arXiv:1311.3085 (2014)")


def _rad_mmse_display(g):
    with mp.workdps(25):
        g = mp.mpf(g)
        f = lambda z: mp.tanh(g + mp.sqrt(g) * z)**2 * mp.npdf(z)
        return 1 - mp.quad(f, [-mp.inf, -mp.sqrt(g), 0, mp.inf])


def spiked_wigner(route):
    from .inference import spiked as S
    gmm = 1 / (1 + G)
    pts_l = (("1.5",), ("2",), ("3",))
    return [
        text("AMP route", "Y = (lambda/N) x x^T + W with W ~ GOE(N) (off-diagonal variance 1/N, "
             "diagonal 2/N), x i.i.d. from a zero-mean unit-variance prior; scalar channel "
             "y = gamma X + sqrt(gamma) G with posterior-mean denoiser F (the PRIORS hook of "
             "inference/spiked.py); Lelarge-Miolane's lambda_LM is lambda^2", source=_MV),
        sym("Gaussian prior: scalar MMSE of the posterior mean F(y) = y/(1 + gamma)",
            sp.Eq(sp.Function("mmse")(G), gmm), source=_LM,
            check=Check((G,), lambda g: S.scalar_mmse("gaussian", float(g)),
                        (("0.3",), ("1",), ("4",)), "rhs")),
        cited("Rademacher prior: scalar MMSE, F(y) = tanh(y) (DAM Eq. (7))",
              r"\mathrm{mmse}(\gamma) = 1 - \mathbb{E}\,\tanh^2\!\left(\gamma + \sqrt{\gamma}\,Z\right)",
              _DAM, check=Check((G,), lambda g: S.scalar_mmse("rademacher", float(g)),
                                (("0.3",), ("1",), ("4",)), display=_rad_mmse_display, tol="1e-10")),
        sym("State evolution, gaussian prior (MV Eq. (2.24))",
            sp.Eq(sp.Symbol("gamma_{t+1}"), LAM**2 * (1 - gmm)), source=_MV,
            check=Check((LAM, G), lambda l, g: float(l)**2 * (1 - S.scalar_mmse("gaussian", float(g))),
                        (("1.5", "0.5"), ("2", "3"), ("3", "1")), "rhs")),
        sym("Gaussian prior: the fixed point, q = gamma*/lambda^2 (LM Sec. 2.3: q* = 1 - 1/lambda_LM)",
            sp.Eq(sp.Symbol("q"), 1 - 1 / LAM**2), source=_LM,
            check=Check((LAM,), lambda l: S.spiked_state_evolution(l, "gaussian").q, pts_l, "rhs")),
        sym("Gaussian prior: matrix MMSE 1 - q^2 (LM Eq. (10))",
            sp.Eq(sp.Symbol("MMSE"), (2 - 1 / LAM**2) / LAM**2), source=_LM,
            check=Check((LAM,), lambda l: S.spiked_state_evolution(l, "gaussian").matrix_mmse,
                        pts_l, "rhs")),
        sym("RS potential, gaussian prior (DAM Eq. (9), lambda_DAM = lambda^2; minimised)",
            sp.Eq(sp.Symbol("Psi"), LAM**2 / 4 + G**2 / (4 * LAM**2) - G / 2 + sp.log(1 + G) / 2),
            source=_DAM,
            check=Check((LAM, G), lambda l, g: S.spiked_potential(l, g, "gaussian"),
                        (("1.5", "1.25"), ("2", "1"), ("3", "8")), "rhs")),
        cited("Bayes AMP with spectral start (MV Eqs. (2.20)-(2.21))",
              r"x^0 = \sqrt{N\lambda^2(\lambda^2-1)}\,\varphi_1,\quad x^{t+1} = Y f_t(x^t) - b_t f_{t-1}"
              r"(x^{t-1}),\quad f_t = \lambda F(\cdot;\gamma_t),\quad b_t = \frac{1}{N}\sum_i f_t'(x^t_i)",
              _MV),
        cited("Threshold for nontrivial estimation (both priors)",
              r"\lambda_c = 1:\quad q^* = 0 \text{ for } \lambda \le 1,\quad q^* > 0 \text{ for } \lambda > 1",
              f"{_LM}, Sec. 2.3, Prop. 4; {_DAM}, Thm 4.3 and Lemma 3.3"),
        text("Finite-N checks", "Bayes AMP on sampled matrices over named seeds: |<xhat, x>|/N, the "
             "cosine, the sign-resolved and the matrix MSE with 99% Student-t bands (statistical)"),
    ]


def sbm(route):
    from .inference import sbm as B
    u = sp.log((A * P + B_ * (1 - P)) / (B_ * P + A * (1 - P)))
    pts = (("5.5", "0.5"), ("4", "2"), ("1", "6"))
    return [
        text("Cavity route", "two groups, sigma_i = +-1 uniform; edge probability a/N inside a group, "
             "b/N across (DKMZ c_in = a, c_out = b, mean degree c = (a + b)/2); BP messages "
             "p = psi^{i->j}_+ in log-ratio form (inference/sbm.py)", source=_DKMZ),
        sym("Message weight of a neighbour (DKMZ Eq. (26), q = 2)", sp.Eq(sp.Symbol("u(p)"), u),
            source=_DKMZ,
            check=Check((A, B_, P), lambda a, b, p: float(B._u(float(a), float(b), np.array(float(p)))),
                        (("5.5", "0.5", "0.3"), ("4", "2", "0.9"), ("1", "6", "0.5")), "rhs")),
        cited("BP update with the external field (DKMZ Eqs. (26)-(28))",
              r"L_{i\to j} = -\Delta h + \sum_{k\in\partial i\setminus j} u(p_{k\to i}),\quad "
              r"\Delta h = \frac{a-b}{N}\sum_k \left(2\psi^{k}_{+} - 1\right),\quad "
              r"L_i = -\Delta h + \sum_{k\in\partial i} u(p_{k\to i})", _DKMZ),
        sym("Kesten-Stigum margin (DKMZ Eq. (44) at q = 2: |a - b| > 2 sqrt(c))",
            sp.Eq(sp.Symbol("tau"), (A - B_)**2 / (2 * (A + B_))), source=_DKMZ,
            check=Check((A, B_), lambda a, b: B.sbm_kesten_stigum(a, b).margin, pts, "rhs")),
        sym("KS point at mean degree c: a/b = 1/eps_c (DKMZ Fig. 1 caption)",
            sp.Eq(sp.Symbol("a_KS/b_KS"), (C + sp.sqrt(C)) / (C - sp.sqrt(C))), source=_DKMZ,
            check=Check((C,), lambda c: B.sbm_ks_line(c).ratio, (("2",), ("3",), ("5",)), "rhs")),
        cited("Kesten-Stigum threshold (theorems)",
              r"(a-b)^2 \le 2(a+b),\ a+b>2:\ \text{no correlated estimator};\qquad "
              r"(a-b)^2 > 2(a+b):\ \text{an efficient correlated estimator exists}", _MNS),
        text("Finite-N checks", "BP from i.i.d. uniform messages on sampled graphs over named seeds: "
             "overlap |(1/N) sum sigma_i sigmahat_i| with a 99% Student-t band (statistical)"),
    ]


__all__ = ["sbm", "spiked_wigner"]
