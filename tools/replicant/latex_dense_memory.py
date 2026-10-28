"""Displayed equations of the `dense_memory` and `dense_memory_ising` catalog entries (replica route).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against dense_memory/*.py at three points
(tests/test_latex.py); the equations the code cannot evaluate in closed form (the all-patterns bound's integral
structure) are cited LaTeX strings.  Equation numbers are those of Lucibello & Mezard, Phys. Rev. Lett. 132, 077301
(2024), arXiv:2304.14964v4; the binary ensemble and the Ising thresholds are marked DERIVED HERE.
"""
import mpmath as mp
import sympy as sp

from .dense_memory import ensembles as EN
from .dense_memory import ising as IS
from .dense_memory import rem_noise as RN
from .dense_memory import thresholds as TH
from .tex import Check, cited, sym, text

A, LM_, RH, M, X, EPS = sp.symbols("alpha lambda rho m x epsilon", positive=True)
_LM = "Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4"
_DH = "Demircigil, Heusel, Lowe, Upgang & Vermet, J. Stat. Phys. 168, 288 (2017), arXiv:1702.01929v2"
PHI = sp.Function("phi")
LAMS, EPSS = sp.Symbol("lambda_*", positive=True), sp.Symbol("epsilon_*", positive=True)
Q = sp.sqrt(1 + 4 * LM_**2 * RH**2)
ZETA_SPH = (Q - 1 - sp.log((1 + Q) / 2)) / 2
_PT = (("0.2", "0.5", "1"), ("0.5", "2", "0.8"), ("0.1", "3", "1.3"))      # (alpha, lambda, rho)
_PTL = (("0.5", "1"), ("1.5", "0.8"), ("3", "1.2"))                         # (lambda, rho)
_PAR = (("0.1", "1"), ("0.5", "0.8"), ("1", "1.5"))                         # (alpha, rho)
_PAR2 = (("0.1", "1"), ("0.5", "0.8"), ("0.2", "1.5"))
_PEPS = (("0.3", "1"), ("0.6", "0.9"), ("0.1", "2"))                        # (epsilon, rho)


def dense_memory(route):
    g, sp_ = "gaussian", "spherical"
    zg = LM_**2 * RH**2 / 2
    sg = EPS**2 / (2 * RH**2)
    ss = -sp.log(1 - EPS**2 / RH**2) / 2
    lam_star_g = sp.sqrt(2 * A) / RH
    lam_star_s = sp.exp(2 * A) * sp.sqrt(1 - sp.exp(-2 * A)) / RH
    alpha1_g = LM_ * (1 - LM_ / 2)
    alpha1_s = LM_ - ZETA_SPH.subs(RH, 1)
    return [
        text("Scope", "E(x) = -(1/lam) log sum_{mu=1}^{P} exp(lam x.xi^mu) + |x|^2/2 with P = e^{alpha N} patterns; "
             "for an x independent of patterns mu >= 2 the sum over them is the free energy of a Random Energy Model "
             "with levels eps_mu = x.xi^mu/N and inverse temperature lam (Eq. (4)); the paper's analysis is at zero "
             "temperature, through E(x) ~ -max(x.xi^1, N phi) + |x|^2/2 (Eq. (6)); no finite-temperature replica-"
             "symmetric free energy of the memory is derived there", source=_LM),
        text("Reuse of the REM entry", "the label-replica calculation below is the derivation of the `rem` entry "
             "(derive_rem.py) with 2^N levels replaced by e^{alpha N} and the Gaussian cumulant by zeta; at alpha = "
             "log 2 and Gaussian levels phi equals -f_REM(beta = lam, N_scale = rho^2) (step dm-rem-reduction)"),
        sym("Gaussian patterns: cumulant generating function (D4)", sp.Eq(sp.Symbol("zeta_g"), zg), source=_LM,
            check=Check((LM_, RH), lambda l, r: EN.zeta(g, l, r), _PTL, "rhs")),
        sym("Gaussian patterns: rate function of the overlap (D3)", sp.Eq(sp.Symbol("s_g"), sg), source=_LM,
            check=Check((EPS, RH), lambda e, r: EN.rate(g, e, r), _PEPS, "rhs")),
        sym("Spherical patterns: cumulant generating function (C3)", sp.Eq(sp.Symbol("zeta_s"), ZETA_SPH), source=_LM,
            check=Check((LM_, RH), lambda l, r: EN.zeta(sp_, l, r), _PTL, "rhs")),
        sym("Spherical patterns: rate function of the overlap (C4)", sp.Eq(sp.Symbol("s_s"), ss), source=_LM,
            check=Check((EPS, RH), lambda e, r: EN.rate(sp_, e, r), _PEPS, "rhs")),
        cited("Rate function and cumulant are Legendre transforms (A13)-(A14); condensation line and noise "
              "(A17)", r"\zeta_\rho(\lambda) = \sup_\epsilon\, \lambda\epsilon - s_\rho(\epsilon),\quad "
              r"\lambda_* = s_\rho'(\epsilon_*),\ \ s_\rho(\epsilon_*) = \alpha,\quad "
              r"\phi_{\alpha,\rho}(\lambda) = \begin{cases} (\alpha+\zeta_\rho(\lambda))/\lambda & \lambda<\lambda_*\\ "
              r"\epsilon_* & \lambda \ge \lambda_* \end{cases}", _LM),
        sym("Gaussian: condensation threshold (D6)", sp.Eq(LAMS, lam_star_g), source=_LM,
            check=Check((A, RH), lambda a, r: RN.condensation_lambda(g, a, r), _PAR, "rhs")),
        sym("Gaussian: noise, uncondensed phase, lam < lam_* (Eq. (5), (D5))",
            sp.Eq(PHI(A, LM_), A / LM_ + LM_ * RH**2 / 2), source=_LM,
            check=Check((A, LM_, RH), lambda a, l, r: RN.phi(g, a, l, r), (("0.1", "0.2", "1"), ("0.3", "0.5", "1"),
                                                                           ("0.5", "0.4", "1.2")), "rhs")),
        sym("Gaussian: noise, frozen phase, lam >= lam_* (Eq. (5), (D5))",
            sp.Eq(PHI(A, LM_), RH * sp.sqrt(2 * A)), source=_LM,
            check=Check((A, RH), lambda a, r: RN.phi(g, a, 50, r), _PAR2, "rhs")),
        sym("Spherical: condensation threshold (C5)", sp.Eq(LAMS, lam_star_s), source=_LM,
            check=Check((A, RH), lambda a, r: RN.condensation_lambda(sp_, a, r), _PAR, "rhs")),
        sym("Spherical: noise, frozen phase (C5)", sp.Eq(PHI(A, LM_), RH * sp.sqrt(1 - sp.exp(-2 * A))), source=_LM,
            check=Check((A, RH), lambda a, r: RN.phi(sp_, a, 50, r), _PAR2, "rhs")),
        sym("Spherical: noise, uncondensed phase (the line above (C5))",
            sp.Eq(PHI(A, LM_), (A + ZETA_SPH) / LM_), source=_LM,
            check=Check((A, LM_, RH), lambda a, l, r: RN.phi(sp_, a, l, r), (("0.1", "0.2", "1"), ("0.3", "0.5", "1"),
                                                                             ("0.5", "0.4", "1.2")), "rhs")),
        text("1RSB over labels (A24)-(A27)", "lam phi = min_{0<m<=1} (alpha + zeta(lam m))/m; stationarity "
             "lam m zeta'(lam m) - zeta(lam m) = alpha; m = 1 for lam < lam_*, m = lam_*/lam above; the second "
             "derivative at the stationary point is lam^2 zeta''(lam m)/m > 0, a minimum (step dm-extremize)",
             source=_LM),
        text("Single-pattern retrieval threshold (Eq. (7))", "alpha_1(lam) = sup{alpha : phi_{alpha,1}(lam) < 1}; "
             "solved here by bisection on phi and, independently, by Legendre duality: s(1) if lam >= s'(1), else "
             "lam - zeta(lam)", source=_LM),
        sym("Gaussian: alpha_1 for lam < 1 (D7); alpha_1 = 1/2 for lam >= 1", sp.Eq(sp.Symbol("alpha_1"), alpha1_g),
            source=_LM, check=Check((LM_,), lambda l: TH.alpha_1(g, l), (("0.2",), ("0.5",), ("0.9",)), "rhs")),
        sym("Spherical: alpha_1 = lam - zeta(lam) (always uncondensed: s' = inf at the edge)",
            sp.Eq(sp.Symbol("alpha_1"), alpha1_s), source=_LM,
            check=Check((LM_,), lambda l: TH.alpha_1(sp_, l), (("0.2",), ("1",), ("5",)), "rhs")),
        cited("Gaussian all-patterns lower bound: alpha_c^lb solves A(alpha, lam) = alpha, with r_0^2 = alpha/(lam(1 - "
              "lam/2)) (lam < 1), 2 alpha (lam >= 1); (D13)",
              r"A = \min\left(\min_{r\le r_0}\tilde I(r),\ "
              r"\min_{r\ge r_0}\left[\tilde I(r)+\tfrac{r^2}{2}-\alpha\right]\right),"
              r"\quad \tilde I(r) = \tfrac{r^2-1}{2}-\log r", _LM),
        text("All-patterns capacity", "alpha_c^lb = log(2)/4 for lam >= 0.70091... (the printed value; computed "
             "0.700917829...); spherical: alpha_c = alpha_1 (C6)-(C7)", source=_LM),
        text("Basins (Sec. IV)", "cos theta_c = phi_{alpha,rho}(lam)/rho; large lam: cos theta_c = rho c_max(alpha) "
             "with c_max = eps*(alpha, 1) the Gilbert-Varshamov cosine (App. B); alpha << 1: sqrt(2 log P/N) (B8)",
             source=_LM),
        sym("Ramsauer et al. bound as restated in App. E, Eq. (E7), spherical patterns",
            sp.Eq(sp.Symbol("alpha_c_rams"), sp.log((2 * LM_ / 5) / sp.LambertW(2 * LM_ / 5)) / 4), source=_LM,
            check=Check((LM_,), lambda l: TH.ramsauer_lower_bound(l), (("0.5",), ("2",), ("30",)), "rhs")),
        text("Scaled dot product (Sec. VI)", "lam = lam~/N^a, P = exp(alpha~ N^{1-a}): alpha~_1 = alpha~_c = lam~, for "
             "any pattern ensemble", source=_LM),
    ]


def dense_memory_ising(route):
    b = "binary"
    zb = sp.log(sp.cosh(LM_))
    sb = ((1 + EPS) * sp.log(1 + EPS) + (1 - EPS) * sp.log(1 - EPS)) / 2
    return [
        text("Scope", "Ising neurons x in {-1,+1}^N, +-1 patterns, F(x) = exp(lam x); the zero-temperature single-flip "
             "dynamics of Demircigil et al. (Eq. (1)) is the descent of E(x) = -(1/lam) log sum_mu exp(lam x.xi^mu) "
             "on the hypercube.  The REM noise with binary levels is DERIVED HERE by the recipe of Lucibello & "
             "Mezard (A16)-(A17) (they only say binary patterns should be treatable); only the fixed-point "
             "exponent below is a published theorem", source=_DH),
        sym("Binary levels: cumulant generating function, E exp(lam x_i xi_i) = cosh(lam) per site (DERIVED HERE)",
            sp.Eq(sp.Symbol("zeta_b"), zb), source=_LM,
            check=Check((LM_,), lambda l: EN.zeta(b, l), (("0.3",), ("1",), ("4",)), "rhs")),
        sym("Binary levels: rate function = the function I of Demircigil et al. Thm 1.3",
            sp.Eq(sp.Symbol("s_b"), sb), source=_DH,
            check=Check((EPS,), lambda e: EN.rate(b, e), (("0.2",), ("0.6",), ("0.95",)), "rhs")),
        sym("Binary levels: noise, uncondensed phase (alpha < log 2 required for a condensed phase)",
            sp.Eq(PHI(A, LM_), (A + zb) / LM_), source=_LM,
            check=Check((A, LM_), lambda a, l: RN.phi(b, a, l),
                        (("0.2", "0.5"), ("0.3", "0.7"), ("0.1", "0.4")), "rhs")),
        sym("Binary levels: energy criterion phi < 1, alpha_1 = lam - log cosh(lam) (DERIVED HERE)",
            sp.Eq(sp.Symbol("alpha_1"), LM_ - zb), source=_LM,
            check=Check((LM_,), lambda l: IS.alpha_1_energy(l), (("0.5",), ("1",), ("3",)), "rhs")),
        sym("Single-flip stability of a typical pattern: the decision sum has fair random signs, so its size is the "
            "noise at 2 lam, alpha_1^flip = 2 lam - log cosh(2 lam) (DERIVED HERE; sampled in sim.sign_noise_mc)",
            sp.Eq(sp.Symbol("alpha_1^flip"), 2 * LM_ - sp.log(sp.cosh(2 * LM_))), source=_DH,
            check=Check((LM_,), lambda l: IS.alpha_1_flip(l), (("0.5",), ("1",), ("3",)), "rhs")),
        sym("Fixed-point exponent of Demircigil et al. (Thm 1.3, Remark 1.4; proved for lam = 1)",
            sp.Eq(sp.Symbol("alpha_fp"), sb.subs(EPS, 1 - 2 * RH) / 2), source=_DH,
            check=Check((RH,), lambda r: IS.fixed_point_exponent(r), (("0.05",), ("0.1",), ("0.25",)), "rhs")),
        text("Mechanism", "log(2)/2 is the birthday threshold of pattern pairs within Hamming distance 1 "
             "(M^2 N / 2^N ~ 1); the exact expected number of such pairs is C(P,2)(1 + N)/2^N", source=_DH),
        text("Polynomial interactions", "refused (unsupported:interaction): F(x) = x^p stores O(N^{p-1}) patterns, so "
             "P = e^{alpha N} is outside its regime; its thermodynamics is the p-spin glass (entries `pspin`, "
             "`ising_pspin`)"),
    ]


__all__ = ["dense_memory", "dense_memory_ising"]
