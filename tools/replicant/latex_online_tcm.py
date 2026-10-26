"""Displayed equations of the `online_tcm` catalog entry (route online, O4).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against online/ at three points
(tests/test_latex.py); the model, the rate equations and the papers' results are cited.
"""
import math

import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym, text

AL, E1, E2, E3, KS = sp.symbols("alpha e_1 e_2 e_3 K", real=True)
_CC = "Copelli & Caticha, J. Phys. A 28, 1615 (1995)"
_CKC = "Copelli, Kinouchi & Caticha, Phys. Rev. E 53, 6341 (1996) (abstract)"


def _kappa_formula():
    with mp.workdps(25):
        H = lambda x: mp.erfc(x / mp.sqrt(2)) / 2
        return 2 / mp.quad(lambda x: mp.npdf(x) * mp.e ** (-x * x / 2) / H(x), [-mp.inf, 0, 5, mp.inf])


def online_tcm(route):
    from .online import tcm as T
    from .online import tcm_flow as TF
    p0 = sp.binomial(KS - 1, (KS - 1) / 2) / 2 ** (KS - 1)
    hebb_rho = lambda a, k, eta: float(TF.integrate_tcm(TF.tcm_flow("hebb", int(k), eta=float(eta)), float(a),
                                                         (0.0, 1e-12), alphas=(float(a),)).rho[0])
    return [
        text("Online route, tree committee machine", "K odd non-overlapping branches of N/K inputs (N(0, 1); CC95 use "
             "+-1), teacher branches |B_k| = 1, label sigma_B = sign(sum_k sign(B_k . xi_k)) (CC95 Eq. (1)), update "
             "J_k <- J_k + (1/N) F_k xi_k (Eq. (6), Omega_k = 0), alpha = examples / N (CKC96's alpha_CM), e_g = "
             "P(student label != teacher label), branch-symmetric overlaps rho_k = rho; the N -> oo limit is "
             "self-averaging by assumption", source=_CC),
        cited("Branch flow (CC95 Eqs. (7)-(8) with R = J_k . B_k, Q = |J_k|^2; the 1/K because |xi_k|^2 = N/K)",
              r"\frac{dR}{d\alpha} = \langle F_k\, y_k\rangle,\quad \frac{dQ}{d\alpha} = 2\sqrt{Q}\,\langle F_k\, x_k"
              r"\rangle + \frac{\langle F_k^2\rangle}{K}", _CC),
        sym("Committee error from the branch errors e_k = arccos(rho_k)/pi, K = 3 (CC95 Eq. (5))",
            sp.Eq(sp.Symbol("e_g"), E1 * E2 * E3 + (E1 + E2 + E3 - E1 * E2 - E1 * E3 - E2 * E3) / 2), source=_CC,
            check=Check((E1, E2, E3), lambda a, b, c: T.eps_committee((float(a), float(b), float(c)), 3),
                        (("0.1", "0.2", "0.3"), ("0.01", "0.4", "0.05"), ("0.5", "0.25", "0")), "rhs")),
        sym("Pseudo-Hebb from rho(0) = 0, K = 3 (CC95 Eq. (30)); checked against the TCM Hebb ODE from Q0 = 1e-12",
            sp.Eq(sp.Symbol("rho"), (1 + 2 * sp.pi / (3 * AL)) ** sp.Rational(-1, 2)), source=_CC,
            check=Check((AL,), lambda a: hebb_rho(a, 3, 1.0), (("1",), ("5",), ("20",)), "rhs", tol="1e-8")),
        sym("Hebb from J = 0 at any odd K (derived here: <sigma_B y_k> = P0 sqrt(2/pi), P0 the tie probability of "
            "the other K - 1 hidden units); K = 3 is Eq. (30)",
            sp.Eq(sp.Symbol("rho"), (1 + sp.pi / (2 * KS * p0 ** 2 * AL)) ** sp.Rational(-1, 2)),
            source="derived here from " + _CC,
            check=Check((AL, KS), lambda a, k: hebb_rho(a, k, 1.0), (("2", "1"), ("5", "5"), ("1", "9")), "rhs",
                        tol="1e-8")),
        sym("Mapped noise of local learning (derived here; CKC96's epsilon(K) not read): branch k sees its "
            "teacher unit through a label flipped with probability (1 - P0)/2, on N/K inputs, at alpha_p = K alpha",
            sp.Eq(sp.Symbol("lambda_K"), (1 - p0) / 2), source="derived here from " + _CKC,
            check=Check((KS,), lambda k: T.effective_noise(int(k)), (("3",), ("5",), ("11",)), "rhs")),
        cited("Optimal weight (CC95 Eqs. (9)-(12), (19)-(20); optimal only in tcm.OPTIMALITY_CLASS: greedy, true rho): F_k = K J_k (<b_k>/rho_k - h_k), the teacher field "
              "averaged over its posterior given sigma_B and all student fields (nonlocal); lambda_k = sqrt(1 - "
              "rho_k^2)/rho_k, y_k = Delta_k/lambda_k; replicant's posterior route agrees with Eq. (21) to 2e-7",
              r"\tilde F_1 = \frac{3}{\sqrt{2\pi}} J_1\lambda_1\sigma_B e^{-\Delta_1^2/2\lambda_1^2}\, "
              r"\frac{D(y_2,y_3)}{H(-y_2)H(-y_3)+H(-y_1)D(y_2,y_3)},\ D = H(-y_2)H(y_3)+H(y_2)H(-y_3)", _CC),
        cited("Asymptotic constant of the optimal rule (CC95 Eq. (32): ~0.88, the perceptron's); checked against "
              "the TCM rate limit", r"e_g \simeq \frac{\kappa}{\alpha},\quad \kappa = 2\Big[\int Dx\, "
              r"\frac{e^{-x^2/2}}{H(x)}\Big]^{-1} \approx 0.88", _CC,
              check=Check((), lambda: TF.rate_limit_constant(TF.tcm_flow("optimal", 3)), ((),),
                          display=_kappa_formula, tol="1e-4")),
        cited("CKC96 (abstract): local TCM learning equals noisy-perceptron learning; here kappa_loc(K) = P0 "
              "C(lambda_K), C = 2/I(lambda) (O1), which grows with K (2.665 at K = 3) while the nonlocal rule "
              "keeps 0.88", r"\rho_{TCM}(\alpha) = \rho_{perc}\big(K\alpha;\ \lambda_K\big),\quad "
              r"e_g \simeq \frac{P_0\,C(\lambda_K)}{\alpha}", _CKC),
        text("Finite-N checks", "online TCM learning in the simulation (online/simulate_tcm.py: CC95 Eqs. (19)-(20) as "
             "printed for the nonlocal rule; no code shared with the ODEs), 99% Student-t bands; an ODE with a wrong "
             "1/K in the eta^2 term falls outside them"),
    ]


__all__ = ["online_tcm"]
