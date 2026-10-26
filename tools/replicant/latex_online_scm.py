"""Displayed equations of the `online_scm` catalog entry (route online, O5).  Same rule as latex_entries.py:
each sympy block carries a tex.Check against online/ at three points (tests/test_latex.py); the model, the
rate equations and the papers' statements without a closed form are cited.
"""
import numpy as np
import sympy as sp

from .tex import Check, cited, sym, text

C11, C12, C13, C14, C22, C23, C24, C33, C34, C44 = sp.symbols("C11 C12 C13 C14 C22 C23 C24 C33 C34 C44", real=True)
KS, ETA = sp.symbols("K eta", positive=True)
_SS = "Saad & Solla, NIPS 8, 302 (1996) and MANNA (1997), restating PRL 74, 4337 and PRE 52, 4225 (1995)"
_BS = "Biehl & Schwarze, J. Phys. A 28, 643 (1995)"
_BRW = "Biehl, Riegler & Woehler, J. Phys. A 29, 4769 (1996)"


def _m(*rows):
    return np.array([[float(x) for x in r] for r in rows])


def online_scm(route):
    from .online import scm as S
    from .online import scm_flow as F
    trunc = lambda k: F.symmetric_fixed_point(S.scm_flow(int(k), int(k), 1.0, truncated=True), "identical")
    l4 = (1 + C11) * (1 + C22) - C12 ** 2
    l0 = l4 * C34 - C23 * C24 * (1 + C11) - C13 * C14 * (1 + C22) + C12 * C13 * C24 + C12 * C14 * C23
    l1 = l4 * (1 + C33) - C23 ** 2 * (1 + C11) - C13 ** 2 * (1 + C22) + 2 * C12 * C13 * C23
    l2 = l4 * (1 + C44) - C24 ** 2 * (1 + C11) - C14 ** 2 * (1 + C22) + 2 * C12 * C14 * C24
    etac = sp.pi * sp.sqrt(sp.Rational(5, 3))
    mode = lambda e, which: getattr(F.optimum_modes(S.scm_flow(1, 1, float(e))), which)
    return [
        text("Online route, soft committee machine", "K student and M teacher hidden units g(x) = erf(x/sqrt 2) with "
             "unit hidden-to-output weights, inputs xi_j i.i.d. N(0, 1), eps = (sigma - tau)^2/2, update J_i <- J_i + "
             "(eta/N) delta_i xi with delta_i = g'(x_i)[sum_n g(y_n) - sum_j g(x_j)], alpha = examples/N, Q_ik = J_i . "
             "J_k, R_in = J_i . B_n, T_nm = B_n . B_m (Saad & Solla's labels: K student, M teacher); Biehl-Schwarze's "
             "Q is |J| and their beta = 1 two-unit model is K = 2, M = 1; the N -> oo limit is self-averaging by "
             "assumption", source=_SS),
        cited("Equations of motion (SS NIPS Eq. (4); MANNA Eq. (4)), with signs s = -1 for student and +1 for teacher units",
              r"\frac{dR_{in}}{d\alpha} = \eta\sum_u s_u I_3(i,n,u),\quad \frac{dQ_{ik}}{d\alpha} = \eta\sum_u s_u"
              r"[I_3(i,k,u) + I_3(k,i,u)] + \eta^2\sum_{uv} s_u s_v I_4(i,k,u,v),\quad \epsilon_g = \tfrac12\sum_{uv}"
              r"s_u s_v I_2(u,v)", _SS),
        sym("I2 = <g(u) g(v)> (from SS NIPS Eq. (2)); checked against online/scm.i2 (and, in the tests, against "
            "Gauss-Hermite)", sp.Eq(sp.Symbol("I_2"), 2 / sp.pi * sp.asin(C12 / sp.sqrt((1 + C11) * (1 + C22)))),
            source=_SS, check=Check((C11, C22, C12), lambda a, b, c: float(S.i2(_m((a, c), (c, b)))),
                                    (("0.5", "0.3", "0.1"), ("1", "1", "0.9"), ("0.2", "2", "-0.4")), "rhs")),
        sym("I3 = <g'(u) v g(w)> (SS MANNA Eq. (5))",
            sp.Eq(sp.Symbol("I_3"), 2 / sp.pi * (C23 * (1 + C11) - C12 * C13) / ((1 + C11) * sp.sqrt((1 + C11) * (1 + C33) - C13 ** 2))),
            source=_SS, check=Check((C11, C12, C13, C23, C33), lambda a, b, c, d, e: float(S.i3(_m((a, b, c), (b, 1, d), (c, d, e)))),
                                    (("0.5", "0.2", "0.1", "0.3", "1"), ("0.3", "-0.1", "0.25", "0.4", "0.6"),
                                     ("1", "0.5", "0.5", "0.5", "1")), "rhs")),
        sym("I4 = <g'(u) g'(v) g(w) g(z)> (SS MANNA Eq. (6))",
            sp.Eq(sp.Symbol("I_4"), 4 / sp.pi ** 2 * sp.asin(l0 / sp.sqrt(l1 * l2)) / sp.sqrt(l4)), source=_SS,
            check=Check((C11, C12, C13, C14, C22, C23, C24, C33, C34, C44),
                        lambda a, b, c, d, e, f, g, h, i, j: float(S.i4(_m((a, b, c, d), (b, e, f, g), (c, f, h, i), (d, g, i, j)))),
                        (("0.5", "0.1", "0.2", "0.1", "0.4", "0.1", "0.3", "1", "0.2", "1"),
                         ("0.3", "0.2", "0.25", "0.25", "0.3", "0.25", "0.25", "1", "0", "1"),
                         ("1", "0.3", "0.6", "0.1", "0.7", "0.2", "0.5", "1", "0.3", "1")), "rhs")),
        sym("Small-eta symmetric plateau (SS NIPS Eqs. (6)-(7); 0.0203 at K = 3); checked against the Newton root of the "
            "eta^2-truncated ODE, where also Q* = C* = 1/(2K - 1), R* = 1/sqrt(K(2K - 1))",
            sp.Eq(sp.Symbol("epsilon_g^*"), KS / sp.pi * (sp.pi / 6 - KS * sp.asin(1 / (2 * KS)))), source=_SS,
            check=Check((KS,), lambda k: trunc(k).eps, (("2",), ("3",), ("4",)), "rhs", tol="1e-9")),
        sym("Escape time of the small-eta symmetric plateau, in units of 1/eta (SS NIPS p. 307); checked against 1/lambda_esc "
            "of the truncated ODE's Jacobian", sp.Eq(sp.Symbol("tau"), sp.pi / (2 * KS) * sp.sqrt(2 * KS - 1) * (2 * KS + 1) ** sp.Rational(3, 2)),
            source=_SS, check=Check((KS,), lambda k: 1 / trunc(k).lam_esc, (("2",), ("3",), ("4",)), "rhs", tol="1e-6")),
        sym("Single graded unit (K = M = 1), the faster asymptotic mode at the optimum (BS Eq. (12)); checked against the "
            "ODE's Jacobian", sp.Eq(sp.Symbol("lambda_1"), 4 * sp.sqrt(5) / 3 * ETA / etac * (ETA / etac - 1)), source=_BS,
            check=Check((ETA,), lambda e: mode(e, "lam1"), (("0.5",), ("2",), ("3.5",)), "rhs", tol="1e-6")),
        sym("Single graded unit, the slower mode (BS Eq. (12)); eta_opt = 2 eta_c/3 where lambda_1 = 2 lambda_2",
            sp.Eq(sp.Symbol("lambda_2"), -2 * sp.sqrt(5) * ETA / (9 * etac)), source=_BS,
            check=Check((ETA,), lambda e: mode(e, "lam2"), (("0.5",), ("2",), ("3.5",)), "rhs", tol="1e-6")),
        cited("eta_max of the realisable SCM to first order in 1/K (SS NIPS Eq. (9)); the exact K = 3 root is 1.6207 "
              "(Fig. 2: 1.62); SS's eta_opt: lambda_1(eta_opt) = 2 lambda_2(eta_opt), which at K = 3 is not the fastest "
              "asymptotic decay (replicant: crossing 0.917, fastest 1.280)",
              r"\eta_{max} = \frac{\pi}{K}\,\frac{75 - 42\sqrt3}{25\sqrt3 - 42} + O(K^{-2})", _SS),
        cited("Plateau length (BRW Eqs. (9)-(11), (16)-(17)): alpha_P is where the deviation of an order parameter from "
              "the fixed point exceeds B; with an R asymmetry X_R = sqrt(2X/N) the length grows as half the naive "
              "tau_esc ln(1/X)", r"\alpha_P - \alpha_0 = \tau_{esc}\Big(D - \tfrac12\ln(2X) - \tfrac12\ln N\Big),\quad "
              r"\tau_{esc} = 1/\lambda_{esc}", _BRW),
        text("Plateau reports and finite-N checks", "online/plateau.py: the unstable fixed point by Newton from the slowest "
             "point, BRW's entry/exit with B = 0.05, 'undetermined' when not left by alpha_max (nothing extrapolated), "
             "the escape alpha recomputed at rtol x 1e-2; online/simulate_scm.py: online gradient descent with no code "
             "shared with the ODEs, 99% Student-t bands; the ODE without its eta^2 term falls outside them"),
    ]


__all__ = ["online_scm"]
