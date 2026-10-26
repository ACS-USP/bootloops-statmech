"""Displayed equations of the `online_linear` catalog entry (route online, O2).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against online/ at three points
(tests/test_latex.py); the update rule, the rate equations and the paper's results are cited.
"""
import sympy as sp

from .tex import Check, cited, sym, text

AL, ETA, R0, Q0, RHO, Q, R, M = sp.symbols("alpha eta R_0 Q_0 rho Q R M", real=True)
_KC = "Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995)"
_HMRT = "Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949 (2022), Eq. (8)"


def _f(*xs):
    return [float(x) for x in xs]


def online_linear(route):
    from .online import linear as L
    from .online import linear_offline as OFF
    m = M / Q0
    top = 2 * sp.sqrt(m) / (1 + m)
    r0 = R0 / sp.sqrt(Q0)
    return [
        text("Online route", "Teacher |B|^2 = M, inputs xi with i.i.d. N(0, 1) components, linear outputs "
             "b = B . xi and h = J . xi, one fresh noiseless example per update, alpha = examples / N; R = J . B, "
             "Q = J . J, rho = R / sqrt(Q M), e_g = (M + Q - 2R)/2 (the quadratic generalization error). The "
             "sphere holds Q fixed by the decay Omega; the N -> oo limit is self-averaging by assumption", source=_KC),
        cited("Update rule and rate equations (KC Eqs. (5), (7)-(8) with I = 1; the sphere chooses Omega so "
              "that dQ/dalpha = 0)",
              r"J \leftarrow \left(1 - \frac{\Omega}{N}\right) J + \frac{\eta}{N} F\, \xi,\quad "
              r"\frac{dR}{d\alpha} = \eta \langle b F\rangle - \Omega R,\quad \frac{dQ}{d\alpha} = 2\eta "
              r"\langle h F\rangle + \eta^2 \langle F^2\rangle - 2\Omega Q", _KC),
        sym("Standard rule F = b - h (KC Eq. (22)) at constant eta, unconstrained, M = 1: dD/dalpha = "
            "(eta^2 - 2 eta) D for D = 2 e_g (derived)",
            sp.Eq(sp.Symbol("e_g"), (1 + Q0 - 2 * R0) * sp.exp((ETA**2 - 2 * ETA) * AL) / 2), source=_KC,
            check=Check((AL, ETA, R0, Q0), lambda al, eta, r0, q0: L.standard_closed_form(
                *_f(al, eta), init=_f(r0, q0))[2],
                (("1", "1", "0", "1"), ("3", "0.5", "0.1", "0.5"), ("2", "1.5", "0.2", "2")), "rhs")),
        sym("Standard rule on the sphere at eta = 1 (KC Eq. (23) solved; rho_max of Eq. (24)), checked "
            "against the integrated ODE",
            sp.Eq(RHO, top + (r0 / sp.sqrt(M) - top) * sp.exp(-(1 + m) * AL / 2)), source=_KC,
            check=Check((AL, R0, Q0, M), lambda al, r0, q0, mm: L.integrate_linear(
                L.linear_flow("standard", constraint="sphere", M=float(mm)), float(al), _f(r0, q0),
                alphas=(float(al),)).rho[0],
                (("1", "0", "1", "1"), ("2.5", "0.1", "4", "1"), ("0.7", "0.2", "0.5", "2")), "rhs", tol="1e-7")),
        sym("Optimal rule F = sqrt(Q/M) b / rho - h (KC Eq. (9)): drho/dalpha = (1 - rho^2)/(2 rho) (Eq. "
            "(17)), solved from rho0 (Eq. (19) at rho0 = 0), checked against the integrated ODE",
            sp.Eq(RHO, sp.sqrt(1 - (1 - sp.Symbol("rho_0")**2) * sp.exp(-AL))), source=_KC,
            check=Check((AL, sp.Symbol("rho_0")), lambda al, rr: L.integrate_linear(
                L.linear_flow("optimal", constraint="sphere"), float(al), (float(rr), 1.0),
                alphas=(float(al),)).rho[0],
                (("1", "0.001"), ("3", "0.2"), ("0.4", "0.5")), "rhs", tol="1e-7")),
        sym("KC's learning-rate parameter, the 'hypothesis temperature' (Eq. (13)); the optimal F is "
            "Gamma rho (b - rho h)/(1 - rho^2) on Q = M = 1 (Eqs. (12), (14), (15))",
            sp.Eq(sp.Symbol("Gamma"), (1 - RHO**2) / RHO**2), source=_KC,
            check=Check((RHO,), lambda rho: float(L.hypothesis_temperature(float(rho))),
                        (("0.2",), ("0.6",), ("0.95",)), "rhs")),
        sym("Greedy learning rate of the standard rule, derived here (not in KC95): argmax over eta of "
            "drho/dalpha = eta A - eta^2 B (1 on the tabula rasa manifold R = Q, where KC p. 9 find the standard "
            "rule optimal)",
            sp.Eq(sp.Symbol("eta^*"), (Q * M - R**2) / (R * (M + Q - 2 * R))),
            source="derived here (online/linear.py greedy_eta), not in KC95",
            check=Check((R, Q, M), lambda r, q, mm: L.greedy_eta(*_f(r, q, mm)),
                        (("0.3", "0.3", "1"), ("0.6", "1", "1"), ("0.5", "2", "1.5")), "rhs")),
        sym("Offline T = 0 solution on the sphere (KC Eq. (47), rho = alpha / r): the standard algorithm "
            "r = sqrt(Q/M) (KC Eq. (27) at Q = M), from the zero-energy-set geometry",
            sp.Eq(RHO, AL * sp.sqrt(M / Q0)), source=_KC,
            check=Check((AL, Q0, M), lambda al, q0, mm: OFF.offline(float(al), "standard", M=float(mm),
                                                                    Q=float(q0)).rho,
                        (("0.5", "1", "1"), ("0.2", "0.5", "1"), ("0.3", "2", "1.5")), "rhs")),
        sym("Offline 'optimal' algorithm r = sqrt(alpha) (KC Eqs. (48)-(49)): the pseudoinverse direction",
            sp.Eq(RHO, sp.sqrt(AL)), source=_KC,
            check=Check((AL,), lambda al: OFF.offline(float(al), "optimal").rho,
                        (("0.25",), ("0.5",), ("0.9",)), "rhs")),
        sym("Pseudoinverse e_g through ridge (E7): gamma = p/n = 1/alpha, sigma2 = 0, |beta|^2 = M; excess risk "
            "M (1 - 1/gamma) = 2 e_g",
            sp.Eq(sp.Symbol("e_g"), M * (1 - AL) / 2), source=_HMRT,
            check=Check((AL, M), lambda al, mm: OFF.ridge_route(float(al), float(mm)).eg,
                        (("0.25", "1"), ("0.5", "2"), ("0.8", "1")), "rhs")),
        cited("Results compared (KC Eqs. (19), (24)-(27), (31), (49); p. 11)",
              r"\rho^{on}_{opt} = \sqrt{1 - e^{-\alpha}},\ \rho^{on}_{std} = 1 - e^{-\alpha},\ "
              r"\rho^{off}_{opt} = \sqrt{\alpha},\ \rho^{off}_{std} = \alpha\ (\alpha \le 1),\ "
              r"1 - \rho^{on}_{opt} \simeq \tfrac{1}{2} e^{-\alpha},\ \rho^{on}_{opt} > \rho^{off}_{std} "
              r"\text{ for } \alpha \lesssim 0.71", _KC),
        text("Finite-N checks", "online learning simulated at finite N over named seeds (online/simulate_linear.py, "
             "no code shared with the ODEs), 99% Student-t bands; a planted ODE without the eta^2 term falls "
             "outside them; offline solutions built from sampled data (statistical)"),
    ]


__all__ = ["online_linear"]
