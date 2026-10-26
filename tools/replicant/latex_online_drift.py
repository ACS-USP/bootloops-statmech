"""Displayed equations of the `online_drift` catalog entry (route online, O3).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against online/ at three points
(tests/test_latex.py); the drift models, the rate equations and the papers' results are cited.
"""
import sympy as sp

from .tex import Check, cited, sym, text

AL, D, RHO, TH, ETA, LAM, R0, Q0, DEL = sp.symbols("alpha D rho theta eta lambda_n R_0 Q_0 Delta", real=True)
_VKC = "Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998)"
_KC93 = "Kinouchi & Caticha, J. Phys. A 26, 6161 (1993)"


def _f(*xs):
    return [float(x) for x in xs]


def online_drift(route):
    from .online import drift as DR
    from .online import drift_rules as VR
    s, c = sp.sin(TH), sp.cos(TH)
    k = 1 - 2 * LAM
    cc = sp.sqrt(sp.pi * D) / k
    a = k * sp.sqrt(2 / sp.pi)
    r_inf = ETA * a / D
    return [
        text("Online route with drift", "The O1 engine (|B| = 1, inputs N(0, 1), label flipped with probability "
             "lambda_n = chi/2, J <- J + (eta/N) F b xi, eps = arccos(rho)/pi against the current teacher); after each "
             "example the teacher drifts: random, B <- (B + sqrt(2D)/N u)/|.| with B(alpha) . B(alpha') = "
             "e^{-D |alpha - alpha'|} (VKC Eqs. (4), (26); KC93 Eqs. (3)-(6), T = D), or deterministic, B turned away "
             "from J by arccos(1 - D/N^2) (VKC Eqs. (28)-(30)); the N -> oo limit is self-averaging by assumption",
             source=_VKC),
        cited("Overlap flow with uncorrelated drift (VKC Eqs. (10), (27), (31); KC93 Eq. (14)): the drift term does not "
              "involve W, so the maximiser W* (Eq. (17)) is the driftless one (VKC Sec. III, p. 9)",
              r"\frac{d\rho}{d\alpha} = \rho\left\langle W\left(\frac{y b}{\rho} - \Delta\right) - \frac{W^2}{2}"
              r"\right\rangle - \begin{cases} D\rho & \text{random} \\ \sqrt{2D(1-\rho^2)} & \text{deterministic}"
              r"\end{cases}", _VKC),
        sym("Annealed Hebb W_AH = sqrt(2/pi) lambda^2 rho (1 - chi) (VKC Eq. (22), chi = 2 lambda_n), per unit |J| as a "
            "function of theta = arccos rho",
            sp.Eq(sp.Symbol("W_AH"), sp.sqrt(2 / sp.pi) * s**2 / c * k), source=_VKC,
            check=Check((TH, LAM), lambda th, lam: float(VR.vkc_modulation("annealed_hebb", 0.3, *_f(th, lam))),
                        (("0.3", "0"), ("0.8", "0.1"), ("1.2", "0.2")), "rhs")),
        sym("Symmetric weight W_SW = sqrt(2/pi) lambda (1 - chi) e^{-Delta^2 / (2 lambda^2)} (VKC Eq. (24))",
            sp.Eq(sp.Symbol("W_SW"), sp.sqrt(2 / sp.pi) * s / c * k * sp.exp(-DEL**2 * c**2 / (2 * s**2))), source=_VKC,
            check=Check((DEL, TH, LAM), lambda d, th, lam: float(VR.vkc_modulation("symmetric", *_f(d, th, lam))),
                        (("0.4", "0.5", "0"), ("-1.1", "0.9", "0.1"), ("2", "1.2", "0.05")), "rhs")),
        sym("Annealed Hebb under random drift: drho/dalpha = (1 - chi)^2 s^4/(pi rho) - D rho (derived from Eqs. (22), "
            "(27)), so s^2 = c rho, c = sqrt(pi D)/(1 - chi); checked against the ODE's stationary state",
            sp.Eq(RHO, (sp.sqrt(cc**2 + 4) - cc) / 2), source="derived here from " + _VKC,
            check=Check((D, LAM), lambda d, lam: float(__import__("math").cos(
                DR.stationary(DR.drift_flow("annealed_hebb", D=float(d), noise=float(lam))).theta)),
                (("0.1", "0"), ("0.02", "0.1"), ("0.4", "0.2")), "rhs", tol="1e-9")),
        sym("Plain Hebb under random drift (derived): dR/dalpha = eta a - D R, a = (1 - 2 lambda_n) sqrt(2/pi); R -> "
            "eta a / D while Q grows like eta^2 alpha, so rho -> 0 (no tracking; VKC p. 17); checked against the ODE",
            sp.Eq(sp.Symbol("R"), r_inf + (R0 - r_inf) * sp.exp(-D * AL)), source="derived here from " + _VKC,
            check=Check((AL, D, ETA, LAM, R0), lambda al, d, eta, lam, r0: float(DR.integrate_drift(
                DR.drift_flow("hebb", D=float(d), eta=float(eta), noise=float(lam)), float(al), (float(r0), 1.0),
                alphas=(float(al),)).R[0]),
                (("2", "0.3", "1", "0", "0"), ("5", "0.1", "0.5", "0.1", "0.2"), ("1", "1", "2", "0.2", "0.5")),
                "rhs", tol="1e-8")),
        cited("KC93 Eqs. (17)-(23), with Eq. (22) as derived here ((1 + lambda^2); printed (1 - lambda^2)): the optimal "
              "rule's rate, and its stationary state I(lambda_oo) = T (random) or sqrt(2D) lambda_oo (deterministic, "
              "from VKC Eq. (31)); lambda = tan(pi e_G)",
              r"\frac{d\rho}{d\alpha} = \rho\,(I(\lambda) - T),\quad I(\lambda) = \frac{\lambda^3}{2\pi}\int Du\,"
              r"\frac{e^{-(1+\lambda^2)u^2/2}}{H(u)}", _KC93),
        cited("Results compared (VKC Tables I-II, Figs. 4, 6, Eq. (34); KC93 Eqs. (24)-(25)); the KC93 bound: "
              "e_G^{opt}(\\alpha) \\le e_G^{W}(\\alpha) for every modulated Hebbian W from the same rho0. "
              "The delta relation holds only for the decoupled rules (W independent of |J|, KC93 p. 6); the "
              "fixed-rate free-norm perceptron gives delta = 3, 4 (not VKC p. 15's 5, 8)",
              r"e_G^\infty \simeq c\,D^{1/\delta},\ \delta = \tfrac{1}{\beta} + 2\ (\text{random}),\ "
              r"\tfrac{2}{\beta} + 2\ (\text{deterministic});\ c_{opt} = \Big[\tfrac{\pi^2}{2}\int Du\,"
              r"\tfrac{e^{-u^2/2}}{H(u)}\Big]^{-1/3} \approx 0.447", _VKC),
        text("Finite-N checks", "online learning with the teacher drifting in the simulation (online/simulate_drift.py, "
             "no code shared with the ODEs), 99% Student-t bands; an ODE without (or with half) the drift term falls "
             "outside them"),
    ]


__all__ = ["online_drift"]
