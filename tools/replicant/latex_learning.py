"""Displayed equations of the `ridge` catalog entry (replica route, E7).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against learning/ridge.py or
learning/ridge_rs.py at three points (tests/test_latex.py).  The Stieltjes transform is displayed as
HMRT print it (m(z) at z = -lambda) and its derivative is sympy's, so the checks compare the
printed formula with the code's rewritten forms (m = 2/(b + sqrt D), m' = m (1 + gamma m)/sqrt D);
theorem statements and the RS saddle equations are cited strings.
"""
import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym, text

G, L, S2, R2, A2 = sp.symbols("gamma lambda sigma^2 r^2 alpha^2", positive=True)
_HMRT = ("Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949 (2022), arXiv:1903.08560")
_DW = "Dobriban & Wager, Ann. Statist. 46, 247 (2018), arXiv:1507.03003"
_LGC = ("Loureiro, Gerbelot, Cui, Goldt, Krzakala, Mezard & Zdeborova, J. Stat. Mech. (2022) 114001, "
        "arXiv:2102.08127")


def _f(*xs):
    return [mp.mpf(str(x)) for x in xs]


def ridge(route):
    from .learning import ridge as R
    from .learning import ridge_rs as RS
    z = sp.Symbol("z")
    m_z = (1 - G - z - sp.sqrt((1 - G - z)**2 - 4 * G * z)) / (2 * G * z)
    m = m_z.subs(z, -L)                               # HMRT's printed m(z) at z = -lambda
    dm = sp.diff(m_z, z).subs(z, -L)                  # m'(z) at z = -lambda
    risk = R2 * L**2 * dm + S2 * G * (m - L * dm)
    pts = (("0.5", "0.3"), ("1", "0.5"), ("2.5", "0.1"))
    pts4 = (("0.5", "0.3", "1", "1"), ("1", "0.5", "1", "2"), ("2.5", "0.1", "0.5", "1"))
    lam_star = S2 * G / R2
    c = (G - 1) / G * A2
    lam_l, alpha = L / G, 1 / G
    V = (-(lam_l + alpha - 1) + sp.sqrt((lam_l + alpha - 1)**2 + 4 * lam_l)) / (2 * lam_l)
    return [
        text("Replica route", "X in R^{n x p} with i.i.d. N(0, 1) entries, y = X beta + sigma xi, "
             "|beta|^2 = r^2, gamma = p/n, ridge (X^T X + n lambda I)^{-1} X^T y; reported: the "
             "excess risk |beta_hat - beta|^2 = bias + variance (the test MSE adds sigma^2). The RS "
             "saddle point is a cited closed form solved by the code; replicant's formal layer does "
             "not derive it; the rigorous proofs are random-matrix theory", source=_HMRT),
        sym("Marchenko-Pastur Stieltjes transform at z = -lambda (HMRT Cor. 5, m(z) as printed)",
            sp.Eq(sp.Symbol("m"), m), source=_HMRT,
            check=Check((G, L), lambda g, lam: R.stieltjes_mp(*_f(g, lam))[0], pts, "rhs")),
        sym("Its derivative m'(z) at z = -lambda (sympy differentiation of the printed m(z))",
            sp.Eq(sp.Symbol("m'"), dm), source=_HMRT,
            check=Check((G, L), lambda g, lam: R.stieltjes_mp(*_f(g, lam))[1], pts, "rhs")),
        cited("Bias and variance at finite n (HMRT Lemma 1; S = X^T X / n, isotropic beta)",
              r"B_X = \lambda^2 \beta^T (S+\lambda)^{-2}\beta \to r^2\lambda^2 m'(-\lambda),\quad "
              r"V_X = \frac{\sigma^2}{n}\mathrm{tr}\, S (S+\lambda)^{-2} \to \sigma^2\gamma\,"
              r"\left(m(-\lambda) - \lambda m'(-\lambda)\right)", _HMRT),
        sym("Limiting ridge risk (HMRT Cor. 5, Eq. (43))", sp.Eq(sp.Symbol("R"), risk), source=_HMRT,
            check=Check((G, L, S2, R2), lambda g, lam, s2, r2: R.ridge_test_error(g, lam, s2, r2),
                        pts4, "rhs")),
        sym("Ridgeless limit, gamma < 1 (HMRT Prop. 2, Eq. (8))", sp.Eq(sp.Symbol("R"), S2 * G / (1 - G)),
            source=_HMRT, check=Check((G, S2), lambda g, s2: R.ridgeless_test_error(g, s2, 1),
                                      (("0.3", "1"), ("0.5", "0.25"), ("0.9", "2")), "rhs")),
        sym("Ridgeless limit, gamma > 1 (HMRT Thm 1, Eqs. (6)-(8)); divergent at gamma = 1",
            sp.Eq(sp.Symbol("R"), R2 * (1 - 1 / G) + S2 / (G - 1)), source=_HMRT,
            check=Check((G, S2, R2), lambda g, s2, r2: R.ridgeless_test_error(g, s2, r2),
                        (("1.5", "1", "1"), ("2", "0.5", "2"), ("4", "0.25", "1")), "rhs")),
        sym("Optimal penalty (HMRT Cor. 5; DW Thm 2.1 at unit noise)", sp.Eq(sp.Symbol("lambda^*"), lam_star),
            source=_HMRT, check=Check((G, S2, R2), lambda g, s2, r2: R.optimal_lambda(g, s2, r2),
                                      (("0.5", "1", "1"), ("2", "1", "5"), ("3", "0.5", "2")), "rhs")),
        sym("Risk at the optimal penalty (HMRT Cor. 5)",
            sp.Eq(sp.Symbol("R^*"), (S2 * G * m).subs(L, lam_star)), source=_HMRT,
            check=Check((G, S2, R2), lambda g, s2, r2: R.optimal_test_error(g, s2, r2),
                        (("0.5", "1", "1"), ("1", "1", "5"), ("3", "0.5", "2")), "rhs")),
        sym("Optimal predictive risk at unit noise, R^* + 1 (DW Eq. (8), alpha^2 = r^2)",
            sp.Eq(sp.Symbol("R^*_{test}"), (1 + c + sp.sqrt((1 - c)**2 + 4 * A2)) / 2), source=_DW,
            check=Check((G, A2), lambda g, a2: R.optimal_test_error(g, 1, a2) + 1,
                        (("0.5", "1"), ("1", "1"), ("2", "5")), "rhs")),
        cited("RS saddle point of the linear student, isotropic (Loureiro et al. Eq. (A.38) and Thm 1 "
              "Eq. (2.8) with label noise; alpha = 1/gamma, lambda_L = lambda/gamma, rho = r^2)",
              r"\hat V = \hat m = \frac{\alpha}{1+V},\ \hat q = \frac{\alpha(\rho+\sigma^2+q-2m)}{(1+V)^2},"
              r"\ V = \frac{1}{\lambda_L+\hat V},\ m = \frac{\hat m\rho}{\lambda_L+\hat V},\ "
              r"q = \frac{\hat q+\hat m^2\rho}{(\lambda_L+\hat V)^2},\ R = \rho+q-2m", _LGC),
        sym("The V root of the saddle point, V (lambda_L + alpha/(1 + V)) = 1, lambda_L = lambda/gamma",
            sp.Eq(sp.Symbol("V"), V), source=_LGC,
            check=Check((G, L), lambda g, lam: RS.ridge_replica_rs(g, lam).V, pts, "rhs")),
        text("Finite-n checks", "SVD fits on sampled designs over named seeds: HMRT's finite-n risk R_X "
             "(Lemma 1 identities on the sampled spectrum) with 99% Student-t bands, extrapolated in "
             "1/n (statistical)"),
    ]


__all__ = ["ridge"]
