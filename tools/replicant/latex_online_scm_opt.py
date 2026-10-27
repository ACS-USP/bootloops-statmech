"""Displayed equations of `online_scm` with rule=optimal (O6; Vicente & Caticha 1997).  Same rule as
latex_entries.py: sympy blocks carry a tex.Check against online/optimal_multilayer.py (tests/test_latex.py)."""
import numpy as np
import sympy as sp

from .tex import Check, cited, sym, text

_VC = "Vicente & Caticha, J. Phys. A 30, L599 (1997) (arXiv cond-mat/9706015v1)"
Q11, Q12, Q22, R1, R2 = sp.symbols("Q11 Q12 Q22 R1 R2", real=True)


def opt_blocks():
    from .online import optimal_multilayer as O
    d1, d2 = sp.sqrt(1 + Q11), sp.sqrt(1 + Q22)
    eg = (sp.asin(Q11 / (1 + Q11)) + sp.asin(Q22 / (1 + Q22)) + 2 * sp.asin(Q12 / (d1 * d2))) / sp.pi \
        - 2 / sp.pi * (sp.asin(R1 / (sp.sqrt(2) * d1)) + sp.asin(R2 / (sp.sqrt(2) * d2))) + sp.Rational(1, 6)
    e21 = lambda a, b, c, r, s: O.eps_vc(np.array([[float(a), float(b)], [float(b), float(c)]]),
                                         np.array([[float(r)], [float(s)]]))
    pl = lambda: O.plateau_k2m2()
    return [
        text("Optimal rule (rule=optimal; VC97)", "VC97's model is the soft committee machine above with M_nm = "
             "delta_nm and the learning rate folded into the modulation: J_k <- J_k + (F_k/N) S; F may read every "
             "student field x_k, the teacher output Sigma_B and the exact order parameters; it is greedy (the "
             "largest decay of e_g per example) and a maximum only where the functional Hessian H is positive "
             "definite (a saddle otherwise)", source=_VC),
        cited("Order-parameter flow and the variational solution (VC97 Eqs. (2), (4)); H_ij = 2 de_g/dQ_ij, "
              "G_kn = -de_g/dR_kn, <y>_{H|V} the mean of the teacher fields given x and Sigma_B",
              r"\frac{dR_{in}}{d\alpha} = \langle y_n F_i\rangle,\ \frac{dQ_{ij}}{d\alpha} = \langle x_iF_j + x_jF_i + "
              r"F_iF_j\rangle,\quad \mathbf F = \mathbf H^{-1}\mathbf G\langle\mathbf y\rangle_{\mathcal H|\mathcal V} - \mathbf x", _VC),
        cited("Closure derived here (not in VC97): with A = H^-1 G and W = <yhat yhat^T> (W = 1 for M = 1, since "
              "Sigma_B = erf(y/sqrt2) fixes y), the flow is algebraic in A and W",
              r"\frac{d\mathbf R}{d\alpha} = \mathbf A\mathbf W - \mathbf R,\quad \frac{d\mathbf Q}{d\alpha} = "
              r"\mathbf A\mathbf W\mathbf A^T - \mathbf Q,\quad \mathbf W = \langle\hat{\mathbf y}\hat{\mathbf y}^T\rangle", _VC),
        sym("Generalisation error (VC97 Eq. (3)) at K = 2, M = 1; checked against optimal_multilayer.eps_vc",
            sp.Eq(sp.Symbol("e_g"), eg), source=_VC,
            check=Check((Q11, Q12, Q22, R1, R2), e21, (("0.4", "0.1", "0.3", "0.2", "0.1"),
                        ("1", "-0.2", "0.5", "0.7", "-0.1"), ("0.38", "0", "5e-7", "0", "0")), "rhs")),
        sym("K = M = 2 plateau of the optimal flow, derived here: identical students with Sigma singular, "
            "R = (1/sqrt6) 1 1^T, Q = (1/3) 1 1^T; e_g (VC97 Fig. 2 reads 0.0118 at N = 5000); checked against the "
            "computed root (plateau_k2m2)",
            sp.Eq(sp.Symbol("e_g^*"), sp.Rational(1, 3) - 4 / sp.pi * sp.asin(sp.Rational(1, 4))), source=_VC,
            check=Check((), lambda: pl().eps, ((),), "rhs", tol="1e-9")),
        sym("Its Hessian eigenvalues: 2(D11 + D12) and 2(D11 - D12) = 0 (VC97: the smallest 'stays very close to "
            "zero in the plateau'); checked against the computed root", sp.Eq(sp.Symbol("lambda_H"),
            4 * sp.sqrt(15) / (5 * sp.pi)), source=_VC, check=Check((), lambda: pl().eigs[1], ((),), "rhs", tol="1e-7")),
    ]


__all__ = ["opt_blocks"]
