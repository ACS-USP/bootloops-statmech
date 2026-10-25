"""Displayed equations of the `xorsat` catalog entry (cavity route): the leaf-removal /
warning-propagation recursion, the 2-core, the complexity and the two thresholds.  Same rule
as latex_entries.py: each sympy block carries a tex.Check against csp/xorsat.py at three
points (tests/test_latex.py); the threshold conditions are cited strings.
"""
import sympy as sp

from .tex import Check, cited, sym, text

K = sp.Symbol("K", positive=True, integer=True)
A = sp.Symbol("alpha", positive=True)
Q = sp.Symbol("q", positive=True)
_MRZ = "Mezard, Ricci-Tersenghi & Zecchina, J. Stat. Phys. 111, 505 (2003)"
_MM = "Mezard & Montanari, Information, Physics, and Computation (OUP 2009), ch. 18"
_PS = "Pittel & Sorkin, Combin. Probab. Comput. 25, 236 (2016), Thm 16"
_DM = "Dubois & Mandler, C. R. Acad. Sci. Paris Ser. I 335, 963 (2002), Thm 3"


def xorsat(route):
    from .csp import xorsat as X
    lam = K * A * Q**(K - 1)
    arb = ((3, "0.9", "0.7"), (4, "0.8", "0.5"), (5, "1.0", "0.9"))
    roots = lambda: tuple((k, a, X.xorsat_core_size(k, a).q)
                          for k, a in ((3, "0.85"), (3, "0.95"), (4, "0.9")))
    core = lambda key: lambda k, a, q: getattr(X.xorsat_core_size(int(k), a), key)
    v_expr = 1 - (1 + lam) * sp.exp(-lam)
    return [
        text("Cavity route", "zero-temperature warning propagation on the factor graph: a check "
             "warns a variable when its K - 1 other variables are frozen by the rest of the graph "
             "(q_hat = q^(K-1)); a variable is frozen along an edge when another check warns it. "
             "This is leaf removal (peeling); the physical solution is the largest root, reached "
             "from q = 1 (csp/xorsat.py; the degree law is the DEGREE_LAWS hook)", source=_MRZ),
        sym("Leaf-removal recursion, Poisson(K alpha) degrees (MRZ Eq. (30), MM Eq. (18.7))",
            sp.Eq(Q, 1 - sp.exp(-lam)), source=f"{_MRZ}; {_MM}",
            check=Check((K, A, Q), lambda k, a, q: X.leaf_map(int(k), a, q), arb, "rhs")),
        sym("Fixed-point density alpha*(q); alpha_d = min over q in (0, 1) (MM Eq. (18.8))",
            sp.Eq(sp.Symbol("alpha^*"), -sp.log(1 - Q) / (K * Q**(K - 1))), source=_MM,
            check=Check((K, Q), lambda k, q: X.fixed_point_alpha(int(k), q),
                        ((3, "0.5"), (3, "0.88"), (4, "0.7")), "rhs")),
        sym("Core variables per variable, lambda = K alpha q^(K-1) (MRZ Eq. (32))",
            sp.Eq(sp.Symbol("V"), v_expr), source=_MRZ,
            check=Check((K, A, Q), core("variables"), roots, "rhs")),
        sym("Core checks per variable (MRZ Eq. (33), MM Eq. (18.15))",
            sp.Eq(sp.Symbol("C"), A * Q**K), source=f"{_MRZ}; {_MM}",
            check=Check((K, A, Q), core("checks"), roots, "rhs")),
        sym("Complexity: ln of the number of core solutions (clusters) per variable",
            sp.Eq(sp.Symbol("Sigma"), sp.log(2) * (v_expr - A * Q**K)), source=f"{_MRZ}, Eq. (13)",
            check=Check((K, A, Q), core("complexity"), roots, "rhs")),
        cited("Dynamical (2-core) threshold: tangency of the recursion",
              r"\alpha_d(K) = \min_{0<q<1} \frac{-\ln(1-q)}{K q^{K-1}},\qquad "
              r"\alpha_d(3) = 0.818469",
              f"{_MM}, Thm 18.1; {_MRZ}, Sec. III.B"),
        cited("Satisfiability threshold: as many core checks as core variables",
              r"\Sigma(\alpha_s) = 0 \iff 1 - (1 + \lambda)e^{-\lambda} = \frac{\lambda}{K}"
              r"\left(1 - e^{-\lambda}\right),\qquad \alpha_s(3) = 0.917935",
              f"{_MRZ}, Eq. (42); {_DM} (K = 3); {_PS} (K >= 3)"),
        text("Finite-N checks", "independent of the closed forms: sampled instances solved by "
             "exact Gaussian elimination over GF(2) (bit-packed rows), the satisfiable fraction "
             "over named seeds with a 99% Clopper-Pearson band, and leaf removal on the sampled "
             "hypergraph (csp/xorsat_finite.py)"),
    ]


__all__ = ["xorsat"]
