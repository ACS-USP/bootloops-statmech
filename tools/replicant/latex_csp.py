"""Displayed equations of the `xorsat` and `ksat` catalog entries (cavity route).  xorsat: the leaf-removal /
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


_BMZ = "Braunstein, Mezard & Zecchina, Random Struct. Algorithms 27, 201 (2005)"
_MMZ = "Mertens, Mezard & Zecchina, Random Struct. Algorithms 28, 340 (2006)"
_DSS = "Ding, Sly & Sun, Ann. of Math. 196, 1 (2022)"


def ksat(route):
    """BP and SP update rules as sympy (checked against the code's functions, the SP factor
    through the live SP_UPDATES hook); the complexity formulas and the threshold cited."""
    from .csp import ksat as S
    from .csp import sp as SP
    PS, PU = sp.symbols("p_s p_u", positive=True)
    pts = (("0.3", "0.6"), ("0.9", "0.05"), ("0.5", "0.5"))
    return [
        text("Cavity route", "survey propagation is the zero-energy (y -> oo) limit of the 1RSB "
             "cavity equations on the factor graph: a survey is the probability, over clusters "
             "of SAT assignments, that clause a sends variable i a warning (all its other "
             "variables are forced to violate it).  The SP factor is the SP_UPDATES hook "
             "(csp/sp.py), audited by enumerating warning configurations", source=_MMZ),
        sym("Annealed (first-moment) entropy, (1/N) ln E Z (DSS Eq. (2.3))",
            sp.Eq(sp.Symbol("s_ann"), sp.log(2) + A * sp.log(1 - 2**(-K))), source=_DSS,
            check=Check((K, A), lambda k, a: S.ksat_annealed_bound(int(k), a),
                        ((3, "2.0"), (3, "4.2"), (4, "9.0")), "rhs")),
        sym("BP cavity probability that j violates a (BMZ Eq. (11)); P^u, P^s products of "
            "(1 - delta) over the same-sign and opposite-sign clauses",
            sp.Eq(sp.Symbol("gamma"), PU / (PU + PS)), source=f"{_BMZ}, Eqs. (10)-(12)",
            check=Check((PU, PS), lambda pu, ps: S.bp_rule(float(pu), float(ps)), pts, "rhs")),
        sym("SP factor of input j: Pi^u / (Pi^u + Pi^s + Pi^0), p_s (p_u) = product of "
            "(1 - eta) over the same-sign (opposite-sign) clauses; eta_(a->i) = product over "
            "j != i (BMZ Eqs. (26)-(27), MMZ Eq. (2))",
            sp.Eq(sp.Symbol("f_SP"), PS * (1 - PU) / (PS + PU - PS * PU)),
            source=f"{_BMZ}; {_MMZ}",
            check=Check((PS, PU), lambda ps, pu: SP.SP_UPDATES["bmz"](float(ps), float(pu)),
                        pts, "rhs")),
        cited("Complexity of an instance (BMZ Eqs. (39)-(41))",
              r"N\Sigma = \sum_a \ln\Big[\prod_{j\in a}(\Pi^u+\Pi^s+\Pi^0)_{j\to a}"
              r" - \prod_{j\in a}\Pi^u_{j\to a}\Big] - \sum_i (n_i - 1)\ln\big[P^+_i + P^-_i"
              r" - P^+_i P^-_i\big]", f"{_BMZ}; {_MMZ}, Eqs. (3)-(5)"),
        cited("Complexity of the SP population (MMZ Eq. (18), with the fixed-point identity (17)); "
              "pi = e^-x and pi' = e^-z with x, z independent sums of phi = -ln(1 - eta) over "
              "Poisson(K alpha / 2) population members (the no-warning probabilities of the two "
              "sign classes), rho = pi' (1 - pi) / (pi + pi' - pi pi') (the SP factor of one "
              "input), the expectations over independent draws",
              r"\Sigma = \Sigma_0 + \alpha\left[K I_{K-1} - (K-1) I_K\right] = "
              r"\mathbb{E}\ln(\pi + \pi' - \pi\pi') - \alpha (K-1)\,\mathbb{E}\ln\Big(1 - "
              r"\prod_{i=1}^{K}\rho_i\Big)", f"{_MMZ}, Eqs. (13), (15), (17), (18)"),
        cited("SAT/UNSAT threshold (cavity conjecture): the root of the complexity",
              r"\Sigma(\alpha_s) = 0,\qquad \alpha_s(3) = 4.26675 \pm 0.00015",
              f"{_MMZ}, Sec. 2.3 and Eq. (43); proved for K >= k0 only: {_DSS}, Theorem 1"),
        text("Finite-N checks", "independent of BP and SP: exhaustive counts of all 2^N "
             "assignments (N <= 22) on named seeds, Jensen's inequality and the exact first "
             "moment; BP exact on tree formulas (BMZ Theorem 2)"),
    ]


__all__ = ["ksat", "xorsat"]
