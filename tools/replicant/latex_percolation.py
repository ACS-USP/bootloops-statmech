"""Displayed equations of the P1 entries percolation_bethe and percolation_graph (catalog_percolation.py).
Each sympy block carries a tex.Check against the percolation/ code (the certified ball's midpoint) at
three points: the fixed-point equation and the giant at the code's own root, the mean cluster size,
and the cluster-size closed forms against the generating-function coefficients (Lagrange inversion).
"""
from fractions import Fraction

import sympy as sp

from .tex import Check, cited, sym, text

K, T, U, C, S = sp.symbols("k t u c S", positive=True)
SS = sp.Symbol("s", positive=True, integer=True)
_NSW = "Newman, Strogatz & Watts, arXiv:cond-mat/0007235, Eqs. (26), (27), (34)-(38)"
_CNSW = "Callaway, Newman, Strogatz & Watts, arXiv:cond-mat/0007300, Eqs. (5)-(7), (12), (14), (15)"
_MS = "Malthe-Sorenssen, Percolation Theory Using Python (2024), Ch. 3, Eqs. (3.2)-(3.27)"
_MR = "Molloy & Reed, Random Struct. Alg. 6, 161 (1995), Theorem 1"


def _giant(law, dil):
    from .percolation.cavity import percolation_giant
    return percolation_giant(law, dil)


def _bethe_pt(k, t, kind="bond"):
    return lambda: (k, t, _giant(("bethe", k), (kind, Fraction(t))).selected.u.mid())


def _ps(law, t, s):
    from .percolation.clusters import cluster_distribution
    return cluster_distribution(law, ("bond", Fraction(t)), s).lagrange[s - 1].mid()


def _sample_blocks():
    return [text("Finite graphs, sampled", "configuration-model (or G(N, p)) graphs over named seeds, "
                 "bond or site dilution, clusters by union-find with path compression and union by "
                 "size; the largest cluster estimates P_inf, the others the finite-cluster statistics; "
                 "means over the seeds with stderr (statistical, never final)", source=_NSW),
            cited("Estimators over n named seeds (of NSW Eqs. (34), (36) on finite graphs)",
                  r"\hat P_\infty = \frac{1}{n}\sum_{i=1}^{n} \frac{|C_1^{(i)}|}{N},\qquad "
                  r"\widehat{\langle s\rangle}^{(i)} = \frac{\sum_{C \neq C_1^{(i)}} |C|^2}"
                  r"{\sum_{C \neq C_1^{(i)}} |C|},\qquad \text{band } t_{0.995,\,n-1}\,"
                  r"\frac{\hat\sigma}{\sqrt n}", "percolation/sampling.py; " + _NSW)]


def percolation_bethe(route):
    if route == "sample":
        return _sample_blocks()
    pts = (_bethe_pt(3, "0.75"), _bethe_pt(4, "0.5"), _bethe_pt(5, "0.3"))
    giant = 1 - U ** K
    mean = 1 + K * T * U ** (K - 2) / (1 - (K - 1) * T * U ** (K - 2))
    gs = K * sp.factorial((K - 1) * SS) / (sp.factorial(SS) * sp.factorial((K - 2) * SS + 2))
    ps = SS * gs * T ** (SS - 1) * (1 - T) ** ((K - 2) * SS + 2)
    return [
        text("Cavity route on the Bethe lattice", "coordination k, occupation t (bond or site); u is the "
             "probability that a branch does not reach infinity; every root in [0, 1] is enclosed by "
             "interval Newton, its stability is f'(u) < 1", source=_MS),
        cited("Generating functions with dilution", r"H_1(x) = 1 - t + t\,x\,G_1(H_1(x)),\qquad "
              r"H_0(x) = x\,G_0(H_1(x))\ \text{(bond)},\qquad G_0 = x^k,\ G_1 = x^{k-1}", _CNSW),
        sym("Fixed point (MS Eq. (3.5)), at the code's certified root", sp.Eq(U, 1 - T + T * U ** (K - 1)),
            source=_MS, check=Check((K, T, U), lambda k, t, u: u, pts, "rhs")),
        cited("Threshold", r"p_c = \frac{1}{k-1}", f"{_MS}, Eq. (3.2)"),
        sym("Giant component (bond; site: times t)", sp.Eq(sp.Symbol("P_inf"), giant), source=_CNSW,
            check=Check((K, T, U), lambda k, t, u: _giant(("bethe", int(k)), ("bond", Fraction(str(t))))
                        .P_inf.mid(), pts, "rhs")),
        sym("Mean finite-cluster size of an occupied vertex (derived here; MS Eq. (3.20) below p_c)",
            sp.Eq(sp.Symbol("<s>"), mean),
            check=Check((K, T, U), lambda k, t, u: _giant(("bethe", int(k)), ("bond", Fraction(str(t))))
                        .mean_finite.mid(), pts, "rhs")),
        sym("Cluster sizes (MS Eq. (3.27); g_s derived here), against the Lagrange coefficients",
            sp.Eq(sp.Symbol("P_s"), ps), source=_MS,
            check=Check((K, T, SS), lambda k, t, s: _ps(("bethe", int(k)), str(t), int(s)),
                        ((3, "0.75", 4), (4, "0.25", 3), (5, "0.5", 2)), "rhs")),
    ]


def percolation_graph(route):
    if route == "sample":
        return _sample_blocks()
    lam = C * T
    pts = tuple((lambda c=c, t=t: (c, t, _giant(("poisson", Fraction(c)), ("bond", Fraction(t)))
                                   .P_inf.mid())) for c, t in (("2", "1"), ("3", "0.5"), ("10", "0.2")))
    return [
        text("Cavity route on configuration-model graphs", "degree law p_k (G0 = sum p_k x^k, G1 = G0'/"
             "G0'(1)), bond or site occupation t; Erdos-Renyi: G0 = G1 = exp(c (x - 1))", source=_NSW),
        cited("Giant component and threshold", r"S = 1 - G_0(u),\quad u = 1 - t + t\,G_1(u),\qquad "
              r"t_c = \frac{1}{G_1'(1)}", _CNSW),
        cited("Molloy-Reed criterion", r"Q = \sum_k k(k-2)\,p_k > 0 \iff \frac{\langle k^2\rangle}"
              r"{\langle k\rangle} > 2", _MR),
        sym("Erdos-Renyi giant (NSW Eq. (37), lambda = c t), at the code's certified S",
            sp.Eq(S, 1 - sp.exp(-lam * S)), source=_NSW,
            check=Check((C, T, S), lambda c, t, s: s, pts, "rhs")),
        sym("Mean finite-cluster size (NSW Eq. (38))", sp.Eq(sp.Symbol("<s>"), 1 / (1 - lam + lam * S)),
            source=_NSW, check=Check((C, T, S), lambda c, t, s: _giant(
                ("poisson", Fraction(str(c))), ("bond", Fraction(str(t)))).mean_finite.mid(), pts, "rhs")),
        sym("Cluster sizes: the Borel law (derived here), against the Lagrange coefficients",
            sp.Eq(sp.Symbol("P_s"), (lam * SS) ** (SS - 1) * sp.exp(-lam * SS) / sp.factorial(SS)),
            check=Check((C, T, SS), lambda c, t, s: _ps(("poisson", Fraction(str(c))), str(t), int(s)),
                        (("2", "0.25", 3), ("1", "1", 4), ("3", "0.5", 2)), "rhs")),
    ]


__all__ = ["percolation_bethe", "percolation_graph"]
