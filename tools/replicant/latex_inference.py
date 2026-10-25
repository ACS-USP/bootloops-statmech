"""Displayed equations of the `ldpc_bec` catalog entry (cavity route): the density-evolution
recursion, the channel-parameter function eps(x) whose minimum is the BP threshold, the bit
erasure probability, the trial (Bethe) entropy whose root is the MAP threshold, and the area
condition.  Same rule as latex_entries.py: each sympy block carries a tex.Check against
inference/ldpc.py at three points (tests/test_latex.py); the threshold statements are cited
strings.
"""
import sympy as sp

from .tex import Check, cited, sym, text

L_, R_ = sp.symbols("l r", positive=True, integer=True)
E, X = sp.symbols("epsilon x", positive=True)
_MMU = "Measson, Montanari & Urbanke, IEEE Trans. Inform. Theory 54, 5277 (2008)"
_MM = "Mezard & Montanari, Information, Physics, and Computation (OUP 2009), Sec. 15.3"
_KRU = "Kudekar, Richardson & Urbanke, arXiv:1001.1826 (2010), Lemma 4"
_LMSS = "Luby, Mitzenmacher, Shokrollahi & Spielman, IEEE Trans. Inform. Theory 47, 569 (2001)"


def ldpc_bec(route):
    from .inference import ldpc as L
    y = 1 - (1 - X)**(R_ - 1)                       # 1 - rho(1 - x), rho(z) = z^(r-1)
    eps_x = X / y**(L_ - 1)
    pts4 = ((3, 6, "0.45", "0.3"), (4, 8, "0.5", "0.7"), (3, 5, "0.6", "0.1"))
    pts3 = ((3, 6, "0.3"), (4, 8, "0.55"), (5, 10, "0.9"))

    def de_points():
        return tuple((l, r, e, L.bec_density_evolution(l, r, e).edge_erasure)
                     for l, r, e in ((3, 6, 0.45), (3, 6, 0.6), (4, 8, 0.5)))

    return [
        text("Cavity route", "belief propagation on the code's factor graph: on the erasure "
             "channel a message is either erased or exact, so BP is the peeling decoder and its "
             "N -> oo limit is density evolution for the erasure probability x of a "
             "variable-to-check message (edge perspective lambda(x) = x^(l-1), rho(x) = "
             "x^(r-1); inference/ldpc.py, the degree polynomials are the DEGREE_PAIRS hook)",
             source=_MM),
        sym("Density evolution, x_0 = epsilon (MM Eq. (15.33))",
            sp.Eq(sp.Symbol("x_{t+1}"), E * y.subs(X, sp.Symbol("x_t"))**(L_ - 1)), source=_MM,
            check=Check((L_, R_, E, sp.Symbol("x_t")), lambda l, r, e, x: L.de_map(int(l), int(r), e, x),
                        pts4, "rhs")),
        sym("Channel parameter of a nonzero DE fixed point; eps_BP = its minimum over x in "
            "(0, 1] (MM Eq. (15.34), MMU Fact 1)",
            sp.Eq(sp.Function("epsilon")(X), eps_x), source=f"{_MM}; {_MMU}",
            check=Check((L_, R_, X), lambda l, r, x: L.eps_of_x(int(l), int(r), x), pts3, "rhs")),
        sym("Bit erasure probability at the DE fixed point x, Lambda(y) = y^l (MM Sec. 15.3.2)",
            sp.Eq(sp.Symbol("P_b"), E * y**L_), source=_MM,
            check=Check((L_, R_, E, X), lambda l, r, e, x:
                        L.bec_density_evolution(int(l), int(r), float(e)).erasure_prob,
                        de_points, "rhs")),
        sym("Trial (Bethe) entropy on the EBP curve, Lambda'(1) = l, Gamma'(1) = r (MMU "
            "Definition 4, Lemma 3)",
            sp.Eq(sp.Symbol("P(x)"), L_ * X * (1 - X)**(R_ - 1) - L_ / R_ * (1 - (1 - X)**R_)
                  + eps_x * y**L_), source=_MMU,
            check=Check((L_, R_, X), lambda l, r, x: L.trial_entropy(int(l), int(r), x), pts3,
                        "rhs")),
        cited("BP threshold (below it peeling recovers every bit w.h.p.)",
              r"\epsilon_{\rm BP} = \min_{0<x\le 1}\frac{x}{\left(1-(1-x)^{r-1}\right)^{l-1}},"
              r"\qquad \epsilon_{\rm BP}(3,6) \approx 0.42944",
              f"{_LMSS}, Thm 2; {_MM}; {_KRU}, Example 5"),
        cited("MAP threshold: Maxwell construction (root of the trial entropy = area theorem)",
              r"P(x^*) = 0,\ \ \epsilon_{\rm MAP} = \epsilon(x^*)\quad\Longleftrightarrow\quad"
              r"\int_{x^*}^{1}\Lambda(y(x))\,\epsilon'(x)\,dx = 1 - \frac{l}{r},\qquad"
              r"\epsilon_{\rm MAP}(3,6) \approx 0.48815",
              f"{_MMU}, Lemmas 2-4, Theorem 10, Examples 4 and 6; {_KRU}"),
        text("Finite-N checks", "independent of density evolution: configuration-model graphs "
             "(multi-edges kept), the peeling decoder on named seeds, the success fraction with "
             "a 99% Clopper-Pearson band and the residual erasure fraction "
             "(inference/ldpc_finite.py)"),
    ]


__all__ = ["ldpc_bec"]
