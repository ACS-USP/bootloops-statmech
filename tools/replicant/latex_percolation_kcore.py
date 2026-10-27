"""Displayed equations of the P3 entries percolation_kcore and percolation_bootstrap
(catalog_percolation_kcore.py).  Each sympy block carries a tex.Check against the percolation/ code at
three points: the Erdos-Renyi core equation and the core size (DGM Eqs. (4), (11)) at the code's
certified root, the birth point at the code's certified argmin, the hypergraph 2-core equation, and
Balogh-Pittel's closed form and final state for bootstrap percolation.
"""
from fractions import Fraction

import sympy as sp

from .tex import Check, cited, sym, text

C, T, X, MU, KK, AL, D, Q, PH = sp.symbols("c t x mu k alpha d q p", positive=True)
KH = sp.Symbol("K", positive=True, integer=True)
L = sp.Symbol("ell", positive=True, integer=True)
_DGM = "Dorogovtsev, Goltsev & Mendes, arXiv:cond-mat/0509102, Eqs. (1), (2), (4)-(8), (11)"
_J09 = "Janson, arXiv:0804.1656, Eq. (4.9), Theorems 4.5, 4.6, 5.1"
_JL = "Janson & Luczak, arXiv:math/0508453, Theorem 1.1 (Pittel, Spencer & Wormald)"
_BP = "Bartha & Pete, arXiv:1509.08454, Eqs. (3.1), (3.2) (Balogh & Pittel 2007)"


_MEMO = {}


def _memo(key, fn):
    """fn() once per key and planted-hook state (a planted error is never served a stale value)."""
    from .percolation import kcore, kcore_threshold
    full = (key, tuple(sorted(kcore._HOOKS.items())), tuple(sorted(kcore_threshold._HOOKS.items())))
    if full not in _MEMO:
        _MEMO[full] = fn()
    return _MEMO[full]


def _fp(law, K, k, t):
    from .percolation.kcore import kcore_fixed_points
    t = Fraction(str(t))
    return _memo(("fp", repr(law), K, k, t), lambda: kcore_fixed_points(law, K, k, ("site", t)))


def _er_pt(c, t, k):
    return lambda: (c, t, k, _fp(("poisson", Fraction(c)), 2, k, t).selected.x.mid())


def _ck(k):
    from .percolation.kcore_threshold import degree_threshold
    return _memo(("ck", k), lambda: degree_threshold(2, k, "site", 1))


def _sample_blocks(what):
    return [text("Finite graphs, sampled", f"{what} over named seeds; the k-core by exact peeling (delete a "
                 "vertex in fewer than k live (hyper)edges until none is left), bootstrap percolation by "
                 "direct spreading, the finally uninfected set checked to equal the (d + 1 - ell)-core of "
                 "the initially uninfected set on every instance; means with stderr (statistical, never "
                 "final)", source=_J09),
            cited("Estimators over n named seeds", r"\hat V = \frac1n \sum_{i=1}^{n} \frac{|\mathrm{core}_k"
                  r"^{(i)}|}{N},\qquad \text{band } t_{0.995,\,n-1}\,\frac{\hat\sigma}{\sqrt n}",
                  "percolation/kcore_sample.py; " + _J09)]


def percolation_kcore(route):
    if route == "sample":
        return _sample_blocks("configuration-model, G(N, p) and XORSAT-ensemble hypergraphs")
    pts = (_er_pt(4, "1", 3), _er_pt(5, "0.9", 3), _er_pt(8, "1", 4))
    phi = 1 - sp.uppergamma(KK - 1, C * X) / sp.gamma(KK - 1)
    core = T * (1 - sp.uppergamma(KK, C * X) / sp.gamma(KK))
    hk = MU / (1 - sp.uppergamma(KK - 1, MU) / sp.gamma(KK - 1))
    hyper = [(lambda K=K, a=a: (K, a, _fp(("poisson", Fraction(a) * K), K, 2, 1).selected.x.mid()))
             for K, a in ((3, "1"), (4, "1"), (3, "0.9"))]
    return [
        text("Cavity route for the k-core", "x is the probability that an end of an edge roots an infinite "
             "(k - 1)-ary subtree; every root of x = F(x) in [0, 1] is isolated and enclosed in arb, and "
             "the largest (the limit of peeling from x = 1) is selected with its certified F'(x) < 1",
             source=_DGM),
        cited("Core equation and core size for a degree law P(q) (site occupation t)",
              r"x = t\sum_{j} \frac{(j+1)P(j+1)}{\langle q\rangle}\,\mathrm{P}(\mathrm{Bin}(j, x) \ge k-1),"
              r"\qquad V = t\sum_q P(q)\,\mathrm{P}(\mathrm{Bin}(q, x) \ge k)", _DGM),
        sym("Erdos-Renyi core equation (DGM Eq. (4)), at the code's certified root",
            sp.Eq(X, T * phi), source=_DGM, check=Check((C, T, KK, X), lambda c, t, k, x: x, pts, "rhs")),
        sym("Erdos-Renyi core size (DGM Eq. (11))", sp.Eq(sp.Symbol("V"), core), source=_DGM,
            check=Check((C, T, KK, X), lambda c, t, k, x: _fp(("poisson", Fraction(str(c))), 2, int(k), t)
                        .core.mid(), pts, "rhs")),
        cited("Birth point (the hybrid transition's jump)", r"t_c = \min_{0<y\le 1}\frac{y}{\Psi(y)},\qquad "
              r"c_k = \min_{\mu>0}\frac{\mu}{\mathrm{P}(\mathrm{Po}(\mu)\ge k-1)}", f"{_J09}; {_JL}"),
        sym("Erdos-Renyi k-core threshold at the code's certified argmin mu*", sp.Eq(sp.Symbol("c_k"), hk),
            source=_JL, check=Check((KK, MU), lambda k, mu: _ck(int(k)).value.mid(), tuple(
                (lambda k=k: (k, _ck(k).s_star.mid())) for k in (3, 4, 5)), "rhs")),
        sym("K-uniform Poisson hypergraph, k = 2 (derived here; the E5 XORSAT leaf-removal equation)",
            sp.Eq(X, 1 - sp.exp(-KH * AL * X ** (KH - 1))), source="csp/xorsat.py (Mezard, Ricci-Tersenghi "
            "& Zecchina, arXiv:cond-mat/0207140, Eq. (30))",
            check=Check((KH, AL, X), lambda K, a, x: x, hyper, "rhs")),
    ]


def _bq(d, ell, q):
    from .percolation.kcore import kcore_fixed_points
    q = Fraction(str(q))
    return _memo(("bq", d, ell, q), lambda: kcore_fixed_points(("bethe", d), 2, d + 1 - ell, ("site", 1 - q)))


def _qc(d):
    from .percolation.kcore_threshold import occupation_threshold
    return 1 - occupation_threshold(("bethe", d), 2, d - 1, "site").value.mid()


def percolation_bootstrap(route):
    if route == "sample":
        return _sample_blocks("random d-regular configuration-model graphs")
    qc = 1 - (D - 2) ** (2 * D - 5) / ((D - 1) ** (D - 2) * (D - 3) ** (D - 3))
    fin = 1 - (1 - Q) * sp.betainc(D - L + 1, L, 0, PH) / sp.beta(D - L + 1, L)    # P(Bin(d, 1-p) <= ell-1)
    pts = tuple((lambda d=d, l=l, q=q: (d, l, q, _bq(d, l, q).selected.x.mid()))
                for d, l, q in ((4, 2, "1/20"), (5, 2, "1/50"), (6, 3, "1/10")))
    return [
        text("Bootstrap percolation as a core", "on a d-regular graph the finally uninfected vertices are the "
             "(d + 1 - ell)-core of the initially uninfected ones, a site percolation with occupation "
             "1 - q (Janson, Sec. 5); the core route of percolation_kcore with the law of degree d",
             source=_J09),
        cited("Critical infection probability", r"q_c = 1 - \inf_{0<p\le 1}\frac{p}{\mathrm{P}(\mathrm{Bin}"
              r"(d-1, 1-p) \le \ell - 1)}", _J09),
        sym("Closed form for ell = 2, d > 3, against the certified minimum", sp.Eq(sp.Symbol("q_c"), qc),
            source=_BP, check=Check((D,), lambda d: _qc(int(d)), ((4,), (5,), (6,)), "rhs")),
        sym("Final infected fraction at the largest root p (Janson Theorem 5.1; the binomial tail "
            "P(Bin(d, 1 - p) <= ell - 1) written as the regularized incomplete beta function)",
            sp.Eq(sp.Symbol("A_f/n"), fin), source=_J09,
            check=Check((D, L, Q, PH), lambda d, l, q, p: 1 - _bq(int(d), int(l), str(q)).core.mid(), pts,
                        "rhs")),
    ]


__all__ = ["percolation_bootstrap", "percolation_kcore"]
