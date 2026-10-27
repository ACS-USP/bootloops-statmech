"""Second route of the k-core values, in mpmath, from Dorogovtsev, Goltsev & Mendes, PRL 96, 040601
(2006), arXiv:cond-mat/0509102v3 (DGM), in their own form: it imports nothing from kcore.py or
kcore_threshold.py (no binomial tails, no hooks) and reads the run's k, K and law afresh.

  Phi_k(R) = sum_(n=0)^(k-2) (1 - R)^n / n! G1^(n)(R)                    (DGM Eq. (4))
             Erdos-Renyi: Gamma(k - 1, c (1 - R)) / Gamma(k - 1)          (DGM, below Eq. (10))
  site p: R = 1 - p + p Phi_k(R), the lowest root R < 1 (DGM Eqs. (2), (4)); bond: the same in
             1 - R = t x (Janson, arXiv:0804.1656, Thm 4.6);  hypergraphs (derived here): 1 - R =
             b x^(K-1), x = a (1 - Phi_k(R))
  M(k) = p sum_q P(q) sum_(n>=k) C(q, n) R^(q-n) (1 - R)^n                (DGM Eq. (1))
             Erdos-Renyi: p {1 - Gamma(k, c (1 - R)) / Gamma(k)}           (DGM Eq. (11))
  threshold: p_c(k) = 1 / max f_k(R), f_k(R) = [1 - Phi_k(R)] / (1 - R)   (DGM Eqs. (5)-(8));
             hypergraphs: max (1 - Phi_k)^m / (1 - R) (derived here); in the ER mean degree,
             min_mu mu / P(Po(mu) >= k - 1)^m (Janson & Luczak, arXiv:math/0508453, Thm 1.1)
G1^(n) is differentiated from the degree law's coefficients p_q (G1 = sum q p_q x^(q-1) / <q>).
Floats at DPS digits: a grid of the function, a local refinement by mp.findroot; a disagreement
with the certified balls beyond TOL is the caller's no-solution:route-disagreement.
  dgm_fixed(law, K, k, kind, t) -> {x, core}
  dgm_threshold(law, K, k, kind, t_ck=None) -> {threshold_t, y_star, core_at_jump[, c_k, mu_star]}
"""
from fractions import Fraction
from math import comb

import mpmath as mp

DPS = 20
TOL = mp.mpf("1e-15")
GRID = 48


class _DGM:
    def __init__(self, law, K, k, kind, t):
        self.K, self.k, self.m = K, k, K - 1
        self.kind, self.t = kind, mp.mpf(t.numerator) / t.denominator
        self.a, self.b = (self.t, mp.mpf(1)) if kind == "site" else (mp.mpf(1), self.t)
        self.c = None
        if law[0] == "poisson":
            self.c = mp.mpf(law[1].numerator) / law[1].denominator
            self.z = self.c
        else:
            self.pq = {q: mp.mpf(w.numerator) / w.denominator for q, w in law[1].items() if w}
            self.z = sum(q * w for q, w in self.pq.items())

    def g1n(self, n, R):
        """G1^(n)(R) from the degree coefficients."""
        s = mp.mpf(0)
        for q, w in self.pq.items():
            e = q - 1 - n
            if e >= 0:
                s += q * w * mp.ff(q - 1, n) * R ** e
        return s / self.z

    def phi_k(self, R):
        if self.c is not None:
            return mp.gammainc(self.k - 1, self.c * (1 - R), mp.inf, regularized=True)
        return sum(((1 - R) ** n / mp.factorial(n) * self.g1n(n, R) for n in range(self.k - 1)), mp.mpf(0))

    def psi(self, y):
        return 1 - self.phi_k(1 - y)

    def M(self, y):
        if self.c is not None:
            return 1 - mp.gammainc(self.k, self.c * y, mp.inf, regularized=True)
        R = 1 - y
        return sum((w * sum(comb(q, n) * R ** (q - n) * y ** n for n in range(self.k, q + 1))
                    for q, w in self.pq.items()), mp.mpf(0))

    def gap(self, x):
        return self.a * self.psi(self.b * x ** self.m) - x


def _largest_root(D):
    xs = [mp.mpf(i) / GRID for i in range(GRID, 0, -1)]
    g = [D.gap(x) for x in xs]
    if abs(g[0]) < mp.mpf(10) ** (5 - DPS):
        return mp.mpf(1)
    for i in range(1, len(xs)):
        if g[i] > 0:
            return mp.findroot(D.gap, (xs[i], xs[i - 1]), solver="anderson")
    i = max(range(len(xs)), key=lambda j: g[j])           # the bump may sit between grid points
    if 0 < i < len(xs) - 1:
        xm = mp.findroot(lambda x: mp.diff(D.gap, x), xs[i])
        if D.gap(xm) > 0:
            return mp.findroot(D.gap, (xm, xs[i - 1]), solver="anderson")
    return mp.mpf(0)


def _argmin(f, lo, hi):
    xs = [lo + (hi - lo) * i / GRID for i in range(1, GRID + 1)]
    v = [f(x) for x in xs]
    i = min(range(len(xs)), key=lambda j: v[j])
    if i == len(xs) - 1 and hi == 1:
        return mp.mpf(1)
    return mp.findroot(lambda x: mp.diff(f, x), xs[i])


def dgm_fixed(law, K, k, kind, t):
    """{x, core} at the run point; law ("poisson", Fraction c) or ("finite", {q: Fraction p_q})."""
    t = Fraction(t)
    with mp.workdps(DPS):
        D = _DGM(law, K, k, kind, t)
        x = _largest_root(D) if t > 0 else mp.mpf(0)
        return {"x": +x, "core": +(D.a * D.M(D.b * x ** D.m)) if x > 0 else mp.mpf(0)}


def dgm_threshold(law, K, k, kind, t_ck=None):
    """{threshold_t, y_star, core_at_jump[, c_k, mu_star]} (discontinuous case (k - 1)(K - 1) >= 2);
    c_k (Poisson laws) at occupation t_ck when given."""
    with mp.workdps(DPS):
        D = _DGM(law, K, k, kind, Fraction(1))
        H = lambda s: s / D.psi(s) ** D.m
        ys = _argmin(H, mp.mpf(0), mp.mpf(1))
        Hs = H(ys)
        tc = Hs if kind == "bond" else Hs ** (mp.mpf(1) / D.m)
        out = {"threshold_t": +tc, "y_star": +ys, "core_at_jump": +((tc if kind == "site" else 1) * D.M(ys))}
        if D.c is not None and t_ck is not None:
            T = mp.mpf(Fraction(t_ck).numerator) / Fraction(t_ck).denominator
            pi = lambda mu: mp.gammainc(k - 1, 0, mu, regularized=True)
            Hm = lambda mu: mu / pi(mu) ** D.m
            mu = _argmin(Hm, mp.mpf(0), mp.mpf(8 * k + 40))
            out["c_k"], out["mu_star"] = +(Hm(mu) / (T ** D.m if kind == "site" else T)), +mu
        return out


__all__ = ["DPS", "TOL", "dgm_fixed", "dgm_threshold"]
