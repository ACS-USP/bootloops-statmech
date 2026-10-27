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
  threshold: p_c(k) = 1 / max f_k(R), f_k(R) = [1 - Phi_k(R)] / (1 - R)   (DGM Eqs. (5)-(8)), the
             stationarity f_k' = 0 with Phi_k' = (1 - R)^(k-2)/(k-2)! G1^(k-1)(R) (derived here);
             hypergraphs: max (1 - Phi_k)^m / (1 - R) (derived here); in the ER mean degree,
             min_mu mu / P(Po(mu) >= k - 1)^m (Janson & Luczak, arXiv:math/0508453, Thm 1.1)
G1^(n) is differentiated from the degree law's coefficients p_q (G1 = sum q p_q x^(q-1) / <q>).
Floats at `dps` digits (DPS by default): a grid of the function, a local refinement by mp.findroot.
The caller (catalog_percolation_kcore._second) compares with the certified balls at a tolerance
scaled by the route's precision and the conditioning 1 / |1 - F'(x)|, reruns at 40 digits before
refusing no-solution:route-disagreement, and skips the route with a note where it fails to compute
(an auxiliary route never blocks or crashes a certified run).
  dgm_fixed(law, K, k, kind, t, dps=DPS) -> {x, core, edges}
  dgm_threshold(law, K, k, kind, t_ck=None, dps=DPS) -> {threshold_t, y_star, core_at_jump[, c_k, mu_star]}
"""
from fractions import Fraction
from math import comb

import mpmath as mp

DPS = 20
TOL = mp.mpf("1e-15")
GRID, ARGMIN_GRID = 48, 24


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

    def dpsi(self, y):
        """Psi'(y) = Phi_k'(1 - y) = y^(k-2)/(k-2)! G1^(k-1)(1 - y) (the sum of Eq. (4) telescopes;
        derived here); ER: d/da Gamma(k - 1, a)/Gamma(k - 1) = -a^(k-2) e^-a/(k-2)!."""
        if self.c is not None:
            a = self.c * y
            return self.c * a ** (self.k - 2) * mp.exp(-a) / mp.factorial(self.k - 2)
        return y ** (self.k - 2) / mp.factorial(self.k - 2) * self.g1n(self.k - 1, 1 - y)

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
    if abs(g[0]) < mp.mpf(10) ** (5 - mp.mp.dps):
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


def _argmin(f, g, lo, hi):
    """Grid minimum of f refined by a bracketed findroot on its stationarity function g (g = 0 iff
    f' = 0); grid points where f is not usable (the tail of Psi cancels at this precision) are
    skipped."""
    xs = [lo + (hi - lo) * i / ARGMIN_GRID for i in range(1, ARGMIN_GRID + 1)]
    v = [f(x) for x in xs]
    ok = [j for j in range(len(xs)) if v[j] is not None]
    if not ok:
        raise ValueError("Psi vanishes on the whole grid at this precision")
    i = min(ok, key=lambda j: v[j])
    if i == len(xs) - 1 and hi == 1:
        return mp.mpf(1)
    if 0 < i < len(xs) - 1 and v[i - 1] is not None and v[i + 1] is not None:   # g changes sign
        return mp.findroot(g, (xs[i - 1], xs[i + 1]), solver="anderson")
    return mp.findroot(g, xs[i])


def _ratio(num, den, m):
    """num / den^m, or None where den is not > 10^(8 - dps): Psi = 1 - Phi_k is then dominated by the
    cancellation of the sum (absolute error ~ 10^-dps), not a usable value."""
    return num / den ** m if den > mp.mpf(10) ** (8 - mp.mp.dps) else None


def dgm_fixed(law, K, k, kind, t, dps=DPS):
    """{x, core, edges} at the run point; law ("poisson", Fraction c) or ("finite", {q: Fraction p_q}).
    edges: Janson's lambda p^2 / (2 pi) at K = 2 (p = b x, pi = b; arXiv:0804.1656 Thms 4.5, 4.6),
    <d> b x^K / K at K >= 3 (derived here)."""
    t = Fraction(t)
    with mp.workdps(dps):
        D = _DGM(law, K, k, kind, t)
        x = _largest_root(D) if t > 0 else mp.mpf(0)
        if x == 0:
            return {"x": mp.mpf(0), "core": mp.mpf(0), "edges": mp.mpf(0)}
        edges = D.z * (D.b * x) ** 2 / (2 * D.b) if K == 2 else D.z * D.b * x ** K / K
        return {"x": +x, "core": +(D.a * D.M(D.b * x ** D.m)), "edges": +edges}


def dgm_threshold(law, K, k, kind, t_ck=None, dps=DPS):
    """{threshold_t, y_star, core_at_jump[, c_k, mu_star]} (discontinuous case (k - 1)(K - 1) >= 2);
    c_k (Poisson laws) at occupation t_ck when given."""
    with mp.workdps(dps):
        D = _DGM(law, K, k, kind, Fraction(1))
        H = lambda s: _ratio(s, D.psi(s), D.m)
        ys = _argmin(H, lambda s: D.psi(s) - D.m * s * D.dpsi(s), mp.mpf(0), mp.mpf(1))
        Hs = H(ys)
        tc = Hs if kind == "bond" else Hs ** (mp.mpf(1) / D.m)
        out = {"threshold_t": +tc, "y_star": +ys, "core_at_jump": +((tc if kind == "site" else 1) * D.M(ys))}
        if D.c is not None and t_ck is not None:
            T = mp.mpf(Fraction(t_ck).numerator) / Fraction(t_ck).denominator
            pi = lambda mu: mp.gammainc(k - 1, 0, mu, regularized=True)
            Hm = lambda mu: _ratio(mu, pi(mu), D.m)
            dpi = lambda mu: mu ** (k - 2) * mp.exp(-mu) / mp.factorial(k - 2)
            mu = _argmin(Hm, lambda mu: pi(mu) - D.m * mu * dpi(mu), mp.mpf(0), mp.mpf(8 * k + 40))
            out["c_k"], out["mu_star"] = +(Hm(mu) / (T ** D.m if kind == "site" else T)), +mu
        return out


__all__ = ["DPS", "TOL", "dgm_fixed", "dgm_threshold"]
