"""The k-core of locally tree-like random (hyper)graphs: every fixed point of the core equation,
certified in arb balls, the selected (largest) one with its computed stability, and the core's size.

Model.  A K-uniform hypergraph (K = 2: a graph), vertex degrees from a law of percolation/laws.py
(("poisson", c), for K >= 3 c = K alpha; ("finite", {d: p_d}); ("bethe", d): the random d-regular
graph), site ("site", t) or bond ("bond", t) diluted.  The k-core: the largest sub-hypergraph in
which every vertex lies in >= k hyperedges.

Equations (K = 2: Dorogovtsev, Goltsev & Mendes, PRL 96, 040601 (2006), arXiv:cond-mat/0509102v3
(DGM), Eqs. (1), (2), (4); Janson & Luczak, Random Struct. Alg. 30, 50 (2007), arXiv:math/0508453
(JL07), Thm 2.3; Janson, arXiv:0804.1656 (J09), Thms 4.5 (site), 4.6 (bond), Thm A.1):
  x = P(an end of an edge is the root of an infinite (k-1)-ary subtree) = 1 - R of DGM, = Janson's p^.
  Psi(y) = sum_j q_j P(Bin(j, y) >= k - 1)    (q_j = (j+1) p_(j+1)/<d>, the excess law: DGM (2));
           Poisson: P(Po(c y) >= k - 1)       (DGM (4) with Phi_k = Gamma(k-1, c(1-R))/Gamma(k-1))
  site:  x = t Psi(x),        vertices in the core V = t sum_d p_d P(Bin(d, x) >= k)   (DGM (1), (2))
  bond:  w = t x = t Psi(w),  V = sum_d p_d P(Bin(d, w) >= k)                          (J09 Thm 4.6)
  edges per vertex: site <d> x^2 / 2, bond <d> t x^2 / 2  (JL07 Thm 2.3: lambda p^2 / 2; J09 4.5, 4.6)
K >= 3 (derived here, the same argument on the factor graph): a hyperedge reached from a vertex
survives iff its K - 1 other ends do, y = b x^(K-1); x = a Psi(y); V = a sum_d p_d P(Bin(d, y) >= k);
hyperedges per vertex (<d>/K) b x^K, with (a, b) = (t, 1) for site and (1, t) for bond.  At k = 2,
Poisson c = K alpha, t = 1 these are csp/xorsat.py's leaf-removal equations (q = x, q_hat = y, V,
C = alpha q^K), the E5 cross-check.  Writing F(x) = a Psi(b x^m), m = K - 1:
  F is increasing, F(0) = 0.  The peeling iteration x_{n+1} = F(x_n) from x_0 = 1 decreases to the
  LARGEST root in [0, 1] (J09 Thm A.1; DGM: "the lowest nontrivial solution R < 1"): the selection.
Discontinuous case (k - 1) m >= 2 (k >= 3, or k = 2 on hypergraphs): F(x) <= a B b^(k-1) x^(m(k-1))
with B = sum_j q_j C(j, k - 1) (Poisson c^(k-1)/(k-1)!) since P(Bin(j, y) >= r) <= C(j, r) y^r, so
no root lies in (0, x0] once a B b^(k-1) x0^(m(k-1)-1) < 1.  On [x0, 1] the roots are isolated by
bisection: an interval is discarded when the monotone enclosure [F(lo) - hi, F(hi) - lo] of F - x
excludes 0; it holds exactly one root when F' - 1 (arb interval extension) excludes 0 and F - x
changes sign at its ends; that root is enclosed by interval Newton (>= 40 digits).  An interval
that cannot be decided at width 1e-30 (a near-double root: t at the jump) is refused
undetermined:near-critical (ambiguous:critical-point when t lies in the certified threshold ball,
kcore_threshold.refuse_at_threshold).  x = 0 is always a root (F'(0) = 0 exactly); x = 1 is one iff F(1) = 1
exactly (decided in rationals).  The selected largest root must have the certified F'(x) < 1
(else ambiguous:critical-point: the largest root is marginal), and an exclusion scan of
(x*, 1] must find no other root (else no-solution:dropped-root; percolation/kcore_audit.py).
Continuous case k = 2, K = 2: x = 1 - u with u = 1 - t + t G1(u) (site; bond: x = (1 - u)/t), the
P1 cavity equation, whose fixed points percolation/cavity.py certifies (reused, not duplicated).
Second route of the selection (rigorous): the peeling iteration h_{n+1} = upper(F(h_n)) from 1 and
l_{n+1} = lower(F(l_n)) from the isolating interval's lower end bracket x*; they must overlap it.

  Model(law, K, k, dilution)       .F(x), .dF(x), .psi(y), .dpsi(y), .d2psi(y), .core(y), .edges(x)
  kcore_fixed_points(law, K, k, dilution) -> KcoreResult
_HOOKS (planted errors, tests): "k_shift" (0): the core order k + shift in every equation (a wrong
threshold k); "drop_hyperedge" (False): y = b x instead of b x^(K-1) (the missing hyperedge term,
invisible at K = 2); "drop_root" (None | "largest"): the largest root dropped from the list.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller
from .kcore_audit import bracket as _bracket
from .kcore_audit import no_root_above as _no_root_above
from .laws import occupation, parse_law

DIGITS = 40
PREC = 4 * DIGITS + 128
_HOOKS = {"k_shift": 0, "drop_hyperedge": False, "drop_root": None}
K_MAX, CORE_MAX = 12, 60
MIN_WIDTH = Fraction(1, 10 ** 30)


def _int(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need an int in {lo} .. {hi}")
    return x


def _btail(j, y, r):
    """P(Bin(j, y) >= r) for an arb y (the shorter of the two sums)."""
    if r <= 0:
        return 0 * y + 1
    if r > j:
        return 0 * y
    if r <= j - r + 1:
        s = 0 * y
        for n in range(r):
            s += comb(j, n) * y ** n * (1 - y) ** (j - n)
        return 1 - s
    s = 0 * y
    for n in range(r, j + 1):
        s += comb(j, n) * y ** n * (1 - y) ** (j - n)
    return s


def _ptail(mu, r):
    """P(Po(mu) >= r) for an arb mu."""
    if r <= 0:
        return 0 * mu + 1
    s, term = 0 * mu, 0 * mu + 1
    for n in range(r):
        s += term
        term = term * mu / (n + 1)
    return 1 - (-mu).exp() * s


def _ppmf(mu, n):
    if n < 0:
        return 0 * mu
    return (-mu).exp() * mu ** n / factorial(n)


class Model:
    """F(x) = a Psi(b x^m) and the core's size, for one (law, K, k, dilution)."""

    def __init__(self, law, K, k, dilution):
        self.law = parse_law(law)
        self.K = _int(K, "hyperedge-size", 2, K_MAX)
        k0 = _int(k, "core-order", 2, CORE_MAX)
        self.kind, self.t = occupation(dilution)
        self.k = k0 + _HOOKS["k_shift"]
        self.m = 1 if _HOOKS["drop_hyperedge"] else self.K - 1
        self.a, self.b = (self.t, Fraction(1)) if self.kind == "site" else (Fraction(1), self.t)
        self.poisson = self.law.kind == "poisson"
        if self.poisson:
            self.c = Fraction(self.law.c)
        else:
            self.q = {j: w for j, w in self.law.excess().items() if w}
            self.p = {d: w for d, w in self.law.degree().items() if w}
            if max(self.q) < self.k - 1:
                raise Refusal("precondition:core-order", f"k = {k0}: every excess degree is < k - 1 = "
                              f"{self.k - 1} (max degree {max(self.p)}), so the k-core is empty for every "
                              "occupation")
        self.z = self.law.mean()
        self.continuous = (self.k - 1) * self.m == 1
        self._cache = {}
        baller()
        from flint import ctx
        self._ctx = ctx

    def _q(self, v):
        """arb_q(v) at the current precision, cached (the rationals of the law are reused often)."""
        key = (v, self._ctx.prec)
        got = self._cache.get(key)
        if got is None:
            got = self._cache[key] = arb_q(v)
        return got

    # ---- Psi and its derivatives at an arb y ------------------------------------------------
    def psi(self, y):
        r = self.k - 1
        if self.poisson:
            return _ptail(self._q(self.c) * y, r)
        return sum((self._q(w) * _btail(j, y, r) for j, w in self.q.items()), 0 * y)

    def dpsi(self, y):
        r = self.k - 1
        if self.poisson:
            c = self._q(self.c)
            return c * _ppmf(c * y, r - 1)
        s = 0 * y
        for j, w in self.q.items():
            if j >= r >= 1:
                s += self._q(w) * j * comb(j - 1, r - 1) * y ** (r - 1) * (1 - y) ** (j - r)
        return s

    def d2psi(self, y):
        r = self.k - 1
        if self.poisson:
            c = self._q(self.c)
            return c * c * (_ppmf(c * y, r - 2) - _ppmf(c * y, r - 1))
        s = 0 * y
        for j, w in self.q.items():
            if j >= r >= 1:
                f = self._q(w) * j * comb(j - 1, r - 1)
                if r >= 2:
                    s += f * (r - 1) * y ** (r - 2) * (1 - y) ** (j - r)
                if j > r:
                    s -= f * (j - r) * y ** (r - 1) * (1 - y) ** (j - r - 1)
        return s

    def bound_B(self):
        """Psi(y) <= B y^(k-1) (an exact rational)."""
        r = self.k - 1
        if self.poisson:
            return self.c ** r / factorial(r)
        return sum(w * comb(j, r) for j, w in self.q.items())

    # ---- the fixed-point map and the core --------------------------------------------------
    def y_of(self, x):
        return self._q(self.b) * x ** self.m

    def F(self, x):
        return self._q(self.a) * self.psi(self.y_of(x))

    def dF(self, x):
        dy = self._q(self.b) * self.m * x ** (self.m - 1) if self.m > 1 else self._q(self.b) + 0 * x
        return self._q(self.a) * self.dpsi(self.y_of(x)) * dy

    def F1_exact(self):
        """F(1) as an exact rational when it is one (finite laws), else None."""
        if self.poisson:
            return None
        if self.b != 1:
            return None
        return self.a * sum(w for j, w in self.q.items() if j >= self.k - 1)

    def core(self, y):
        """Vertices in the core per vertex, at the hyperedge probability y."""
        if self.poisson:
            v = _ptail(self._q(self.c) * y, self.k)
        else:
            v = sum((self._q(w) * _btail(d, y, self.k) for d, w in self.p.items()), 0 * y)
        return self._q(self.a) * v

    def edges(self, x):
        """Hyperedges in the core per vertex: (<d>/K) b x^K."""
        return self._q(self.z) / self.K * self._q(self.b) * x ** self.K


@dataclass(frozen=True)
class Root:
    x: Ball
    jacobian: Ball      # F'(x)
    stable: bool        # F'(x) < 1, certified


@dataclass(frozen=True)
class KcoreResult:
    model: object
    roots: tuple        # ascending x
    selected: Root      # the largest root
    y: Ball             # the hyperedge-side probability b x^m at the selection
    core: Ball          # vertices in the core / N
    edges: Ball         # hyperedges in the core / N
    bracket: dict | None
    phase: str


def _x0(M):
    """An exact x0 > 0 with no root of F(x) = x in (0, x0] (discontinuous case)."""
    from flint import arb
    coef = M.a * M.bound_B() * M.b ** (M.k - 1)
    e = M.m * (M.k - 1) - 1
    j = 1
    while coef * Fraction(1, 2 ** j) ** e >= 1:
        j += 1
    return arb(2) ** (-j)


def _isolate(M):
    """[(lo, hi)] arb-exact intervals each holding exactly one root of F(x) = x in (0, 1]."""
    from flint import arb
    one = arb(1)
    exact1 = M.F1_exact() == 1
    # (0, x0]: no root (the bound F(x) <= a B b^(k-1) x^(m(k-1)))
    out, stack = [], [(_x0(M), one)]
    wmin = arb_q(MIN_WIDTH)
    while stack:
        lo, hi = stack.pop()
        Flo, Fhi = M.F(lo), M.F(hi)
        if Fhi.upper() < lo or Flo.lower() > hi:
            continue                                   # F(I) misses I: no root
        box = lo.union(hi)
        d = M.dF(box) - 1
        glo, ghi = Flo - lo, Fhi - hi
        if not d.contains(0):
            if exact1 and hi == one:
                out.append((lo, hi, True))
                continue
            if glo > 0 and ghi < 0 or glo < 0 and ghi > 0:
                out.append((lo, hi, False))
                continue
            if glo > 0 and ghi > 0 or glo < 0 and ghi < 0:
                continue
        if hi - lo < wmin:
            raise Refusal("undetermined:near-critical", f"F(x) = x cannot be decided on [{lo.str(12)}, "
                          f"{hi.str(12)}]: a near-double root (the occupation is at the core's jump "
                          "to within the resolution 1e-30)")
        mid = ((lo + hi) / 2).mid()
        stack += [(lo, mid), (mid, hi)]
    return sorted(out, key=lambda iv: float(iv[0].mid()))


def _newton(M, lo, hi):
    """The root in [lo, hi] (F - x changes sign, F' - 1 excludes 0) to >= DIGITS digits."""
    from flint import arb
    x = ((lo + hi) / 2).mid()
    up = (M.F(lo) - lo) < 0                 # F - x increasing through the root
    for _ in range(200):
        g = M.F(arb(x)) - x
        if (g < 0) == up:
            lo = x
        elif (g > 0) == up:
            hi = x
        xn = (x - g / (M.dF(arb(x)) - 1)).mid()
        if not (lo < xn < hi):
            xn = ((lo + hi) / 2).mid()
        done = abs(xn - x) < arb(2) ** (16 - PREC)
        x = xn
        if done:
            break
    r = arb(2) ** (-(PREC // 2))
    for _ in range(40):
        box = arb(x, r)
        N = x - (M.F(arb(x)) - x) / (M.dF(box) - 1)
        if box.contains_interior(N):
            return N
        r *= 4
    raise Refusal("undetermined:near-critical", f"interval Newton did not contract near x = {x.str(15)}")


def _continuous(M, law, dilution):
    from .cavity import percolation_fixed_points
    fps = percolation_fixed_points(law, dilution)
    roots = []
    with baller().hygiene.ctx_guard(prec=PREC):
        for f in fps:
            w = 1 - f.u.arb
            x = w if M.kind == "site" or M.t == 0 else w / arb_q(M.t)      # bond: w = t x
            roots.append(Root(ball_of(x), f.jacobian, f.stable))
    return sorted(roots, key=lambda r: float(r.x.mid()))


def kcore_fixed_points(law, K, k, dilution) -> KcoreResult:
    M = Model(law, K, k, dilution)
    dil = occupation(dilution)
    from flint import arb
    bracket = None
    if M.continuous:
        roots = _continuous(M, law, dil)
        ivs = None
    else:
        with baller().hygiene.ctx_guard(prec=PREC):
            try:
                ivs = _isolate(M)
            except Refusal as r:
                from .kcore_threshold import refuse_at_threshold
                refuse_at_threshold(law, K, k, M.kind, M.t, r)
                raise
            roots = [Root(ball_of(arb(0)), ball_of(arb(0)), True)]
            for lo, hi, at1 in ivs:
                x = arb(1) if at1 else _newton(M, lo, hi)
                jac = M.dF(x)
                roots.append(Root(ball_of(x), ball_of(jac), bool(jac < 1)))
    if _HOOKS["drop_root"] == "largest" and len(roots) > 1:
        roots = roots[:-1]
    sel = roots[-1]
    with baller().hygiene.ctx_guard(prec=PREC):
        x = sel.x.arb
        iv = ivs[len(roots) - 2] if ivs is not None and len(roots) > 1 else None
        if ivs is not None:
            _no_root_above(M, iv[1] if iv is not None else _x0(M), sel.x)
        elif x == 0 and not sel.stable:
            raise Refusal("no-solution:dropped-root", "x = 0 is unstable (t G1'(1) > 1): the "
                          "nontrivial root of the 2-core equation is missing from the list")
        if not sel.jacobian.arb < 1:
            raise Refusal("ambiguous:critical-point", f"the largest fixed point x = {sel.x.render(12)} has "
                          f"F'(x) = {sel.jacobian.render(12)}, not certified < 1: a marginal root (the "
                          "occupation is at a core transition); not certified")
        if iv is not None and not iv[2]:
            bracket = _bracket(M, iv[0], x, sel.jacobian.arb)
        y = M.y_of(x)
        core, edges = M.core(y), M.edges(x)
        yb, cb, eb = ball_of(y), ball_of(core), ball_of(edges)
    phase = "core" if x > 0 else "no core"
    return KcoreResult(M, tuple(roots), sel, yb, cb, eb, bracket, phase)


__all__ = ["KcoreResult", "Model", "Root", "kcore_fixed_points"]
