"""The cluster-size distribution P_s (the probability that an occupied vertex lies in a finite cluster
of exactly s vertices) by two algorithms on the generating functions, both in arb balls:

  series iteration (NSW Sec. II C after Eq. (27); CNSW after Eq. (9)): iterate the truncated power
      series H1 <- 1 - t + t x G1(H1) n + 1 times from H1 = 1, then H0 = x G0(H1) and P_s = [x^s] H0.
      (NSW's text says "s + 1 iterations of Eq. (27), starting with H1 = 1"; the equation iterated must
      be (26), the H1 equation, as CNSW's "iterating Eq. (5) n + 1 times" has it: an apparent slip in
      NSW.)  Each iteration fixes one more coefficient (n iterations suffice for P_n).
  Lagrange inversion (derived here): with a = 1 - t and v = H1 - a, v = x t G1(a + v), and
      P_1 = G0(a),   P_s = (1/(s - 1)) [v^(s-2)] G0'(a + v) (t G1(a + v))^(s-1)   (s >= 2).
Both read the law's G0 / G1 (percolation/laws.py), so a wrong G0 or G1 moves both: the independent
checks are the closed forms (closed_forms.py: the Bethe lattice's s g_s t^(s-1) (1-t)^((k-2)s+2) and
the Borel law of the Poisson graph) and the sampled graphs.  For site dilution the distribution over
occupied vertices equals the bond one at the same t (on a tree the perimeter edges and perimeter sites
are in one-to-one correspondence; derived here); over all vertices it is t P_s.
Sum rule (NSW after Eq. (33)): sum_s P_s = H0(1) = 1 - S_occ, so every partial sum is <= 1 - S_occ;
a partial sum above it (beyond the balls) is refused no-solution:sum-rule.

  cluster_distribution(law, dilution, s_max) -> ClusterDistribution(series, lagrange) (lists of
      Balls, P_1 .. P_s_max); the two must overlap (else no-solution:route-disagreement)
_HOOKS["extra_iterations"] (1): the series iterations beyond n; planted -1 leaves P_n unconverged.
"""
from dataclasses import dataclass

from ..core import Refusal
from ..lattice.ball import arb_q, ball_of, baller
from .laws import occupation, parse_law

S_MAX = 60
_HOOKS = {"extra_iterations": 1}       # NSW's s + 1 iterations; planted -1 (n - 1): P_n wrong (tests)
_PREC = 256


def _mul(a, b, n):
    out = [0 * a[0]] * (n + 1)
    for i, ai in enumerate(a[:n + 1]):
        if ai == 0:
            continue
        for j, bj in enumerate(b[:n + 1 - i]):
            out[i + j] = out[i + j] + ai * bj
    return out


def _compose(L, which, H, n):
    """G(H(x)) truncated at x^n, G = G0 or G1 of the law, H a series (list of arbs)."""
    zero = 0 * H[0]
    if L.kind == "poisson":
        c = arb_q(L.c)
        e = [(c * (H[0] - 1)).exp()] + [zero] * n
        for m in range(1, n + 1):          # E' = c H' E: m e_m = c sum_k k h_k e_(m-k)
            s = zero
            for k in range(1, m + 1):
                if k < len(H):
                    s = s + k * H[k] * e[m - k]
            e[m] = c * s / m
        return e
    coeffs = L.degree() if which == "G0" else L.excess()
    out = [zero] * (n + 1)
    for j in range(max(coeffs), -1, -1):     # Horner in series arithmetic
        out = _mul(out, H, n)
        out[0] = out[0] + arb_q(coeffs.get(j, 0))
    return out


def _series(L, t, n):
    T = arb_q(t)
    H = [arb_q(1)] + [arb_q(0)] * n
    for _ in range(n + _HOOKS["extra_iterations"]):     # m iterations fix H1 to x^(m-1); P_n needs n
        g = _compose(L, "G1", H, n)
        H = [1 - T] + [T * c for c in g[:n]]
    H0 = [arb_q(0)] + _compose(L, "G0", H, n)[:n]
    return H0[1:]


def _lagrange(L, t, n):
    T = arb_q(t)
    a = 1 - T
    out = [L.G0(a)]
    if n < 2:
        return out
    d0 = L.taylor(a, n - 2, "dG0")
    g1 = [T * c for c in L.taylor(a, n - 2, "G1")]
    power = [arb_q(1)] + [arb_q(0)] * (n - 2)
    for s in range(2, n + 1):
        power = _mul(power, g1, n - 2)        # (t G1)^(s-1)
        prod = _mul(d0, power, s - 2)
        out.append(prod[s - 2] / (s - 1))
    return out


@dataclass(frozen=True)
class ClusterDistribution:
    s_max: int
    series: tuple
    lagrange: tuple


def cluster_distribution(law, dilution, s_max, s_occupied=None):
    L = parse_law(law)
    _, t = occupation(dilution)
    if isinstance(s_max, bool) or not isinstance(s_max, int) or not 1 <= s_max <= S_MAX:
        raise Refusal("precondition:s-max", f"s_max = {s_max!r}: need an int in 1 .. {S_MAX}")
    with baller().hygiene.ctx_guard(prec=_PREC):
        ser, lag = _series(L, t, s_max), _lagrange(L, t, s_max)
        for s, (a, b) in enumerate(zip(ser, lag), 1):
            if not a.overlaps(b):
                raise Refusal("no-solution:route-disagreement", f"P_{s}: series iteration {a.str(12)}"
                              f" vs Lagrange inversion {b.str(12)}")
        if s_occupied is not None:
            total = sum(lag[1:], lag[0])
            if total > 1 - s_occupied.arb:
                raise Refusal("no-solution:sum-rule", f"sum_(s <= {s_max}) P_s = {total.str(12)} exceeds "
                              f"1 - S = {(1 - s_occupied.arb).str(12)}")
        return ClusterDistribution(s_max, tuple(ball_of(x) for x in ser), tuple(ball_of(x) for x in lag))


__all__ = ["ClusterDistribution", "S_MAX", "cluster_distribution"]
