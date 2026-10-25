"""Random K-XORSAT: leaf removal and the zero-temperature cavity (1RSB, y -> oo) closed forms.

Model.  N boolean variables, M = alpha N parity checks x_{i1} + ... + x_{iK} = y mod 2; each
check draws K distinct variables uniformly among the C(N, K) K-tuples, independently of the
other checks, and a uniform right-hand side y (Mezard, Ricci-Tersenghi & Zecchina 2003, Eq. (1),
gamma = alpha, p = K).  As N -> oo a variable's degree is Poisson(c), c = K alpha.

Recursion (leaf removal = zero-temperature warning propagation on the factor graph).  Let q be
the probability that a variable, reached along an edge from a check, still has another check
after leaf removal (it is "frozen" by the rest of the graph), and q_hat the probability that a
check sends a warning (all its K - 1 other variables are frozen):
    q_hat = q^(K-1),          q = 1 - G1(1 - q_hat; c)
with G0 the variable-degree generating function and G1 = G0'/G0'(1) the excess-degree one
(G0 = G1 = exp(c (z - 1)) for the Poisson law).  For the Poisson law this is MRZ Eq. (30)
(m = q) and Eq. (11) (1 - t = q_hat, lambda = c q_hat); the trivial root q = 0 always exists.
The physical root is the LARGEST one (leaf removal started from the full graph, q = 1).
  2-core:   variables  V = 1 - G0(1 - q_hat) - q_hat G0'(1 - q_hat)   (degree >= 2 in the core)
            checks     C = alpha q^K                                   (all K variables frozen)
  Poisson:  V = 1 - (1 + lambda) e^-lambda,  C = (lambda / K)(1 - e^-lambda)  (MRZ Eqs. (32)-(33))
  complexity Sigma = ln 2 (V - C): ln of the number of core solutions per variable (MRZ Eq. (13)
  and Sec. III.E: clusters <-> core solutions, 2^(N_c - M_c) of them).
  alpha_d   the smallest alpha with a nontrivial root: tangency Phi = dPhi/dq = 0 of
            Phi(q, alpha) = 1 - G1(1 - q^(K-1); c) - q; there the core jumps from 0 to V_d > 0.
  alpha_s   Sigma = 0 on the nontrivial branch (N_c = M_c, MRZ Eq. (42)).

The degree law is a HOOK: DEGREE_LAWS[name] (default "poisson") supplies g0, dg0, g1, dg1 and
the mean degree; every threshold is computed from it.  Results are cached per (K, law object),
so planting a different law object never returns a stale value.

  xorsat_thresholds(K, law="poisson") -> XorsatThresholds(alpha_d, alpha_s, q_d, lambda_d,
                                          core_d, q_s, lambda_s)
  xorsat_bp_threshold(K) / xorsat_sat_threshold(K) -> mpf
  xorsat_core_size(K, alpha) -> XorsatCore(q, q_hat, lam, variables, checks, complexity, trivial)
  xorsat_complexity(K, alpha) -> mpf
Refusals: precondition:integer-k-at-least-3, precondition:positive-alpha,
unsupported:degree-law, no-solution:bracket, no-solution:audit.
Precision: mp.workdps(DPS) inside every call (global precision untouched).
"""
import numbers
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp

from ..core import Refusal

DPS = 30
_GRID = 64          # sign scan of Phi on [q_d, 1] for the largest root


class PoissonLaw:
    """Variable degree ~ Poisson(c), c = mean_factor(K) * alpha (the true law: K alpha).
    G0(z) = G1(z) = exp(c (z - 1)).  `mean_factor` exists so a test can plant a wrong law."""

    def __init__(self, mean_factor=None, name="poisson"):
        self.mean_factor = mean_factor or (lambda k: k)
        self.name = name

    def mean(self, K, alpha):
        return self.mean_factor(K) * alpha

    def g0(self, z, c):
        return mp.exp(c * (z - 1))

    def dg0(self, z, c):
        return c * mp.exp(c * (z - 1))

    g1, dg1 = g0, dg0


DEGREE_LAWS = {"poisson": PoissonLaw()}


@dataclass(frozen=True)
class XorsatCore:
    K: int
    alpha: object
    q: object             # P(a variable reached from a check is frozen), largest root
    q_hat: object         # P(a check sends a warning) = q^(K-1)
    lam: object           # c q_hat: Poisson mean of the core degrees (MRZ lambda*)
    variables: object     # core variables / N
    checks: object        # core checks / N
    complexity: object    # ln 2 (variables - checks)
    trivial: bool         # True below alpha_d (empty core, q = 0)


@dataclass(frozen=True)
class XorsatThresholds:
    K: int
    alpha_d: object       # 2-core appears (dynamical / clustering threshold)
    alpha_s: object       # Sigma = 0 (satisfiability threshold)
    q_d: object           # the double root at alpha_d
    lambda_d: object      # MRZ lambda*(gamma_d)
    core_d: object        # V at alpha_d+ (the jump of the core)
    q_s: object
    lambda_s: object      # MRZ lambda_c; Pittel-Sorkin mu*


def _k(K):
    if isinstance(K, bool) or not isinstance(K, numbers.Integral) or K < 3:
        raise Refusal("precondition:integer-k-at-least-3",
                      f"K = {K!r}: the closed forms here are for integer K >= 3 (K = 2 has a "
                      "continuous core transition, alpha_d = alpha_s = 1/2, not implemented)")
    return int(K)


def _alpha(alpha):
    try:
        a = alpha if isinstance(alpha, mp.mpf) else mp.mpf(str(alpha))
    except (TypeError, ValueError):
        a = None
    if a is None or not mp.isfinite(a) or a <= 0:
        raise Refusal("precondition:positive-alpha",
                      f"alpha = {alpha!r}: the check density M/N must be a positive number")
    return a


def _law(name):
    if name not in DEGREE_LAWS:
        raise Refusal("unsupported:degree-law",
                      f"degree law {name!r}; implemented: {', '.join(sorted(DEGREE_LAWS))}")
    return DEGREE_LAWS[name]


def _phi(law, K, a, q):
    """Phi(q, alpha) = leaf-removal map minus q."""
    return 1 - law.g1(1 - q**(K - 1), law.mean(K, a)) - q


def _dphi_dq(law, K, a, q):
    return law.dg1(1 - q**(K - 1), law.mean(K, a)) * (K - 1) * q**(K - 2) - 1


def _alpha_star(law, K, q):
    """The alpha at which q is a fixed point (Phi increasing in alpha), by a bracketed root."""
    hi = mp.mpf(1)
    while _phi(law, K, hi, q) < 0:
        hi *= 2
        if hi > 1e6:
            raise Refusal("no-solution:bracket", f"no alpha makes q = {q} a fixed point")
    return mp.findroot(lambda a: _phi(law, K, a, q), (hi / 2**10, hi), solver="anderson")


@lru_cache(maxsize=None)
def _tangency(K, law):
    """(alpha_d, q_d): min over q of alpha*(q), refined as the double root Phi = dPhi/dq = 0."""
    with mp.workdps(DPS):
        qs = [mp.mpf(i) / 40 for i in range(1, 40)]
        with mp.workdps(15):
            al = [_alpha_star(law, K, q) for q in qs]
        i = min(range(len(al)), key=lambda j: al[j])
        if i in (0, len(al) - 1):
            raise Refusal("no-solution:bracket", f"alpha*(q) has no interior minimum (K = {K})")
        a, q = mp.findroot([lambda a, q: _phi(law, K, a, q), lambda a, q: _dphi_dq(law, K, a, q)],
                           (al[i], qs[i]))
        if not (0 < q < 1 and abs(_phi(law, K, a, q)) < mp.mpf(10)**(5 - DPS)
                and _alpha_star(law, K, q - mp.mpf("1e-3")) > a < _alpha_star(law, K, q + mp.mpf("1e-3"))):
            raise Refusal("no-solution:audit", f"tangency (alpha, q) = ({a}, {q}) failed its audit")
        return +a, +q


def _core(K, a, law):
    ad, qd = _tangency(K, law)
    c = law.mean(K, a)
    if a < ad:
        z = mp.mpf(0)
        return XorsatCore(K, a, z, z, z, z, z, z, True)
    if a == ad:
        q = qd
    else:
        grid = [qd + (1 - qd) * mp.mpf(i) / _GRID for i in range(_GRID + 1)]
        vals = [_phi(law, K, a, q) for q in grid]
        if vals[0] <= 0 or vals[-1] >= 0:
            raise Refusal("no-solution:bracket", f"Phi(q_d) = {vals[0]}, Phi(1) = {vals[-1]} at "
                          f"alpha = {a}: no sign change on [q_d, 1]")
        i = max(j for j in range(_GRID) if vals[j] > 0)
        q = mp.findroot(lambda x: _phi(law, K, a, x), (grid[i], grid[i + 1]), solver="anderson")
        if abs(_phi(law, K, a, q)) > mp.mpf(10)**(5 - DPS):
            raise Refusal("no-solution:audit", f"leaf-removal root q = {q} at alpha = {a}")
    qh = q**(K - 1)
    v = 1 - law.g0(1 - qh, c) - qh * law.dg0(1 - qh, c)
    ch = a * q**K
    return XorsatCore(K, a, +q, +qh, +(c * qh), +v, +ch, +(mp.log(2) * (v - ch)), False)


@lru_cache(maxsize=None)
def _thresholds(K, law):
    with mp.workdps(DPS):
        ad, qd = _tangency(K, law)
        sigma = lambda a: _core(K, a, law).complexity
        cd = _core(K, ad, law)
        if cd.complexity <= 0:
            raise Refusal("no-solution:bracket", f"Sigma(alpha_d) = {cd.complexity} <= 0 (K = {K})")
        hi = ad + mp.mpf("0.05")
        while sigma(hi) > 0:
            hi += mp.mpf("0.05")
            if hi > 8:
                raise Refusal("no-solution:bracket", f"Sigma stays positive up to alpha = 8 (K = {K})")
        a_s = mp.findroot(sigma, (ad, hi), solver="anderson")
        cs = _core(K, a_s, law)
        if abs(cs.complexity) > mp.mpf(10)**(5 - DPS):
            raise Refusal("no-solution:audit", f"Sigma(alpha_s = {a_s}) = {cs.complexity}")
        return XorsatThresholds(K, +ad, +a_s, +qd, +cd.lam, +cd.variables, +cs.q, +cs.lam)


def xorsat_thresholds(K, law="poisson") -> XorsatThresholds:
    return _thresholds(_k(K), _law(law))


def xorsat_bp_threshold(K, law="poisson"):
    """alpha_d(K): the leaf-removal (BP / warning-propagation) fixed point becomes nontrivial."""
    return xorsat_thresholds(K, law).alpha_d


def xorsat_sat_threshold(K, law="poisson"):
    """alpha_s(K): the complexity of the core vanishes (N_c = M_c)."""
    return xorsat_thresholds(K, law).alpha_s


def xorsat_core_size(K, alpha, law="poisson") -> XorsatCore:
    K, a, lw = _k(K), _alpha(alpha), _law(law)
    with mp.workdps(DPS):
        return _core(K, +a, lw)


def xorsat_complexity(K, alpha, law="poisson"):
    return xorsat_core_size(K, alpha, law).complexity


def fixed_point_alpha(K, q, law="poisson"):
    """alpha*(q): the density at which q is a leaf-removal fixed point (alpha_d = its minimum)."""
    K, lw = _k(K), _law(law)
    with mp.workdps(DPS):
        return +_alpha_star(lw, K, mp.mpf(str(q)) if not isinstance(q, mp.mpf) else q)


def leaf_map(K, alpha, q, law="poisson"):
    """The recursion's right-hand side 1 - G1(1 - q^(K-1); c) at (K, alpha, q)."""
    K, a, lw = _k(K), _alpha(alpha), _law(law)
    with mp.workdps(DPS):
        q = q if isinstance(q, mp.mpf) else mp.mpf(str(q))
        return +(_phi(lw, K, a, q) + q)


__all__ = ["DEGREE_LAWS", "DPS", "PoissonLaw", "XorsatCore", "XorsatThresholds", "fixed_point_alpha",
           "leaf_map", "xorsat_bp_threshold", "xorsat_complexity", "xorsat_core_size",
           "xorsat_sat_threshold", "xorsat_thresholds"]
