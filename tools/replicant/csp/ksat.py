"""Random K-SAT: instances, exhaustive counts, the annealed (first-moment) bound, and belief
propagation on an instance.

Ensemble.  N boolean variables, M = round(alpha N) clauses; each clause is the OR of K literals
on K distinct variables drawn uniformly (rows with a repeated variable are redrawn), each
literal negated with probability 1/2, clauses independent (Braunstein, Mezard & Zecchina 2005,
Sec. V: "k-tuples of variable indices at random (with no repetitions)", signs with probability
0.5).  numpy default_rng(seed) per instance, so a named seed names an instance.  As N -> oo a
variable's number of unnegated (negated) occurrences is Poisson(K alpha / 2), independently
(Mertens, Mezard & Zecchina 2006, Sec. 3).  Ding, Sly & Sun draw literals independently (a
variable may repeat in a clause) and M ~ Poisson(n alpha): the same N -> oo local structure.

  random_ksat(K, N, alpha, seed) -> KSatFormula(K, N, alpha, seed, var (M, K), sign (M, K))
      sign +1: the variable appears unnegated (x), -1: negated (not x)
  ksat_annealed_bound(K, alpha) -> mpf   (1/N) ln E Z = ln 2 + alpha ln(1 - 2^-K): every
      assignment violates a clause with probability 2^-K, clauses independent, so
      E Z = 2^N (1 - 2^-K)^M exactly (DSS, Eq. (2.3)).  An upper bound on the quenched entropy
      (1/N) E ln Z (Jensen), and by Markov's inequality on the threshold:
  ksat_first_moment_threshold(K) -> mpf  alpha_1 = ln 2 / -ln(1 - 2^-K), the root of the bound
  ksat_sp_support_threshold(K) / ksat_sp_support(K, alpha)   alpha_t and (t, tau) of MMZ Eq.
      (39): the fraction t of surveys that are exactly zero (a property of the support only)
  ksat_exact_marginals(f) -> (Z, ones)   exhaustive enumeration of all 2^N assignments
      (N <= 22); ones[i] = number of solutions with x_i = 1.   ksat_exact_count(f) -> Z
  ksat_count_reference(f) -> Z            recursive backtracking count (pure Python), for tests
  ksat_bp(f, *, iters, damping, seed, tol) -> BPSATResult   BMZ Sec. III.B, Eqs. (10)-(14)
  ksat_bp_converged_fraction(K, N, alpha, seeds, *, iters, damping) -> ConvergedFraction
      (99% Clopper-Pearson band over the named seeds)

BP (BMZ Eqs. (10)-(12)).  delta_{a->i} = prod_{j in a, j != i} gamma_{j->a}: the probability
that every other variable of a violates it; gamma_{j->a} = P^u / (P^u + P^s) with
P^u = prod over the other clauses where j has the SAME sign as in a of (1 - delta_{b->j}) and
P^s the same product over the clauses where j has the OPPOSITE sign.  Synchronous (parallel)
updates, delta <- damping delta_old + (1 - damping) delta_new, started from uniform random
messages (seed); converged when max |delta_new - delta_old| < tol within `iters` sweeps (BMZ
use sequential random-order sweeps: a convergence statement here is about this schedule).
Marginal P(x_i = 1) by Eq. (13), entropy per variable S/N by Eq. (14); both exact on trees
(BMZ Theorem 2), an approximation on loopy instances.  `converged` is never set without the
residual test passing.
Refusals: precondition:integer-k-at-least-2 (-3 for the support threshold), :positive-alpha, :enough-variables (N >= K),
:named-seeds, :exhaustive-size, :damping (0 <= damping < 1), :iterations.
"""
import math
import numbers
from dataclasses import dataclass

import mpmath as mp
import numpy as np

from ..core import Refusal
from . import _checks
from .xorsat import _alpha
from .xorsat_finite import clopper_pearson

EXHAUSTIVE_MAX = 22
# Measured window of tests/test_ksat.py::test_bp_converges_below_alpha_d_and_not_above.
# Converged instances out of seeds 0-19 (synchronous BP, tol 1e-8, damping 0.5, 1000 sweeps):
#   N = 2000:  3.5 -> 20, 3.8 -> 8, 3.9 -> 2, 4.0 / 4.1 / 4.2 -> 0  (damping 0.8, 2000 sweeps:
#              3.8 -> 8, 3.9 -> 2, 4.0 -> 0: the edge is not a damping artefact)
#   N = 6000:  3.7 -> 18, 3.8 -> 3, 3.9 -> 0
# Undamped synchronous sweeps fail much earlier (N = 2000, 300 sweeps: 3.0 -> 20, 3.5 -> 11,
# 3.7 -> 0): that edge is the schedule's, not the instance's.  The measured edge (3.7 - 3.9)
# brackets alpha_d = 3.86 (KMRSZ 2007) but is not a measurement of it.
BP_WINDOW = {"N": 2000, "seeds": tuple(range(20)), "iters": 1000, "damping": 0.5,
             "below": "3.5", "above": "4.0"}


def _k(K):
    if isinstance(K, bool) or not isinstance(K, numbers.Integral) or K < 2:
        raise Refusal("precondition:integer-k-at-least-2",
                      f"K = {K!r}: clauses need an integer number K >= 2 of literals")
    return int(K)


def _n(N, K):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < K:
        raise Refusal("precondition:enough-variables",
                      f"N = {N!r}: need an integer N >= K = {K} (K distinct variables per clause)")
    return int(N)


@dataclass(frozen=True, eq=False)
class KSatFormula:
    K: int
    N: int
    alpha: float
    seed: object
    var: np.ndarray       # (M, K) int64 variable indices, distinct within a row
    sign: np.ndarray      # (M, K) int8: +1 unnegated, -1 negated

    @property
    def M(self):
        return len(self.var)


def random_ksat(K, N, alpha, seed):
    K = _k(K)
    seed = _checks.seed(seed)
    N, a = _n(N, K), float(_alpha(alpha))
    M = int(round(a * N))
    rng = np.random.default_rng(seed)
    var = rng.integers(0, N, size=(M, K))
    while True:                                  # redraw rows with a repeated variable
        s = np.sort(var, axis=1)
        bad = np.flatnonzero((s[:, 1:] == s[:, :-1]).any(axis=1))
        if bad.size == 0:
            break
        var[bad] = rng.integers(0, N, size=(bad.size, K))
    sign = np.where(rng.random((M, K)) < 0.5, -1, 1).astype(np.int8)
    return KSatFormula(K, N, a, seed, var.astype(np.int64), sign)


def ksat_annealed_bound(K, alpha):
    K, a = _k(K), _alpha(alpha)
    with mp.workdps(30):
        return +(mp.log(2) + mp.mpf(a) * mp.log(1 - mp.mpf(2)**-K))


def ksat_first_moment_threshold(K):
    K = _k(K)
    with mp.workdps(30):
        return +(mp.log(2) / -mp.log(1 - mp.mpf(2)**-K))


# ---- support of the SP surveys (MMZ Sec. 5.1) ----------------------------------------------
def _k3(K):
    if isinstance(K, bool) or not isinstance(K, numbers.Integral) or K < 3:
        raise Refusal("precondition:integer-k-at-least-3",
                      f"K = {K!r}: the support threshold is a tangency only for K >= 3 (K = 2: "
                      "a continuous onset at alpha = 1, not implemented)")
    return int(K)


def _support_alpha(K, u):
    """alpha at which u = 1 - t is a fixed point of u = (1 - exp(-K alpha u / 2))^(K-1)."""
    return -2 * mp.log(1 - u**(mp.mpf(1) / (K - 1))) / (K * u)


def ksat_sp_support_threshold(K):
    """alpha_t(K): the smallest alpha where MMZ Eq. (39), t = 1 - (1 - tau)^(K-1),
    tau = exp(-(K alpha / 2)(1 - t)), has a solution t < 1 (a survey is exactly zero iff some
    input variable has no non-zero survey from its opposite-sign clauses).  Below it every
    survey is exactly 0; above it the non-zero surveys may still decay to 0 (they do below the
    SP onset ~3.9 for K = 3).  Tangency min over u of alpha*(u), audited on both sides."""
    K = _k3(K)
    with mp.workdps(30):
        us = [mp.mpf(i) / 50 for i in range(1, 50)]
        u0 = min(us, key=lambda u: _support_alpha(K, u))
        um = mp.findroot(lambda u: mp.diff(lambda v: _support_alpha(K, v), u), u0)
        at = _support_alpha(K, um)
        if not (0 < um < 1 and _support_alpha(K, um - mp.mpf("1e-3")) > at
                and _support_alpha(K, um + mp.mpf("1e-3")) > at):
            raise Refusal("no-solution:audit", f"K = {K}: tangency u = {um} is not a minimum")
        return +at


def ksat_sp_support(K, alpha):
    """(t, tau) of MMZ Eq. (39) at alpha: the stable (smallest-t) solution, reached by
    iterating u -> (1 - exp(-K alpha u / 2))^(K-1) from u = 1; t = 1 (all surveys zero) below
    alpha_t."""
    K, a = _k(K), _alpha(alpha)
    with mp.workdps(30):
        g, u = K * a / 2, mp.mpf(1)
        for _ in range(100000):
            new = (1 - mp.exp(-g * u))**(K - 1)
            if abs(new - u) < mp.mpf("1e-25") or new < mp.mpf("1e-20"):
                u = new
                break
            u = new
        if u < mp.mpf("1e-20"):
            return mp.mpf(1), mp.mpf(1)
        u = mp.findroot(lambda v: (1 - mp.exp(-g * v))**(K - 1) - v, u)
        return +(1 - u), +mp.exp(-g * u)


# ---- exhaustive counts -----------------------------------------------------------------
def ksat_exact_marginals(f):
    if f.N > EXHAUSTIVE_MAX:
        raise Refusal("precondition:exhaustive-size",
                      f"N = {f.N}: exhaustive enumeration is limited to N <= {EXHAUSTIVE_MAX}")
    x = np.arange(1 << f.N, dtype=np.uint32)
    bits = [((x >> np.uint32(i)) & np.uint32(1)).astype(bool) for i in range(f.N)]
    alive = np.ones(x.size, dtype=bool)
    for row, sg in zip(f.var.tolist(), f.sign.tolist()):
        sat = np.zeros(x.size, dtype=bool)
        for v, s in zip(row, sg):
            sat |= bits[v] if s > 0 else ~bits[v]
        alive &= sat
    z = int(alive.sum())
    return z, [int((alive & b).sum()) for b in bits]


def ksat_exact_count(f):
    return ksat_exact_marginals(f)[0]


def ksat_count_reference(f):
    """Backtracking over x_0, x_1, ...: a clause whose literals are all assigned and false
    prunes the branch; independent of the vectorised enumeration."""
    clauses = [list(zip(r, s)) for r, s in zip(f.var.tolist(), f.sign.tolist())]
    last = [max(v for v, _ in c) for c in clauses]
    by_last = [[c for c, l in zip(clauses, last) if l == i] for i in range(f.N)]
    x = [0] * f.N

    def rec(i):
        if i == f.N:
            return 1
        total = 0
        for val in (0, 1):
            x[i] = val
            if all(any((x[v] == 1) == (s > 0) for v, s in c) for c in by_last[i]):
                total += rec(i + 1)
        return total

    return rec(0)


# ---- belief propagation ----------------------------------------------------------------
@dataclass(frozen=True)
class BPSATResult:
    converged: bool
    iterations: int
    residual: float
    marginals: np.ndarray    # P(x_i = 1), BMZ Eq. (13)
    entropy: float           # S / N, BMZ Eq. (14); nan when BP gives no positive count
    messages: np.ndarray     # delta_{a->i}, shape (M, K)
    note: str = ""


def _group_logs(f, m):
    """Per (variable, sign) group: sum of ln(1 - m) over its edges and the number of exact
    zeros (m = 1), which are counted, never logged.  Group key 2 i + 1: x_i unnegated."""
    key = 2 * f.var + (f.sign > 0)
    one = m >= 1.0
    with np.errstate(divide="ignore"):
        lg = np.where(one, 0.0, np.log1p(-np.where(one, 0.0, m)))
    L = np.bincount(key.ravel(), weights=lg.ravel(), minlength=2 * f.N)
    Z = np.bincount(key.ravel(), weights=one.ravel().astype(float), minlength=2 * f.N)
    return key, lg, one, L, Z


def group_products(f, m):
    """For messages m (M, K) on the edges, (same, opp) (M, K): for edge (a, i) the product of
    (1 - m_{b->i}) over the OTHER clauses b where i has the same sign as in a, and over the
    clauses where i has the opposite sign."""
    key, lg, one, L, Z = _group_logs(f, m)
    opp = key ^ 1
    same = np.where(Z[key] - one > 0, 0.0, np.exp(L[key] - lg))
    other = np.where(Z[opp] > 0, 0.0, np.exp(L[opp]))
    return same, other


def var_products(f, m):
    """(P+, P-) per variable: prod of (1 - m) over all the clauses where it appears unnegated
    (negated)."""
    _, _, _, L, Z = _group_logs(f, m)
    P = np.where(Z > 0, 0.0, np.exp(L))
    return P[1::2], P[0::2]


def others_product(g):
    """out[:, k] = prod over k' != k of g[:, k'] (K small: direct loop, zeros safe)."""
    K = g.shape[1]
    return np.stack([np.prod(np.delete(g, k, axis=1), axis=1) for k in range(K)], axis=1)


def bp_rule(pu, ps):
    """BMZ Eq. (11): gamma = P^u / (P^u + P^s), the probability that the variable violates
    the clause in the cavity graph."""
    return pu / (pu + ps)


def _bp_iter_args(iters, damping):
    if isinstance(iters, bool) or not isinstance(iters, numbers.Integral) or iters < 1:
        raise Refusal("precondition:iterations", f"iters = {iters!r}: need an integer >= 1")
    if not (isinstance(damping, (int, float)) and 0 <= damping < 1):
        raise Refusal("precondition:damping", f"damping = {damping!r}: need 0 <= damping < 1")
    return int(iters), float(damping)


def ksat_bp(f, *, iters=1000, damping=0.0, seed=0, tol=1e-8):
    iters, damping = _bp_iter_args(iters, damping)
    rng = np.random.default_rng(_checks.seed(seed))
    d = rng.random(f.var.shape)
    res, it, ok, note = math.inf, 0, False, ""
    with np.errstate(invalid="ignore", divide="ignore"):
        for it in range(1, iters + 1):
            pu, ps = group_products(f, d)          # P^u: same-sign others, P^s: opposite
            new = others_product(bp_rule(pu, ps))
            if not np.all(np.isfinite(new)):
                note = "contradictory messages (P^u = P^s = 0 on an edge): BP has no fixed point here"
                break
            new = damping * d + (1 - damping) * new
            res = float(np.max(np.abs(new - d))) if new.size else 0.0
            d = new
            if res < tol:
                ok = True
                break
        pu, ps = group_products(f, d)
        pp, pm = var_products(f, d)
        marg = pm / (pm + pp)
        clause = np.prod(pu + ps, axis=1) - np.prod(pu, axis=1)
        deg = np.bincount(f.var.ravel(), minlength=f.N)
        s = np.sum(np.log(clause)) + np.sum((1 - deg) * np.log(pp + pm))
    ent = float(s) / f.N if np.isfinite(s) else math.nan
    if not ok and not note:
        note = f"not converged: residual {res:.3g} >= tol {tol:g} after {it} sweeps"
    return BPSATResult(ok, it, res, marg, ent, d, note)


@dataclass(frozen=True)
class ConvergedFraction:
    K: int
    N: int
    alpha: float
    seeds: tuple
    n: int
    n_conv: int
    fraction: float
    lo: float             # exact 99% Clopper-Pearson interval
    hi: float
    iters: int
    damping: float


def ksat_bp_converged_fraction(K, N, alpha, seeds, *, iters=300, damping=0.0, tol=1e-8):
    seeds = _checks.seeds(seeds)
    iters, damping = _bp_iter_args(iters, damping)
    k = sum(ksat_bp(random_ksat(K, N, alpha, s), iters=iters, damping=damping, seed=s,
                    tol=tol).converged for s in seeds)
    lo, hi = clopper_pearson(k, len(seeds))
    return ConvergedFraction(_k(K), int(N), float(_alpha(alpha)), seeds, len(seeds), k,
                             k / len(seeds), lo, hi, iters, damping)


__all__ = ["BPSATResult", "bp_rule", "ksat_sp_support", "ksat_sp_support_threshold", "BP_WINDOW", "ConvergedFraction", "KSatFormula", "ksat_annealed_bound",
           "ksat_bp", "ksat_bp_converged_fraction", "ksat_count_reference", "ksat_exact_count",
           "ksat_exact_marginals", "ksat_first_moment_threshold", "random_ksat"]
