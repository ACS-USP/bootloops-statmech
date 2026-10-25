"""Random K-XORSAT at finite N: sampled instances, exact GF(2) elimination, leaf removal.

Independent of xorsat.py (it imports only its input checks): nothing here reads the degree-law
hook or any closed form.  Ensemble (as in xorsat.py): M = round(alpha N) checks, each on K
distinct variables drawn uniformly, independent checks, uniform right-hand sides; numpy
default_rng(seed) per instance, so a seed names an instance.

  xorsat_instance(K, N, alpha, seed) -> (idx (M, K) int64, rhs (M,) uint8)
  gf2_solve(idx, rhs, N) -> (satisfiable, rank)   forward Gaussian elimination over GF(2) on
      the augmented matrix [A | b], rows bit-packed in uint64 words (bit j of a row = variable
      j, bit N = right-hand side); unsatisfiable iff a zero coefficient row keeps b = 1
  gf2_rank_reference(idx, N)   rank of A by a different algorithm (an XOR basis of Python
      integers keyed by the leading bit), for tests
  xorsat_gf2_satisfiable(K, N, alpha, seed) -> bool
  xorsat_sat_probability(K, N, alpha, seeds) -> SatProbability: the satisfiable fraction over
      the named seeds, its binomial stderr and the exact 99% Clopper-Pearson interval (its
      larger half-width `halfwidth99` is the reported uncertainty: the stderr is 0 when every
      seed agrees)
  leaf_removal_core(idx, N) -> (core variables, core checks)   repeatedly delete a variable of
      degree 1 with its check (degree-0 variables are not in the core); what is left is the
      2-core, independent of the deletion order
  xorsat_sampled_core(K, N, alpha, seed) -> (V, C) the core fractions of one instance
Statistical: a fraction over seeds is a binomial estimate, never certified.
Refusals: precondition:integer-k-at-least-3, :positive-alpha, :enough-variables (N >= max(12,
4K): smaller systems are not a finite-size check of anything), :named-seeds (seeds must be non-negative
integers: numpy's default_rng refuses negative ones).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import _checks
from .xorsat import _alpha, _k

CONF = 0.99


def _n(N, K):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < max(12, 4 * K):
        raise Refusal("precondition:enough-variables",
                      f"N = {N!r}: need an integer N >= max(12, 4K) = {max(12, 4 * K)}")
    return int(N)


_seed, _seeds = _checks.seed, _checks.seeds      # shared with csp/ksat.py


def xorsat_instance(K, N, alpha, seed):
    seed = _seed(seed)
    K = _k(K)
    N, a = _n(N, K), float(_alpha(alpha))
    M = int(round(a * N))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(M, K))
    while True:                                  # resample rows with a repeated variable
        s = np.sort(idx, axis=1)
        bad = np.flatnonzero((s[:, 1:] == s[:, :-1]).any(axis=1))
        if bad.size == 0:
            break
        idx[bad] = rng.integers(0, N, size=(bad.size, K))
    rhs = rng.integers(0, 2, size=M, dtype=np.uint8)
    return idx.astype(np.int64), rhs


def _packed(idx, rhs, N):
    M = len(rhs)
    W = (N + 1 + 63) // 64
    A = np.zeros((M, W), dtype=np.uint64)
    rows = np.repeat(np.arange(M), idx.shape[1])
    cols = idx.reshape(-1)
    np.bitwise_xor.at(A, (rows, cols >> 6), np.left_shift(np.uint64(1), (cols & 63).astype(np.uint64)))
    hit = np.flatnonzero(np.asarray(rhs) & 1)
    A[hit, N >> 6] ^= np.uint64(1) << np.uint64(N & 63)
    return A


def gf2_solve(idx, rhs, N):
    """(satisfiable, rank of A) of A x = b over GF(2); idx (M, K) variable indices per row."""
    idx = np.asarray(idx, dtype=np.int64)
    A = _packed(idx, rhs, N)
    M, r = A.shape[0], 0
    for j in range(N):
        if r == M:
            break
        w, b = j >> 6, np.uint64(1) << np.uint64(j & 63)
        nz = np.flatnonzero(A[r:, w] & b)
        if nz.size == 0:
            continue
        if nz[0]:                    # swap the first row with bit j up; the old row r has no bit j
            p = r + int(nz[0])
            A[[r, p]] = A[[p, r]]
        if nz.size > 1:
            A[r + nz[1:], w:] ^= A[r, w:]
        r += 1
    bad = A[r:, N >> 6] & (np.uint64(1) << np.uint64(N & 63))
    return (not bool(bad.any())), r


def gf2_rank_reference(idx, N):
    """Rank of the coefficient matrix by an XOR basis of Python integers (leading bit)."""
    basis = {}
    for row in np.asarray(idx).tolist():
        x = 0
        for j in row:
            x ^= 1 << int(j)
        while x:
            top = x.bit_length() - 1
            if top not in basis:
                basis[top] = x
                break
            x ^= basis[top]
    return len(basis)


def xorsat_gf2_satisfiable(K, N, alpha, seed):
    idx, rhs = xorsat_instance(K, N, alpha, seed)
    return gf2_solve(idx, rhs, int(N))[0]


@dataclass(frozen=True)
class SatProbability:
    K: int
    N: int
    alpha: float
    seeds: tuple
    n: int
    n_sat: int
    fraction: float
    stderr: float         # binomial sqrt(p (1 - p) / n) (0 when every seed agrees)
    lo: float             # exact 99% Clopper-Pearson interval
    hi: float
    halfwidth99: float    # max(fraction - lo, hi - fraction): the reported uncertainty


def clopper_pearson(k, n, conf=CONF):
    """Exact two-sided interval: quantiles of Beta(k, n - k + 1) and Beta(k + 1, n - k)."""
    from scipy.special import betaincinv
    t = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(betaincinv(k, n - k + 1, t))
    hi = 1.0 if k == n else float(betaincinv(k + 1, n - k, 1 - t))
    return lo, hi


def xorsat_sat_probability(K, N, alpha, seeds):
    seeds = _seeds(seeds)
    K = _k(K)
    N, a = _n(N, K), float(_alpha(alpha))
    k = sum(xorsat_gf2_satisfiable(K, N, alpha, s) for s in seeds)
    n = len(seeds)
    p = k / n
    lo, hi = clopper_pearson(k, n)
    return SatProbability(K, N, a, seeds, n, k, p, math.sqrt(p * (1 - p) / n), lo, hi,
                          max(p - lo, hi - p))


def leaf_removal_core(idx, N):
    """(core variables, core checks) after leaf removal on the hypergraph of rows `idx`."""
    idx = np.asarray(idx, dtype=np.int64)
    M = len(idx)
    if M == 0:
        return 0, 0
    flat = idx.reshape(-1)
    deg = np.bincount(flat, minlength=N).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    start = np.concatenate(([0], np.cumsum(np.bincount(flat, minlength=N))))
    check_of = (order // idx.shape[1]).tolist()
    start, rows = start.tolist(), idx.tolist()
    alive = [True] * M
    deg_l = deg.tolist()
    stack = [v for v in range(N) if deg_l[v] == 1]
    while stack:
        v = stack.pop()
        if deg_l[v] != 1:
            continue
        for e in range(start[v], start[v + 1]):
            m = check_of[e]
            if alive[m]:
                break
        alive[m] = False
        for u in rows[m]:
            deg_l[u] -= 1
            if deg_l[u] == 1:
                stack.append(u)
    return sum(d >= 2 for d in deg_l), sum(alive)


def xorsat_sampled_core(K, N, alpha, seed):
    idx, _ = xorsat_instance(K, N, alpha, seed)
    v, c = leaf_removal_core(idx, int(N))
    return v / N, c / N


__all__ = ["SatProbability", "clopper_pearson", "gf2_rank_reference", "gf2_solve",
           "leaf_removal_core", "xorsat_gf2_satisfiable", "xorsat_instance", "xorsat_sampled_core",
           "xorsat_sat_probability"]
