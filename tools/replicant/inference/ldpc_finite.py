"""Regular (l, r) LDPC codes at finite N on the BEC: configuration-model graphs and the peeling
decoder.

Independent of the density-evolution code in ldpc.py (it imports only its input checks):
nothing here reads the DEGREE_PAIRS hook or any closed form.

Ensemble (configuration model).  N variables with l sockets each, M = N l / r checks with r
sockets each (N l must be divisible by r); a uniform random permutation (numpy default_rng(seed))
matches variable socket s (variable s // l) to check socket perm[s] (check perm[s] // r).
Multi-edges (a variable meeting the same check twice) are KEPT, not resampled: the decoder
runs on the multigraph, where a double edge counts as two erased sockets of that check (BP
on the multigraph; over GF(2) the pair would cancel, so this is never optimistic).  Their
number stays O(1) as N grows (measured: mean 4.6 per graph for (3, 6), N = 20000, 60
seeds), a vanishing fraction of the N l edges, so they do not change the N -> oo limit;
`multi_edges` reports the count per graph.

Channel.  Each bit is erased independently with probability eps, drawn from
default_rng((seed, 1)) so the graph and the erasures of one seed are independent streams.
Peeling (= BP on the BEC, Mezard & Montanari 2009, Sec. 15.3.1 and Exercise 15.11): a check
with exactly one erased socket determines that bit; repeat until no such check is left.  The
erased bits that remain form the largest stopping set inside the erased set.

  ldpc_ensemble_graph(l, r, N, seed) -> LdpcGraph(var_checks (N, l), M, multi_edges)
  ldpc_peeling_decode(l, r, N, eps, seed) -> PeelResult(success, residual, residual_fraction,
      erased, erased_fraction, multi_edges, erased_mask, residual_mask)
  ldpc_peeling_success(l, r, N, eps, seeds) -> PeelStats: success fraction over named seeds
      with its exact 99% Clopper-Pearson interval (`halfwidth99` = the larger half-width is the
      reported uncertainty) and the residual erasure fraction (mean, stderr)
Statistical: fractions over seeds are binomial estimates, never certified.
Refusals: precondition:degree-pair, :erasure-probability, :enough-variables (N an integer >=
max(100, r)), :socket-count (N l not divisible by r), :named-seeds.
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from ..csp import _checks
from ..csp.xorsat_finite import clopper_pearson
from .ldpc import _eps, _pair

N_MIN = 100


@dataclass(frozen=True, eq=False)
class LdpcGraph:
    l: int
    r: int
    N: int
    M: int
    seed: int
    var_checks: np.ndarray     # (N, l): the check of each variable socket
    multi_edges: int           # sum over variables of (l - number of distinct checks)


@dataclass(frozen=True, eq=False)
class PeelResult:
    l: int
    r: int
    N: int
    eps: float
    seed: int
    success: bool              # every erased bit recovered
    residual: int              # erased bits left (the stopping set)
    residual_fraction: float   # residual / N
    erased: int
    erased_fraction: float
    multi_edges: int
    erased_mask: np.ndarray
    residual_mask: np.ndarray


@dataclass(frozen=True)
class PeelStats:
    l: int
    r: int
    N: int
    eps: float
    seeds: tuple
    n: int
    n_success: int
    fraction: float
    lo: float                  # exact 99% Clopper-Pearson interval of the success fraction
    hi: float
    halfwidth99: float
    residual_mean: float
    residual_stderr: float
    multi_edges_mean: float


def _n(N, l, r):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < max(N_MIN, r):
        raise Refusal("precondition:enough-variables",
                      f"N = {N!r}: need an integer N >= max({N_MIN}, r) = {max(N_MIN, r)}")
    if (N * l) % r:
        raise Refusal("precondition:socket-count",
                      f"N l = {N * l} is not divisible by r = {r}: the configuration model needs "
                      "as many check sockets as variable sockets (M = N l / r)")
    return int(N)


def ldpc_ensemble_graph(l, r, N, seed) -> LdpcGraph:
    seed = _checks.seed(seed)
    l, r = _pair(l, r)
    N = _n(N, l, r)
    perm = np.random.default_rng(seed).permutation(N * l)
    vc = (perm // r).reshape(N, l).astype(np.int64)
    s = np.sort(vc, axis=1)
    multi = int((s[:, 1:] == s[:, :-1]).sum())
    return LdpcGraph(l, r, N, N * l // r, seed, vc, multi)


def ldpc_peeling_decode(l, r, N, eps, seed) -> PeelResult:
    seed = _checks.seed(seed)
    l, r = _pair(l, r)
    e = float(_eps(eps))
    g = ldpc_ensemble_graph(l, r, N, seed)
    N, M, vc = g.N, g.M, g.var_checks
    erased = np.random.default_rng((seed, 1)).random(N) < e
    ids = np.flatnonzero(erased)
    cnt = np.bincount(vc[ids].ravel(), minlength=M)                 # erased sockets per check
    tot = np.bincount(vc[ids].ravel(), weights=np.repeat(ids, l), minlength=M)
    cnt, tot = cnt.tolist(), [int(t) for t in tot.tolist()]          # sum of their variables
    rows, left = vc.tolist(), erased.tolist()
    stack = [c for c in range(M) if cnt[c] == 1]
    while stack:
        c = stack.pop()
        if cnt[c] != 1:
            continue
        v = tot[c]                       # the one erased variable left on check c
        left[v] = False
        for c2 in rows[v]:
            cnt[c2] -= 1
            tot[c2] -= v
            if cnt[c2] == 1:
                stack.append(c2)
    res = np.array(left, dtype=bool)
    k = int(res.sum())
    return PeelResult(l, r, N, e, seed, k == 0, k, k / N, int(ids.size), ids.size / N,
                      g.multi_edges, erased, res)


def ldpc_peeling_success(l, r, N, eps, seeds) -> PeelStats:
    seeds = _checks.seeds(seeds)
    runs = [ldpc_peeling_decode(l, r, N, eps, s) for s in seeds]
    n, k = len(runs), sum(x.success for x in runs)
    p = k / n
    lo, hi = clopper_pearson(k, n)
    xs = [x.residual_fraction for x in runs]
    m = math.fsum(xs) / n
    se = math.sqrt(math.fsum((x - m)**2 for x in xs) / (n - 1) / n) if n > 1 else math.nan
    return PeelStats(runs[0].l, runs[0].r, runs[0].N, runs[0].eps, seeds, n, k, p, lo, hi,
                     max(p - lo, hi - p), m, se, sum(x.multi_edges for x in runs) / n)


__all__ = ["LdpcGraph", "N_MIN", "PeelResult", "PeelStats", "ldpc_ensemble_graph",
           "ldpc_peeling_decode", "ldpc_peeling_success"]
