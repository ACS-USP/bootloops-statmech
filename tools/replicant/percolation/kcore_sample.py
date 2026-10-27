"""Finite random (hyper)graphs, k-cores by exact peeling, and bootstrap percolation by direct
spreading: the statistical route of the P3 entries.  Nothing from kcore.py, kcore_threshold.py or
kcore_dgm.py is imported (no equation is shared); the graphs are percolation/sampling.py's samplers
(configuration model, exact G(N, p), named seeds) and, for K >= 3, the E5 XORSAT ensemble
(csp/xorsat_finite.xorsat_instance: M = round(alpha N) hyperedges on K distinct uniform vertices;
consumed, never edited).

Peeling (the k-core of a finite multi-hypergraph, independent of the order): repeatedly delete a
live vertex lying in fewer than k live hyperedges, with its hyperedges (a loop counts twice).  What
is left is the k-core: its vertices / N and hyperedges / N.  Dilution: site (a vertex is kept with
probability t, its hyperedges die with it) or bond (a hyperedge is kept with probability t).
Bootstrap percolation (threshold ell on the d-regular configuration model, each vertex infected
with probability q): an uninfected vertex with >= ell infected neighbours (with multiplicity)
becomes infected, until nothing changes.
Exact identities, checked on every instance (a correct sampler never violates them; a violation is
refused no-solution:sampler-invariant):
  bootstrap: the finally uninfected set equals the (d + 1 - ell)-core of the initially uninfected
      set (Janson, arXiv:0804.1656, Sec. 5: "it is easily seen"), by the two algorithms above;
  K >= 3, k = 2, no dilution: the peeled core equals csp/xorsat_finite.leaf_removal_core's
      (variables and checks) on the same instance;
  every core vertex lies in >= k core hyperedges.
Over the named seeds: means, stderr, 99% Student-t half-widths (statistical, never final).
  sample_kcore(law, K, k, dilution, N, seeds) -> SampleKcore
  sample_bootstrap(d, ell, q, N, seeds) -> SampleKcore
_HOOKS: "peel_k_shift" (0): the peeler's k + shift (a wrong threshold k); "spread_shift" (0): the
bootstrap threshold ell + shift (tests).
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..core import Refusal
from . import sampling as S

_HOOKS = {"peel_k_shift": 0, "spread_shift": 0}


def _incidence(edges, N):
    """For each vertex, the list of hyperedge indices containing it (with multiplicity)."""
    inc = [[] for _ in range(N)]
    for e, row in enumerate(edges):
        for v in row:
            inc[v].append(e)
    return inc


def peel(edges, alive, k):
    """(vertex mask, hyperedge mask) of the k-core; edges a list of K-tuples, alive a bool list."""
    k = k + _HOOKS["peel_k_shift"]
    N = len(alive)
    alive = list(alive)
    elive = [all(alive[v] for v in row) for row in edges]
    inc = _incidence(edges, N)
    deg = [0] * N
    for e, row in enumerate(edges):
        if elive[e]:
            for v in row:
                deg[v] += 1
    stack = [v for v in range(N) if alive[v] and deg[v] < k]
    while stack:
        v = stack.pop()
        if not alive[v]:
            continue
        alive[v] = False
        for e in inc[v]:
            if elive[e]:
                elive[e] = False
                for u in edges[e]:
                    deg[u] -= 1
                    if alive[u] and deg[u] < k:
                        stack.append(u)
    return alive, elive, deg


def spread(edges, infected, ell):
    """The final infected mask of bootstrap percolation with threshold ell (graphs, K = 2)."""
    ell = ell + _HOOKS["spread_shift"]
    N = len(infected)
    inf = list(infected)
    nbr = [[] for _ in range(N)]
    for a, b in edges:
        nbr[a].append(b)
        nbr[b].append(a)
    cnt = [sum(inf[u] for u in nbr[v]) for v in range(N)]
    stack = [v for v in range(N) if not inf[v] and cnt[v] >= ell]
    while stack:
        v = stack.pop()
        if inf[v]:
            continue
        inf[v] = True
        for u in nbr[v]:
            cnt[u] += 1
            if not inf[u] and cnt[u] >= ell:
                stack.append(u)
    return inf


def _hyper(K, alpha, N, seed):
    from ..csp.xorsat_finite import xorsat_instance
    idx, _ = xorsat_instance(K, N, str(float(alpha)), seed)
    return idx


def _instance(law, K, k, dil, N, seed):
    rng = np.random.default_rng(seed)
    if K == 2:
        model = S._float_law(law)
        kind, ks, ps = model
        edges = (S._er_edges(N, ks, rng) if kind == "poisson"
                 else S._config_edges(S._degrees(kind, ks, ps, N, rng), rng))
    else:
        alpha = Fraction(law[1]) / K
        edges = _hyper(K, alpha, N, seed)
    dkind, t = dil
    if dkind == "site":
        occ = S._occupy(N, t, rng).tolist()
        rows = edges.tolist()
    else:
        occ = [True] * N
        rows = [r for r, keep in zip(edges.tolist(), S._occupy(len(edges), t, rng).tolist()) if keep]
    alive, elive, deg = peel(rows, occ, k)
    if any(alive[v] and deg[v] < k for v in range(N)):
        raise Refusal("no-solution:sampler-invariant", "a core vertex lies in fewer than k core hyperedges")
    if K >= 3 and k == 2 and t == 1:
        from ..csp.xorsat_finite import leaf_removal_core
        v2, c2 = leaf_removal_core(np.asarray(rows, dtype=np.int64).reshape(-1, K), N)
        if (v2, c2) != (sum(alive), sum(elive)):
            raise Refusal("no-solution:sampler-invariant", f"peeled 2-core ({sum(alive)}, {sum(elive)}) "
                          f"!= csp leaf removal ({v2}, {c2}) on the same hypergraph")
    return {"core_fraction": sum(alive) / N, "core_edges": sum(elive) / N}


def _bootstrap_instance(d, ell, q, N, seed):
    rng = np.random.default_rng(seed)
    edges = S._config_edges(np.full(N, d), rng)
    infected = S._occupy(N, q, rng).tolist()
    final = spread(edges.tolist(), infected, ell)
    core, _, _ = peel(edges.tolist(), [not x for x in infected], d + 1 - ell)
    if [not x for x in final] != core:
        raise Refusal("no-solution:sampler-invariant", f"the finally uninfected set ({N - sum(final)}) is "
                      f"not the ({d + 1 - ell})-core of the initially uninfected set ({sum(core)})")
    return {"final_infected": sum(final) / N, "uninfected_core": sum(core) / N}


@dataclass(frozen=True)
class SampleKcore:
    N: int
    seeds: tuple
    mean: dict
    stderr: dict
    halfwidth99: dict


def _stats(runs, N, seeds):
    mean, se, hw = {}, {}, {}
    for key in runs[0]:
        xs = np.array([r[key] for r in runs], dtype=float)
        mean[key], se[key] = float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(xs.size))
        hw[key] = S._t(0.995, xs.size - 1) * se[key]
    return SampleKcore(N, seeds, mean, se, hw)


def _n(N):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not S.N_MIN <= N <= S.N_MAX:
        raise Refusal("precondition:n", f"N = {N!r}: need an int in {S.N_MIN} .. {S.N_MAX}")
    return int(N)


def sample_kcore(law, K, k, dilution, N, seeds):
    N, seeds = _n(N), S._seeds(seeds)
    if K >= 3 and law[0] != "poisson":
        raise Refusal("unsupported:hypergraph-law", f"K = {K}: only the Poisson (XORSAT) ensemble is "
                      "sampled for hypergraphs")
    if not (isinstance(dilution, tuple) and len(dilution) == 2 and dilution[0] in ("bond", "site")):
        raise Refusal("precondition:dilution", f"{dilution!r}: need ('bond' | 'site', t)")
    t = float(S._q(dilution[1], "occupation"))
    if not 0 <= t <= 1:
        raise Refusal("precondition:occupation", f"t = {dilution[1]!r}: need 0 <= t <= 1")
    runs = [_instance(law, K, k, (dilution[0], t), N, s) for s in seeds]
    return _stats(runs, N, seeds)


def sample_bootstrap(d, ell, q, N, seeds):
    N, seeds = _n(N), S._seeds(seeds)
    if (N * d) % 2:
        raise Refusal("precondition:n", f"N d = {N} x {d} is odd: no d-regular configuration")
    q = float(S._q(q, "infection"))
    runs = [_bootstrap_instance(d, ell, q, N, s) for s in seeds]
    return _stats(runs, N, seeds)


__all__ = ["SampleKcore", "peel", "sample_bootstrap", "sample_kcore", "spread"]
