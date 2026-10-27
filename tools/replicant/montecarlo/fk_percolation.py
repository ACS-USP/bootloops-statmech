"""The Fortuin-Kasteleyn machinery at q = 1 is bond percolation (ruling A-R7): the random-cluster weight
p^{open} (1 - p)^{closed} q^{C} (Salas & Sokal 1997 Eq. (2.5); Beffara & Duminil-Copin Eq. (1.1)) at q = 1
is independent bond percolation, and its self-dual point p_sd(1) = 1/2 (BDC Theorem 1) is P2's square-bond
p_c.  So cluster.sw_sweep itself, run with q = 1 on P2's open W x H square lattice (percolation/lattice2d
.bonds, NZ's index rules), must give P2's exact crossing polynomial (percolation/enumerate2d.py, built
from a planar embedding and label propagation: no shared code with sw_sweep or the union-find).

  fk_crossing_counts(W, H) -> [A_0 .. A_M]: every one of the 2^M bond configurations is fed to sw_sweep
      through a scripted generator (uniform 0 for an open bond, 1 for a closed one: open iff r < p for any
      0 < p < 1), and the union-find left by the sweep decides the horizontal crossing (a cluster with a
      column-0 and a column-(W - 1) vertex); A_n counts the crossing configurations with n open bonds
  fk_crossing_sampled(W, H, p, samples, seed) -> (crossings, samples): sw_sweep at q = 1 with a real
      generator (numpy default_rng(seed)): a binomial sample of R(p)
  p2_tie(W, H) -> (holds, detail): fk_crossing_counts == enumerate2d.crossing_counts("square", "bond",
      W, H) exactly (integers), with R(1/2) printed
M = 2 W H - W - H <= 20 (else precondition:exact-size).
"""
from fractions import Fraction

import numpy as np

from ..core import Refusal
from ..percolation import enumerate2d as EN
from ..percolation.lattice2d import bonds
from ..percolation.unionfind import UnionFind
from .cluster import sw_sweep

M_MAX = 20


class _Scripted:
    """A stand-in generator: random(M) returns the scripted bond uniforms, integers(q, N) zeros."""

    def __init__(self, r):
        self.r = r

    def random(self, n):
        return np.asarray(self.r[:n], float)

    def integers(self, q, size):
        return np.zeros(size, np.int64)


def _setup(W, H):
    u, v = bonds("square", W, H)
    if len(u) > M_MAX:
        raise Refusal("precondition:exact-size", f"{W} x {H}: {len(u)} bonds > {M_MAX}")
    left = [y * W for y in range(H)]
    right = [y * W + W - 1 for y in range(H)]
    return u.tolist(), v.tolist(), left, right


def _crosses(uf, left, right):
    roots = {uf.find(a) for a in left}
    return any(uf.find(b) in roots for b in right)


def fk_crossing_counts(W, H):
    u, v, left, right = _setup(W, H)
    M, N = len(u), W * H
    uf = UnionFind(N)
    counts = [0] * (M + 1)
    for c in range(2 ** M):
        r = [0.0 if (c >> k) & 1 else 1.0 for k in range(M)]
        sw_sweep([0] * N, 1, 0.5, u, v, _Scripted(r), uf)
        if _crosses(uf, left, right):
            counts[bin(c).count("1")] += 1
    return counts


def fk_crossing_sampled(W, H, p, samples, seed):
    u, v, left, right = _setup(W, H)
    N = W * H
    uf = UnionFind(N)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(samples):
        sw_sweep([0] * N, 1, float(p), u, v, rng, uf)
        hits += _crosses(uf, left, right)
    return hits, samples


def p2_tie(W, H):
    mine = fk_crossing_counts(W, H)
    theirs = EN.crossing_counts("square", "bond", W, H)
    r = EN.crossing_probability(mine, Fraction(1, 2))
    if mine == theirs:
        return True, (f"FK at q = 1 on the open {W} x {H} square lattice: crossing counts {mine} = P2's enumeration; "
                      f"R(p_sd(1) = 1/2) = {r}")
    return False, f"FK at q = 1: {mine} != P2's {theirs}"


__all__ = ["fk_crossing_counts", "fk_crossing_sampled", "p2_tie"]
