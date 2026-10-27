"""Swendsen-Wang and Wolff cluster updates for the q-state Potts model (q = 2: Ising) on the L x L torus,
on the shared union-find (percolation/unionfind.py, ruling A-R2: the reference implementation, used by
every small-lattice exact check).  Pure Python lists in the hot loop; the vectorised labelling for
L >= 16 is cluster_fast.py, checked against this one trajectory-for-trajectory.

Swendsen-Wang (Salas & Sokal 1997, arXiv:hep-lat/9605018, Sec. 2.1 and p. 5): every bond between equal
colours opens with probability p = 1 - e^{-K} (fk.py; K = 2 beta for Ising), the open bonds are joined
by union-find, and every cluster gets a uniform new colour in 0 .. q - 1 (for Ising: each cluster
flipped with probability 1/2, Liu, Polkovnikov & Sandvik 2014, arXiv:1310.6327, Sec. III B).
Wolff (LPS Sec. III C: the same bond rule, grown from a uniform random seed site, flipped with
probability one): the cluster of the seed is grown by testing each bond from a cluster site to a
neighbour of the seed's old colour a once, opening it with probability p; every cluster site takes a
new colour b uniform among the q - 1 others (Ising: b = 1 - a).  Detailed balance holds for both for
any couplings J_ij >= 0.  With frustrated or random-sign couplings the construction stays valid (a bond
opens with probability 1 - e^{-2 beta |J_ij|} only on a satisfied edge, and the Edwards-Sokal marginal is
still the Boltzmann weight) but it loses its efficiency: the satisfied-edge clusters can percolate where
the spins do not order, so a cluster flip no longer follows the slow modes (a statement of this package,
no source fetched for it; not implemented here: this module takes the uniform ferromagnet).

Random numbers (numpy default_rng per named seed), consumed in a fixed order so that cluster_fast.py
reproduces a trajectory exactly: SW draws rng.random(2N) for the bonds (bond k = (u[k], v[k]) of
exact_small.torus_bonds), then rng.integers(q, size=N), the new colour of the cluster with canonical
index c being entry c, clusters numbered by their first site in 0 .. N - 1.  Wolff draws the seed,
then (q >= 3) the new colour, then uniforms in blocks of 64 as bonds are tested.

  neighbours(L) -> list of the 4 neighbour ids of each site (multigraph at L = 2)
  sw_sweep(s, q, p, u, v, rng, uf) -> (n_sat before the update, n_clusters, sum of |C|^2)
      s: list of colours, updated in place; uf: a UnionFind(N) reused across sweeps
  wolff_step(s, q, p, nbrs, rng, record=None) -> (cluster size, change of n_sat, old colour a, new b);
      with `record` a list, every bond test is appended as (i, j, opened) (the union-find cross-check,
      wolff_unionfind_check)
  wolff_unionfind_check(before, q, after, record, L) -> None: the grown cluster is exactly the
      union-find component of the seed over the opened bonds, every tested bond is a torus bond between
      the old colour, every untested aligned bond leaving the cluster is absent, and the cluster now has
      one new colour; else no-solution:sampler-invariant
_HOOKS (tests, planted errors): "flip" "cluster" | "site" (a new colour per site, not per cluster);
"growth" "aligned" | "any" (Wolff adds neighbours of any colour); fk._HOOKS["coupling_factor"] (the
bond probability).
"""
from ..core import Refusal
from ..percolation.unionfind import UnionFind
from .exact_small import torus_bonds

_HOOKS = {"flip": "cluster", "growth": "aligned"}


def neighbours(L):
    u, v = torus_bonds(L)
    nb = [[] for _ in range(L * L)]
    for a, b in zip(u.tolist(), v.tolist()):
        nb[a].append(b)
        nb[b].append(a)
    return nb


def bond_lists(L):
    u, v = torus_bonds(L)
    return u.tolist(), v.tolist()


def sw_sweep(s, q, p, u, v, rng, uf):
    N = len(s)
    r = rng.random(len(u)).tolist()
    uf.reset()
    union = uf.union
    n_sat = 0
    for k, a in enumerate(u):
        b = v[k]
        if s[a] == s[b]:
            n_sat += 1
            if r[k] < p:
                union(a, b)
    colours = rng.integers(q, size=N).tolist()
    find = uf.find
    canon = {}
    per_site = _HOOKS["flip"] == "site"
    for i in range(N):
        c = canon.setdefault(find(i), len(canon))
        s[i] = colours[i] if per_site else colours[c]
    if sum(uf.set_sizes()) != N or uf.n_sets != len(canon):
        raise Refusal("no-solution:sampler-invariant", "SW: cluster sizes do not partition the N sites")
    return n_sat, len(canon), uf.sum_sq


class _Uniforms:
    __slots__ = ("rng", "buf", "k")

    def __init__(self, rng):
        self.rng, self.buf, self.k = rng, [], 0

    def __call__(self):
        if self.k == len(self.buf):
            self.buf, self.k = self.rng.random(64).tolist(), 0
        self.k += 1
        return self.buf[self.k - 1]


def wolff_step(s, q, p, nbrs, rng, record=None):
    N = len(s)
    i0 = int(rng.integers(N))
    a = s[i0]
    b = 1 - a if q == 2 else (a + 1 + int(rng.integers(q - 1))) % q
    uni = _Uniforms(rng)
    inside = bytearray(N)
    inside[i0] = 1
    stack, size = [i0], 1
    aligned_only = _HOOKS["growth"] == "aligned"
    while stack:
        i = stack.pop()
        for j in nbrs[i]:
            if inside[j] or (aligned_only and s[j] != a):
                continue
            ok = uni() < p
            if record is not None:
                record.append((i, j, ok))
            if ok:
                inside[j] = 1
                stack.append(j)
                size += 1
    d_sat = 0
    for i in range(N):
        if inside[i]:
            for j in nbrs[i]:
                if not inside[j]:
                    d_sat += (s[j] == b) - (s[j] == a)
    for i in range(N):
        if inside[i]:
            s[i] = b
    if record is not None:
        record.append(("seed", i0, a))
    return size, d_sat, a, b


def wolff_unionfind_check(before, q, after, record, L):
    """Cluster-for-cluster check of one recorded Wolff step against the shared UnionFind."""
    tag, i0, a = record[-1]
    trials = record[:-1]
    N = L * L
    nb = neighbours(L)
    uf = UnionFind(N)
    tested = {}
    for i, j, ok in trials:
        if j not in nb[i] or before[i] != a or before[j] != a:
            raise Refusal("no-solution:sampler-invariant", f"Wolff tested ({i}, {j}): not a bond between "
                          f"sites of the old colour {a}")
        tested[(i, j)] = tested.get((i, j), 0) + 1
        if ok:
            uf.union(i, j)
    comp = {x for x in range(N) if uf.connected(x, i0)}
    changed = {x for x in range(N) if after[x] != before[x]}
    if comp != changed or len({after[x] for x in comp}) != 1 or after[i0] == a:
        raise Refusal("no-solution:sampler-invariant", f"Wolff cluster {sorted(changed)} is not the union-find "
                      f"component {sorted(comp)} of seed {i0}, recoloured to one new colour")
    for i in comp:
        for j in set(nb[i]):
            if j not in comp and before[j] == a and tested.get((i, j), 0) != nb[i].count(j):
                raise Refusal("no-solution:sampler-invariant", f"Wolff: aligned bond ({i}, {j}) leaving the "
                              "cluster was not tested once per bond")


__all__ = ["bond_lists", "neighbours", "sw_sweep", "wolff_step", "wolff_unionfind_check"]
