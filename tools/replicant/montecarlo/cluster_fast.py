"""Vectorised Swendsen-Wang for larger tori (ruling A-R2): the same update as cluster.sw_sweep, with the
clusters labelled by scipy.sparse.csgraph.connected_components instead of the shared union-find, and the
same random numbers in the same order (cluster.py's docstring), so that on a named seed the two give the
same trajectory bit for bit; tests/test_mc_cluster.py checks this cluster-for-cluster (every sweep's
partition and colours) on named seeds in the default tier.  Used by the runs for L >= FAST_FROM only.

  sw_sweep_fast(s, q, p, u, v, rng) -> (n_sat, n_clusters, sum |C|^2, labels)   s: int ndarray, in place
"""
import numpy as np

from .cluster import _HOOKS

FAST_FROM = 16


def canonical_labels(labels):
    """Relabel components by their first site: the cluster of site 0 is 0, the next new one 1, ..."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[inv.ravel()]


def sw_sweep_fast(s, q, p, u, v, rng):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = s.size
    r = rng.random(u.size)
    eq = s[u] == s[v]
    open_ = eq & (r < p)
    g = coo_matrix((np.ones(int(open_.sum()), np.int8), (u[open_], v[open_])), shape=(N, N))
    n, lab = connected_components(g, directed=False)
    lab = canonical_labels(lab)
    colours = rng.integers(q, size=N)
    if _HOOKS["flip"] != "none":
        s[:] = colours if _HOOKS["flip"] == "site" else colours[lab]
    sizes = np.bincount(lab, minlength=n)
    return int(eq.sum()), int(n), int((sizes.astype(np.int64) ** 2).sum()), lab


__all__ = ["FAST_FROM", "canonical_labels", "sw_sweep_fast"]
