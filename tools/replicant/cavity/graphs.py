"""Sparse-graph laws, coupling laws and sampled instances for cavity/sparse.py.

Conventions (shared with cavity/sparse.py):
  degree law ("regular", k): every spin has k + 1 neighbours; a cavity spin (one link
             removed) has k further neighbours (excess degree k, branching B = k).
  degree law ("poisson", c): Erdos-Renyi, each pair linked with probability c/(N - 1);
             degree ~ Poisson(c) as N -> oo; excess degree also Poisson(c), B = c.
  degree law ("tree", n):   a uniformly random labelled tree on n vertices (BP is exact).
  J law ("pm1", J):   J_ij = +J or -J with probability 1/2 each (J > 0).
  J law ("gauss", s): J_ij ~ N(0, s^2) (s > 0).
Seeds: every sampler takes a numpy.random.Generator.
"""
from dataclasses import dataclass

import numpy as np

from ..core import Refusal

DEGREE_LAWS = ("regular", "poisson", "tree")
J_LAWS = ("pm1", "gauss")


def degree_law(law, *, allow_tree=False):
    """Validate and normalise a degree law -> (kind, parameter)."""
    kinds = DEGREE_LAWS if allow_tree else DEGREE_LAWS[:2]
    if not (isinstance(law, tuple) and len(law) == 2 and law[0] in kinds):
        raise Refusal("unsupported:degree-law", f"{law!r}: supported {kinds} as (kind, param)")
    kind, p = law
    try:
        float(p)
    except (TypeError, ValueError):
        raise Refusal("unsupported:degree-law", f"{law!r}: parameter must be a number")
    if kind in ("regular", "tree"):
        if isinstance(p, bool) or not float(p).is_integer() or int(p) < 1:
            raise Refusal("unsupported:degree-law", f"{law!r}: {kind} needs an integer >= 1 "
                          "(regular: k = excess degree, connectivity k + 1)")
        return kind, int(p)
    if not float(p) > 0:
        raise Refusal("unsupported:degree-law", f"{law!r}: poisson needs mean degree c > 0")
    return kind, float(p)


def j_law(law):
    """Validate a coupling law -> (kind, scale)."""
    if not (isinstance(law, tuple) and len(law) == 2 and law[0] in J_LAWS):
        raise Refusal("unsupported:J-law", f"{law!r}: supported {J_LAWS} as (kind, scale)")
    try:
        scale = float(law[1])
    except (TypeError, ValueError):
        raise Refusal("unsupported:J-law", f"{law!r}: scale must be a number")
    if not scale > 0:
        raise Refusal("unsupported:J-law", f"{law!r}: scale must be > 0")
    return law[0], float(law[1])


def sample_couplings(law, size, rng):
    kind, s = j_law(law)
    if kind == "pm1":
        return s * rng.choice(np.array([-1.0, 1.0]), size=size)
    return s * rng.standard_normal(size)


def branching(law):
    """Mean excess degree B (regular: k; Poisson: c) and mean degree."""
    kind, p = degree_law(law)
    return (p, p + 1) if kind == "regular" else (p, p)


def sample_excess_degrees(law, size, rng):
    kind, p = degree_law(law)
    return np.full(size, p, dtype=int) if kind == "regular" else rng.poisson(p, size)


def sample_degrees(law, size, rng):
    kind, p = degree_law(law)
    return np.full(size, p + 1, dtype=int) if kind == "regular" else rng.poisson(p, size)


@dataclass(frozen=True)
class Graph:
    """n vertices; edges (m, 2) int array of (i, j), i != j, no repeats; J (m,) couplings."""
    n: int
    edges: np.ndarray
    J: np.ndarray

    def degrees(self):
        return np.bincount(self.edges.ravel(), minlength=self.n)

    def check(self):
        e = np.asarray(self.edges)
        if e.ndim != 2 or (e.size and e.shape[1] != 2) or len(self.J) != len(e):
            raise Refusal("precondition:graph", "edges must be (m, 2) with m couplings")
        if e.size and (e.min() < 0 or e.max() >= self.n or np.any(e[:, 0] == e[:, 1])):
            raise Refusal("precondition:graph", "edge endpoints out of range or self-loop")
        if len({tuple(sorted(x)) for x in e.tolist()}) != len(e):
            raise Refusal("precondition:graph", "repeated edge (multigraph)")
        return self


def random_tree(n, rng):
    """Vertex i > 0 attaches to a uniform earlier vertex (a random recursive tree)."""
    if n < 1:
        raise Refusal("precondition:graph", "tree needs n >= 1")
    parents = [int(rng.integers(i)) for i in range(1, n)]
    return np.array([[p, i] for i, p in zip(range(1, n), parents)], dtype=int).reshape(-1, 2)


def random_regular(n, d, rng, tries=200):
    """Configuration model with rejection (no self-loops, no multi-edges)."""
    if (n * d) % 2 or d >= n:
        raise Refusal("precondition:graph", f"no simple {d}-regular graph on n = {n}")
    for _ in range(tries):
        stubs = rng.permutation(np.repeat(np.arange(n), d)).reshape(-1, 2)
        if np.any(stubs[:, 0] == stubs[:, 1]):
            continue
        key = np.sort(stubs, 1)
        if len(np.unique(key[:, 0] * n + key[:, 1])) == len(key):
            return key
    raise Refusal("no-solution:graph", f"configuration model: no simple graph in {tries} tries")


def erdos_renyi(n, c, rng):
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(len(iu)) < c / (n - 1)
    return np.stack([iu[keep], ju[keep]], 1).astype(int)


def sample_graph(law, J, n, rng):
    """A finite instance of the law with couplings from J law `J`."""
    kind, p = degree_law(law, allow_tree=True)
    if kind == "tree":
        edges = random_tree(p, rng)
        n = p
    elif not (isinstance(n, int) and n >= 2):
        raise Refusal("precondition:n", f"sampled {kind} graph needs an integer n >= 2, got {n!r}")
    elif kind == "regular":
        edges = random_regular(n, p + 1, rng)
    else:
        edges = erdos_renyi(n, p, rng)
    return Graph(n, edges, sample_couplings(J, len(edges), rng)).check()
