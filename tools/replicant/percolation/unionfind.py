"""Disjoint sets (union-find) with path compression and union by size: the one implementation the
percolation code shares (P1 finite random graphs; P2 Newman-Ziff; later the Swendsen-Wang and Wolff
cluster algorithms).  Pure Python lists (no numpy in the hot path: a find is a few list reads).

Interface (elements are the integers 0 .. n - 1):
  UnionFind(n)              n singletons
  uf.find(x) -> int         the root of x's set; compresses the whole path to the root (two passes)
  uf.union(x, y) -> int     merge the sets of x and y (the smaller root is attached under the larger;
                            ties: y's root under x's) and return the root of the merged set; a no-op
                            returning the common root when they are already one set
  uf.connected(x, y) -> bool
  uf.size(x) -> int         the size of x's set
  uf.n_sets                 the current number of sets
  uf.largest                the size of the largest set (kept up to date by union: O(1), the
                            observable Newman-Ziff reads after every added bond)
  uf.add_edges(edges)       union over an iterable of (x, y) pairs (any int pairs, e.g. an (m, 2)
                            numpy array); returns the number of merges that joined two sets
  uf.labels() -> list       the root of every element (each path compressed)
  uf.set_sizes() -> list    the size of every set, one entry per set, in root order
  uf.reset()                back to n singletons (reuse the allocation across Monte Carlo sweeps)
Invariants: parent[r] == r exactly for roots; sizes are stored at roots only.  A precondition
failure (n < 0 or an element out of range) raises Refusal precondition:union-find-element, never a
silent index wrap-around (Python's negative indexing).
"""
from ..core import Refusal


class UnionFind:
    __slots__ = ("n", "parent", "_size", "n_sets", "largest")

    def __init__(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise Refusal("precondition:union-find-element", f"n = {n!r}: need an int >= 0")
        self.n = n
        self.reset()

    def reset(self):
        self.parent = list(range(self.n))
        self._size = [1] * self.n
        self.n_sets = self.n
        self.largest = 1 if self.n else 0

    def _check(self, x):
        if not 0 <= x < self.n:
            raise Refusal("precondition:union-find-element", f"element {x!r} not in 0 .. {self.n - 1}")

    def find(self, x):
        parent = self.parent
        if not 0 <= x < self.n:
            self._check(x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:              # second pass: every node on the path points at the root
            parent[x], x = r, parent[x]
        return r

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        size = self._size
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        size[rx] += size[ry]
        self.n_sets -= 1
        if size[rx] > self.largest:
            self.largest = size[rx]
        return rx

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
        return self._size[self.find(x)]

    def add_edges(self, edges):
        before = self.n_sets
        for x, y in edges:
            self.union(int(x), int(y))
        return before - self.n_sets

    def labels(self):
        return [self.find(x) for x in range(self.n)]

    def set_sizes(self):
        return [self._size[r] for r in range(self.n) if self.parent[r] == r]


__all__ = ["UnionFind"]
