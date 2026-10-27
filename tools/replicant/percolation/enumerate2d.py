"""Exhaustive enumeration of small 2D lattices (P2): the exact crossing probability as a polynomial in
p, the planar duality identities it must satisfy, and an exact cross-check of the Newman-Ziff sweep.
Shares no code with lattice2d.py: the graph is built from a planar embedding (vertices at their
coordinates, an edge between every two vertices at Euclidean distance 1), not from index rules, and
clusters are found by label propagation over all 2^M configurations at once (numpy), not union-find.

Embeddings of the W x H region (vertex (x, y), the same labelling as lattice2d.py):
  square       (x, y)
  triangular   (x + (y mod 2)/2, y sqrt(3)/2)
  honeycomb    (x sqrt(3)/2, 3y/2 + 1/2 [x + y even])   (the brick wall drawn as hexagons)
The horizontal crossing joins a vertex that is leftmost in its row to one that is rightmost in its
row; the vertical one the bottom row to the top row.  M = number of bonds (bond) or sites (site);
M <= M_MAX (else precondition:exact-size).

  crossing_counts(lattice, dilution, W, H, direction="h") -> [A_0 .. A_M]: A_n = the number of
      crossing configurations with n occupied elements, so R(p) = sum_n A_n p^n (1 - p)^(M - n)
  crossing_probability(counts, p) -> Fraction (p rational)
  duality_identity(lattice, dilution, W, H) -> (holds, detail): exact polynomial identities
      square bond: R_h[W x H](p) + R_h[(H + 1) x (W - 1)](1 - p) = 1 (Bollobas & Riordan,
                   arXiv:math/0410359, Lemma 3 / Corollary 4: exactly one of an open horizontal
                   crossing of R and a closed vertical crossing of the dual rectangle; rotated)
      triangular site: R_h(p) + R_v(1 - p) = 1 on the same region (the Hex-board argument: a
                   triangulated disc has exactly one of an open left-right or a closed top-bottom
                   crossing; derived here, cf. Bollobas & Riordan, arXiv:math/0412510, Sec. 4.3)
      else (None, "no planar duality identity for this lattice at finite size")
      _HOOKS["dual_shape"] (tests): the dual rectangle is (H + a) columns x (W + b) rows, (a, b) =
      (1, -1); planted (0, -1)
  nz_cross_check(lattice, dilution, W, H, orders, n_stars) -> None: for every NZ order, the first n
      at which the first n elements cross (from the enumeration's table) must equal the sweep's n*;
      an NZ bond that is not an edge of the embedding, or a different n*, is refused
      no-solution:nz-enumeration
"""
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

from ..core import Refusal

M_MAX = 20
_HOOKS = {"dual_shape": (1, -1)}


def _coords(lattice, W, H):
    s3 = math.sqrt(3)
    pts = {}
    for y in range(H):
        for x in range(W):
            if lattice == "square":
                pts[(x, y)] = (x, y)
            elif lattice == "triangular":
                pts[(x, y)] = (x + (y % 2) / 2, y * s3 / 2)
            else:
                pts[(x, y)] = (x * s3 / 2, 1.5 * y + (0.5 if (x + y) % 2 == 0 else 0.0))
    return pts


@lru_cache(maxsize=64)
def graph(lattice, W, H):
    """(vertex keys, edges as index pairs) of the embedded region."""
    pts = _coords(lattice, W, H)
    keys = sorted(pts, key=lambda k: (k[1], k[0]))
    xy = np.array([pts[k] for k in keys], float)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    edges = tuple((i, j) for i in range(len(keys)) for j in range(i + 1, len(keys)) if abs(d[i, j] - 1) < 1e-9)
    return tuple(keys), edges


def _ends(keys, direction):
    rows = {}
    for i, (x, y) in enumerate(keys):
        rows.setdefault(y if direction == "h" else x, []).append((x if direction == "h" else y, i))
    lo = [min(r)[1] for r in rows.values()]
    hi = [max(r)[1] for r in rows.values()]
    return lo, hi


@lru_cache(maxsize=64)
def _table(lattice, dilution, W, H, direction):
    """Boolean array over all 2^M configurations: does it cross?  (bit e of the index = element e.)"""
    keys, edges = graph(lattice, W, H)
    V = len(keys)
    M = len(edges) if dilution == "bond" else V
    if M > M_MAX:
        raise Refusal("precondition:exact-size", f"{lattice} {dilution} {W} x {H}: {M} elements > {M_MAX} "
                      "(2^M configurations)")
    conf = np.arange(2 ** M, dtype=np.int64)
    bit = lambda e: ((conf >> e) & 1).astype(bool)
    occ = [bit(i) for i in range(V)] if dilution == "site" else [np.ones(conf.size, bool)] * V
    opened = [bit(e) if dilution == "bond" else occ[a] & occ[b] for e, (a, b) in enumerate(edges)]
    lab = np.tile(np.arange(V, dtype=np.int16), (conf.size, 1))
    changed = True
    while changed:
        changed = False
        for e, (a, b) in enumerate(edges):
            m = np.minimum(lab[:, a], lab[:, b])
            upd = opened[e] & ((lab[:, a] != m) | (lab[:, b] != m))
            if upd.any():
                changed = True
                lab[upd, a] = m[upd]
                lab[upd, b] = m[upd]
    lo, hi = _ends(keys, direction)
    out = np.zeros(conf.size, bool)
    for i in lo:
        for j in hi:
            out |= occ[i] & occ[j] & (lab[:, i] == lab[:, j])
    return out


def crossing_counts(lattice, dilution, W, H, direction="h"):
    t = _table(lattice, dilution, W, H, direction)
    M = int(round(math.log2(t.size)))
    pop = np.zeros(t.size, np.int64)
    for e in range(M):
        pop += (np.arange(t.size) >> e) & 1
    return [int(c) for c in np.bincount(pop[t], minlength=M + 1)]


def crossing_probability(counts, p):
    p = Fraction(p)
    M = len(counts) - 1
    return sum(a * p ** n * (1 - p) ** (M - n) for n, a in enumerate(counts))


def _poly(counts):
    """Coefficients in the monomial basis (exact ints): sum_n A_n p^n (1 - p)^(M - n)."""
    M = len(counts) - 1
    c = [0] * (M + 1)
    for n, a in enumerate(counts):
        for k in range(M - n + 1):
            c[n + k] += a * math.comb(M - n, k) * (-1) ** k
    return c


def _reflect(c):
    """Coefficients of q(p) = poly(1 - p)."""
    out = [0] * len(c)
    for j, a in enumerate(c):
        for k in range(j + 1):
            out[k] += a * math.comb(j, k) * (-1) ** k
    return out


def duality_identity(lattice, dilution, W, H):
    if lattice == "square" and dilution == "bond" and W >= 2:
        a = _poly(crossing_counts("square", "bond", W, H))
        dh, dw = _HOOKS["dual_shape"]
        b = _reflect(_poly(crossing_counts("square", "bond", H + dh, W + dw)))
        what = f"R_h[{W} x {H}](p) + R_h[{H + dh} x {W + dw}](1 - p)"
    elif lattice == "triangular" and dilution == "site":
        a = _poly(crossing_counts("triangular", "site", W, H))
        b = _reflect(_poly(crossing_counts("triangular", "site", W, H, "v")))
        what = f"R_h[{W} x {H}](p) + R_v[{W} x {H}](1 - p)"
    else:
        return None, "no planar duality identity for this lattice at finite size"
    n = max(len(a), len(b))
    s = [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]
    holds = s[0] == 1 and not any(s[1:])
    terms = " + ".join(f"{c} p^{i}" for i, c in enumerate(s) if c) or "0"
    return holds, f"{what} = 1 identically (exact polynomials)" if holds else f"{what} = {terms}, not 1"


def nz_cross_check(lattice, dilution, W, H, orders, n_stars, nz_bonds=None):
    """Exact comparison of NZ sweeps with the enumeration: `orders` are the sweeps' element orders,
    `n_stars` their n*; for bond, `nz_bonds` = the sweep's (u, v) arrays (vertex ids y W + x)."""
    table = _table(lattice, dilution, W, H, "h")
    keys, edges = graph(lattice, W, H)
    if dilution == "bond":
        index = {frozenset((keys[a][1] * W + keys[a][0], keys[b][1] * W + keys[b][0])): e
                 for e, (a, b) in enumerate(edges)}
        u, v = nz_bonds
        if len(u) != len(edges):
            raise Refusal("no-solution:nz-enumeration", f"the sweep has {len(u)} bonds, the embedding "
                          f"{len(edges)}")
        try:
            elem = [index[frozenset((int(a), int(b)))] for a, b in zip(u, v)]
        except KeyError as exc:
            raise Refusal("no-solution:nz-enumeration", f"sweep bond {set(exc.args[0])} is not an edge "
                          "of the embedded lattice")
    else:
        pos = {k[1] * W + k[0]: i for i, k in enumerate(keys)}
        elem = [pos[s] for s in range(W * H)]
    for order, ns in zip(orders, n_stars):
        mask, first = 0, 0 if table[0] else None
        for n, k in enumerate(order, 1):
            if first is not None:
                break
            mask |= 1 << elem[int(k)]
            if table[mask]:
                first = n
        if first != ns:
            raise Refusal("no-solution:nz-enumeration", f"{lattice} {dilution} {W} x {H}: the sweep's "
                          f"n* = {ns}, the enumeration's {first}")


__all__ = ["M_MAX", "crossing_counts", "crossing_probability", "duality_identity", "graph",
           "nz_cross_check"]
