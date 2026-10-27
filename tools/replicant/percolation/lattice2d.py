"""Newman-Ziff sweeps on 2D lattices (P2): the Monte Carlo route.  It shares no code with the
closed-form routes (exact2d.py, cardy.py) or with the exhaustive enumeration (enumerate2d.py, which
builds its graphs from a planar embedding, not from these index rules); it uses the shared
union-find (unionfind.py, ruling P-R2).

Lattices, W columns x H rows, vertex id y W + x, open boundaries on all four sides:
  square       bonds (x, y)-(x + 1, y), (x, y)-(x, y + 1)
  triangular   the square bonds plus, between rows y and y + 1, (x, y)-(x - 1, y + 1) for even y and
               (x, y)-(x + 1, y + 1) for odd y: odd rows shifted right by 1/2, so the region is a
               rectangle of width W - 1/2 and height (H - 1) sqrt(3)/2
  honeycomb    the brick wall: horizontal bonds, and (x, y)-(x, y + 1) only when x + y is even
Bond or site percolation.  The crossing is horizontal: an occupied cluster containing a vertex of
column 0 and one of column W - 1 (W = 1: any occupied vertex crosses).  Each vertex carries the flags
1 (left column) | 2 (right column); a root carries the union of its set's flags (kept by the caller
of union: NZ's displacement offsets are needed for wrapping on a torus, not for open crossings).

The sweep (Newman & Ziff 2000, PRL 85, 4104, the algorithm of pp. 3-5 and Eq. (1)): the M elements
(bonds or sites) are occupied one at a time in a uniformly random order (numpy default_rng(seed)
per seed); after n of them the configuration is a uniform sample of the microcanonical ensemble
with n occupied.  Recorded per sweep: n* (the first n with a crossing), and when `full` the largest
cluster |C1|(n) and S2(n) = sum over the other clusters of |C|^2, n = 0 .. M.  The canonical value
at p is the binomial convolution Q(p) = sum_n C(M, n) p^n (1 - p)^(M - n) Q_n (NZ Eq. (1)); for the
crossing Q_n = P(n* <= n).  Observables: R(p) (crossing probability), P_inf(p) = E|C1|/N and
chi(p) = E S2/N (the mean cluster size, largest cluster excluded), all linear in the configuration,
so Eq. (1) is exact for them.
Exact invariants, checked on every sweep (refused no-solution:sampler-invariant): with every element
occupied the lattice is one cluster of N vertices that crosses; |C1| never decreases; the sum of
squared sizes is N^2 at the end.
_HOOKS (tests): "neighbourhood" None | "diagonal" (planted: (x, y)-(x + 1, y + 1) bonds added to any
lattice); "wrap_crossing" False | True (planted wrong boundary condition: the crossing direction made
periodic, bonds (W - 1, y)-(0, y)); planted observables: "site_s2_singletons" False (the N - n
unoccupied singletons left in the site S2), "p_inf_norm" "M" (P_inf = E|C1|/M), "chi_norm" False (chi
not divided by N).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from .unionfind import UnionFind

LATTICES = ("square", "triangular", "honeycomb")
DILUTIONS = ("bond", "site")
L_MAX = 2048
_HOOKS = {"neighbourhood": None, "wrap_crossing": False, "site_s2_singletons": True, "p_inf_norm": "N",
          "chi_norm": True}


def check_lattice(lattice, dilution):
    if not isinstance(lattice, str) or lattice not in LATTICES:
        raise Refusal("precondition:lattice", f"lattice = {lattice!r}: one of {', '.join(LATTICES)}")
    if not isinstance(dilution, str) or dilution not in DILUTIONS:
        raise Refusal("precondition:dilution", f"dilution = {dilution!r}: bond or site")


def check_size(W, H, name="lattice-size", lo=1, hi=L_MAX):
    for v in (W, H):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise Refusal(f"precondition:{name}", f"W x H = {W!r} x {H!r}: need ints in {lo} .. {hi}")
    return int(W), int(H)


def bonds(lattice, W, H):
    """(u, v) int arrays of the lattice's bonds (index rules above; planted hooks applied)."""
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x, y = x.ravel(), y.ravel()
    i = y * W + x
    us, vs = [], []

    def add(mask, dx, dy):
        us.append(i[mask])
        vs.append((y[mask] + dy) * W + x[mask] + dx)

    add(x < W - 1, 1, 0)
    if lattice == "honeycomb":
        add((y < H - 1) & ((x + y) % 2 == 0), 0, 1)
    else:
        add(y < H - 1, 0, 1)
    if lattice == "triangular":
        add((y < H - 1) & (y % 2 == 0) & (x >= 1), -1, 1)
        add((y < H - 1) & (y % 2 == 1) & (x < W - 1), 1, 1)
    if _HOOKS["neighbourhood"] == "diagonal":
        add((y < H - 1) & (x < W - 1), 1, 1)
    if _HOOKS["wrap_crossing"] and W > 2:
        add(x == W - 1, -(W - 1), 0)
    return np.concatenate(us), np.concatenate(vs)


def physical_aspect(lattice, dilution, W, H):
    """Width / height of the W x H region as a continuum rectangle: THE convention of every Cardy
    comparison in this package (tests, GUIDE).  Each site owns its Voronoi cell: triangular, a hexagon
    of width 1 in rows sqrt(3)/2 apart, so W / (H sqrt(3)/2); square site W / H; honeycomb (brick
    wall), columns sqrt(3)/2 and rows 3/2 apart, W / (sqrt(3) H).  Square bond: (W - 1) / H, the
    convention under which Bollobas & Riordan's self-dual (n + 1) x n bond rectangle (W = n + 2,
    H = n + 1) has aspect 1.  The region's lattice points span (W - 1/2) x (H - 1) sqrt(3)/2 on the
    triangular lattice; the O(1/L) ambiguity is the systematic the Cardy tests allow (one spacing)."""
    if lattice == "triangular":
        return 2 * W / (math.sqrt(3) * H)
    if lattice == "honeycomb":
        return W / (math.sqrt(3) * H)
    return (W - 1) / H if dilution == "bond" else W / H


def _flags(W, H):
    x = np.tile(np.arange(W), H)
    return ((x == 0) * 1 + (x == W - 1) * 2).tolist()


def sweep(lattice, dilution, W, H, rng, full=True, order=None):
    """One NZ sweep: dict(n_star, M, N[, largest, s2]) (arrays over n = 0 .. M)."""
    u, v = bonds(lattice, W, H)
    N = W * H
    M = len(u) if dilution == "bond" else N
    perm = rng.permutation(M) if order is None else np.asarray(order)
    uf = UnionFind(N)
    find, union = uf.find, uf.union
    flag = _flags(W, H)
    n_star = None
    largest = [0] * (M + 1) if full else None
    s2 = [0] * (M + 1) if full else None
    if dilution == "bond":
        if W == 1 and N:
            n_star = 0
        if full:
            largest[0], s2[0] = 1 if N else 0, max(N - 1, 0)
        uu, vv = u[perm].tolist(), v[perm].tolist()
        for n in range(1, M + 1):
            a, b = find(uu[n - 1]), find(vv[n - 1])
            if a != b:
                r = union(a, b)
                f = flag[a] | flag[b]
                flag[r] = f
                if f == 3 and n_star is None:
                    n_star = n
                    if not full:
                        break
            if full:
                big = uf.largest
                largest[n], s2[n] = big, uf.sum_sq - big * big
    else:
        nbr = [[] for _ in range(N)]
        for a, b in zip(u.tolist(), v.tolist()):
            nbr[a].append(b)
            nbr[b].append(a)
        occ = [False] * N
        for n, s in enumerate(perm.tolist(), 1):
            occ[s] = True
            f = flag[s]
            for t in nbr[s]:
                if occ[t]:
                    a, b = find(s), find(t)
                    if a != b:
                        r = union(a, b)
                        f = flag[a] | flag[b] | f
                        flag[r] = f
            if f == 3 and n_star is None:
                n_star = n
                if not full:
                    break
            if full:
                big = uf.largest
                largest[n] = big
                s2[n] = uf.sum_sq - ((N - n) if _HOOKS["site_s2_singletons"] else 0) - big * big
    out = {"n_star": n_star, "M": M, "N": N, "order": perm}
    if full:
        if n_star is None or largest[M] != N or uf.sum_sq != N * N or \
                any(largest[k] > largest[k + 1] for k in range(M)):
            raise Refusal("no-solution:sampler-invariant", f"{lattice} {dilution} {W} x {H}: all occupied "
                          f"gives largest {largest[M]} of N = {N}, sum of squares {uf.sum_sq}, crossing "
                          f"{n_star is not None}, or |C1| decreased")
        out["largest"], out["s2"] = np.array(largest, float), np.array(s2, float)
    elif n_star is None:
        raise Refusal("no-solution:sampler-invariant", f"{lattice} {dilution} {W} x {H}: no crossing "
                      "with every element occupied")
    return out


def binomial_weights(M, p):
    """C(M, n) p^n (1 - p)^(M - n), n = 0 .. M (exact deltas at p = 0, 1)."""
    from scipy.stats import binom
    if p <= 0 or p >= 1:
        w = np.zeros(M + 1)
        w[0 if p <= 0 else M] = 1.0
        return w
    return binom.pmf(np.arange(M + 1), M, p)


def crossing_cdf(n_stars, M):
    """P(n* <= n), n = 0 .. M, from a list of n* (the microcanonical crossing probability)."""
    c = np.bincount(np.asarray(n_stars, int), minlength=M + 1).astype(float)
    return np.cumsum(c) / len(n_stars)


@dataclass(frozen=True)
class LatticeSample:
    lattice: str
    dilution: str
    W: int
    H: int
    p: float
    seeds: tuple
    sweeps: int
    mean: dict          # R, P_inf, chi: means over the seeds of the per-seed canonical values
    stderr: dict
    halfwidth99: dict
    n_stars: dict       # seed -> [n*]


def check_seeds(seeds):
    if isinstance(seeds, (str, bytes)) or not hasattr(seeds, "__iter__"):
        raise Refusal("precondition:named-seeds", f"seeds = {seeds!r}: need a list of ints >= 0")
    out = tuple(seeds)
    if len(out) < 2 or len(set(out)) != len(out) or any(
            isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in out):
        raise Refusal("precondition:named-seeds", f"seeds = {seeds!r}: need >= 2 distinct ints >= 0")
    return tuple(int(s) for s in out)


def check_sweeps(sweeps):
    if isinstance(sweeps, bool) or not isinstance(sweeps, (int, np.integer)) or not 1 <= sweeps <= 10**6:
        raise Refusal("precondition:sweeps", f"sweeps = {sweeps!r}: need an int in 1 .. 10^6")
    return int(sweeps)


def sample_lattice(lattice, dilution, p, W, H, seeds, sweeps, cross_check=None):
    """Per named seed, `sweeps` NZ sweeps (numpy default_rng(seed)); the canonical R, P_inf, chi at p
    per seed; their mean, stderr and 99% Student-t half-width over the seeds.  `cross_check(runs,
    bonds)`, if given, is called once per seed with the seed's sweeps (an exact comparison, e.g. with
    the enumeration of a small lattice)."""
    from scipy.special import stdtrit
    check_lattice(lattice, dilution)
    W, H = check_size(W, H)
    if W * H > 4 * 10**6:
        raise Refusal("precondition:lattice-size", f"W x H = {W * H} > 4e6")
    p = float(p)
    if not 0 <= p <= 1 or math.isnan(p):
        raise Refusal("precondition:occupation", f"p = {p!r}: need 0 <= p <= 1")
    seeds, sweeps = check_seeds(seeds), check_sweeps(sweeps)
    rows, stars = [], {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        runs = [sweep(lattice, dilution, W, H, rng) for _ in range(sweeps)]
        M, N = runs[0]["M"], runs[0]["N"]
        w = binomial_weights(M, p)
        stars[seed] = [r["n_star"] for r in runs]
        if cross_check is not None:
            cross_check(runs, bonds(lattice, W, H))
        norm_p = N if _HOOKS["p_inf_norm"] == "N" else M
        rows.append({"R": float(w @ crossing_cdf(stars[seed], M)),
                     "P_inf": float(w @ np.mean([r["largest"] for r in runs], 0)) / norm_p,
                     "chi": float(w @ np.mean([r["s2"] for r in runs], 0)) / (N if _HOOKS["chi_norm"] else 1)})
    mean, se, hw = {}, {}, {}
    t = float(stdtrit(len(seeds) - 1, 0.995))
    for k in rows[0]:
        xs = np.array([r[k] for r in rows])
        mean[k], se[k] = float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))
        hw[k] = t * se[k]
    return LatticeSample(lattice, dilution, W, H, p, seeds, sweeps, mean, se, hw, stars)


__all__ = ["LATTICES", "LatticeSample", "binomial_weights", "bonds", "check_lattice", "check_seeds",
           "check_size", "check_sweeps", "crossing_cdf", "physical_aspect", "sample_lattice", "sweep"]
