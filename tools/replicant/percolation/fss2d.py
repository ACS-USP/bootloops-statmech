"""Finite-size scaling of the Newman-Ziff crossing probability (P2): p_c from the crossings of
R_L(p) and R_2L(p), and 1/nu from a collapse.  Statistical: bootstrap bands over named seeds, with a
stated systematic that is an assumption, not a bound; never final.

Data: for each L, `sweeps` NZ sweeps per named seed (lattice2d.sweep, stopped at the first crossing)
on the open L x L region; n* per sweep.  Per seed, the canonical R_L(p) (NZ Eq. (1)) on a fine grid of
p; the estimate is the mean over seeds; the bootstrap resamples seeds (numpy default_rng(boot_seed)).
  crossing p_c     the p where R_L - R_L' changes sign (linear interpolation on the grid) for
                   consecutive sizes L < L'; the estimate is the largest pair's, the band its 99%
                   bootstrap interval (0.5 / 99.5 percentiles of BOOT = 200 replicates), the systematic
                   |p_x(largest) - p_x(next)| (the drift between successive crossings; an assumption)
  collapse 1/nu    the levels R* in LEVELS give p_L(R*) (R_L(p_L) = R*); with y = 1/nu the scaled
                   z = (p_L(R*) - p_c) L^y must not depend on L.  chi2(p_c, y) = sum over levels and L of
                   (z - mean_L z)^2 / var z, var z = L^(2y) var p_L(R*) (bootstrap variance; the levels
                   of one L are correlated, so chi2 is a collapse measure, not chi-square distributed);
                   (p*, y*) minimise it jointly (nested golden sections); bands: the 0.5 / 99.5
                   percentiles over BOOT_JOINT = 100 bootstrap replicates (near their extremes);
                   systematics (assumptions): the larger of the shift when the smallest L is dropped
                   and the spread between estimators (p*: the crossing estimate; y*: the width slope)
  width slope      a second 1/nu estimator: the standard deviation of n*/M scales as L^(-1/nu); minus
                   the log-slope between the smallest and largest L, with its bootstrap band
  collapse_test(p_c, y)   "collapses" when p_c and y each lie in the joint collapse's band widened by
                   its systematic, else "fails" (a printed diagnostic; tests assert it on fixed seeds)
fss(lattice, dilution, Ls, seeds, sweeps) -> FSSResult.  `contrast(interval, a, b)`: whether the band
separates two values (printed, never asserted for unresolvable contrasts).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import Refusal
from .lattice2d import binomial_weights, bonds, check_lattice, check_seeds, check_sweeps, crossing_cdf, sweep

LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
Y_LO, Y_HI = 0.3, 1.5
GRID = 401
BOOT = 200
BOOT_JOINT = 100
_GOLD = (math.sqrt(5) - 1) / 2


def _check_Ls(Ls):
    if isinstance(Ls, (str, bytes)) or not hasattr(Ls, "__iter__"):
        raise Refusal("precondition:fss-sizes", f"Ls = {Ls!r}: need >= 3 increasing ints in 4 .. 2048")
    out = tuple(Ls)
    if len(out) < 3 or any(isinstance(L, bool) or not isinstance(L, (int, np.integer)) or not 4 <= L <= 2048
                           for L in out) or list(out) != sorted(set(out)):
        raise Refusal("precondition:fss-sizes", f"Ls = {Ls!r}: need >= 3 increasing ints in 4 .. 2048")
    return tuple(int(L) for L in out)


def collect(lattice, dilution, Ls, seeds, sweeps):
    """{L: (M, {seed: array of n*})}."""
    out = {}
    for L in Ls:
        per = {}
        for s in seeds:
            rng = np.random.default_rng([s, L])
            per[s] = np.array([sweep(lattice, dilution, L, L, rng, full=False)["n_star"] for _ in range(sweeps)])
        out[L] = (len(bonds(lattice, L, L)[0]) if dilution == "bond" else L * L, per)
    return out


def _curves(M, per, grid):
    """Per-seed canonical R(p) on the grid: array (seeds, grid)."""
    C = np.array([crossing_cdf(ns, M) for ns in per.values()])
    return np.array([C @ binomial_weights(M, p) for p in grid]).T


def _cross(grid, a, b):
    """The sign change of a - b inside the transition (both curves in [0.05, 0.95]) nearest R = 1/2
    (tail noise flips the sign of a - b where both are ~0 or ~1)."""
    d = a - b
    mid = (a + b) / 2
    sg = np.sign(d)
    idx = np.nonzero(((sg[:-1] * sg[1:] < 0) | (sg[:-1] == 0)) & (np.minimum(a, b)[:-1] > 0.05)
                     & (np.maximum(a, b)[:-1] < 0.95))[0]
    if idx.size == 0:
        return math.nan
    i = idx[np.argmin(np.abs(mid[idx] - 0.5))]
    if sg[i] == 0:
        return float(grid[i])
    return float(grid[i] - d[i] * (grid[i + 1] - grid[i]) / (d[i + 1] - d[i]))


def _level(grid, curve, r):
    i = np.searchsorted(curve, r)
    if i == 0 or i >= len(grid):
        return math.nan
    return float(grid[i - 1] + (r - curve[i - 1]) * (grid[i] - grid[i - 1]) / (curve[i] - curve[i - 1]))


def _golden(f, a, b, tol=1e-4):
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = f(d)
    return (a + b) / 2


@dataclass
class FSSResult:
    lattice: str
    dilution: str
    Ls: tuple
    seeds: tuple
    sweeps: int
    crossings: dict                 # (L, L') -> (estimate, lo, hi) 99% bootstrap
    p_c: tuple                      # crossing estimate: (estimate, lo, hi, systematic)
    p_c_collapse: tuple             # joint collapse: (p*, lo, hi, systematic)
    inv_nu: tuple                   # joint collapse: (y*, lo, hi, systematic)
    width_slope: tuple              # (estimate, lo, hi)
    levels: dict = field(repr=False, default_factory=dict)   # L -> p_L(R*) per level (full data)
    var0: dict = field(repr=False, default_factory=dict)     # L -> bootstrap variance of p_L(R*)

    def chi2(self, p_c, y, Ls=None, levels=None):
        """sum over levels and L of (z - mean_L z)^2 / var z, z = (p_L(R*) - p_c) L^y (var z =
        L^(2y) var p_L(R*), the bootstrap variance); (chi2, dof)."""
        Ls, levels = Ls or self.Ls, levels or self.levels
        z = np.array([(np.asarray(levels[L], float) - p_c) * L ** y for L in Ls])
        var = np.array([self.var0[L] * float(L) ** (2 * y) for L in Ls])
        ok = np.all(np.isfinite(z), 0) & np.all(var > 0, 0)
        dev = z[:, ok] - z[:, ok].mean(0)
        return float((dev ** 2 / var[:, ok]).sum()), int(ok.sum() * (len(Ls) - 1))

    def collapse_test(self, p_c, y):
        """Does (p_c, y = 1/nu) lie in the joint collapse's box: each coordinate inside its 99%
        bootstrap interval widened by its systematic?  A printed diagnostic in the run path."""
        inside = lambda t, v: t[1] - t[3] <= v <= t[2] + t[3]
        p, q = self.p_c_collapse, self.inv_nu
        return {"chi2": self.chi2(p_c, y)[0], "chi2_best": self.chi2(p[0], q[0])[0],
                "p_c box": (p[1] - p[3], p[2] + p[3]), "1/nu box": (q[1] - q[3], q[2] + q[3]),
                "verdict": "collapses" if inside(p, p_c) and inside(q, y) else "fails"}


def _joint(res, levels, Ls, lo, hi, tol=1e-5):
    """(p*, y*) minimising chi2 (nested golden sections, p in [lo, hi], y in [Y_LO, Y_HI])."""
    y_of = lambda pc: _golden(lambda y: res.chi2(pc, y, Ls, levels)[0], Y_LO, Y_HI, tol)
    pb = _golden(lambda pc: res.chi2(pc, y_of(pc), Ls, levels)[0], lo, hi, tol)
    return pb, y_of(pb)


def fss(lattice, dilution, Ls, seeds, sweeps, boot_seed=20261027):
    check_lattice(lattice, dilution)
    Ls, seeds, sweeps = _check_Ls(Ls), check_seeds(seeds), check_sweeps(sweeps)
    data = collect(lattice, dilution, Ls, seeds, sweeps)
    lo_q = min(np.quantile(np.concatenate(list(per.values())) / M, 0.002) for M, per in data.values())
    hi_q = max(np.quantile(np.concatenate(list(per.values())) / M, 0.998) for M, per in data.values())
    grid = np.linspace(lo_q, hi_q, GRID)
    curves = {L: _curves(M, per, grid) for L, (M, per) in data.items()}
    rng = np.random.default_rng(boot_seed)
    n = len(seeds)
    picks = [rng.integers(0, n, n) for _ in range(BOOT)]
    mean = {L: c.mean(0) for L, c in curves.items()}
    crossings = {}
    for L1, L2 in zip(Ls, Ls[1:]):
        est = _cross(grid, mean[L1], mean[L2])
        bs = [_cross(grid, curves[L1][k].mean(0), curves[L2][k].mean(0)) for k in picks]
        crossings[(L1, L2)] = tuple(float(x) for x in (est, *np.nanpercentile(bs, [0.5, 99.5])))
    keys = list(crossings)
    pc_est = (*crossings[keys[-1]], float(abs(crossings[keys[-1]][0] - crossings[keys[-2]][0])))
    levels = {L: [_level(grid, mean[L], r) for r in LEVELS] for L in Ls}
    boot = {L: np.array([[_level(grid, curves[L][k].mean(0), r) for r in LEVELS] for k in picks]) for L in Ls}
    var0 = {L: np.nanvar(b, axis=0, ddof=1) for L, b in boot.items()}
    res = FSSResult(lattice, dilution, Ls, seeds, sweeps, crossings, pc_est, (), (), (), levels, var0)
    w = max(pc_est[2] - pc_est[1] + 2 * pc_est[3], 2e-3)
    lo, hi = pc_est[1] - 3 * w, pc_est[2] + 3 * w
    p_all, y_all = _joint(res, levels, Ls, lo, hi)
    p_cut, y_cut = _joint(res, levels, Ls[1:], lo, hi)
    fits = np.array([_joint(res, {L: boot[L][b] for L in Ls}, Ls, lo, hi, 1e-4) for b in range(BOOT_JOINT)])
    widths = {L: np.std(np.concatenate(list(data[L][1].values())) / data[L][0], ddof=1) for L in Ls}
    slope = -math.log(widths[Ls[-1]] / widths[Ls[0]]) / math.log(Ls[-1] / Ls[0])
    bsl = []
    for k in picks:
        wd = [np.std(np.concatenate([data[L][1][seeds[i]] for i in k]) / data[L][0], ddof=1) for L in (Ls[0], Ls[-1])]
        bsl.append(-math.log(wd[1] / wd[0]) / math.log(Ls[-1] / Ls[0]))
    res.width_slope = tuple(float(x) for x in (slope, *np.percentile(bsl, [0.5, 99.5])))
    sys_p = max(abs(p_all - p_cut), abs(p_all - pc_est[0]))
    sys_y = max(abs(y_all - y_cut), abs(y_all - slope))
    res.p_c_collapse = tuple(float(x) for x in (p_all, *np.percentile(fits[:, 0], [0.5, 99.5]), sys_p))
    res.inv_nu = tuple(float(x) for x in (y_all, *np.percentile(fits[:, 1], [0.5, 99.5]), sys_y))
    return res


def contrast(interval, a, b):
    """Does the interval (est, lo, hi[, systematic]) separate a from b?  {"a_inside", "b_inside",
    "resolves"}: resolves only when exactly one of them is inside the band widened by the systematic."""
    est, lo, hi = interval[:3]
    s = interval[3] if len(interval) > 3 else 0.0
    ins = lambda v: lo - s <= v <= hi + s
    return {"a_inside": bool(ins(a)), "b_inside": bool(ins(b)), "resolves": bool(ins(a) != ins(b)),
            "difference": float(abs(a - b)), "band": (float(lo - s), float(hi + s))}


__all__ = ["FSSResult", "LEVELS", "collect", "contrast", "fss"]
