"""Finite-size scaling of the Newman-Ziff crossing probability (P2): p_c from the crossings of
R_L(p) and R_2L(p), and 1/nu from a collapse.  Statistical: bootstrap bands over named seeds, with a
stated systematic that is an assumption, not a bound; never final.

Data: for each L, `sweeps` NZ sweeps per named seed (lattice2d.sweep, stopped at the first crossing)
on the open L x L region; n* per sweep.  Per seed, the canonical R_L(p) (NZ Eq. (1)) on a fine grid of
p; the estimate is the mean over seeds.  Bootstrap: BOOT = 1000 resamplings of the seeds (numpy
default_rng(boot_seed)); every band is the LEVEL = 99% percentile interval of the replicates and every
stderr their standard deviation.
  crossing p_c     the p where R_L - R_L' changes sign (linear interpolation on the grid) for
                   consecutive sizes L < L'; the estimate is the largest pair's
  collapse         the levels R* in LEVELS give p_L(R*) (R_L(p_L) = R*); with y = 1/nu the scaled
                   z = (p_L(R*) - p_c) L^y must not depend on L.  chi2(p_c, y) = sum over levels and L of
                   (z - mean_L z)^2 / var z, var z = L^(2y) var p_L(R*) (bootstrap variance; the levels
                   of one L are correlated, so chi2 is a collapse measure, not chi-square distributed);
                   z is linear in p_c, so for each y the minimising p_c is closed form; (p*, y*) by a
                   golden section in y
  systematic       ONE rule for every lattice (an assumption, not a bound).  Resolved drift: an
                   alternative estimate (crossing: the previous pair's; collapse: the fit without the
                   smallest L) counts only beyond its own noise, max(0, |d| - Z sd(d)) with sd(d) the
                   bootstrap deviation of the paired difference.  For p_c (both estimators) the
                   systematic is the larger of the resolved drift and WINDOW_FRACTION = 1/10 of the
                   critical window (the standard deviation of n*/M at the largest L, measured to ~1%):
                   corrections to scaling are assumed to move an estimator by at most a tenth of the
                   window.  For 1/nu: the resolved drift.
  width slope      a second 1/nu estimator: the standard deviation of n*/M scales as L^(-1/nu); minus
                   the log-slope between the smallest and largest L, with its bootstrap band (printed)
  collapse_test(p_c, y), crossing_test(p_c)   inside the band widened by the systematic, or not (printed
                   diagnostics; the slow tests assert them on two named seed sets)
Validated resolution (tests/test_percolation_lattice_slow.py; Ls 16 .. 128, 48 seeds x 25 sweeps, seed
sets 0 .. 47 and 100 .. 147, all four solved lattices): the collapse accepts (exact p_c, 3/4) and
rejects p_c +- 0.005 and nu = 1; the crossing band accepts the exact p_c and rejects p_c +- 0.01 (it
does not resolve +- 0.005 on every set).
Undetermined: when the largest pair's crossing is not finite, or fewer than MIN_FINITE = 90% of its
bootstrap replicates are (no sign change of R_L - R_L' inside the transition window: too few seeds or
sweeps), its band is all NaN and FSSResult.undetermined["p_c_crossing"] says why (undetermined:crossing);
likewise a collapse whose fit is not finite (undetermined:collapse) and a width slope with too few
resamples of nonzero width (undetermined:width).
fss(lattice, dilution, Ls, seeds, sweeps, boot_seed=20261027) -> FSSResult.  `contrast(interval, a, b)`:
whether the band separates two values (printed, never asserted for unresolvable contrasts).
"""
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core import Refusal
from .lattice2d import binomial_weights, bonds, check_lattice, check_seeds, check_sweeps, crossing_cdf, sweep

LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
Y_LO, Y_HI = 0.3, 1.5
GRID = 401
BOOT = 1000
LEVEL = 0.99
Z = 2.576
WINDOW_FRACTION = 0.1
MIN_FINITE = 0.9         # an estimate needs >= 90% finite bootstrap replicates, else undetermined
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
    crossings: dict                 # (L, L') -> (estimate, lo, hi) bootstrap percentile band
    p_c: tuple                      # crossing estimate: (estimate, lo, hi, systematic, stderr)
    p_c_collapse: tuple             # joint collapse: (p*, lo, hi, systematic, stderr)
    inv_nu: tuple                   # joint collapse: (y*, lo, hi, systematic, stderr)
    width_slope: tuple              # (estimate, lo, hi)
    levels: dict = field(repr=False, default_factory=dict)   # L -> p_L(R*) per level (full data)
    var0: dict = field(repr=False, default_factory=dict)     # L -> bootstrap variance of p_L(R*)
    window: float = 0.0             # std of n*/M at the largest L (the critical window)
    undetermined: dict = field(default_factory=dict)        # value -> "undetermined:crossing: why"

    def chi2(self, p_c, y, Ls=None, levels=None):
        """sum over levels and L of (z - mean_L z)^2 / var z, z = (p_L(R*) - p_c) L^y (var z =
        L^(2y) var p_L(R*), the bootstrap variance); (chi2, dof)."""
        Ls, levels = Ls or self.Ls, levels or self.levels
        return _chi2(Ls, levels, self.var0, y, p_c)[:2]

    def collapse_test(self, p_c, y):
        """Does (p_c, y = 1/nu) lie in the joint collapse's box: each coordinate inside its bootstrap band
        widened by its resolved drift?  A printed diagnostic in the run path."""
        inside = lambda t, v: t[1] - t[3] <= v <= t[2] + t[3]
        p, q = self.p_c_collapse, self.inv_nu
        return {"chi2": self.chi2(p_c, y)[0], "chi2_best": self.chi2(p[0], q[0])[0],
                "p_c box": (p[1] - p[3], p[2] + p[3]), "1/nu box": (q[1] - q[3], q[2] + q[3]),
                "verdict": "collapses" if inside(p, p_c) and inside(q, y) else "fails"}

    def crossing_test(self, p_c):
        """Does p_c lie in the crossing band widened by its resolved drift?"""
        e, lo, hi, s = self.p_c[:4]
        return "inside" if lo - s <= p_c <= hi + s else "outside"


def _chi2(Ls, levels, var0, y, p_c=None):
    """(chi2, dof, p): z = (P - p) s_L with s_L = L^y is linear in p, so for a given y the p minimising
    chi2 is closed form (used when p_c is None)."""
    P = np.array([np.asarray(levels[L], float) for L in Ls])
    sc = np.array([float(L) ** y for L in Ls])
    v = np.array([var0[L] for L in Ls]) * sc[:, None] ** 2
    ok = np.all(np.isfinite(P), 0) & np.all(v > 0, 0)
    P, v = P[:, ok], v[:, ok]
    A = P * sc[:, None]
    A = A - A.mean(0)
    B = (sc - sc.mean())[:, None]
    if p_c is None:
        p_c = float((A * B / v).sum() / (B * B / v).sum())
    return float(((A - p_c * B) ** 2 / v).sum()), int(ok.sum() * (len(Ls) - 1)), p_c


def _joint(Ls, levels, var0, tol=1e-5):
    """(p*, y*) minimising chi2: golden section in y, p in closed form."""
    y = _golden(lambda y: _chi2(Ls, levels, var0, y)[0], Y_LO, Y_HI, tol)
    return _chi2(Ls, levels, var0, y)[2], y


def _resolved(est, reps, what="no sign change of R_L - R_L' inside the transition window"):
    """None if the estimate is finite and at least MIN_FINITE of its bootstrap replicates are, else why
    not (the crossing or level is not inside the transition window of the curves)."""
    reps = np.asarray(reps, float)
    frac = float(np.isfinite(reps).mean()) if reps.size else 0.0
    if math.isfinite(est) and frac >= MIN_FINITE:
        return None
    return (f"estimate {est:.5g}, {100 * frac:.0f}% of the {reps.size} bootstrap replicates finite (need a "
            f"finite estimate and >= {100 * MIN_FINITE:.0f}%): {what}; more seeds or sweeps")


def _band(est, reps, other=None, other_reps=None, floor=0.0):
    """(est, lo, hi, systematic, stderr): the LEVEL percentile band of the bootstrap replicates; the
    systematic is max(floor, the resolved drift max(0, |est - other| - Z sd(replicate differences))).
    An unresolved estimate (_resolved) is all NaN: no band is taken of too few replicates."""
    reps = np.asarray(reps, float)
    if _resolved(est, reps) is not None:
        return (math.nan,) * 5
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    sysm = 0.0
    if other is not None:
        d = np.asarray(other_reps[0], float) - np.asarray(other_reps[1], float)
        d = d[np.isfinite(d)]
        if math.isfinite(other) and d.size >= 2:      # else no resolved drift can be measured
            sysm = max(0.0, abs(est - other) - Z * float(np.std(d, ddof=1)))
    sysm = max(sysm, floor)
    return tuple(float(x) for x in (est, lo, hi, sysm, np.std(reps, ddof=1)))


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
    picks = rng.integers(0, n, (BOOT, n))
    wts = np.zeros((BOOT, n))
    np.add.at(wts, (np.repeat(np.arange(BOOT), n), picks.ravel()), 1.0 / n)
    mean = {L: c.mean(0) for L, c in curves.items()}
    rep = {L: wts @ c for L, c in curves.items()}                # (BOOT, GRID) replicate mean curves
    crossings, cross_reps = {}, {}
    for L1, L2 in zip(Ls, Ls[1:]):
        cross_reps[(L1, L2)] = np.array([_cross(grid, rep[L1][b], rep[L2][b]) for b in range(BOOT)])
        est = _cross(grid, mean[L1], mean[L2])
        fin = cross_reps[(L1, L2)][np.isfinite(cross_reps[(L1, L2)])]
        crossings[(L1, L2)] = tuple(float(x) for x in (est, *(np.percentile(fin, [50 * (1 - LEVEL), 50 * (
            1 + LEVEL)]) if fin.size else (math.nan, math.nan))))
    k1, k0 = list(crossings)[-1], list(crossings)[-2]
    M_top, per_top = data[Ls[-1]]
    window = float(np.std(np.concatenate(list(per_top.values())) / M_top, ddof=1))
    floor = WINDOW_FRACTION * window
    pc_est = _band(crossings[k1][0], cross_reps[k1], crossings[k0][0], (cross_reps[k1], cross_reps[k0]), floor)
    status = {"p_c_crossing": _resolved(crossings[k1][0], cross_reps[k1])}
    levels = {L: [_level(grid, mean[L], r) for r in LEVELS] for L in Ls}
    boot = {L: np.array([[_level(grid, rep[L][b], r) for r in LEVELS] for b in range(BOOT)]) for L in Ls}
    with warnings.catch_warnings():         # a level never reached by a replicate is all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        var0 = {L: np.nanvar(b, axis=0, ddof=1) for L, b in boot.items()}
    res = FSSResult(lattice, dilution, Ls, seeds, sweeps, crossings, pc_est, (), (), (), levels, var0)
    res.undetermined = {k: f"undetermined:crossing: {why}" for k, why in status.items() if why}
    p_all, y_all = _joint(Ls, levels, var0)
    p_cut, y_cut = _joint(Ls[1:], levels, var0)
    fa = np.array([_joint(Ls, {L: boot[L][b] for L in Ls}, var0, 1e-4) for b in range(BOOT)])
    fc = np.array([_joint(Ls[1:], {L: boot[L][b] for L in Ls}, var0, 1e-4) for b in range(BOOT)])
    res.p_c_collapse = _band(p_all, fa[:, 0], p_cut, (fa[:, 0], fc[:, 0]), floor)
    res.window = window
    res.inv_nu = _band(y_all, fa[:, 1], y_cut, (fa[:, 1], fc[:, 1]))
    no_levels = "no level R* inside every size's curve"
    for k, est, reps in (("p_c_collapse", p_all, fa[:, 0]), ("inv_nu_collapse", y_all, fa[:, 1])):
        why = _resolved(est, reps, no_levels)
        if why:
            res.undetermined[k] = f"undetermined:collapse: {why}"
    widths = {L: np.std(np.concatenate(list(data[L][1].values())) / data[L][0], ddof=1) for L in Ls}
    lslope = lambda w0, w1: -math.log(w1 / w0) / math.log(Ls[-1] / Ls[0]) if w0 > 0 and w1 > 0 else math.nan
    slope = lslope(widths[Ls[0]], widths[Ls[-1]])
    bsl = []
    for k in picks:          # a resample of identical sweeps has width 0: no slope (NaN replicate)
        wd = [np.std(np.concatenate([data[L][1][seeds[i]] for i in k]) / data[L][0], ddof=1) for L in (Ls[0], Ls[-1])]
        bsl.append(lslope(*wd))
    bsl = np.array(bsl)
    why = _resolved(slope, bsl, "a zero width of n*/M in the resamples")
    if why:
        res.undetermined["inv_nu_width_slope"] = f"undetermined:width: {why}"
        res.width_slope = (math.nan,) * 3
    else:
        fin = bsl[np.isfinite(bsl)]
        res.width_slope = tuple(float(x) for x in (slope, *np.percentile(fin, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])))
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
