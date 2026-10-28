"""Ferrenberg-Swendsen histogram reweighting of canonical energy series (M2): single histogram (a run at beta0
reweighted to beta near beta0) and multiple histogram (runs at several beta combined), the reliable window from
histogram overlap, and its refusal.

Sources.  The Ferrenberg-Swendsen papers (PRL 61, 2635 (1988); PRL 63, 1195 (1989)) were not fetched (not freely
available); the equations are taken from Kumar, Bouzida, Swendsen, Kollman & Rosenberg, J. Comput. Chem. 13,
1011 (1992) (PDF read, pages rendered), who state them "essentially those of Ferrenberg and Swendsen", and from
Berg, arXiv:cond-mat/9909236 (PDF read):
  single histogram: P_beta(E) = N_beta0(E) e^{-(beta - beta0) E} / sum_E N_beta0(E) e^{-(beta - beta0) E} (Kumar
      Eqs. (17), (18) with one temperature change and no restraint; Berg Eq. (2.15) with w = e^{-beta0 E}); on the
      series: <O>_beta = sum_t O_t e^{-(beta - beta0) E_t} / sum_t e^{-(beta - beta0) E_t}
  multiple histogram: P_beta(E) = sum_k g_k^{-1} N_k(E) e^{-beta E} / sum_m n_m g_m^{-1} e^{f_m - beta_m E},
      e^{-f_j} = sum_E P_{beta_j}(E) (Kumar Eqs. (19), (20), iterated from f = 0 (their text below Eq. (21));
      the statistical-inefficiency factors g_k = 1 + 2 tau are those of the original Ferrenberg-Swendsen
      equations (Kumar's footnote to Eq. (19) and Eq. (22)); here g_k = 2 tau_int(E) with stats.tau_int's
      tau_int = 1/2 + sum rho (the same 1 + 2 sum_{t >= 1} rho)).  Gauge f_0 = 0; iterate until max |df| < 1e-12.
Window rule (stated; Berg below Eq. (2.15): reweighting works only where the sampled histogram includes the
target's).  Overlap O(beta) = sum_E min(p0(E), p_beta(E)), p0 the pooled (all seeds) normalised histogram of the
run at beta0 and p_beta the same histogram reweighted to beta.  beta is reliable iff O(beta) >= O_MIN = 1/2: for a
Gaussian histogram of width sigma_E, O = 2 Phi(-|beta - beta0| sigma_E / 2), so the rule allows |beta - beta0|
<= 1.349 / sigma_E, a reweighted mean within 1.35 sigma_E of the sampled one.  The window [lo, hi] is found by
bisection to 1e-9 relative on each side, unbounded (inf) when O stays >= O_MIN out to |beta - beta0| = 1e3 (a
histogram whose extreme level already holds half the mass) (O is assumed to fall monotonically away from beta0:
the tests check it on
a grid).  Multiple histogram: the union of the runs' windows, which must chain (each window meets the next in
beta), else precondition:histogram-overlap (the free energies f_m are not determined).  A beta outside the
window: precondition:reweight-window.  Both are input-validity refusals, never sampler refusals; with
enforce=False (the run path's implicit default beta) a beta outside the window is returned with in_window False.

  window(series, beta0) -> (lo, hi)                         series: one 1-d energy array per named seed
  overlap(series, beta0, beta) -> float
  single_histogram(series, beta0, beta, N) -> Reweight(u, C per seed and bands, overlap, window)
  multiple_histogram(runs, beta, N) -> Reweight (or [Reweight] for a list / tuple / array of beta)
      runs: [(beta_m, [series per seed])], the same seeds in every run; per seed its own f_m (so seeds stay independent), bands over seeds
_HOOKS["factor_scale"] (1.0; planted e.g. 1.25: e^{-1.25 (beta - beta0) E}, a wrong reweighting factor);
_HOOKS["free_energies"] ("solve"; planted "zero": the f_m left at 0, not solved).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import stats as ST

O_MIN = 0.5
CAP = 1e3                # |beta - beta0| beyond which a side is called unbounded
_HOOKS = {"factor_scale": 1.0, "free_energies": "solve"}


@dataclass(frozen=True)
class Reweight:
    beta: float
    per_seed: dict          # "u", "C" -> tuple over seeds
    bands: dict
    overlap: float          # single: O(beta); multiple: the largest run overlap
    window: tuple
    text: str
    in_window: bool = True  # False only with enforce=False and beta outside the window (caller labels it)


def _series(series):
    out = [np.asarray(s, float).ravel() for s in series]
    if len(out) < 2 or any(s.size < 8 or not np.isfinite(s).all() for s in out):
        raise Refusal("precondition:time-series", "need >= 2 named-seed energy series of >= 8 finite samples")
    return out


def _lse(a, w=None):
    top = a.max()
    return top + math.log(float(np.exp(a - top).sum() if w is None else (w * np.exp(a - top)).sum()))


def _hist(series):
    E, c = np.unique(np.concatenate(series), return_counts=True)
    return E, c / c.sum()


def _shift(beta0, beta):
    return -_HOOKS["factor_scale"] * (float(beta) - float(beta0))


def log_factor(beta0, beta, E):
    """ln of the single-histogram factor e^{-(beta - beta0) E} (hook included)."""
    return _shift(beta0, beta) * np.asarray(E, float)


def _overlap(E, p0, beta0, beta):
    a = np.log(p0) + log_factor(beta0, beta, E)
    return float(np.minimum(p0, np.exp(a - _lse(a))).sum())


def overlap(series, beta0, beta):
    return _overlap(*_hist(_series(series)), beta0, beta)


def window(series, beta0):
    E, p0 = _hist(_series(series))
    if E.size < 2:
        raise Refusal("precondition:histogram", "a one-level energy histogram: no reweighting window")
    sd = math.sqrt(float(p0 @ (E - p0 @ E) ** 2))
    edges = []
    for sign in (-1, 1):
        lo, hi = 0.0, 1.0 / sd
        while _overlap(E, p0, beta0, beta0 + sign * hi) >= O_MIN and hi <= CAP:
            lo, hi = hi, 2 * hi
        if hi > CAP:                     # the overlap never falls below O_MIN: unbounded on this side
            edges.append(sign * math.inf)
            continue
        while hi - lo > 1e-9 * max(1.0, abs(beta0)):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _overlap(E, p0, beta0, beta0 + sign * mid) >= O_MIN else (lo, mid)
        edges.append(float(beta0) + sign * lo)
    return max(edges[0], 0.0), edges[1]


def _moments(E, logw, beta, N):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    u = float(w @ E)
    return u / N, float(beta) ** 2 * float(w @ (E - u) ** 2) / N


def _bands(per, conf):
    return {k: ST.band(v, conf) for k, v in per.items()}


def _inside(beta, win, what, enforce=True):
    """beta inside win?  Outside: refused precondition:reweight-window when enforced (an explicit request),
    else False (the caller labels the estimate undetermined)."""
    if win[0] <= float(beta) <= win[1]:
        return True
    if enforce:
        raise Refusal("precondition:reweight-window", f"beta = {float(beta):.6g} outside the reliable window "
                      f"[{win[0]:.6g}, {win[1]:.6g}] of {what} (overlap rule O >= {O_MIN:g})")
    return False


def single_histogram(series, beta0, beta, N, conf=0.99, enforce=True):
    """enforce=False: a beta outside the window gives the estimate with in_window False (not refused)."""
    s = _series(series)
    win = window(s, beta0)
    ok = _inside(beta, win, f"the run at beta0 = {float(beta0):.6g}", enforce)
    per = {"u": [], "C": []}
    for x in s:
        u, C = _moments(x, log_factor(beta0, beta, x), beta, N)
        per["u"].append(u)
        per["C"].append(C)
    per = {k: tuple(v) for k, v in per.items()}
    b, O = _bands(per, conf), overlap(s, beta0, beta)
    text = (f"single histogram beta0 = {float(beta0):.6g} -> beta = {float(beta):.6g}: u = {b['u'].mean:.6g} +- "
            f"{b['u'].half_width:.2g}, C = {b['C'].mean:.5g} +- {b['C'].half_width:.2g} ({conf:.0%} Student-t over "
            f"{len(s)} seeds); overlap {O:.3f}, window [{win[0]:.6g}, {win[1]:.6g}]"
            + ("" if ok else "; beta OUTSIDE the window"))
    return Reweight(float(beta), per, b, O, win, text, ok)


def _lse_axis(a, axis):
    top = a.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(a - top).sum(axis=axis, keepdims=True))).squeeze(axis)


def _free_energies(levels, counts, n_eff, betas):
    """Kumar Eqs. (19), (20) on one seed's level histograms: f (len R), f_0 = 0."""
    f = np.zeros(len(betas))
    if _HOOKS["free_energies"] != "solve":
        return f
    num = np.log(counts.sum(axis=0))                     # ln sum_k N_k(E) / g_k
    lnn, BE = np.log(n_eff), np.outer(levels, betas)
    for _ in range(100000):
        lp = num - _lse_axis(lnn + f - BE, 1)            # ln P_{beta = 0}(E), Eq. (19)
        new = -_lse_axis(lp[:, None] - BE, 0)            # Eq. (20)
        new -= new[0]
        if np.abs(new - f).max() < 1e-12:
            return new
        f = new
    raise Refusal("no-solution:convergence", "Ferrenberg-Swendsen free energies did not converge in 1e5 iterations")


def multiple_histogram(runs, beta, N, conf=0.99, enforce=True):
    """beta: one inverse temperature (-> Reweight) or a sequence (-> [Reweight], the f_m solved once per seed).
    enforce=False: a target outside the combined window gives in_window False (windows that do not chain are
    always refused precondition:histogram-overlap: the f_m are not determined)."""
    many = isinstance(beta, (list, tuple, np.ndarray))
    try:
        targets = [float(b) for b in (beta if many else (beta,))]       # floats, Fractions, mpf
    except (TypeError, ValueError):
        raise Refusal("precondition:beta", f"beta = {beta!r}: need a number or a sequence of numbers")
    runs = sorted(((float(b), _series(s)) for b, s in runs), key=lambda r: r[0])
    if len(runs) < 2 or len({len(s) for _, s in runs}) != 1:
        raise Refusal("precondition:histogram-runs", "need >= 2 runs at distinct beta with the same named seeds")
    wins = [window(s, b) for b, s in runs]
    for (b1, _), (b2, _), w1, w2 in zip(runs, runs[1:], wins, wins[1:]):
        if w2[0] > w1[1]:
            raise Refusal("precondition:histogram-overlap", f"runs at beta = {b1:.6g} and {b2:.6g} do not overlap: "
                          f"windows [{w1[0]:.6g}, {w1[1]:.6g}] and [{w2[0]:.6g}, {w2[1]:.6g}]")
    win = (wins[0][0], wins[-1][1])
    oks = [_inside(b, win, "the combined runs", enforce) for b in targets]
    betas = np.array([b for b, _ in runs])
    per = [{"u": [], "C": []} for _ in targets]
    for i in range(len(runs[0][1])):
        xs = [s[i] for _, s in runs]
        g = np.array([2 * ST.tau_int(x).tau for x in xs])
        levels = np.unique(np.concatenate(xs))
        counts = np.array([np.bincount(np.searchsorted(levels, x), minlength=levels.size) for x in xs]) / g[:, None]
        n_eff = np.array([x.size for x in xs]) / g
        f = _free_energies(levels, counts, n_eff, betas)
        den = _lse_axis(np.log(n_eff) + f - np.outer(levels, betas), 1)
        for b, out in zip(targets, per):
            u, C = _moments(levels, np.log(counts.sum(axis=0)) - b * levels - den, b, N)
            out["u"].append(u)
            out["C"].append(C)
    res = []
    for b, out, ok in zip(targets, per, oks):
        out = {k: tuple(v) for k, v in out.items()}
        bd = _bands(out, conf)
        O = max(overlap(s, bm, b) for bm, s in runs)
        text = (f"multiple histogram over beta = {', '.join(f'{x:.6g}' for x in betas)} -> beta = {b:.6g}: u = "
                f"{bd['u'].mean:.6g} +- {bd['u'].half_width:.2g}, C = {bd['C'].mean:.5g} +- {bd['C'].half_width:.2g} "
                f"({conf:.0%} Student-t over {len(runs[0][1])} seeds); window [{win[0]:.6g}, {win[1]:.6g}]"
                + ("" if ok else "; beta OUTSIDE the window"))
        res.append(Reweight(b, out, bd, O, win, text, ok))
    return res if many else res[0]


__all__ = ["O_MIN", "Reweight", "log_factor", "multiple_histogram", "overlap", "single_histogram", "window"]
