"""Statistics of Markov-chain Monte Carlo time series: the one module the advanced-Monte-Carlo tasks share
(M1 cluster algorithms, M2 Wang-Landau / multicanonical / reweighting, M3 population annealing; ruling
A-R4).  numpy only; every function takes plain arrays and returns a frozen dataclass with a `text`.

Interface
  autocorrelation(x) -> ndarray            normalised rho(t), t = 0 .. n - 1 (FFT, mean subtracted)
  tau_int(x, c=6.0) -> TauInt              integrated autocorrelation time with the self-consistent
                                           window rule below; tau, its stderr, the window M
  binned(x, n_blocks=32) -> Binned         mean and stderr from n_blocks block means
  binning_curve(x) -> [(block, stderr)]    stderr of the mean at blocks 1, 2, 4, ... (a plateau
                                           diagnostic: it rises to sqrt(2 tau_int) x naive and stays)
  jackknife(f, series, n_blocks=32) -> Jackknife
                                           leave-one-block-out estimate and stderr of a nonlinear
                                           f(list of block-restricted arrays) (e.g. a variance)
  band(values, conf=0.99) -> Band          Student-t band of the mean of independent named-seed runs
                                           (amp/_stats.Band; df = runs - 1)
  t_quantile(prob, df)                     the Student-t quantile used by every band
  equilibration(traces, conf=0.99) -> Equilibration
                                           drift diagnostic over named seeds (below)
  chi2_gof(observed, probs, alpha, min_expected=5) -> Chi2
                                           Pearson's goodness of fit against exact cell probabilities
  power_fit(Ls, taus) -> PowerFit          tau ~ A L^z per seed, z in a Student-t band over seeds, and
                                           a log-or-power curvature diagnostic (below)

Window rule (tau_int).  rho(t) is the normalised autocorrelation; tau_hat(M) = 1/2 + sum_{t=1}^{M} rho(t)
(Salas & Sokal, arXiv:hep-lat/9605018v2 (1997), Eq. (4.11)); M is the smallest
integer with M >= c tau_hat(M), c = 6 (their p. 13: "the choice c ~ 6 is reasonable" for a roughly
exponential rho, the rule of their ref. [32], Madras & Sokal 1988, not fetched); the stderr is
sqrt(2 (2M + 1) / n) tau_hat (their Eq. (4.13), valid for tau << M << n).  `reliable` is False when no
window is found below n / 4 or when n < 100 tau_hat (Salas & Sokal report the procedure works for
n >~ 10^4 tau; 100 is this package's weaker floor, stated, not theirs).

Equilibration.  traces is (runs, T): the measured series of one observable after burn-in, one row per
named seed.  Per run d_r = mean(second half) - mean(first half); equilibrated iff the 99% Student-t
band of the mean of d_r over runs contains 0 (needs >= 2 runs).  A drift outside the band labels the
run "undetermined"; this is a diagnostic, never a refusal of a correct sampler (standing ruling).

power_fit.  taus is (runs, len(Ls)); per run the OLS slope of ln tau on ln L is z_r; z is reported as
band(z_r).  Curvature: the local slopes between the two smallest and the two largest L, s_lo and s_hi,
per run; their difference s_hi - s_lo in a Student-t band.  A pure power law has s_hi = s_lo; tau ~ a + b
ln L has s_hi < s_lo.  `separable_from_zero` = the z band excludes 0; `separable_from_log` = the
difference band contains 0 AND excludes the log model's predicted difference (computed from the
per-run log fit tau = a + b ln L at the same L).  When either is False the caller prints, never asserts.
"""
import math
from dataclasses import dataclass

import numpy as np

from functools import lru_cache

from ..amp._stats import Band
from ..amp._stats import t_quantile as _t_quantile
from ..core import Refusal

WINDOW_C = 6.0
RUN_LENGTH_FLOOR = 100


@dataclass(frozen=True)
class TauInt:
    tau: float
    stderr: float
    window: int
    n: int
    reliable: bool
    text: str


@dataclass(frozen=True)
class Binned:
    mean: float
    stderr: float
    block: int
    n_blocks: int


@dataclass(frozen=True)
class Jackknife:
    estimate: float
    stderr: float
    n_blocks: int


@dataclass(frozen=True)
class Equilibration:
    equilibrated: bool
    drift: float
    half_width: float
    text: str


@dataclass(frozen=True)
class Chi2:
    stat: float
    df: int
    critical: float
    p_value: float
    alpha: float
    ok: bool
    cells: int
    text: str


@dataclass(frozen=True)
class PowerFit:
    z: Band
    slope_difference: Band
    log_difference: float
    separable_from_zero: bool
    separable_from_log: bool
    text: str


def _series(x, least=2):
    a = np.asarray(x, dtype=float).ravel()
    if a.size < least or not np.isfinite(a).all():
        raise Refusal("precondition:time-series", f"need >= {least} finite samples, got {a.size}")
    return a


def autocorrelation(x):
    a = _series(x)
    n = a.size
    d = a - a.mean()
    f = np.fft.rfft(d, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    if acf[0] <= 0:
        return np.r_[1.0, np.zeros(n - 1)]       # a constant series: no correlation to measure
    return acf / acf[0]


def tau_int(x, c=WINDOW_C):
    a = _series(x, 8)
    n = a.size
    rho = autocorrelation(a)
    cum = 0.5 + np.cumsum(rho[1:])               # cum[M - 1] = tau_hat(M)
    Ms = np.arange(1, n)
    ok = np.nonzero(Ms >= c * cum)[0]
    found = ok.size > 0 and Ms[ok[0]] < n // 4
    M = int(Ms[ok[0]]) if ok.size else n - 1
    tau = float(max(cum[M - 1], 0.5))
    se = math.sqrt(2 * (2 * M + 1) / n) * tau
    reliable = bool(found and n >= RUN_LENGTH_FLOOR * tau)
    text = (f"tau_int {tau:.4g} +- {se:.2g} (window M = {M}, c = {c:g}, n = {n}"
            + ("" if reliable else "; UNRELIABLE: " + ("no window below n/4" if not found else
                                                         f"n < {RUN_LENGTH_FLOOR} tau")) + ")")
    return TauInt(tau, se, M, n, reliable, text)


def binned(x, n_blocks=32):
    a = _series(x, n_blocks)
    b = a.size // n_blocks
    means = a[:b * n_blocks].reshape(n_blocks, b).mean(axis=1)
    return Binned(float(a.mean()), float(means.std(ddof=1) / math.sqrt(n_blocks)), b, n_blocks)


def binning_curve(x):
    a = _series(x, 4)
    out, b = [], 1
    while a.size // b >= 4:
        k = a.size // b
        m = a[:k * b].reshape(k, b).mean(axis=1)
        out.append((b, float(m.std(ddof=1) / math.sqrt(k))))
        b *= 2
    return out


def jackknife(f, series, n_blocks=32):
    arrs = [_series(s, n_blocks) for s in series]
    n = min(a.size for a in arrs)
    b = n // n_blocks
    if b < 1:
        raise Refusal("precondition:time-series", f"{n} samples < {n_blocks} blocks")
    arrs = [a[:b * n_blocks] for a in arrs]
    full = float(f(arrs))
    idx = np.arange(b * n_blocks).reshape(n_blocks, b)
    loo = np.array([f([a[np.delete(idx, k, 0).ravel()] for a in arrs]) for k in range(n_blocks)])
    se = math.sqrt((n_blocks - 1) / n_blocks * float(((loo - loo.mean()) ** 2).sum()))
    return Jackknife(n_blocks * full - (n_blocks - 1) * float(loo.mean()), se, n_blocks)


@lru_cache(maxsize=256)
def t_quantile(prob, df):
    return _t_quantile(prob, df)


def band(values, conf=0.99):
    """amp/_stats.student_band's band (the same Band), with the t quantile cached."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        raise Refusal("precondition:two-samples", f"a Student-t band needs at least two independent runs, got {n}")
    mean = math.fsum(xs) / n
    se = math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (n - 1)) / math.sqrt(n)
    t = t_quantile((1 + conf) / 2, n - 1)
    return Band(mean, se, t * se, n - 1, conf, t)


def equilibration(traces, conf=0.99):
    tr = np.asarray(traces, dtype=float)
    if tr.ndim != 2 or tr.shape[0] < 2 or tr.shape[1] < 4:
        raise Refusal("precondition:windows", f"traces of shape {tr.shape}: need (runs >= 2, T >= 4)")
    h = tr.shape[1] // 2
    d = tr[:, tr.shape[1] - h:].mean(axis=1) - tr[:, :h].mean(axis=1)
    bd = band(d, conf)
    ok = abs(bd.mean) <= bd.half_width
    text = (f"second - first half {bd.mean:+.3g} +- {bd.half_width:.2g} ({conf:.0%} Student-t, "
            f"{tr.shape[0]} seeds, {tr.shape[1]} samples each): " + ("no drift" if ok else
                                                                      "DRIFT: undetermined"))
    return Equilibration(bool(ok), float(bd.mean), float(bd.half_width), text)


def chi2_gof(observed, probs, alpha, min_expected=5.0):
    """Any count in a cell of exact probability 0 fails outright; cells with expected count < min_expected
    are pooled into one cell; df = cells - 1 (the probabilities are exact, nothing is fitted)."""
    from scipy.special import chdtrc, chdtri
    o = np.asarray(observed, dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    if o.shape != p.shape or o.sum() <= 0 or abs(p.sum() - 1) > 1e-9 or (p < 0).any():
        raise Refusal("precondition:chi2-cells", "observed and probabilities must align; p sums to 1")
    if not 0 < alpha < 1:
        raise Refusal("precondition:chi2-alpha", f"alpha = {alpha!r}: need 0 < alpha < 1")
    if (o[p == 0] > 0).any():
        return Chi2(math.inf, 0, 0.0, 0.0, alpha, False, int((p > 0).sum()),
                    "observed counts in a cell of exact probability 0")
    o, p = o[p > 0], p[p > 0]
    n = o.sum()
    e = n * p
    small = e < min_expected
    oo, ee = list(o[~small]), list(e[~small])
    if small.any():
        oo.append(o[small].sum())
        ee.append(e[small].sum())
    oo, ee = np.array(oo), np.array(ee)
    stat = float(((oo - ee) ** 2 / ee).sum())
    df = len(ee) - 1
    if df < 1:
        raise Refusal("precondition:chi2-cells", "fewer than two cells after pooling")
    crit = float(chdtri(df, alpha))
    pv = float(chdtrc(df, stat))
    return Chi2(stat, df, crit, pv, alpha, stat <= crit, len(ee),
                f"chi2 = {stat:.4g} on {df} df (critical {crit:.4g} at alpha = {alpha:g}; p = {pv:.3g}; "
                f"{int(n)} samples, {len(ee)} cells)")


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def power_fit(Ls, taus, conf=0.99):
    Ls = np.asarray(Ls, float)
    T = np.asarray(taus, float)
    if Ls.ndim != 1 or Ls.size < 3 or T.ndim != 2 or T.shape[1] != Ls.size or T.shape[0] < 2 \
            or (T <= 0).any() or (np.diff(Ls) <= 0).any():
        raise Refusal("precondition:fit-sizes", "need >= 3 increasing L and (runs >= 2, len(L)) taus > 0")
    lx = np.log(Ls)
    z = band([_slope(lx, np.log(t)) for t in T], conf)
    sl = lambda t, i, j: (math.log(t[j]) - math.log(t[i])) / (lx[j] - lx[i])
    k = Ls.size
    diff = band([sl(t, k - 2, k - 1) - sl(t, 0, 1) for t in T], conf)
    pred = []
    for t in T:                                   # the log model a + b ln L fitted per run
        b = _slope(lx, t)
        a = float(t.mean() - b * lx.mean())
        f = a + b * lx
        if (f <= 0).any():
            pred.append(0.0)
            continue
        pred.append(sl(f, k - 2, k - 1) - sl(f, 0, 1))
    logd = float(np.mean(pred))
    from_zero = z.mean - z.half_width > 0 or z.mean + z.half_width < 0
    from_log = abs(diff.mean) <= diff.half_width and abs(logd - diff.mean) > diff.half_width
    text = (f"z = {z.mean:.3f} +- {z.half_width:.3f} ({conf:.0%} Student-t over {T.shape[0]} seeds, L = "
            f"{', '.join(str(int(x)) for x in Ls)}); local-slope change {diff.mean:+.3f} +- {diff.half_width:.3f} "
            f"(a log law a + b ln L predicts {logd:+.3f}); "
            + ("separable from 0" if from_zero else "NOT separable from z = 0") + ", "
            + ("separable from a logarithm" if from_log else "NOT separable from a logarithm"))
    return PowerFit(z, diff, logd, bool(from_zero), bool(from_log), text)


__all__ = ["Band", "Binned", "Chi2", "Equilibration", "Jackknife", "PowerFit", "TauInt", "autocorrelation",
           "band", "binned", "binning_curve", "chi2_gof", "equilibration", "jackknife", "power_fit",
           "t_quantile", "tau_int"]
