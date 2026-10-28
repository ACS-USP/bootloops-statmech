"""Population annealing (M3): Hukushima & Iba's algorithm as described by Machta (arXiv:1006.0252, Sec. II, Eqs.
(1)-(4)) and Wang, Machta & Katzgraber (WMK, arXiv:1508.05647, Sec. II A, Eqs. (1), (2), (4)); Hukushima & Iba, AIP
Conf. Proc. 690, 200 (2003), not fetched (cited through both).

Algorithm (WMK Sec. II A).  R replicas start iid uniform at beta = 0 (an exact equilibrium sample, ln Omega = N ln 2).
The schedule is `temperatures` steps evenly spaced in beta from 0 to beta (WMK Sec. V B).  At a step beta -> beta':
  Q = (1/R~) sum_j exp(-(beta' - beta) E_j)            (WMK Eq. (2): the population estimate of Z(beta')/Z(beta))
  tau_j = (R / R~) exp(-(beta' - beta) E_j) / Q          (WMK Eq. (1), sum_j tau_j = R)
  n_j = floor(tau_j) or ceil(tau_j), the ceiling with probability tau_j - floor(tau_j)   (WMK's nearest-integer
      resampling: E n_j = tau_j, the population size R~ fluctuates about R)
then `sweeps` Metropolis sweeps at beta' (2D Ising: M1's checkerboard metropolis.py; SK / +-J: sk_metropolis.py),
and the observables are measured.  The free-energy estimator (Machta Eq. (4), WMK Eq. (4)) is
  ln Z~(beta_k) = -beta_k F~(beta_k) = N ln 2 + sum_{l <= k} ln Q_l.
Bias at finite R (WMK Sec. IV A, Eqs. (12)-(16), (20), (23)): exp(-beta F~) is the weight whose average over runs
gives the free energy (Machta Eq. (7), WMK Eq. (6)), so E[ln Z~] = ln Z - var(beta F~)/2 - (higher cumulants): one run
UNDERESTIMATES ln Z by about var(beta F~)/2 = rho_f / (2R).  `summary` reports the simple mean over seeds (biased),
that bias estimated as -s^2/2 (s^2 the sample variance of ln Z~ over seeds), and Machta's weighted estimator
ln[(1/M) sum_r exp(ln Z~_r)] with a jackknife over seeds (stats.jackknife, one seed per block), which removes the
leading bias.  Normalisation.  WMK's Eq. (2) divides by the actual size R~ (a ratio estimator); then e^{ln Z~} is
biased at O(temperatures / R): measured E[Z~ / Z] = 1.0153 +- 0.0020 at R = 5 (Gaussian SK N = 16, beta = 2, 10
temperatures, 1 sweep, 200000 runs; the reviewer's independent code: 1.022 +- 0.002).  Dividing by the nominal R
instead, ln Z~_nom = ln Z~ + sum_{k < K} ln(R~_k / R), makes e^{ln Z~_nom} exactly unbiased for Z whatever the
population sizes (a martingale argument, E n_j = tau_j; derived here and by the reviewer, not printed in the
sources): measured 0.9995 +- 0.0027 on the same runs.  The WMK form has the smaller variance (sd of ln Z~ 0.84
against 1.01 there), so the simple mean uses it (`ln_z`), and Machta's weighted average, whose point is an unbiased
average of e^{ln Z~}, uses the nominal form (`ln_z_nominal`).  An extinct run contributes e^{-inf} = 0 to that
average (Machta Eq. (7) averages over all M runs) and is left out of the simple mean.  A population that dies out
only at the last temperature (every n_j = 0 after the last resampling) keeps a finite ln Z~ (that Q was computed
from the population before the resampling) and has a NaN energy; it is left out of the simple mean (which needs
both finite), enters the weighted average with that finite ln Z~, and the run is labelled undetermined (extinct).
Families (WMK Sec. IV B): replica i of the initial population founds family i; n_i is the fraction of the final
population descending from it;
  rho_t = R~ sum_i n_i^2 (Eq. (28)), R~ / rho_t the effective number of independent measurements;
  S_f = -sum_i n_i ln n_i (Eq. (30)), e^{S_f} the effective number of families, rho_s = R~ / e^{S_f} (Eq. (31)).
Labels.  WMK (Sec. V B) require R >= 100 rho_s, i.e. e^{S_f} >= 100, and recommend rho_t (Sec. VI B: "it may be
preferable to use rho_t ... as an equilibration criterion"); here a seed passes when R~ / rho_t >= FAMILY_MIN = 100 at
the final beta (rho_t >= rho_s, so this implies WMK's criterion; the factor 100 is theirs, its use with rho_t this
package's choice).  Machta (Sec. III): if var(beta F~) "is smaller than about 0.5 then a modest ensemble of
independent runs will be sufficient"; here the sample variance over seeds must be <= VAR_MAX = 1/2.  A run is
"equilibrated" when every seed passes, var <= 1/2 and no population died out; otherwise "undetermined" (never
refused: a correct sampler is not refused at runtime).

  Ising2D(L), SK(J)        the systems (energy, uniform start, sweeps per named-seed block)
  anneal(system, beta, R, temperatures, sweeps, seeds) -> PARun
  family_stats(counts) -> (rho_t, R / rho_t, rho_s, e^{S_f}) for family sizes `counts`
  summary(run) -> PASummary   estimators, diagnostics, status
Random numbers: seed s uses numpy default_rng([s, 0x5041]) for its start, resampling and sweep seeds; its blocks
never draw from another seed's stream, so its results do not depend on the other seeds in the run (tested).
_HOOKS (tests): "reweight_scale" 1.0 (planted c: exp(-c (beta' - beta) E) in Q and tau, a wrong reweighting factor);
"resample" True (False: n_j = 1, the resampling step missing); "families" "track" ("reset": every replica refounds
its own family at each step, a broken family bookkeeping); "q_offset" 0 (1: Q divides by R~ + 1, a normalisation off
by one, ln Z~ low by about temperatures / R).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal

FAMILY_MIN = 100
VAR_MAX = 0.5
STREAM = 0x5041
_HOOKS = {"reweight_scale": 1.0, "resample": True, "families": "track", "q_offset": 0}


class Ising2D:
    """H = -sum s s over the 2N bonds of the L x L torus (even L >= 4: the checkerboard sweep of metropolis.py)."""

    def __init__(self, L):
        self.L, self.N = L, L * L

    def init(self, rng, n):
        return rng.choice(np.array([-1, 1], np.int64), size=(n, self.L, self.L))

    def energy(self, s):
        return -(s * (np.roll(s, 1, 1) + np.roll(s, 1, 2))).sum(axis=(1, 2)).astype(float)

    def sweep(self, s, beta, sweeps, rngs, sizes):
        from .metropolis import metropolis_run
        if not sweeps:
            return s
        out, a = [], 0
        for g, n in zip(rngs, sizes):
            sub = int(g.integers(2 ** 62))
            out.append(metropolis_run(self.L, beta, sweeps, (sub,), init=s[a:a + n])["final"] if n else s[a:a])
            a += n
        return np.concatenate(out)


class SK:
    """H = -sum_{i<j} J_ij s_i s_j for one coupling matrix J (sk_metropolis.py)."""

    def __init__(self, J):
        self.J = np.asarray(J, float)
        self.N = self.J.shape[0]

    def init(self, rng, n):
        return rng.choice((-1.0, 1.0), size=(n, self.N))

    def energy(self, s):
        from .sk_metropolis import sk_energy
        return sk_energy(self.J, s)

    def sweep(self, s, beta, sweeps, rngs, sizes):
        from .sk_metropolis import sk_sweeps
        return sk_sweeps(self.J, s, beta, sweeps, rngs, sizes)


@dataclass
class PARun:
    N: int
    R: int
    betas: np.ndarray          # (temperatures + 1,)
    seeds: tuple
    ln_z: np.ndarray           # (seeds, temperatures + 1): ln Z~ of each seed at each beta (WMK's Q, actual R~)
    energy: np.ndarray         # (seeds, temperatures + 1): population mean of H / N after the sweeps
    sizes: np.ndarray          # (seeds, temperatures + 1): R~
    families: np.ndarray       # (seeds, 4): rho_t, R~ / rho_t, rho_s, e^{S_f} at the final beta
    extinct: np.ndarray        # (seeds,) bool

    @property
    def ln_z_nominal(self):
        """(seeds,) ln Z~ at the final beta with Q divided by the nominal R: e^{ln Z~_nom} unbiased for Z."""
        with np.errstate(divide="ignore"):
            return self.ln_z[:, -1] + np.log(self.sizes[:, :-1] / self.R).sum(axis=1)


def family_stats(counts):
    c = np.asarray(counts, float)
    c = c[c > 0]
    Rt = c.sum()
    n = c / Rt
    rho_t = Rt * float((n * n).sum())
    eS = math.exp(float(-(n * np.log(n)).sum()))
    return rho_t, Rt / rho_t, Rt / eS, eS


def log_weight(beta, beta_next, E):
    """The reweighting factor's logarithm -(beta' - beta) E (WMK Eq. (1)); the "reweight_scale" plant multiplies it."""
    return -_HOOKS["reweight_scale"] * (beta_next - beta) * E


def _resample(x, R, rng):
    """ln Q and the copy numbers for log-weights x = -(beta' - beta) E of one seed's population."""
    m = float(x.max())
    w = np.exp(x - m)
    lnQ = m + math.log(float(w.sum()) / (len(w) + _HOOKS["q_offset"]))
    if not _HOOKS["resample"]:
        return lnQ, np.ones(len(x), np.int64)
    tau = R * w / w.sum()
    fl = np.floor(tau)
    return lnQ, (fl + (rng.random(len(x)) < tau - fl)).astype(np.int64)


def anneal(system, beta, R, temperatures, sweeps, seeds):
    seeds = tuple(int(x) for x in seeds)
    M, K = len(seeds), int(temperatures)
    betas = np.linspace(0.0, float(beta), K + 1)
    rngs = [np.random.default_rng([x, STREAM]) for x in seeds]
    sizes = np.full(M, R, np.int64)
    s = np.concatenate([system.init(g, R) for g in rngs])
    fam = np.arange(M * R)
    ln_z, en, sz = np.zeros((M, K + 1)), np.zeros((M, K + 1)), np.zeros((M, K + 1), np.int64)
    ln_z[:, 0], sz[:, 0] = system.N * math.log(2), R
    E = system.energy(s)
    en[:, 0] = [E[a:a + n].mean() / system.N for a, n in zip(np.r_[0, np.cumsum(sizes)[:-1]], sizes)]
    extinct = np.zeros(M, bool)
    for k in range(1, K + 1):
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        copies, new = np.zeros(len(s), np.int64), np.zeros(M, np.int64)
        for b in range(M):
            a, n = starts[b], sizes[b]
            if n == 0:
                ln_z[b, k] = -math.inf
                continue
            lnQ, copies[a:a + n] = _resample(log_weight(betas[k - 1], betas[k], E[a:a + n]), R, rngs[b])
            ln_z[b, k], new[b] = ln_z[b, k - 1] + lnQ, copies[a:a + n].sum()
        sizes = new
        s, fam = np.repeat(s, copies, axis=0), np.repeat(fam, copies)
        if _HOOKS["families"] == "reset":
            fam = np.arange(len(fam))
        s = system.sweep(s, betas[k], sweeps, rngs, sizes)
        E = system.energy(s)
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        for b in range(M):
            a, n = starts[b], sizes[b]
            extinct[b] |= n == 0
            en[b, k] = E[a:a + n].mean() / system.N if n else math.nan
        sz[:, k] = sizes
    if int(sizes.sum()) != len(s) or len(fam) != len(s):
        raise Refusal("no-solution:sampler-invariant", "population annealing: block sizes != population size")
    fs = np.full((M, 4), math.nan)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    for b in range(M):
        if sizes[b]:
            fs[b] = family_stats(np.unique(fam[starts[b]:starts[b] + sizes[b]], return_counts=True)[1])
    return PARun(system.N, R, betas, seeds, ln_z, en, sz, fs, extinct)


@dataclass
class PASummary:
    ln_z_mean: object          # stats.Band of ln Z~ / N over seeds (the simple, biased mean)
    ln_z_weighted: float       # Machta Eq. (7), jackknifed, / N
    ln_z_weighted_stderr: float
    bias_estimate: float       # -s^2 / (2N): the leading bias of the simple mean (WMK Eq. (16))
    var_beta_f: float          # sample variance of ln Z~ over seeds (= var(beta F~))
    energy: object             # stats.Band of the population mean H / N
    status: str
    reasons: tuple
    text: str


def weighted_ln_z(x):
    """Machta Eq. (7): ln[(1/M) sum_r exp(x_r)] (log-sum-exp; an x_r = -inf, an extinct run, contributes 0)."""
    x = np.asarray(x, float)
    m = float(x.max())
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.exp(x - m).mean()))


def _jackknife_weighted(x):
    """Leave-one-seed-out jackknife of weighted_ln_z (stats.jackknife's estimate and stderr, one seed per block), in
    O(M) and admitting the -inf of an extinct run (which stats' time-series check refuses).  With a single finite
    entry (one surviving run) a leave-one-out value is -inf and the stderr is returned as NaN, with the plain
    weighted estimate; `summary` never reaches this case (fewer than two survivors return earlier, undetermined
    with NaN values), only a direct caller can."""
    from .stats import Jackknife
    x = np.asarray(x, float)
    n, m = len(x), float(x.max())
    w = np.exp(x - m)
    with np.errstate(divide="ignore"):
        loo = m + np.log(np.maximum(w.sum() - w, 0.0) / (n - 1))
    if not np.isfinite(loo).all():              # one surviving run: its removal leaves no weight
        return Jackknife(weighted_ln_z(x), math.nan, n)
    se = math.sqrt((n - 1) / n * float(((loo - loo.mean()) ** 2).sum()))
    return Jackknife(n * weighted_ln_z(x) - (n - 1) * float(loo.mean()), se, n)


def _nan_band():
    from ..amp._stats import Band
    return Band(math.nan, math.nan, math.nan, 0, 0.99, math.nan)


def summary(run):
    from . import stats as ST
    N, M = run.N, len(run.seeds)
    xn = run.ln_z_nominal
    ok = np.isfinite(run.ln_z[:, -1]) & np.isfinite(run.energy[:, -1])
    x, en = run.ln_z[ok, -1], run.energy[ok, -1]
    reasons = []
    if run.extinct.any():
        reasons.append(f"population died out for seeds {[s for s, e in zip(run.seeds, run.extinct) if e]}")
    low = [(s, float(f[1])) for s, f in zip(run.seeds, run.families) if not f[1] >= FAMILY_MIN]
    alive = [t for t in low if t[1] == t[1]]
    if alive:
        worst = min(alive, key=lambda t: t[1])
        reasons.append(f"too few families (R~ / rho_t < {FAMILY_MIN}) for {len(alive)} of {M} seeds (lowest: seed "
                       f"{worst[0]}, R~ / rho_t = {worst[1]:.3g})")
    if len(x) < 2:                      # labelled, never refused: a correct sampler is not refused at runtime
        reasons.append(f"fewer than two surviving seeds ({len(x)} of {M})")
        why = "UNDETERMINED: " + "; ".join(reasons)
        return PASummary(_nan_band(), math.nan, math.nan, math.nan, math.nan, _nan_band(), "undetermined",
                         tuple(reasons), f"population annealing: R = {run.R}, {M} seeds; {why}")
    var = float(np.var(x, ddof=1))
    if not var <= VAR_MAX:
        reasons.append(f"var(beta F~) over seeds {var:.3g} > {VAR_MAX}")
    jk = _jackknife_weighted(xn)
    band, eb = ST.band(x / N), ST.band(en)
    status = "undetermined" if reasons else "equilibrated"
    fam = run.families[np.isfinite(run.families[:, 1])]
    text = (f"population annealing: R = {run.R}, {len(run.betas) - 1} temperatures to beta = {run.betas[-1]:.6g}, "
            f"{M} seeds; ln Z / N = {band.mean:.6g} +- {band.stderr:.2g} (simple mean, biased by about "
            f"{-var / (2 * N):.2g}), weighted (Machta Eq. (7), nominal-R normalisation, jackknife) "
            f"{jk.estimate / N:.6g} +- {jk.stderr / N:.2g}; var(beta F~) = {var:.3g}; R~ / rho_t min "
            f"{fam[:, 1].min():.4g}, e^S_f min {fam[:, 3].min():.4g}; "
            + (f"UNDETERMINED: {'; '.join(reasons)}" if reasons else "equilibrated (families and var(beta F~))"))
    return PASummary(band, jk.estimate / N, jk.stderr / N, -var / (2 * N), var, eb, status, tuple(reasons), text)


def check_params(R, temperatures, sweeps, seeds):
    """The run path's parameters; sweeps >= 1 (sweeps = 0 is pure resampling with no Metropolis move: the
    population never equilibrates at the new temperature; anneal itself accepts it, the tests use it)."""
    from ..percolation.lattice2d import check_seeds
    for name, v, lo, hi in (("population", R, 2, 10 ** 6), ("temperatures", temperatures, 1, 10 ** 5),
                            ("sweeps", sweeps, 1, 10 ** 4)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise Refusal(f"precondition:{name}", f"{name} = {v!r}: need an int in {lo} .. {hi}")
    return int(R), int(temperatures), int(sweeps), check_seeds(seeds)


__all__ = ["FAMILY_MIN", "Ising2D", "PARun", "PASummary", "SK", "VAR_MAX", "anneal", "check_params",
           "family_stats", "log_weight", "summary", "weighted_ln_z"]
