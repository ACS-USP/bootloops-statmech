"""Multicanonical runs of the 2D Ising torus with weights from a Wang-Landau estimate, canonical averages reweighted
from them, and the detailed-balance / ergodicity chi^2 of the fixed-weight walk (M2).

Weights.  Berg, "Introduction to multicanonical Monte Carlo simulations", arXiv:cond-mat/9909236 (Fields Inst.
Commun. 26, 2000), PDF read: the multicanonical weight w_mu(E) approximates w_{1/n}(E) = 1/n(E) (Eqs. (1.5),
(1.6)), sampled with the Metropolis acceptance min[1, w(k')/w(k)] (Eq. (2.11)); here w_mu = exp(-S_WL(E)) with S_WL
the seed's own Wang-Landau estimate (one level per interval, the finest partition of Berg & Neuhaus, PRL 68, 9
(1992), arXiv:hep-lat/9202004, PDF read, Eq. (7): e^{alpha_k + beta_k S} for S_k < S <= S_k+1).  Levels the WL run
never visited get w = 0 (never entered).
Reweighting.  Berg Eq. (2.15): <O>(beta) = sum_i O_i w_mu^{-1}(E_i) e^{-beta E_i} / sum_i w_mu^{-1}(E_i) e^{-beta E_i}
(Berg & Neuhaus Eq. (10) for the action density); "this relation is rigorous, because the weights ... are exactly
known" (Berg below Eq. (1.8)), so the estimate does not inherit the WL error, only the sampling error.  O = E
gives u, the second moment C = beta^2 Var(E) / N (Berg Eq. (2.16)), and |m| from the per-level mean of |M|.
Window rule (Berg below Eq. (2.15): "re-weighting ... can only work in a temperature range where the multicanonical
histogram includes the canonical histogram"), stated here: beta is reliable iff the canonical law the weights
predict, P_w(E) ~ e^{S_WL(E) - beta E}, puts at most MASS_MAX = 10^-3 of its mass on levels some seed visited
fewer than H_MIN = 20 times (a count's relative error 1/sqrt(20) ~ 0.22); otherwise precondition:reweight-window
(an input-validity refusal).
Status "equilibrated" iff every seed's walkers visited every level and stats.equilibration of the per-sweep
energy (one trace per seed: the mean over its walkers) shows no drift; else "undetermined".

  muca_run(L, ln_g, k, seeds, walkers, sweeps, burn_in) -> MucaRun     ln_g (seeds, levels) on the levels k
  reweight(run, beta) -> Reweighted(u, C, abs_m per seed; bands; mass_poor)
  stationarity_chi2(L, ln_g, samples, seed, alpha) / ergodicity_chi2(L, ln_g, chains, sweeps, seed, alpha) -> Chi2
      against the exact law pi(s) ~ exp(-ln_g(E(s))) over exact_small's classes (L <= 4): one sweep (N random-site
      proposals) applied to `samples` exact draws; or `chains` walkers from all +1 after `sweeps` sweeps
_HOOKS["weight_power"] (1.0; planted e.g. 0.9: the reweighting factor w_mu^{-0.9}, a wrong reweighting factor).
Second stream: the multicanonical walkers of seed s draw from numpy default_rng((s, MUCA_STREAM)), not WL's stream.
"""
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from . import stats as ST
from .flat_walk import Walkers

MUCA_STREAM = 0x4D55
_HOOKS = {"weight_power": 1.0}
H_MIN = 20
MASS_MAX = 1e-3


@dataclass
class MucaRun:
    L: int
    k: np.ndarray
    E: np.ndarray
    ln_w: np.ndarray        # (seeds, levels): ln of 1 / w_mu = S_WL
    H: np.ndarray           # (seeds, levels) visits after burn-in
    msum: np.ndarray        # (seeds, levels) sum of |M|
    seeds: tuple
    walkers: int
    sweeps: int
    burn_in: int
    equilibration: ST.Equilibration
    status: str
    text: str


@dataclass(frozen=True)
class Reweighted:
    beta: float
    per_seed: dict          # "u", "C", "abs_m" -> tuple over seeds
    bands: dict
    mass_poor: float
    text: str


def _full(L, k, vals, fill):
    out = np.full(L * L + 1, fill, float)
    out[k] = vals
    return out


def muca_run(L, ln_g, k, seeds, walkers, sweeps, burn_in):
    ln_g = np.asarray(ln_g, float)
    n, N = len(seeds), L * L
    if ln_g.shape != (n, len(k)):
        raise Refusal("precondition:weights", f"ln_g of shape {ln_g.shape}: need (seeds, levels) = ({n}, {len(k)})")
    if 0 not in set(int(x) for x in k):
        raise Refusal("precondition:weights", "the weights do not cover the ground level (start state)")
    S = np.concatenate([np.tile(_full(L, k, row, np.inf), (walkers, 1)) for row in ln_g]).ravel()
    wk = Walkers(L, [(int(s), MUCA_STREAM) for s in seeds], walkers, init=np.ones((n * walkers, N), np.int64))
    K = N + 1
    H, msum = np.zeros(wk.R * K), np.zeros(wk.R * K)
    trace = np.empty((n, sweeps))
    for t in range(burn_in + sweeps):
        if t == burn_in:
            H[:], msum[:] = 0, 0
        wk.run(N, S, None, H, msum)
        if t >= burn_in:
            trace[:, t - burn_in] = (4 * wk.k - 2 * N).reshape(n, walkers).mean(axis=1) / N
    wk.check()
    Hs = H.reshape(n, walkers, K).sum(axis=1)[:, k]
    Ms = msum.reshape(n, walkers, K).sum(axis=1)[:, k]
    eq = ST.equilibration(trace)
    visited = bool((Hs > 0).all())
    ok = visited and eq.equilibrated
    text = (f"multicanonical: {n} seeds x {walkers} walkers x {sweeps} sweeps (burn-in {burn_in}), {len(k)} levels, "
            f"fewest visits {int(Hs.min())}; {eq.text}"
            + ("" if visited else "; UNDETERMINED: a seed never visited some level"))
    return MucaRun(L, np.asarray(k), 4 * np.asarray(k) - 2 * N, ln_g, Hs, Ms, tuple(seeds), walkers, sweeps, burn_in,
                   eq, "equilibrated" if ok else "undetermined", text)


def _logp(a):
    a = a - a.max(axis=-1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(axis=-1, keepdims=True)


def reweight(run, beta, conf=0.99):
    b = float(beta)
    N = run.L * run.L
    E = run.E.astype(float)
    pred = _logp(run.ln_w - b * E)                       # P_w per seed
    poor = (run.H < H_MIN).any(axis=0)
    mass = float(pred[:, poor].sum(axis=1).max()) if poor.any() else 0.0
    if mass > MASS_MAX:
        raise Refusal("precondition:reweight-window", f"beta = {b:.6g}: the weights put {mass:.3g} > {MASS_MAX:g} of "
                      f"the canonical mass on levels visited fewer than {H_MIN} times (E = "
                      f"{run.E[poor].tolist()}); outside the multicanonical run's reliable window")
    with np.errstate(divide="ignore"):
        P = _logp(np.log(run.H) + _HOOKS["weight_power"] * run.ln_w - b * E)   # Berg Eq. (2.15), level histogram
    u = P @ E
    var = (P * (E[None, :] - u[:, None]) ** 2).sum(axis=1)
    mbar = np.where(run.H > 0, run.msum / np.maximum(run.H, 1), 0.0) / N
    per = {"u": tuple(u / N), "C": tuple(b * b * var / N), "abs_m": tuple((P * mbar).sum(axis=1))}
    bands = {key: ST.band(v, conf) for key, v in per.items()}
    text = (f"beta = {b:.6g}: u = {bands['u'].mean:.6g} +- {bands['u'].half_width:.2g}, C = {bands['C'].mean:.5g} +- "
            f"{bands['C'].half_width:.2g} ({conf:.0%} Student-t over {len(run.seeds)} seeds); predicted mass on poorly "
            f"visited levels {mass:.2g} (rule: <= {MASS_MAX:g})")
    return Reweighted(b, per, bands, mass, text)


# ---- detailed balance and ergodicity of the fixed-weight walk ------------------------------------------------
def _exact_law(L, ln_g_levels, k):
    from .exact_small import enumerate_torus
    en = enumerate_torus(L, 2)
    N = L * L
    lev = (2 * N - en.n_sat) // 2
    S = _full(L, k, ln_g_levels, np.inf)
    w = np.exp(-(S[lev] - np.min(ln_g_levels)))
    pi = w / w.sum()
    return en, S, pi


def _chi2(en, finals, pi, alpha):
    st = (1 - finals) // 2                            # spins +1 -> colour 0, -1 -> colour 1
    idx = st @ (2 ** np.arange(st.shape[1], dtype=np.int64))
    cls = en.cls[idx]
    return ST.chi2_gof(np.bincount(cls, minlength=en.n_cls), np.bincount(en.cls, pi, en.n_cls), alpha)


def stationarity_chi2(L, ln_g, k, samples, seed, alpha):
    en, S, pi = _exact_law(L, ln_g, k)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pi.size, size=samples, p=pi)
    spins = 1 - 2 * en.states[idx].astype(np.int64)
    wk = Walkers(L, [(int(seed), MUCA_STREAM)], samples, init=spins)
    wk.run(L * L, np.tile(S, samples))
    wk.check()
    return _chi2(en, wk.spins(), pi, alpha)


def ergodicity_chi2(L, ln_g, k, chains, sweeps, seed, alpha):
    en, S, pi = _exact_law(L, ln_g, k)
    wk = Walkers(L, [(int(seed), MUCA_STREAM)], chains, init=np.ones((chains, L * L), np.int64))
    wk.run(sweeps * L * L, np.tile(S, chains))
    wk.check()
    return _chi2(en, wk.spins(), pi, alpha)


__all__ = ["H_MIN", "MASS_MAX", "MucaRun", "Reweighted", "ergodicity_chi2", "muca_run", "reweight",
           "stationarity_chi2"]
