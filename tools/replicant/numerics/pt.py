"""Parallel tempering Monte Carlo for SK-type couplings (E11), numpy-vectorised.

  geometric_betas(lo, hi, R)                 R inverse temperatures in geometric progression
  pt_sk(J, betas, sweeps, seed) -> PTResult  one PT run per disorder sample (J: N x N or
                                             S x N x N, all samples advanced together)
  equilibration(result, beta_index, start=None) -> Equilibration   window-drift diagnostic

Model: H = -sum_{i<j} J_ij s_i s_j (h = 0), s_i = +-1, J symmetric with zero diagonal (any
law: Gaussian N(0, 1/N) or +-1/sqrt N).  Energies are per spin.

Algorithm.  Every sample carries one replica per beta.  A sweep visits the spins i = 0..N-1 in
order; spin i of every replica is proposed for a flip with dE = 2 s_i h_i (h = J s, the local
fields, updated by a rank-one step after each accepted flip) and accepted with the Metropolis
probability min(1, exp(-beta dE)).  After each sweep neighbouring betas exchange their
configurations with probability min(1, exp((beta_k - beta_{k+1}) (E_k - E_{k+1}))), the even
pairs after even sweeps and the odd pairs after odd sweeps.  Both moves satisfy detailed
balance for prod_k exp(-beta_k H(s_k)) (Hukushima & Nemoto, J. Phys. Soc. Jpn. 65, 1604
(1996), arXiv:cond-mat/9512035; not fetched here, the rule is standard and is checked
numerically against exact enumeration in tests/test_pt.py).  Sequential (not random) site
order keeps the stationary measure (each single-site update leaves it invariant).

Random numbers: numpy.random.default_rng(seed); the initial spins are iid +-1 (an infinite-
temperature start).  The energy is tracked by the accepted dE and recomputed from the final
spins (final_energy_per_spin) as a bookkeeping check.

Statistics: one run per disorder sample; time averages within a run are correlated, so error
bars are taken over independent samples (or independent seeds), never over sweeps.  Results
are statistical, never certified.  `equilibration` compares the sample-mean energy in the
windows [T/4, T/2) and [T/2, T) (a two-window split) by a 99% Student-t band over samples;
a drift outside the band is reported as not equilibrated, which callers turn into an
"undetermined" label rather than a number.  Replica round trips (a replica travelling from
the lowest to the highest beta and back) are counted per sample as a second diagnostic.
"""
import math
from dataclasses import dataclass

import numpy as np

from ..amp._stats import student_band
from ..core import Refusal

# planted-error hooks (tests): acceptance_scale 0.8 = wrong Metropolis rule; exchange_sign -1 =
# the exchange criterion with its sign reversed (favours the hotter state at the larger beta)
_HOOKS = {"acceptance_scale": 1.0, "exchange_sign": 1.0}


@dataclass
class PTResult:
    betas: np.ndarray
    energy_per_spin_trace: np.ndarray    # sweeps x samples x betas (by temperature slot)
    min_energy_per_spin: np.ndarray      # samples: lowest energy seen at any beta
    min_sweep: np.ndarray                # samples: first sweep at which it was seen
    swap_acceptance: np.ndarray          # betas - 1: exchange acceptance rate per pair
    round_trips: np.ndarray              # samples: completed lowest -> highest -> lowest trips
    final_energy_per_spin: np.ndarray    # samples x betas, recomputed from the final spins
    N: int
    sweeps: int


@dataclass
class Equilibration:
    equilibrated: bool
    drift: float          # mean over samples of (late window - early window), per spin
    half_width: float     # 99% Student-t half width of that mean
    text: str


def geometric_betas(lo, hi, R):
    return tuple(float(x) for x in np.geomspace(lo, hi, R))


def _check(J, betas, sweeps, seed):
    J = np.asarray(J, dtype=float)
    if J.ndim == 2:
        J = J[None]
    if J.ndim != 3 or J.shape[1] != J.shape[2] or J.shape[1] < 2 or not np.isfinite(J).all() \
            or not np.array_equal(J, J.transpose(0, 2, 1)) or \
            np.any(np.diagonal(J, axis1=1, axis2=2) != 0):
        raise Refusal("precondition:coupling-matrix",
                      "J must be a finite symmetric N x N (or S x N x N) array, zero diagonal")
    b = np.asarray(betas, dtype=float)
    if b.ndim != 1 or len(b) < 1 or not np.isfinite(b).all() or (b <= 0).any() or \
            (np.diff(b) <= 0).any():
        raise Refusal("precondition:betas", f"betas {betas!r}: need positive, strictly increasing")
    if not (isinstance(sweeps, int) and not isinstance(sweeps, bool) and sweeps >= 1):
        raise Refusal("precondition:sweeps", f"sweeps = {sweeps!r}: need an int >= 1")
    ok = isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0
    ok = ok or (isinstance(seed, (list, tuple)) and len(seed) > 0 and
                all(isinstance(x, (int, np.integer)) and x >= 0 for x in seed))
    if not ok:
        raise Refusal("precondition:seed", f"seed = {seed!r}: need an int >= 0 or a list of them")
    return J, b


def pt_sk(J, betas, sweeps, seed):
    J, b = _check(J, betas, sweeps, seed)
    S, N, R = J.shape[0], J.shape[1], len(b)
    rng = np.random.default_rng(seed)
    Jt = np.ascontiguousarray(J.transpose(1, 0, 2))           # N x S x N: row i of each sample
    s = rng.choice((-1.0, 1.0), size=(S, R, N))
    h = np.einsum("srn,snm->srm", s, J)
    E = -0.5 * np.einsum("srn,srn->sr", s, h)
    beff = _HOOKS["acceptance_scale"] * b[None, :]
    trace = np.empty((sweeps, S, R))
    best, best_sweep = E.min(axis=1).copy(), np.zeros(S, dtype=int)
    tries, accepts = np.zeros(max(R - 1, 0)), np.zeros(max(R - 1, 0))
    rep = np.tile(np.arange(R), (S, 1))                        # replica label in each slot
    heading = np.zeros((S, R), dtype=int)                       # per replica: 0 none, 1 up, -1 down
    trips = np.zeros(S, dtype=int)
    rows = np.arange(S)[:, None]
    for sw in range(sweeps):
        u = rng.random((N, S, R))
        for i in range(N):
            si = s[:, :, i]
            dE = 2.0 * si * h[:, :, i]
            acc = u[i] < np.exp(np.minimum(0.0, -beff * dE))
            d = np.where(acc, -2.0 * si, 0.0)
            s[:, :, i] = si + d
            h += d[:, :, None] * Jt[i][:, None, :]
            E += np.where(acc, dE, 0.0)
        if R > 1:
            k = np.arange(sw % 2, R - 1, 2)
            if len(k):
                x = _HOOKS["exchange_sign"] * (b[k] - b[k + 1])[None, :] * (E[:, k] - E[:, k + 1])
                ok = rng.random((S, len(k))) < np.exp(np.minimum(0.0, x))
                tries[k] += S
                accepts[k] += ok.sum(axis=0)
                lo, hi = k[None, :].repeat(S, 0), (k + 1)[None, :].repeat(S, 0)
                swp = np.where(ok)
                a, c = lo[swp], hi[swp]
                r0 = swp[0]
                s[r0, a], s[r0, c] = s[r0, c].copy(), s[r0, a].copy()
                h[r0, a], h[r0, c] = h[r0, c].copy(), h[r0, a].copy()
                E[r0, a], E[r0, c] = E[r0, c].copy(), E[r0, a].copy()
                rep[r0, a], rep[r0, c] = rep[r0, c].copy(), rep[r0, a].copy()
            # round trips: a replica at the lowest beta heads up; reaching the highest beta
            # it heads down; arriving back at the lowest after heading down completes a trip
            low, top = rep[:, 0], rep[:, R - 1]
            done = heading[rows[:, 0], low] == -1
            trips += done
            heading[rows[:, 0], low] = 1
            heading[rows[:, 0], top] = np.where(heading[rows[:, 0], top] == 1, -1,
                                                heading[rows[:, 0], top])
        trace[sw] = E / N
        m = E.min(axis=1)
        better = m < best - 1e-12 * np.abs(best)
        best_sweep[better] = sw
        best = np.minimum(best, m)
    final = -0.5 * np.einsum("srn,srn->sr", s, np.einsum("srn,snm->srm", s, J)) / N
    rate = np.where(tries > 0, accepts / np.maximum(tries, 1), np.nan)
    return PTResult(b, trace, best / N, best_sweep, rate, trips, final, N, sweeps)


def equilibration(result, beta_index, start=None):
    """Sample-mean energy at betas[beta_index] in [start, T/2) against [T/2, T) (default
    start = T/4); not equilibrated if the 99% Student-t band of the per-sample
    difference excludes 0."""
    T = result.sweeps
    start = T // 4 if start is None else start
    mid = T // 2
    tr = result.energy_per_spin_trace[:, :, beta_index]
    if mid <= start or T <= mid or tr.shape[1] < 2:
        raise Refusal("precondition:windows", f"cannot split sweeps [{start}, {T}) in two")
    d = tr[mid:].mean(axis=0) - tr[start:mid].mean(axis=0)
    band = student_band(d)
    ok = abs(band.mean) <= band.half_width
    text = (f"beta = {result.betas[beta_index]:.4g}: late - early window energy {band.mean:+.2e}"
            f" +- {band.half_width:.2e} (99% t, {len(d)} samples, windows [{start}, {mid}), "
            f"[{mid}, {T})); round trips per sample: median {int(np.median(result.round_trips))}")
    return Equilibration(bool(ok), float(band.mean), float(band.half_width), text)


def sk_ground_energy_finite_size(*a, **k):
    """numerics/finite_size.py (kept there to keep this file small); re-exported here."""
    from .finite_size import sk_ground_energy_finite_size as f
    return f(*a, **k)
