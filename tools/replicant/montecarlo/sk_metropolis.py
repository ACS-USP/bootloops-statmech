"""Single-spin-flip Metropolis for SK-type couplings at one beta, on a population of replicas that share J (M3), in
the pattern of numerics/pt.py (E11): a sweep visits the spins i = 0 .. N-1 in order, proposes the flip of spin i in
every replica with dE = 2 s_i h_i (h = s J, the local fields, updated by a rank-one step after each accepted flip)
and accepts with probability min(1, exp(-beta dE)).  Each single-site update leaves exp(-beta H) invariant, so the
sequential sweep does.  H = -sum_{i<j} J_ij s_i s_j (h = 0), J symmetric with zero diagonal (Gaussian N(0, 1/N) or
+-1/sqrt N, numerics/finite_size.py's couplings).

  sk_sweeps(J, s, beta, sweeps, rngs, sizes) -> s   s (R, N) float +-1, the rows grouped in consecutive blocks of
      `sizes` (one block per named seed); block b draws its uniforms only from rngs[b] (an (N, sizes[b]) array per
      sweep), so a block's trajectory does not depend on the other blocks
  sk_energy(J, s) -> (R,) energies;  exact_log_z(J, beta), exact_energy(J, beta): ln Z and <H> / N by enumeration
      of the 2^N states (numerics/enumeration.gray_states, N <= 20)
  stationarity_chi2(J, beta, samples, seed, alpha) -> stats.Chi2   `samples` exact Boltzmann draws, one sweep each,
      the resulting states against the exact probabilities (pi P = pi)
Exact invariant (no-solution:sampler-invariant): the energy tracked through the accepted dE equals the energy
recomputed from the final spins (1e-9 relative).
_HOOKS["acceptance_scale"] (tests): 1.0; planted 0.95 / 1.05 (a wrong acceptance rule).
"""
import math

import numpy as np

from ..core import Refusal

_HOOKS = {"acceptance_scale": 1.0}
MAX_N_EXACT = 20


def sk_energy(J, s):
    return -0.5 * np.einsum("rn,rn->r", s, s @ J)


def sk_sweeps(J, s, beta, sweeps, rngs, sizes):
    N = J.shape[0]
    if not sweeps or not len(s):
        return s
    b = _HOOKS["acceptance_scale"] * float(beta)
    h = s @ J
    E0 = -0.5 * np.einsum("rn,rn->r", s, h)
    dE_sum = np.zeros(len(s))
    for _ in range(sweeps):
        u = np.concatenate([g.random((N, n)) for g, n in zip(rngs, sizes)], axis=1)
        with np.errstate(divide="ignore"):
            thr = -np.log(u)                     # u < exp(-b dE)  <=>  b dE < -ln u  (dE <= 0 always accepted)
        for i in range(N):
            dE = 2.0 * s[:, i] * h[:, i]
            k = np.flatnonzero(b * dE < thr[i])          # the accepted flips: update only their rows
            d = -2.0 * s[k, i]
            s[k, i] += d
            h[k] += d[:, None] * J[i][None, :]
            dE_sum[k] += dE[k]
    E1 = sk_energy(J, s)
    if not np.allclose(E0 + dE_sum, E1, rtol=1e-9, atol=1e-9 * N):
        raise Refusal("no-solution:sampler-invariant", "SK Metropolis: tracked energy != recomputed energy")
    return s


def _states(N):
    if N > MAX_N_EXACT:
        raise Refusal("precondition:enumeration-size", f"N = {N}: exact SK enumeration here is for N <= {MAX_N_EXACT}")
    from ..numerics.enumeration import gray_states
    return gray_states(N)


def _log_weights(J, beta):
    S = _states(J.shape[0])
    E = sk_energy(J, S)
    return S, E, -float(beta) * E


def exact_log_z(J, beta):
    _, _, lw = _log_weights(J, beta)
    top = lw.max()
    return float(top + math.log(np.exp(lw - top).sum()))


def exact_energy(J, beta):
    _, E, lw = _log_weights(J, beta)
    w = np.exp(lw - lw.max())
    return float((w @ E) / w.sum() / J.shape[0])


def stationarity_chi2(J, beta, samples, seed, alpha):
    from .stats import chi2_gof
    S, _, lw = _log_weights(J, beta)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), size=samples, p=p)
    fin = sk_sweeps(J, S[idx].copy(), beta, 1, [rng], [samples])
    N = J.shape[0]
    code = ((1 - fin) // 2).astype(np.int64) @ (2 ** np.arange(N, dtype=np.int64))
    order = ((1 - S) // 2).astype(np.int64) @ (2 ** np.arange(N, dtype=np.int64))
    probs = np.zeros(2 ** N)
    probs[order] = p
    return chi2_gof(np.bincount(code, minlength=2 ** N), probs, alpha)


__all__ = ["exact_energy", "exact_log_z", "sk_energy", "sk_sweeps", "stationarity_chi2"]
