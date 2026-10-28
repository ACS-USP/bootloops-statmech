"""Finite-N checks of the dense associative memory that share no formula with rem_noise.py / thresholds.py
(statistical or exact-by-enumeration; named seeds, numpy Generators).

  rem_phi_mc(ens, alpha, lam, N, seeds)   the quenched REM noise phi at FINITE N from the EXACT density of
                                          the overlaps (Gaussian: N(0, rho^2/N); spherical: the Beta-type law
                                          of a cosine on S^{N-1}; binary: Binomial), by sampling the number
                                          of patterns in each overlap bin (Poisson, deterministic above 1e12)
                                          -- P = e^{alpha N} levels without drawing them one by one
  sign_noise_mc(alpha, lam, N, seeds)     the same for a sum of RANDOM-SIGN terms (binary patterns: the
                                          single-flip decision of Demircigil et al.): (1/lam N) log |sum|
  gd_retrieval(ens, alpha, lam, N, seed)  the paper's gradient descent x <- sum_mu softmax(lam x.xi)_mu xi^mu
                                          (Eq. (2), rate 1) from x = xi^1: Delta = |x_inf - xi^1|^2/N
  ising_flip_trial(N, P, lam, seed)       P random +-1 patterns, zero-temperature single-flip dynamics of
                                          Demircigil et al. (their Eq. (1) with F = exp(lam .)): which
                                          patterns are fixed points, and the number of pattern pairs within
                                          Hamming distance 1
  enumerate_overlap_law(N)                Gray-code enumeration (numerics/enumeration.py) of all 2^N Ising
                                          configurations against one +-1 pattern: the exact overlap
                                          histogram and sum_x exp(lam x.xi)

The sampler is exact for the model, not a simulation of the formulas: its only inputs are N, alpha, lam and
the exact finite-N law of one overlap.  It converges to phi at rate O(log N / N) (the Derrida logarithmic
correction in the condensed phase), so tests compare at two sizes.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from ..core import Refusal
from . import ensembles as EN
from .ensembles import check_ensemble

_DET = 1e12          # bins with a mean count above this are not sampled (relative fluctuation 1e-6)
_BIN = 0.05          # bin width in units of 1/(lam N): exp(lam N eps) varies by at most 5% per bin


def _log_density(ens, N, eps, rho):
    """ln of the exact density of eps = x.xi/N (per unit eps) at finite N; binary: ln pmf at integer k."""
    if ens == "gaussian":
        return -0.5 * np.log(2 * np.pi * rho**2 / N) - eps**2 * N / (2 * rho**2)
    u = np.clip(eps / rho, -1 + 1e-15, 1 - 1e-15)          # spherical: cosine of the angle
    lc = gammaln(N / 2) - 0.5 * math.log(math.pi) - gammaln((N - 1) / 2)
    return lc + (N - 3) / 2 * np.log1p(-u * u) - math.log(rho)


def _grid(ens, alpha, lam, N, rho):
    """(eps, ln p_j) of the bins that can hold a pattern: the whole support below ~1.2 eps* + 6/sqrt(N)."""
    from .rem_noise import eps_star
    if ens == "binary":
        k = np.arange(N + 1)
        lp = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) - N * math.log(2)
        return (N - 2 * k) / N, lp
    es = float(eps_star(ens, alpha, rho))
    hi = es * 1.2 + 6 * rho / math.sqrt(N)
    if ens == "spherical":                                     # the only ensemble with a finite edge, |eps| < rho
        hi = min(hi, rho * (1 - 1e-9))
    d = _BIN / (lam * N)
    n = int(2 * hi / d) + 2
    if n > 3_000_000:
        raise Refusal("precondition:size", f"N = {N}, lam = {lam}: {n} bins; lower N * lam")
    eps = np.linspace(-hi, hi, n)
    return eps, _log_density(ens, N, eps, rho) + math.log(eps[1] - eps[0])


def _log_counts(rng, ln_mu):
    """ln of the number of patterns in each bin: Poisson-sampled where the mean is below _DET, and the mean itself
    (kept in log space, so that e^{alpha N} beyond float range is fine) above it; -inf for an empty bin."""
    ln_n = np.full(ln_mu.shape, -np.inf)
    big = ln_mu > math.log(_DET)
    ln_n[big] = ln_mu[big]
    small = ~big
    n = rng.poisson(np.exp(ln_mu[small]))
    with np.errstate(divide="ignore"):
        ln_n[small] = np.log(n)
    return ln_n


def rem_phi_mc(ens, alpha, lam, N, seeds, rho=1):
    """List of one quenched phi_hat per seed: (1/(lam N)) log sum_mu exp(lam N eps_mu) over P = e^{alpha N}
    patterns whose overlaps are drawn bin by bin from the exact law (module docstring)."""
    r, a, l = float(check_ensemble(ens, rho)), float(alpha), float(lam)
    eps, lp = _grid(ens, a, l, N, r)
    out = []
    for s in seeds:
        ln_n = _log_counts(np.random.default_rng(s), a * N + lp)
        keep = np.isfinite(ln_n)
        out.append(float(logsumexp(ln_n[keep] + l * N * eps[keep]) / (l * N)))
    return out


def sign_noise_mc(alpha, lam, N, seeds):
    """(1/(lam N)) log |sum_mu s_mu exp(lam N eps_mu)| for i.i.d. fair signs s_mu and binary overlaps: a bin
    with n patterns contributes w (2 Bin(n, 1/2) - n), sampled exactly for n < 1e6 and as w sqrt(n) N(0,1)
    above.  Converges to phi_binary(alpha, 2 lam) (the sum's size is the square root of the sum of squares)."""
    eps, lp = _grid("binary", alpha, lam, N, 1)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        ln_n = _log_counts(rng, alpha * N + lp)
        occupied = np.isfinite(ln_n)
        mid = occupied & (ln_n < math.log(1e6))
        big = ln_n >= math.log(1e6)
        ln_abs, sgn = np.full(ln_n.shape, -np.inf), np.zeros(ln_n.shape)
        n_mid = np.rint(np.exp(ln_n[mid])).astype(np.int64)
        net = 2 * rng.binomial(n_mid, 0.5) - n_mid
        with np.errstate(divide="ignore"):
            ln_abs[mid], sgn[mid] = np.log(np.abs(net)), np.sign(net)
        z = rng.standard_normal(int(big.sum()))
        ln_abs[big], sgn[big] = 0.5 * ln_n[big] + np.log(np.abs(z)), np.sign(z)
        keep = np.isfinite(ln_abs)
        t = lam * N * eps[keep] + ln_abs[keep]
        m = t.max()
        tot = float(np.sum(sgn[keep] * np.exp(t - m)))
        out.append(float((m + math.log(abs(tot))) / (lam * N)))
    return out


def gd_retrieval(ens, alpha, lam, N, seed, rho=1, max_steps=400, tol=1e-10):
    """Delta = |x_inf - xi^1|^2 / N for the gradient descent of the paper's Eq. (2) started at x = xi^1
    (P = round(e^{alpha N}) patterns; gaussian or spherical, |xi|^2 = N for the latter)."""
    if ens == "binary":
        raise Refusal("unsupported:ensemble-dynamics",
                      "the gradient descent of Eq. (2) is for continuous neurons; use ising_flip_trial")
    check_ensemble(ens, rho)
    P = int(round(math.exp(alpha * N)))
    if P * N > 8e7:
        raise Refusal("precondition:size", f"P N = {P * N:.3g} entries; lower alpha N")
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((P, N))
    if ens == "spherical":
        xi *= math.sqrt(N) / np.linalg.norm(xi, axis=1, keepdims=True)
    x = xi[0].copy()
    for _ in range(max_steps):
        z = lam * (xi @ x)
        a = np.exp(z - z.max())
        a /= a.sum()
        xn = a @ xi
        done = np.linalg.norm(xn - x) <= tol * math.sqrt(N)
        x = xn
        if done:
            break
    return float(np.sum((x - xi[0]) ** 2) / N)


@dataclass(frozen=True)
class FlipTrial:
    stable: np.ndarray            # per pattern: fixed point of the single-flip dynamics at every site
    min_distance: int             # smallest Hamming distance between two distinct pattern INDICES
    n_close_pairs: int            # pairs of patterns at Hamming distance <= 1


def ising_flip_trial(N, P, lam, seed):
    """Zero-temperature single-flip dynamics of Demircigil-Heusel-Lowe-Upgang-Vermet (arXiv:1702.01929,
    Eq. (1)) with F(x) = exp(lam x): T_i(sigma) = sgn[sum_mu F(xi_i^mu + m_mu^(i)) - F(-xi_i^mu + m_mu^(i))],
    m_mu^(i) = sum_{j != i} xi_j^mu sigma_j.  Pattern sigma = xi^nu is a fixed point iff
    sum_mu c_{mu i} exp(lam (o_mu - c_{mu i})) has the sign of +1 at every site i, with c = xi^mu xi^nu
    and o_mu = xi^mu . xi^nu; an exact tie counts as NOT stable."""
    rng = np.random.default_rng(seed)
    xi = rng.choice(np.array([-1.0, 1.0]), size=(P, N))
    overlap = xi @ xi.T                                                  # P x P, = N - 2 Hamming
    ham = ((N - overlap) / 2).astype(int)
    np.fill_diagonal(ham, N + 1)
    stable = np.empty(P, bool)
    for nu in range(P):
        c = xi * xi[nu]                                                  # P x N: c_{mu i}
        o = c.sum(axis=1)
        S = (c * np.exp(lam * (o[:, None] - N) - lam * c)).sum(axis=0)   # scaled by exp(-lam N)
        stable[nu] = bool(np.all(S > 1e-9 * math.exp(-lam)))             # > 0 at every site; ties are zero
    return FlipTrial(stable, int(ham.min()), int((ham[np.triu_indices(P, 1)] <= 1).sum()))


def enumerate_overlap_law(N):
    """Gray-code enumeration (numerics/enumeration.py gray_states) of all 2^N Ising configurations
    against the pattern xi = (+1, ..., +1): ({overlap x.xi: number of configurations}, the sum of
    exp(lam x.xi) as a function of lam).  Exact integers; N <= 24."""
    from ..numerics.enumeration import gray_states
    if not (isinstance(N, int) and 1 <= N <= 24):
        raise Refusal("precondition:size", f"N = {N!r}: need an int in [1, 24]")
    ov = gray_states(N).sum(axis=1).astype(int)
    vals, cnt = np.unique(ov, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, cnt)}
    return hist, (lambda lam: sum(c * math.exp(lam * v) for v, c in hist.items()))


__all__ = ["FlipTrial", "enumerate_overlap_law", "gd_retrieval", "ising_flip_trial", "rem_phi_mc",
           "sign_noise_mc"]
