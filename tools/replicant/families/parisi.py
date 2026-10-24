"""Full RSB for SK (Parisi): K-step RSB staircases, extrapolated in K.  Slow tier only.

  krsb_ground_energy(K)              -> KLevel   T = 0, K RSB steps (raw, audited)
  parisi_ground_energy(*, grid, dps) -> FRSBResult(value, error_estimate, "numerical")
  sk_frsb(beta, *, grid=6)           -> FRSBSolution  finite T, h = 0

Conventions (SK of families/sk.py: J_ij ~ N(0, 1/N), h = 0, beta = 1/T, energies per
spin).  Parisi order parameter: a staircase q(x) with K steps, 0 <= q_0 < ... < q_K and
Parisi parameters 0 = m_0 < m_1 < ... < m_K < m_{K+1} = 1.  The K-step functional
(Parisi, J. Phys. A 13, L115 (1980); Mezard, Parisi & Virasoro (1987) ch. III):
  -beta f = ln 2 + beta^2/4 [1 - 2 q_K + sum_{k=0}^{K} (m_{k+1} - m_k) q_k^2]
            + E_z phi_0(beta sqrt(q_0) z)
  phi_K(y) = ln cosh y,  phi_{k-1}(y) = (1/m_k) ln E_z exp(m_k phi_k(y + beta sigma_k z)),
  sigma_k^2 = q_k - q_{k-1};  K = 1 is sk.py's 1RSB functional (checked in the tests).
T = 0 (q_K -> 1, m_k -> 0 at fixed mu_k = beta m_k; the limit sk.py takes for 1RSB):
  E = (1/4) sum_{k=0}^{K-1} (mu_{k+1} - mu_k)(1 - q_k^2) - E_z Phi_0(sqrt(q_0) z),   mu_0 = 0,
  Phi_K(h) = |h|,  Phi_{k-1}(h) = (1/mu_k) ln E_z exp(mu_k Phi_k(h + sigma_k z)),  q_K = 1.
Variational direction (n -> 0 rule of sk.py, = Guerra's bound): f and E are MAXIMISED over
(q, m) / (q, mu); each extra step can only raise them, so E_1 < E_2 < ... < E_inf from
below (Guerra 2003 / Auffinger-Chen 2017: a K-step value bounds the true one).  The
infinite-K limit is the Parisi formula, proven for SK (Talagrand, Ann. Math. 163, 221
(2006); Panchenko, The SK model (2013)).
Gradient (adjoint: Phi backward, the tilted field law P_k forward; m_k = Phi_k'):
  dE/dq_k  = (mu_{k+1} - mu_k)/2 (E_{P_k} m_k^2 - q_k)       (Parisi: q_k = E_{P_k} m_k^2)
  dE/dmu_k = (q_k^2 - q_{k-1}^2)/4 - (E_{P_k} Phi_k - E_{P_{k-1}} Phi_{k-1})/mu_k
(finite T: the same with beta^2/2, beta^2/4 factors and the sign of -beta f).
Numerics.  Uniform h-grid on [-L, L] (L = 7; 8 in the check), extended linearly
(Phi ~ |h| + c) where a kernel reaches beyond; each level is a log-sum-exp of the sampled,
normalised Gaussian kernel (trapezoid, spectrally accurate for dh <= sigma/2); the T = 0
innermost level is closed form (Gaussian times Phi, as in sk.py).  Maximisation: Newton
with a central-difference Hessian of the analytic gradient, seeded from level K - 1.
Levels K <= 9 at T = 0 (K = 10 did not converge to the gradient tolerance in 40 Newton
steps from the level-9 seed, so it is refused), K <= 5 at finite T.
Audit per level: max |gradient| <= 1e-7, Hessian negative definite, and the energy
recomputed at dh/2, L = 8 (its change enters the error budget).
Extrapolation in K (our own convergence study; the banked value is only a target):
E_inf ~ E_K + C K^-s with (E_inf, C, s) fitted to levels K-2, K-1, K.  The fitted s grows
with K (2.55 at K = 4, 3.25 at K = 8, 3.33 at K = 9), so this tail over-estimates; the
stated error estimate is the WHOLE extrapolated tail E_inf - E_K plus grid and optimiser terms,
i.e. no digit beyond the raw variational bound E_K is claimed without that band.

Honest label: the n -> 0 continuation is physics-level, but for SK the Parisi value is a
theorem; the numbers carry discretisation (K, grid) error, stated as error_estimate.
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp

from ..core import Refusal

FRSB_CAVEAT = ("numerical: K-step Parisi staircase (n -> 0 replica functional, = Guerra's "
               "bound) extrapolated in K; the full-RSB value is a theorem for SK (Talagrand "
               "2006), the number carries discretisation error (error_estimate), not proof")
PARISI_CITE = ("Parisi, PRL 43, 1754 (1979); J. Phys. A 13, L115 (1980); Talagrand, Ann. "
               "Math. 163, 221 (2006); Panchenko, The Sherrington-Kirkpatrick Model (2013)")
_L, _DH_MAX, _GTOL, _K_MAX, _K_MAX_T = 7.0, 0.05, 1e-7, 9, 5


@dataclass(frozen=True)
class KLevel:
    K: int
    E: float
    q: tuple
    mu: tuple
    grad_max: float
    hess_max_eig: float      # < 0: local maximum
    grid_change: float       # |E(dh/2, L = 8) - E(dh, L = 7)|
    dh: float


@dataclass(frozen=True)
class FRSBResult:
    value: object
    error_estimate: object
    label: str = "numerical"
    levels: tuple = ()
    exponent: float = None
    note: str = ""
    caveat: str = FRSB_CAVEAT


@dataclass(frozen=True)
class FRSBSolution:
    """Finite-T K-step staircase: qs = (q_0..q_K), ms = (m_1..m_K); f extrapolated in K."""
    beta: float
    qs: tuple
    ms: tuple
    f: object
    e: object
    error_estimate: object
    f_levels: tuple = ()
    grad_max: float = None
    hess_max_eig: float = None
    caveat: str = FRSB_CAVEAT


# ---- one K-step functional on a grid ------------------------------------------------
def _inner(h, s, mu):
    """Phi_{K-1}, its h-derivative and d/dmu for Phi_K = |h| (closed form)."""
    la, lb = mu * h + log_ndtr(h / s + mu * s), -mu * h + log_ndtr(-h / s + mu * s)
    lS = np.logaddexp(la, lb)
    G = mu * mu * s * s / 2 + lS
    m = np.exp(la - lS) - np.exp(lb - lS)
    lc = -h * h / (2 * s * s) - mu * mu * s * s / 2 - 0.5 * math.log(2 * math.pi)
    dG = mu * s * s + h * m + 2 * s * np.exp(lc - lS)
    return G / mu, m, dG / mu - G / mu**2


def _conv(h, Phi, M, s, u, dh):
    """One level: Phi -> (1/u) ln E_z e^{u Phi(h + s z)} on h, with the transfer matrix."""
    n, p = len(h) // 2, int(math.ceil((u * s * s + 12 * s) / dh))
    ramp = dh * np.arange(1, p + 1)
    Pe = np.concatenate([Phi[0] + ramp[::-1], Phi, Phi[-1] + ramp])   # Phi ~ |h| + c
    Me = np.concatenate([-np.ones(p), M, np.ones(p)])
    he = dh * np.arange(-n - p, n + p + 1)
    lk = -(he[None, :] - h[:, None])**2 / (2 * s * s)
    A = lk - logsumexp(lk, axis=1, keepdims=True) + u * Pe[None, :]
    lse = logsumexp(A, axis=1)
    W = np.exp(A - lse[:, None])
    return lse / u, W @ Me, W, p


def _levels(q, mu, *, T0=True, beta=None, dh, L=_L, grad=True):
    """T = 0: (E, dE/dq, dE/dmu).  Finite T (T0=False, h-grid in y = beta h, q has K+1
    entries, mu are the m's): (F = -beta f, dF/dq, dF/dm)."""
    K = len(mu)
    qq = np.concatenate([q, [1.0]]) if T0 else np.asarray(q)
    b = 1.0 if T0 else beta
    sig = b * np.sqrt(np.diff(qq))[:K]
    n = int(round(b * L / dh))
    h = dh * np.arange(-n, n + 1)
    Phi, M, W, pad = [None] * (K + 1), [None] * (K + 1), [None] * (K + 1), [0] * (K + 1)
    if T0:
        Phi[K - 1], M[K - 1], dmuK = _inner(h, sig[K - 1], mu[K - 1])
        top = K - 1
    else:
        a = np.abs(h)
        Phi[K], M[K], top = a + np.log1p(np.exp(-2 * a)) - math.log(2), np.tanh(h), K
    for k in range(top, 0, -1):
        Phi[k - 1], M[k - 1], W[k], pad[k] = _conv(h, Phi[k], M[k], sig[k - 1], mu[k - 1], dh)
    s0 = b * math.sqrt(q[0])
    g = -h * h / (2 * s0 * s0)
    g = np.exp(g - logsumexp(g))
    root = g @ Phi[0]
    mus, qk = np.asarray(mu, float), np.asarray(q, float)
    dm = np.diff(np.concatenate([[0.0], mus] + ([] if T0 else [[1.0]])))
    if T0:
        val = 0.25 * np.sum(dm * (1 - qk**2)) - root
    else:
        quad = 1 - 2 * qk[K] + np.sum(dm * qk**2)
        val = math.log(2) + b * b * quad / 4 + root
    if not grad:
        return val
    P, EP, Em2 = g, [], []
    for k in range(top + 1):
        EP.append(P @ Phi[k])
        Em2.append(P @ M[k]**2)
        if k < top:
            P = (P @ W[k + 1])[pad[k + 1]:pad[k + 1] + 2 * n + 1]
    Em2 = np.array(Em2)
    if T0:
        gq = 0.5 * dm * (Em2 - qk)
        gmu = [(qk[k]**2 - qk[k - 1]**2) / 4 - (EP[k] - EP[k - 1]) / mu[k - 1]
               for k in range(1, K)] + [(1 - qk[K - 1]**2) / 4 - P @ dmuK]
        return val, gq, np.array(gmu)
    s2 = b * b * dm * (qk - Em2) / 2
    gm = np.array([b * b * (qk[k - 1]**2 - qk[k]**2) / 4 + (EP[k] - EP[k - 1]) / mu[k - 1]
                   for k in range(1, K + 1)])
    return val, s2, gm


# ---- maximisation ---------------------------------------------------------------------
def _ordered(v, K, T0):
    q, mu = v[:len(v) - K], v[len(v) - K:]
    return (q[0] > 0 and np.all(np.diff(q) > 0) and q[-1] < 1 and mu[0] > 0
            and np.all(np.diff(mu) > 0) and (T0 or mu[-1] < 1))


def _grad_fn(K, T0, beta, dh, L=_L):
    nq = K if T0 else K + 1
    sign = 1.0 if T0 else -1.0 / beta     # maximise E, or f = -F/beta

    def G(v):
        out = _levels(v[:nq], v[nq:], T0=T0, beta=beta, dh=dh, L=L)
        return sign * out[0], sign * np.concatenate([out[1], out[2]])
    return G


def _hessian(G, v):
    H = np.empty((len(v), len(v)))
    for i in range(len(v)):
        d = 1e-5 * max(abs(v[i]), 0.05)
        e = np.zeros(len(v))
        e[i] = d
        H[:, i] = (G(v + e)[1] - G(v - e)[1]) / (2 * d)
    return (H + H.T) / 2


def _maximise(G, v, K, T0, maxit=40):
    """Newton on the analytic gradient (Hessian recomputed every step, shifted if not
    negative definite); a step is halved until it stays ordered and does not lower the
    objective (beyond 1e-14)."""
    for _ in range(maxit):
        val, g = G(v)
        if np.abs(g).max() < 1e-10:
            break
        H = _hessian(G, v)
        top = np.linalg.eigvalsh(H).max()
        if top >= 0:
            H = H - (top + 1e-6) * np.eye(len(v))
        step, lam = np.linalg.solve(H, -g), 1.0
        while not _ordered(v + lam * step, K, T0) or G(v + lam * step)[0] < val - 1e-14:
            lam /= 2
            if lam < 1e-6:
                return v
        v = v + lam * step
    return v


def _seed(prev, K, nq_prev):
    """Level-K seed: the level-(K-1) staircase resampled at fractions (k + 1/2)/K
    (logit q and log mu, linear in the fraction, linearly extrapolated at the ends)."""
    q, mu = prev[:nq_prev], prev[nq_prev:]
    def resample(y, n_new, spread):
        t0 = (np.arange(len(y)) + 0.5) / len(y)
        t = (np.arange(n_new) + 0.5) / n_new
        if len(y) == 1:            # one level: spread it (logit q +- 1, log mu +- 0.6)
            return y[0] + spread * np.linspace(-1, 1, n_new)
        out = np.interp(t, t0, y)
        lo, hi = t < t0[0], t > t0[-1]
        out[lo] += (t[lo] - t0[0]) * (y[1] - y[0]) / (t0[1] - t0[0])
        out[hi] += (t[hi] - t0[-1]) * (y[-1] - y[-2]) / (t0[-1] - t0[-2])
        return out
    lq = resample(np.log(q / (1 - q)), len(q) + 1, 1.0)
    lm = resample(np.log(mu / (1 - mu)) if mu[-1] < 1 and nq_prev > len(mu) else np.log(mu),
                  K, 0.6)
    qn = 1 / (1 + np.exp(-lq))
    mn = 1 / (1 + np.exp(-lm)) if nq_prev > len(mu) else np.exp(lm)
    return np.concatenate([qn, mn])


def _dh_for(v, nq, beta):
    qq = np.concatenate([v[:nq], [1.0]]) if beta is None else v[:nq]
    s = np.sqrt(np.concatenate([[qq[0]], np.diff(qq)]))
    return min(_DH_MAX, (1 if beta is None else beta) * s.min() / 2)


def _audit(G, v, refusal_detail):
    val, g = G(v)
    eig = float(np.linalg.eigvalsh(_hessian(G, v)).max())
    if np.abs(g).max() > _GTOL or eig >= 0:
        raise Refusal("no-solution", f"{refusal_detail}: max |grad| = {np.abs(g).max():.2e} "
                      f"(tol {_GTOL}), largest Hessian eigenvalue {eig:.2e} (need < 0)")
    return val, float(np.abs(g).max()), eig


@lru_cache(maxsize=None)
def krsb_ground_energy(K: int) -> KLevel:
    """T = 0 energy with K RSB steps (maximised over 2K parameters), audited."""
    if not (isinstance(K, int) and 1 <= K <= _K_MAX):
        raise Refusal("precondition:grid-range", f"K = {K!r}: levels 1..{_K_MAX} are validated")
    v = (np.array([0.48, 1.36]) if K == 1
         else _seed(np.array(krsb_ground_energy(K - 1).q + krsb_ground_energy(K - 1).mu),
                    K, K - 1))
    dh = _dh_for(v, K, None)
    G = _grad_fn(K, True, None, dh)
    v = _maximise(G, v, K, True)
    E, gmax, eig = _audit(G, v, f"T = 0 {K}-RSB point")
    fine = _levels(v[:K], v[K:], dh=dh / 2, L=_L + 1, grad=False)
    return KLevel(K, float(E), tuple(map(float, v[:K])), tuple(map(float, v[K:])), gmax, eig,
                  float(abs(fine - E)), dh)


def _extrapolate(vals, Ks, what):
    """(E_inf, s) from the last three levels, E_K = E_inf - C K^-s."""
    (a, b, c), (K1, K2, K3) = vals[-3:], Ks[-3:]
    if not (a < b < c):
        raise Refusal("no-solution", f"{what}: levels not increasing ({a}, {b}, {c}); the "
                      "variational sequence must rise with K")
    r = (c - b) / (b - a)
    f = lambda s: (K2**-s - K3**-s) / (K1**-s - K2**-s) - r
    if f(0.05) * f(30) > 0:
        raise Refusal("no-solution", f"{what}: no power law K^-s fits levels {Ks[-3:]}")
    s = brentq(f, 0.05, 30)
    if s <= 1:
        raise Refusal("no-solution", f"{what}: fitted exponent s = {s:.3g} <= 1 (no "
                      "convergent tail)")
    C = (c - b) / (K2**-s - K3**-s)
    return c + C * K3**-s, s


def parisi_ground_energy(*, grid: int, dps: int = 20) -> FRSBResult:
    """Full-RSB T = 0 energy: levels K = grid-2, grid-1, grid extrapolated in K.

    error_estimate = (E_inf - E_grid) + max grid change + gradient-based optimiser bound
    (a discretisation error, not a statistical one).  Values are float64 computations
    returned as mpf at `dps` digits of storage (dps sets no extra accuracy)."""
    if not isinstance(grid, int) or grid < 3:
        raise Refusal("precondition:grid-too-small", f"grid = {grid!r}: the K-extrapolation "
                      "fits three levels (E_inf, C, s), so grid >= 3")
    levels = tuple(krsb_ground_energy(K) for K in range(1, grid + 1))
    Es = [lv.E for lv in levels]
    Einf, s = _extrapolate(Es, list(range(1, grid + 1)), "T = 0 K-RSB")
    grid_err = max(lv.grid_change for lv in levels[-3:])
    err = (Einf - Es[-1]) + grid_err + max(lv.grad_max for lv in levels[-3:])
    with mp.workdps(dps):
        return FRSBResult(mp.mpf(Einf), mp.mpf(err), "numerical", levels, s,
                          f"E_{grid} = {Es[-1]:.10f} (variational lower bound); tail fit s = "
                          f"{s:.3f} over K = {grid - 2}..{grid}; grid change {grid_err:.1e}")


# ---- finite temperature --------------------------------------------------------------
@lru_cache(maxsize=None)
def _frsb_level(beta: float, K: int):
    if K == 1:
        from .sk import sk_1rsb
        s = sk_1rsb(beta)
        v = np.array([max(float(s.q0), 0.05), float(s.q1), float(s.m)])
    else:
        prev = _frsb_level(beta, K - 1)[0]
        v = _seed(prev, K, K)
    dh = _dh_for(v, K + 1, beta)
    G = _grad_fn(K, False, beta, dh)
    v = _maximise(G, v, K, False)
    f, gmax, eig = _audit(G, v, f"beta = {beta} {K}-RSB point")
    fine = -_levels(v[:K + 1], v[K + 1:], T0=False, beta=beta, dh=dh / 2, L=_L + 1,
                    grad=False) / beta
    d = 1e-4          # e = -dF/dbeta at fixed (q, m): stationary, so no implicit terms
    Fb = [_levels(v[:K + 1], v[K + 1:], T0=False, beta=beta + t, dh=dh, grad=False)
          for t in (d, -d)]
    return v, float(f), gmax, eig, float(abs(fine - f)), float(-(Fb[0] - Fb[1]) / (2 * d))


def sk_frsb(beta, *, grid: int = 5) -> FRSBSolution:
    """K-step Parisi staircases at h = 0, K = 1..grid, f extrapolated in K (as at T = 0;
    error_estimate = whole tail + grid change + gradient bound).  e = -d(-beta f)/d beta
    at the level-grid point (stationary, so no implicit terms), not extrapolated.
    Exercised at beta = 2 (tests); grid <= 5 (the K = 6 Newton solve from the level-5
    seed was not reached at beta = 2, so larger grids are refused, not attempted)."""
    b = float(beta)
    if not 1 < b <= 3:
        raise Refusal("precondition:temperature-range", f"beta = {beta}: validated for "
                      "1 < beta <= 3 (spin-glass phase; the seed is sk_1rsb's grid range)")
    if not isinstance(grid, int) or grid < 3:
        raise Refusal("precondition:grid-too-small", f"grid = {grid!r}: need >= 3 levels")
    if grid > _K_MAX_T:
        raise Refusal("precondition:grid-range", f"grid = {grid}: finite T validated up to "
                      f"{_K_MAX_T} levels")
    lv = [_frsb_level(b, K) for K in range(1, grid + 1)]
    fs = [x[1] for x in lv]
    finf, _ = _extrapolate(fs, list(range(1, grid + 1)), f"beta = {b} K-RSB")
    err = (finf - fs[-1]) + max(x[4] for x in lv[-3:]) + max(x[2] for x in lv[-3:])
    v = lv[-1][0]
    return FRSBSolution(b, tuple(map(float, v[:grid + 1])), tuple(map(float, v[grid + 1:])),
                        mp.mpf(finf), mp.mpf(lv[-1][5]), mp.mpf(err), tuple(fs), lv[-1][2],
                        lv[-1][3])
