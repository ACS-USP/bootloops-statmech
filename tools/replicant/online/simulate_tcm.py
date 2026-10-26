"""Finite-N online learning in the tree committee machine over named seeds (an independent route).

It builds the teacher and the student and applies the update itself, importing nothing from tcm.py,
tcm_flow.py, odes.py, fields.py or optimal.py (tests/test_online_tcm_sim.py inspects the imports); the
modulations are written out here again.  Per seed: K branches of n = N/K inputs, teacher branches B_k
uniform on the unit sphere of R^n, student J_k = R0 B_k + sqrt(Q0 - R0^2) v_k (v_k a unit vector
orthogonal to B_k); P = round(alpha N) examples xi with i.i.d. N(0, 1) components; label sigma_B =
sign(sum_k sign(B_k . xi_k)); update J_k <- J_k + (1/N) F_k xi_k.  The optimal rules read each branch's
current overlap rho_k = J_k . B_k / |J_k| exactly (CC95's first simulation, their Eq. (26), feeds rho
from the ODE instead); a seed whose rho_k reaches 0 (W ~ 1/rho) is refused undetermined:rho-sign.
  hebb           F_k = eta sigma_B
  local_optimal  F_k = K |J_k| (s_k / c_k) E[z | sigma_B, x_k], the branch posterior with the tie
                 probability P0 = C(K-1, (K-1)/2) / 2^(K-1) of the other hidden units
  optimal        CC95 Eqs. (19)-(20) as printed (K = 3): F_k = (3 / sqrt(2 pi)) |J_k| lambda_k sigma_B
                 e^{-Delta_k^2 / (2 lambda_k^2)} g(Delta_1/lambda_1, Delta_2/lambda_2, Delta_3/lambda_3)
                 for branch 1 (cyclically for 2, 3), g = D(y2, y3) / (H(-y2) H(-y3) + H(-y1) D(y2, y3)),
                 D(y2, y3) = H(-y2) H(y3) + H(y2) H(-y3), Delta_k = sigma_B h_k / |J_k|, lambda_k = s_k / c_k.
eps of a student is exact for Gaussian inputs: the branch errors e_k = atan2(|J_k - (J_k . B_k) B_k|,
J_k . B_k) / pi combined into the committee error by a polynomial in the two partial sums.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

NAMES = ("hebb", "local_optimal", "optimal")
_BLOCK = 64


@dataclass(frozen=True)
class TcmSimResult:
    alpha: float
    eps_mean: float
    band: Band
    eps_seeds: tuple
    K: int
    N: int
    seeds: tuple


def _committee_error(e):
    """e: (S, K) branch errors -> (S,) P(majority votes differ), by multiplying the per-branch
    polynomials sum_{a, b = +-1} p(a, b) X^a Y^b and summing the coefficients with opposite signs."""
    S, K = e.shape
    out = np.zeros(S)
    for s in range(S):
        poly = np.zeros((1, 1))
        poly[0, 0] = 1.0
        for k in range(K):
            ek = e[s, k]
            f = np.array([[(1 - ek) / 2, 0, ek / 2], [0, 0, 0], [ek / 2, 0, (1 - ek) / 2]])  # rows a, cols b in -1, 0, +1
            m = poly.shape[0]
            new = np.zeros((m + 2, m + 2))
            for i in (0, 2):
                for j in (0, 2):
                    new[i:i + m, j:j + m] += f[i, j] * poly
            poly = new
        idx = np.arange(poly.shape[0]) - (poly.shape[0] - 1) // 2
        out[s] = float(np.sum(poly[np.ix_(idx > 0, idx < 0)]) + np.sum(poly[np.ix_(idx < 0, idx > 0)]))
    return out


def _phi(u):
    return np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def _local(x, sig, rho, p0):
    s = np.sqrt(1 - rho * rho)
    u = x * rho / s
    lp = -0.5 * u * u - 0.5 * math.log(2 * math.pi)
    wp = np.exp(log_ndtr(u)) * (p0 * (sig > 0) + (1 - p0) / 2)
    wm = np.exp(log_ndtr(-u)) * (p0 * (sig < 0) + (1 - p0) / 2)
    g = (wp * np.exp(lp - log_ndtr(u)) - wm * np.exp(lp - log_ndtr(-u))) / (wp + wm)
    return s / rho * g


def _cc95(delta, lam):
    """CC95 Eqs. (19)-(20) per unit |J_k| (K = 3), branches along the last axis."""
    H = lambda t: ndtr(-t)
    y = delta / lam
    out = np.empty_like(y)
    for k in range(3):
        y1, y2, y3 = y[..., k], y[..., (k + 1) % 3], y[..., (k + 2) % 3]
        D = H(-y2) * H(y3) + H(y2) * H(-y3)
        out[..., k] = (3 / math.sqrt(2 * math.pi) * lam[..., k] * np.exp(-0.5 * y1 * y1)
                       * D / (H(-y2) * H(-y3) + H(-y1) * D))
    return out


def simulate_tcm(rule, K, N, alpha_max, seeds, eta=1.0, init=(0.0, 1.0)):
    """TcmSimResult of online TCM learning at N = K n over the named seeds (module docstring)."""
    if rule not in NAMES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(NAMES)}")
    if isinstance(K, bool) or not isinstance(K, int) or K < 1 or K % 2 == 0:
        raise Refusal("precondition:K", f"K={K!r}: an odd number of hidden units")
    if rule == "optimal" and K != 3:
        raise Refusal("unsupported:K", "CC95's nonlocal rule (Eqs. (19)-(20)) is written for K = 3")
    if isinstance(N, bool) or not isinstance(N, int) or N % K or N // K < 10:
        raise Refusal("precondition:size", f"N={N!r}: need an int multiple of K = {K} with N/K >= 10")
    seeds = _checks.seeds(seeds)
    if isinstance(eta, bool) or not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    if rule != "hebb" and eta != 1:
        raise Refusal("precondition:optimal-eta", "the optimal modulations are the optimum at eta = 1 only")
    if not (isinstance(alpha_max, (int, float)) and math.isfinite(alpha_max) and alpha_max > 0):
        raise Refusal("precondition:alpha", f"alpha_max={alpha_max!r}: need alpha > 0")
    try:
        R0, Q0 = (float(v) for v in init)
    except (TypeError, ValueError):
        raise Refusal("precondition:state-point", f"init={init!r}: need (R0, Q0)") from None
    if not (math.isfinite(R0) and math.isfinite(Q0)) or not (Q0 > 0 and abs(R0) < math.sqrt(Q0)) or (rule != "hebb" and R0 <= 0):
        raise Refusal("precondition:initial-overlap", f"(R0, Q0) = ({R0}, {Q0}): need Q0 > 0, |R0| < sqrt(Q0), "
                      "and R0 > 0 for the optimal rules")
    n, P = N // K, round(alpha_max * N)
    p0 = math.comb(K - 1, (K - 1) // 2) / 2 ** (K - 1)
    rngs = [np.random.default_rng(s) for s in seeds]
    B = np.stack([g.standard_normal((K, n)) for g in rngs])
    B /= np.linalg.norm(B, axis=2, keepdims=True)
    V = np.stack([g.standard_normal((K, n)) for g in rngs])
    V -= np.einsum("skn,skn->sk", V, B)[..., None] * B
    V /= np.linalg.norm(V, axis=2, keepdims=True)
    J = R0 * B + math.sqrt(Q0 - R0 * R0) * V
    step = 0
    while step < P:
        m = min(_BLOCK, P - step)
        xs = np.stack([g.standard_normal((m, K, n)) for g in rngs], axis=1)      # (m, S, K, n)
        for i in range(m):
            xi = xs[i]
            tau = np.where(np.einsum("skn,skn->sk", B, xi) >= 0, 1.0, -1.0)
            sig = np.where(tau.sum(axis=1) > 0, 1.0, -1.0)[:, None]
            if rule == "hebb":
                F = eta * np.broadcast_to(sig, tau.shape)
            else:
                norm = np.linalg.norm(J, axis=2)
                rho = np.einsum("skn,skn->sk", J, B) / norm
                if np.any(rho <= 0):
                    bad = sorted({seeds[j] for j in np.flatnonzero((rho <= 0).any(axis=1))})
                    raise Refusal("undetermined:rho-sign", f"seeds {bad} reached rho_k <= 0 at alpha = {step / N:.4g} "
                                  f"(N = {N}): the optimal modulations need rho > 0; start well above 1/sqrt(N/K)")
                x = np.einsum("skn,skn->sk", J, xi) / norm
                if rule == "local_optimal":
                    F = K * norm * _local(x, sig, rho, p0)
                else:
                    F = norm * sig * _cc95(sig * x, np.sqrt(1 - rho * rho) / rho)
            J += (F / N)[..., None] * xi
            step += 1
    R = np.einsum("skn,skn->sk", J, B)
    e = np.arctan2(np.linalg.norm(J - R[..., None] * B, axis=2), R) / math.pi
    last = _committee_error(e)
    band = student_band(last)
    return TcmSimResult(float(alpha_max), band.mean, band, tuple(float(v) for v in last), K, N, seeds)


__all__ = ["TcmSimResult", "simulate_tcm"]
