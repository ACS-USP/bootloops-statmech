"""Finite-N online learning with VC97's optimal modulation in the soft committee machine (O6; an independent route).

It imports nothing from optimal_multilayer.py, optimal_ml_flow.py, scm*.py or any other ODE module
(tests/test_online_optimal_ml_sim.py inspects the imports); the algorithm is implemented again from VC97's
Eqs. (1), (3), (4):
  networks  teacher B (M x N) with orthonormal rows; student J = R B + W, W orthogonal to the teacher with
            W W^T = Q - R R^T, so the initial overlaps are exact (as in simulate_scm);
  example   S ~ N(0, 1)^N, x = J S, y = B S, Sigma_B = sum_n erf(y_n / sqrt 2);
  F         F = H^-1 G yhat - x at the current exact overlaps (Q, R): e_g is Eq. (3) typed again, H = 2 D and G by
            central differences of it (step 1e-5), and yhat = <y | x, Sigma_B>:
              M = 1: yhat = sqrt 2 erfinv(Sigma_B);
              M = 2: given x, y ~ N(mu, Sigma), mu = R^T Q^-1 x, Sigma = 1 - R^T Q^-1 R; on the level curve
                     the parameter is p = y1 - y2 and y1 + y2 = q(p) solves the (increasing) constraint by 30
                     safeguarded Newton steps; trapezoid in p on mu_p +- 10 sd_p (n_p points), weight
                     N(y; mu, Sigma) / (g'(y1) + g'(y2));
  update    J <- J + (F / N) S; the overlaps are recomputed from J every 32 examples, updated exactly between.
Bands: amp/_stats.student_band (99% Student-t) over the named seeds at each checkpoint.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfinv

from ..amp._stats import student_band
from ..core import Refusal
from ..csp import _checks

_R2 = math.sqrt(2.0)
_FD = 1e-5


@dataclass(frozen=True)
class OptSimResult:
    alphas: tuple
    eps: np.ndarray              # (seeds, checkpoints)
    bands: tuple
    N: int
    seeds: tuple
    Q: np.ndarray                # (seeds, checkpoints, K, K)
    R: np.ndarray


def _eg(Q, R):
    """Eq. (3), vectorised over leading axes."""
    d = np.sqrt(1 + np.diagonal(Q, axis1=-2, axis2=-1))
    a = np.arcsin(np.clip(Q / (d[..., :, None] * d[..., None, :]), -1, 1)).sum((-2, -1)) / math.pi
    b = np.arcsin(np.clip(R / (_R2 * d[..., :, None]), -1, 1)).sum((-2, -1))
    return a - 2 / math.pi * b + R.shape[-1] / 6


def _coefficients(Q, R):
    """A = H^-1 G per seed by central differences of _eg (Q_ij, Q_ji moved together for i != j)."""
    S, K, M = R.shape
    Hs, Gs = np.empty((S, K, K)), np.empty((S, K, M))
    for i in range(K):
        for j in range(i, K):
            E = np.zeros((K, K))
            E[i, j] = E[j, i] = _FD
            v = (_eg(Q + E, R) - _eg(Q - E, R)) / (2 * _FD)
            Hs[:, i, j] = Hs[:, j, i] = 2 * (v if i == j else v / 2)
        for n in range(M):
            E = np.zeros((K, M))
            E[i, n] = _FD
            Gs[:, i, n] = -(_eg(Q, R + E) - _eg(Q, R - E)) / (2 * _FD)
    return np.linalg.solve(Hs, Gs)


def _yhat2(mu, Sig, sB, n_p):
    """M = 2 posterior mean on the level curve (module docstring); mu (S, 2), Sig (S, 2, 2), sB (S,)."""
    Si = np.linalg.inv(Sig)
    mp_ = mu[:, 0] - mu[:, 1]
    sp_ = np.sqrt(np.maximum(Sig[:, 0, 0] + Sig[:, 1, 1] - 2 * Sig[:, 0, 1], 1e-300))
    p = mp_[:, None] + sp_[:, None] * np.linspace(-10, 10, n_p)[None, :]
    s = sB[:, None]
    q = np.zeros_like(p)
    lo, hi = np.full_like(p, -60.0), np.full_like(p, 60.0)
    for _ in range(30):
        y1, y2 = (q + p) / 2, (q - p) / 2
        f = erf(y1 / _R2) + erf(y2 / _R2) - s
        lo, hi = np.where(f < 0, q, lo), np.where(f >= 0, q, hi)
        d = (np.exp(-y1 * y1 / 2) + np.exp(-y2 * y2 / 2)) / math.sqrt(2 * math.pi)
        step = q - f / np.maximum(d, 1e-300)
        q = np.where((step > lo) & (step < hi), step, (lo + hi) / 2)
    y1, y2 = (q + p) / 2, (q - p) / 2
    d1, d2 = y1 - mu[:, 0:1], y2 - mu[:, 1:2]
    lw = -0.5 * (Si[:, 0, 0, None] * d1 * d1 + 2 * Si[:, 0, 1, None] * d1 * d2 + Si[:, 1, 1, None] * d2 * d2)
    lw = lw - np.log(np.exp(-y1 * y1 / 2) + np.exp(-y2 * y2 / 2))
    w = np.exp(lw - lw.max(1, keepdims=True))
    w[:, 0] *= 0.5
    w[:, -1] *= 0.5
    return np.stack([(w * y1).sum(1), (w * y2).sum(1)], -1) / w.sum(1)[:, None]


def _place(rng, K, M, N, Q, R):
    O = np.linalg.qr(rng.standard_normal((N, M + K)))[0].T
    B, E = O[:M], O[M:]
    G = Q - R @ R.T
    w, V = np.linalg.eigh((G + G.T) / 2)
    return R @ B + (V * np.sqrt(np.clip(w, 0, None))) @ V.T @ E, B


def simulate_optimal(K, M, N, alphas, seeds, init, n_p=201):
    """OptSimResult: e_g at each checkpoint alpha over the named seeds (module docstring)."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 20:
        raise Refusal("precondition:size", f"N={N!r}: need an int >= 20")
    if (K, M) not in ((1, 1), (2, 1), (3, 1), (2, 2)):
        raise Refusal("unsupported:M", f"(K, M) = ({K}, {M}): simulated for M = 1 (K <= 3) and K = M = 2")
    al = tuple(float(a) for a in alphas)
    if not al or al[0] <= 0 or any(b <= a for a, b in zip(al, al[1:])) or not all(map(math.isfinite, al)):
        raise Refusal("precondition:alpha", f"alphas={alphas!r}: need increasing checkpoints > 0")
    seeds = _checks.seeds(seeds)
    Q0, R0 = (np.array(v, dtype=float) for v in init)
    if Q0.shape != (K, K) or R0.shape != (K, M) or np.linalg.eigvalsh(np.block([[Q0, R0], [R0.T, np.eye(M)]])).min() < -1e-12:
        raise Refusal("precondition:initial-overlap", "need Q (K x K), R (K x M) with [[Q, R], [R^T, 1]] PSD")
    rngs = [np.random.default_rng(s) for s in seeds]
    nets = [_place(g, K, M, N, Q0, R0) for g in rngs]
    J, B = np.stack([n[0] for n in nets]), np.stack([n[1] for n in nets])
    eps, Qs, Rs, step = [], [], [], 0
    for target in (round(a * N) for a in al):
        while step < target:
            m = min(32, target - step)
            Q, R = np.einsum("skn,sln->skl", J, J), np.einsum("skn,smn->skm", J, B)
            xs = np.stack([g.standard_normal((m, N)) for g in rngs], axis=1)
            for i in range(m):
                xi = xs[i]
                x, y = np.einsum("skn,sn->sk", J, xi), np.einsum("smn,sn->sm", B, xi)
                sB = erf(y / _R2).sum(1)
                if M == 1:
                    yh = (_R2 * erfinv(np.clip(sB, -1 + 1e-16, 1 - 1e-16)))[:, None]
                else:
                    Bm = np.linalg.solve(Q, R).transpose(0, 2, 1)
                    Sig = np.eye(M) - Bm @ R
                    yh = _yhat2(np.einsum("smk,sk->sm", Bm, x), (Sig + Sig.transpose(0, 2, 1)) / 2, sB, n_p)
                F = np.einsum("skm,sm->sk", _coefficients(Q, R), yh) - x
                n2 = np.einsum("sn,sn->s", xi, xi)
                Q = Q + (F[:, :, None] * x[:, None, :] + x[:, :, None] * F[:, None, :]) / N \
                    + F[:, :, None] * F[:, None, :] * (n2 / N ** 2)[:, None, None]
                R = R + F[:, :, None] * y[:, None, :] / N
                J += F[:, :, None] * xi[:, None, :] / N
            step += m
        Q, R = np.einsum("skn,sln->skl", J, J), np.einsum("skn,smn->skm", J, B)
        eps.append(_eg(Q, R))
        Qs.append(Q)
        Rs.append(R)
    e = np.array(eps).T
    return OptSimResult(al, e, tuple(student_band(e[:, k]) for k in range(len(al))), N, seeds,
                        np.stack(Qs, 1), np.stack(Rs, 1))


__all__ = ["OptSimResult", "simulate_optimal"]
