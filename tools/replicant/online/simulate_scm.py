"""Finite-N online gradient descent in the soft committee machine over named seeds (O5; an independent route).

It builds the networks and applies the update itself, importing nothing from scm.py, scm_flow.py, plateau.py
or any other ODE module (tests/test_online_scm_sim.py inspects the imports).  Per seed:
  teacher   B (M x N) with orthonormal rows (QR of a Gaussian matrix), scaled to |B_n|^2 = T_nn (isotropic:
            T = delta exactly; graded: T_nn = n);
  student   J = R T^-1 B + W with W (K x N) orthogonal to the teacher and W W^T = Q - R T^-1 R^T (symmetric
            square root times orthonormal directions), so J J^T = Q and J B^T = R exactly (as BRW impose
            their Eq. (16) by Gram-Schmidt);
  update    for each of round(alpha N) examples xi ~ N(0, 1)^N: x = J xi, y = B xi, g(u) = erf(u / sqrt 2),
            delta_i = g'(x_i)(sum_n g(y_n) - sum_j g(x_j)), J_i <- J_i + (eta / N) delta_i xi;
  eps       exact for Gaussian inputs, typed again from SS NIPS Eq. (2) on the current overlaps:
            eps = (1/pi)[sum_ik asin(Q_ik/sqrt((1+Q_ii)(1+Q_kk))) + sum_nm asin(T_nm/sqrt((1+T_nn)(1+T_mm)))
                         - 2 sum_in asin(R_in/sqrt((1+Q_ii)(1+T_nn)))].
Bands: amp/_stats.student_band (99% Student-t) over the seeds at each checkpoint alpha.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from ..amp._stats import student_band
from ..core import Refusal
from ..csp import _checks

_BLOCK = 64


@dataclass(frozen=True)
class ScmSimResult:
    alphas: tuple
    eps: np.ndarray              # (seeds, checkpoints)
    bands: tuple                 # one Band per checkpoint
    N: int
    seeds: tuple
    initial: tuple = ()


def _eps(J, B):
    Q, R, T = J @ J.T, J @ B.T, B @ B.T
    q, t = np.diag(Q), np.diag(T)
    a = np.arcsin(Q / np.sqrt(np.outer(1 + q, 1 + q))).sum() + np.arcsin(T / np.sqrt(np.outer(1 + t, 1 + t))).sum()
    return float((a - 2 * np.arcsin(R / np.sqrt(np.outer(1 + q, 1 + t))).sum()) / math.pi)


def _place(rng, K, M, N, T, Q, R):
    B = np.linalg.qr(rng.standard_normal((N, M + K)))[0].T          # M + K orthonormal rows
    D, E = B[:M] * np.sqrt(np.diag(T))[:, None], B[M:]
    G = Q - R @ np.linalg.inv(T) @ R.T
    w, V = np.linalg.eigh((G + G.T) / 2)
    return R @ np.linalg.inv(T) @ D + (V * np.sqrt(np.clip(w, 0, None))) @ V.T @ E, D


def simulate_scm(K, M, N, eta, alphas, seeds, init=None, teacher="isotropic", return_initial=False):
    """ScmSimResult: eps at each checkpoint alpha over the named seeds (module docstring)."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 20:
        raise Refusal("precondition:size", f"N={N!r}: need an int >= 20")
    if isinstance(eta, bool) or not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    al = tuple(float(a) for a in alphas)
    if not al or al[0] <= 0 or any(b <= a for a, b in zip(al, al[1:])) or not all(map(math.isfinite, al)):
        raise Refusal("precondition:alpha", f"alphas={alphas!r}: need increasing checkpoints > 0")
    seeds = _checks.seeds(seeds)
    if teacher not in ("isotropic", "graded"):
        raise Refusal("unsupported:teacher", f"teacher {teacher!r}: isotropic | graded")
    T = np.eye(M) if teacher == "isotropic" else np.diag(np.arange(1.0, M + 1))
    if init is None:
        raise Refusal("precondition:state-point", "the simulation needs the initial overlaps (Q, R)")
    Q, R = (np.array(v, dtype=float) for v in init)
    G = np.block([[Q, R], [R.T, T]])
    if Q.shape != (K, K) or R.shape != (K, M) or np.linalg.eigvalsh(G).min() < -1e-12:
        raise Refusal("precondition:initial-overlap", "need Q (K x K), R (K x M) with [[Q, R], [R^T, T]] positive "
                      "semidefinite")
    rngs = [np.random.default_rng(s) for s in seeds]
    nets = [_place(g, K, M, N, T, Q, R) for g in rngs]
    J = np.stack([n[0] for n in nets])                               # (S, K, N)
    Bt = np.stack([n[1] for n in nets])                              # (S, M, N)
    start = tuple((j @ j.T, j @ b.T) for j, b in zip(J, Bt)) if return_initial else ()
    c, out, step, targets = math.sqrt(2 / math.pi), [], 0, [round(a * N) for a in al]
    for target in targets:
        while step < target:
            m = min(_BLOCK, target - step)
            xs = np.stack([g.standard_normal((m, N)) for g in rngs], axis=1)     # (m, S, N)
            for i in range(m):
                xi = xs[i]
                x = np.einsum("skn,sn->sk", J, xi)
                y = np.einsum("smn,sn->sm", Bt, xi)
                err = erf(y / math.sqrt(2)).sum(axis=1) - erf(x / math.sqrt(2)).sum(axis=1)
                delta = c * np.exp(-x * x / 2) * err[:, None]
                J += (eta / N) * delta[:, :, None] * xi[:, None, :]
            step += m
        out.append([_eps(j, b) for j, b in zip(J, Bt)])
    eps = np.array(out).T
    return ScmSimResult(al, eps, tuple(student_band(eps[:, k]) for k in range(len(al))), N, seeds, start)


__all__ = ["ScmSimResult", "simulate_scm"]
