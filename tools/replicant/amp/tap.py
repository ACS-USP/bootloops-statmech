"""TAP free energy of SK and TAP fixed points (finite N).

phi_TAP(m) = -beta F_TAP(m) / N, with Q = <m, m> (amp/sk.py conventions):
    phi_TAP = (1/N) sum_i H2(m_i) + beta h <m> + (beta / 2N) m.J m + Onsager,
    H2(m) = -((1+m)/2) ln((1+m)/2) - ((1-m)/2) ln((1-m)/2)      (binary entropy),
    onsager="averaged":  (beta^2 / 4) (1 - Q)^2
    onsager="plefka":    (beta^2 / 4N) sum_{i != j} J_ij^2 (1 - m_i^2)(1 - m_j^2)
    onsager="none":      0 (naive mean field; the positive control).
The averaged form is Fan, Mei & Montanari, "TAP free energy, spin glasses, and variational
inference", Ann. Probab. 49(1), 1-45 (2021), doi:10.1214/20-AOP1443 (Crossref),
arXiv:1808.07890, eq. (1.6) (their F = -phi_TAP,
Y = J, lambda = 0), whose stationarity condition is their eq. (1.7), the TAP equations
m = tanh(beta h + beta J m - beta^2 (1 - Q) m) (Thouless, Anderson & Palmer, Phil. Mag. 35,
593 (1977), not fetched).  The "plefka" form is the second-order term of Plefka's expansion
(J. Phys. A 15, 1971 (1982), not fetched; described in Fan-Mei-Montanari Sec. 1 as the
first three terms of the expansion) before J_ij^2 is replaced by its mean 1/N; the two
differ by O(1/N) per spin.  No numerical value is banked: the papers state expressions.

At an AMP fixed point (amp/sk.py) phi_TAP is stationary, and in the RS-stable phase it
approaches the RS -beta f as N -> oo (the tests check both against comparison targets).
A TAP fixed point is ONE solution of the TAP equations: at low temperature (h = 0,
beta > 1) they have exponentially many (Bray & Moore 1980, per Fan-Mei-Montanari Sec. 2,
secondary), and m = 0 solves them at h = 0 for every beta; which one Newton reaches depends
on the start.  Finite-N results are statistical, never certified.
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np

from ..core import Refusal
from .sk import check_matrix, sk_amp
from .state_evolution import _check_beta_h

ONSAGER = ("averaged", "plefka", "none")


@dataclass
class TAPResult:
    m: np.ndarray
    q: float
    phi: float              # phi_TAP per spin (averaged Onsager term)
    residual: float         # max_i |m_i - tanh(beta h + beta (J m)_i - beta^2 (1 - Q) m_i)|
    grad_rms: float         # rms of N d phi_TAP / d m_i
    newton_steps: int
    converged: bool
    amp_converged: bool
    hessian_max_eig: object = None    # < 0: local maximum of phi_TAP (minimum of F_TAP)
    label: str = "statistical"
    note: str = ""


def _prep(m, J, beta, h, onsager):
    J = check_matrix(J)
    b, hh = (float(v) for v in _check_beta_h(beta, h))
    m = np.asarray(m, dtype=float)
    if m.shape != (J.shape[0],):
        raise Refusal("precondition:shape", f"m has shape {m.shape}, J is {J.shape}")
    if not np.all(np.abs(m) <= 1):
        raise Refusal("precondition:magnetization-range", "some |m_i| > 1")
    if onsager not in ONSAGER:
        raise Refusal("precondition:known-onsager",
                      f"onsager={onsager!r}; known: {', '.join(ONSAGER)}")
    return m, J, b, hh


def _xlogx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def sk_tap_free_energy(m, J, beta, h=0, *, onsager="averaged"):
    """phi_TAP(m) = -beta F_TAP / N as an mpf (float64 arithmetic, ~1e-13 relative)."""
    m, J, b, hh = _prep(m, J, beta, h, onsager)
    N = len(m)
    ent = -(_xlogx((1 + m) / 2) + _xlogx((1 - m) / 2))
    val = math.fsum(ent) / N + b * hh * math.fsum(m) / N + b * float(m @ (J @ m)) / (2 * N)
    if onsager == "averaged":
        val += b * b * (1 - float(m @ m) / N)**2 / 4
    elif onsager == "plefka":
        v = 1 - m * m
        val += b * b * float(v @ ((J * J) @ v)) / (4 * N)
    return mp.mpf(val)


def sk_tap_gradient(m, J, beta, h=0, *, onsager="averaged"):
    """N d phi_TAP / d m_i (zero exactly at a solution of the TAP equations)."""
    m, J, b, hh = _prep(m, J, beta, h, onsager)
    if not np.all(np.abs(m) < 1):
        raise Refusal("precondition:open-interval", "the gradient needs |m_i| < 1 (atanh)")
    g = -np.arctanh(m) + b * hh + b * (J @ m)
    if onsager == "averaged":
        g -= b * b * (1 - float(m @ m) / len(m)) * m
    elif onsager == "plefka":
        g -= b * b * m * ((J * J) @ (1 - m * m))
    return g


def _residual(m, J, b, hh):
    Q = float(m @ m) / len(m)
    return float(np.max(np.abs(m - np.tanh(b * hh + b * (J @ m) - b * b * (1 - Q) * m))))


def tap_fixed_point(J, beta, h=0, *, seed=0, iters=200, tol=1e-10, max_newton=30,
                    hessian=False) -> TAPResult:
    """AMP (`iters` steps, init="random", seed) then Newton on the TAP equations (averaged
    Onsager term) with step halving to stay in (-1, 1) and decrease |grad|.  Refuses
    `no-convergence:tap` when the residual stays above tol."""
    J = check_matrix(J)
    b, hh = (float(v) for v in _check_beta_h(beta, h))
    amp = sk_amp(J, b, h=hh, iters=iters, seed=seed)
    m, N = np.clip(amp.m, -1 + 1e-15, 1 - 1e-15), J.shape[0]
    steps = 0
    for steps in range(1, max_newton + 1):
        if _residual(m, J, b, hh) < tol:
            steps -= 1
            break
        g = sk_tap_gradient(m, J, b, hh)
        Q = float(m @ m) / N
        H = b * J + np.diag(-1 / (1 - m * m) - b * b * (1 - Q)) + (2 * b * b / N) * np.outer(m, m)
        step = np.linalg.solve(H, g)
        t, g0 = 1.0, float(g @ g)
        while t > 1e-6:
            trial = m - t * step
            if np.all(np.abs(trial) < 1) and \
                    float(sk_tap_gradient(trial, J, b, hh) @ sk_tap_gradient(trial, J, b, hh)) < g0:
                break
            t /= 2
        else:
            break
        m = trial
    res = _residual(m, J, b, hh)
    if not res < tol:
        raise Refusal("no-convergence:tap",
                      f"Newton from the AMP iterate (seed {seed}, {iters} steps, AMP converged: "
                      f"{amp.converged}) left residual {res:.3g} > {tol:g} after {steps} steps")
    g = sk_tap_gradient(m, J, b, hh)
    eig = None
    if hessian:
        Q = float(m @ m) / N
        H = b * J + np.diag(-1 / (1 - m * m) - b * b * (1 - Q)) + (2 * b * b / N) * np.outer(m, m)
        eig = float(np.linalg.eigvalsh(H)[-1])
    note = ("one solution of the TAP equations (averaged Onsager term), reached by Newton from "
            f"the AMP iterate of seed {seed}; statistical, not certified; near and below T_c "
            "(h = 0, beta > 1) the TAP equations have exponentially many solutions and this "
            "need not be the relevant one (m = 0 always solves them at h = 0)")
    return TAPResult(m, float(m @ m) / N, float(sk_tap_free_energy(m, J, b, hh)), res,
                     float(math.sqrt(g @ g / N)), steps, True, amp.converged, eig, note=note)
