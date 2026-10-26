"""The tree committee machine (TCM) for online learning: architecture, field averages, the mapping.

Architecture (Copelli & Caticha, J. Phys. A 28, 1615 (1995), CC95, Sec. 2, read from the preprint
IFUSP/P-1126; Copelli, Kinouchi & Caticha, Phys. Rev. E 53, 6341 (1996), CKC96, abstract only): K odd
non-overlapping branches of N/K inputs each, xi = (xi_1, ..., xi_K); teacher branches B_k with |B_k| = 1
(CC95: B_k^2 = 1), hidden units tau_k = sign(y_k), y_k = B_k . xi_k, and the label sigma_B = sign(sum_k
tau_k) (CC95 Eq. (1)); the student likewise with J_k and x_k = J_k . xi_k / |J_k|.  CC95 use Boolean
inputs S_kj = +-1; replicant uses i.i.d. N(0, 1) components (the same N -> oo field statistics, CC95
p. 5 after Mato & Parga).  Update J_k <- J_k + (1/N) F_k xi_k (CC95 Eq. (6) with Omega_k = 0; a
learning rate is folded into F); alpha = examples / N = CKC96's alpha_CM (examples per adjustable
weight); eps = P(student label != teacher label) on a fresh input (CC95 e_g, Eqs. (3)-(5)), against
the noiseless teacher.  The state is branch symmetric (rho_k = rho, |J_k| = J), which the flow keeps
(CC95 Sec. 4).

Field averages (computed here, sharing no code with O1's fields.py: O-R2).  For branch k, with
theta = arccos rho, s = sin theta, c = cos theta, y = c x + s z (z a unit Gaussian independent of x):
P(tau = +1 | x) = Phi(u), u = x cot theta, and E[z | tau, x] = +phi(u)/Phi(u) or -phi(u)/Phi(-u).
The label depends on tau_k through the other K - 1 hidden units only via S = sum_{j != k} tau_j:
sigma_B = tau_k if S = 0 (a tie among the others), else sign(S).
  local_optimal  the student branch knows only (sigma_B, x_k): the other units enter through the
                 distribution of S (tie_distribution, a DP over their internal representations), so
                 P(sigma_B | tau_k) = P0 [sigma_B = tau_k] + (1 - P0) / 2 with P0 = P(S = 0).
  optimal        CC95's nonlocal rule (K = 3): the branch knows (sigma_B, x_1, x_2, x_3); the others'
                 tau_j given x_j are Bernoulli(Phi(u_j)), enumerated exactly.
Both use g = E[z | sigma_B, information], F_k = K |J| (s / c) g, which is CC95's F* = K J_k (<b_k>/rho_k
- h_k) (their Eqs. (9)-(12)) written with b = y.  The averages over x (and the others' x_j) are Gauss-
Legendre in u (|u| <= 12, where the integrand lives as rho -> 1), the others' |u_j| > 12 being exact
point masses (their tau_j is then decided).  averages(rule, theta, K) returns, per unit scale c(Q)
(sqrt Q for the optimal rules, 1 for Hebb), Averages(ay, ax, a2, drive) = (<F y>, <F x>, <F^2>,
<F (y - c x)>), with <F y> = <F E[y | sigma_B, info]>.
  hebb           F_k = sigma_B: <sigma_B y_k> by Gauss-Legendre in y with the internal-representation
                 average over S; <sigma_B x_k> = c <sigma_B y_k> (x = c y + s w).

The mapping (CKC96: "local learning in a TCM with K hidden units is simply related to learning in a
simple perceptron with a corresponding noise level"; their epsilon(K) was NOT read): for a local rule
the joint law of (x_k, y_k, sigma_B) is that of a perceptron whose label is flipped with probability
lambda_K = (1 - P0) / 2 (derived here: given only x_k, sign(S) is +-1 with equal probability and
independent of tau_k), on N/K inputs, in time alpha_p = K alpha, with F_perceptron = F_TCM / K.
effective_noise(K) and perceptron_alpha(alpha, K) are that mapping; the TCM route (averages, tcm_flow)
never calls them (tests/test_online_tcm.py inspects the source), so the CKC96 identity is tested
between two independent routes.

_PLANT: private planted-error hooks (tests only): "mapping" multiplies lambda_K, "time" the factor K.
"""
import math
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..core import Refusal

RULES = ("hebb", "local_optimal", "optimal")
Averages = namedtuple("Averages", "ay ax a2 drive")
_PLANT = {"mapping": 1.0, "time": 1.0}
_U = 12.0                    # |u| cut: Phi(-12) ~ 2e-33, the others' tau is decided beyond it

OPTIMALITY_CLASS = {
    "local_optimal": (
        "maximises drho_k/dalpha at every alpha among modulated Hebbian updates J_k <- J_k + (1/N) F_k xi_k "
        "of a tree committee machine (noiseless TCM teacher) whose F_k depends on the current example only "
        "through the teacher label sigma_B and the branch's OWN field h_k (and |J_k|), and that may use the "
        "TRUE overlap rho (branch-symmetric state); it is O1's optimal perceptron rule at the mapped noise "
        "lambda_K (CKC96, abstract; derived here); not optimal against rules that read the other branches' "
        "fields (the nonlocal 'optimal'), several examples, or batch learning"),
    "optimal": (
        "CC95's expected-stability rule (their Eqs. (9)-(20)), K = 3: maximises drho_k/dalpha at every alpha "
        "among modulated Hebbian updates whose F_k depends on the current example through sigma_B and ALL "
        "student fields h_1, ..., h_K (and |J_k|), and that may use the TRUE overlaps rho_k (CC95 also run it "
        "with rho read off the norms, their Eq. (27)); greedy per example (CC95 Sec. 3); not claimed optimal "
        "against rules using several examples, nor against batch learning"),
}


def _check_k(K):
    if isinstance(K, bool) or not isinstance(K, int) or K < 1 or K % 2 == 0:
        raise Refusal("precondition:K", f"K={K!r}: the committee needs an odd number K >= 1 of hidden units")


@lru_cache(maxsize=None)
def _ties(K):
    """Distribution of S = sum of the other K - 1 teacher hidden units (each +-1 with probability 1/2,
    the sign of a symmetric Gaussian field), by a DP over their internal representations."""
    dist = {0: 1.0}
    for _ in range(K - 1):
        new = {}
        for s, p in dist.items():
            for t in (1, -1):
                new[s + t] = new.get(s + t, 0.0) + 0.5 * p
        dist = new
    return dist


def tie_distribution(K):
    """{S: P(S)} for the other K - 1 teacher hidden units."""
    _check_k(K)
    return dict(_ties(K))


def tie_probability(K):
    """P0 = P(the other K - 1 hidden units tie), the probability that branch k decides the label."""
    _check_k(K)
    return _ties(K).get(0, 0.0)


def effective_noise(K):
    """THE MAPPING (not used by the TCM route): the output-noise level lambda_K = (1 - P0)/2 of the
    perceptron equivalent to local learning in branch k."""
    return _PLANT["mapping"] * (1 - tie_probability(K)) / 2


def perceptron_alpha(alpha, K):
    """THE MAPPING's time: a branch has N/K inputs, so alpha_p = K alpha."""
    _check_k(K)
    return _PLANT["time"] * K * alpha


def eps_committee(e, K):
    """P(majority of student units != majority of teacher units) when branch k's units disagree with
    probability e_k, independently (a DP over the pairs of partial sums): CC95 Eq. (5) for K = 3.
    e: a number (all branches) or a sequence of K branch errors."""
    _check_k(K)
    es = [float(e)] * K if np.ndim(e) == 0 else [float(v) for v in e]
    if len(es) != K or not all(0 <= v <= 1 for v in es):
        raise Refusal("precondition:K", f"eps_committee needs {K} branch errors in [0, 1], got {e!r}")
    dist = {(0, 0): 1.0}
    for ek in es:
        new = {}
        for (sb, sj), p in dist.items():
            for db, dj, q in ((1, 1, (1 - ek) / 2), (-1, -1, (1 - ek) / 2), (1, -1, ek / 2), (-1, 1, ek / 2)):
                key = (sb + db, sj + dj)
                new[key] = new.get(key, 0.0) + p * q
        dist = new
    return float(sum(p for (sb, sj), p in dist.items() if sb * sj < 0))


@lru_cache(maxsize=None)
def _leg(n):
    return np.polynomial.legendre.leggauss(n)


def _u_nodes(theta, n):
    """Gauss-Legendre nodes in u = x cot theta with the weights of Dx, and the Gaussian mass beyond."""
    t = math.tan(theta)
    lim = min(_U, 9.0 / t)
    u, w = _leg(n)
    u, w = u * lim, w * lim
    wx = np.exp(-0.5 * (u * t) ** 2) / math.sqrt(2 * math.pi) * t * w
    return u, wx, 0.5 * math.erfc(lim * t / math.sqrt(2))


def _m(u):
    """E[z | tau = +1, x] and E[z | tau = -1, x] at u = x cot theta (in log space)."""
    lp = -0.5 * u * u - 0.5 * math.log(2 * math.pi)
    return np.exp(lp - log_ndtr(u)), -np.exp(lp - log_ndtr(-u))


def _local_grid(theta, K, n=96):
    """(weights of P(sigma_B, x), g = E[z | sigma_B, x_k], x_k) for the local rule."""
    p0 = _ties(K).get(0, 0.0)
    u, wx, _ = _u_nodes(theta, n)
    mp_, mm = _m(u)
    qp, qm = ndtr(u), ndtr(-u)
    ws, gs, xs = [], [], []
    for sig in (1, -1):
        lp = qp * (p0 * (sig == 1) + (1 - p0) / 2)
        lm = qm * (p0 * (sig == -1) + (1 - p0) / 2)
        z = lp + lm
        ws.append(wx * z)
        gs.append((lp * mp_ + lm * mm) / z)
        xs.append(u * math.tan(theta))
    return np.concatenate(ws), np.concatenate(gs), np.concatenate(xs)


def _nonlocal_grid(theta, n=64):
    """Same for CC95's rule at K = 3: branch 1 on Gauss-Legendre nodes in u_1, branches 2, 3 on nodes
    in u_j plus their decided point masses; the internal representations (tau_2, tau_3) enumerated."""
    u1, w1, _ = _u_nodes(theta, n)
    uo, wo, tail = _u_nodes(theta, n)
    q = np.concatenate([ndtr(uo), [1.0, 0.0]])
    wq = np.concatenate([wo, [tail, tail]])
    q2, q3 = q[:, None], q[None, :]
    tie = q2 * (1 - q3) + (1 - q2) * q3                  # P(S = 0 | x_2, x_3)
    plus, minus = q2 * q3, (1 - q2) * (1 - q3)           # P(S = +2), P(S = -2)
    w23 = (wq[:, None] * wq[None, :])[None]
    mp_, mm = (v[:, None, None] for v in _m(u1))
    qp, qm = ndtr(u1)[:, None, None], ndtr(-u1)[:, None, None]
    x1 = np.broadcast_to((u1 * math.tan(theta))[:, None, None], (n, n + 2, n + 2))
    ws, gs, xs = [], [], []
    for sig, other in ((1, plus), (-1, minus)):
        lp = qp * (tie * (sig == 1) + other)
        lm = qm * (tie * (sig == -1) + other)
        z = lp + lm
        ws.append((w1[:, None, None] * w23 * z).ravel())
        gs.append(np.where(z > 0, (lp * mp_ + lm * mm) / np.where(z > 0, z, 1.0), 0.0).ravel())
        xs.append(x1.ravel())
    return np.concatenate(ws), np.concatenate(gs), np.concatenate(xs)


@lru_cache(maxsize=None)
def _hebb_sy(K, n=200):
    """<sigma_B y_k>: Gauss-Legendre in y on [-10, 0] and [0, 10] (the label jumps at y = 0) with the
    average over S."""
    t, w = _leg(n)
    y = np.concatenate([5 * (t + 1), -5 * (t + 1)])
    w = np.concatenate([5 * w, 5 * w]) * np.exp(-0.5 * y * y) / math.sqrt(2 * math.pi)
    sign_y = np.where(y >= 0, 1.0, -1.0)
    lab = sum(p * np.sign(sign_y + s) for s, p in _ties(K).items())
    return float(np.sum(w * lab * y))


def averages(rule, theta, K):
    """Averages(ay, ax, a2, drive) of F per unit scale for branch k at overlap theta (see the module
    docstring)."""
    _check_k(K)
    if rule not in RULES:
        raise Refusal("unsupported:rule", f"rule {rule!r}: {', '.join(RULES)}")
    if not (isinstance(theta, (int, float)) and 0 < theta < math.pi):
        raise Refusal("precondition:overlap", f"theta={theta!r}: need 0 < theta < pi")
    c, s = math.cos(theta), math.sin(theta)
    if rule == "hebb":
        sy = _hebb_sy(K)
        return Averages(sy, c * sy, 1.0, sy * (1 - c * c))
    if not c > 0:
        raise Refusal("precondition:overlap", f"theta={theta!r}: the optimal rules carry 1/rho: need rho > 0")
    if rule == "optimal" and K != 3:
        raise Refusal("unsupported:K", f"the nonlocal rule is implemented for K = 3 (CC95), not K = {K}")
    w, g, x = _nonlocal_grid(theta) if rule == "optimal" else _local_grid(theta, K)
    amp = K * s / c                                       # F = sqrt(Q) * amp * g
    f = c * x + s * g                                     # E[y | sigma_B, information]
    return Averages(amp * float(np.sum(w * g * f)), amp * float(np.sum(w * g * x)),
                    amp * amp * float(np.sum(w * g * g)), amp * s * float(np.sum(w * g * g)))


__all__ = ["Averages", "OPTIMALITY_CLASS", "RULES", "averages", "effective_noise", "eps_committee",
           "perceptron_alpha", "tie_distribution", "tie_probability"]
