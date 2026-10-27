"""The variationally optimal online algorithm of the soft committee machine (O6; Vicente & Caticha 1997).

Paper: R. Vicente and N. Caticha, "Functional optimization of online algorithms in multilayer neural networks",
J. Phys. A 30, L599-L605 (1997), doi:10.1088/0305-4470/30/17/002 [VC97]; read: arXiv cond-mat/9706015v1 (the
authors' TeX source and PostScript figures; the IOP typeset version was not fetched).
Convention (VC97 p. 2, which is Saad & Solla's soft committee machine of online/scm.py with the learning rate
folded into F): K student units J_k, M teacher units B_n with B_n . B_m = delta_nm, g(u) = erf(u / sqrt 2),
Sigma_J = sum_k g(x_k), Sigma_B = sum_n g(y_n), x_k = J_k . S, y_n = B_n . S, S i.i.d. N(0, 1) (VC97: a uniform
spherical distribution; the same field statistics as N -> oo), e_g = <(Sigma_B - Sigma_J)^2> / 2 (Eq. (3)),
update J_k <- J_k + (F_k / N) S (Eq. (1)), alpha = examples / N, R_kn = J_k . B_n, Q_ij = J_i . J_j and
  dR_in/dalpha = <y_n F_i>,  dQ_ij/dalpha = <x_i F_j + x_j F_i + F_i F_j>                      (Eq. (2)).
The variational solution (Eq. (4)): F = H^-1 G <y>_{H|V} - x, with H_ij = delta^2 de_g/dalpha / delta F_i delta F_j
= 2 D_ij, D_ij = de_g/dQ_ij (Q_ij and Q_ji counted as separate entries, so D is symmetric) and G_kn = -de_g/dR_kn;
the visible information V is (x_1 .. x_K, Sigma_B), the hidden H is y.  Derived here (not printed in VC97): with
A = H^-1 G (K x M), yhat = <y | x, Sigma_B> and W = <yhat yhat^T> (M x M), Eq. (2) closes to
  dR = A W - R,  dQ = A W A^T - Q,
because <y yhat^T> = <yhat yhat^T> and <x yhat^T> = R.  Given x, y is Gaussian with mean mu = R^T Q^+ x and
covariance Sigma = 1 - R^T Q^+ R, and mu ~ N(0, 1 - Sigma), so W depends on Sigma only:
  M = 1: Sigma_B = g(y) fixes y, so W = 1 exactly;
  M = 2: yhat is a mean over the level curve g(y1) + g(y2) = Sigma_B (info_matrix: two branches split at
         y1 = y2, each a graph with slope <= 1, windowed three times on the Gaussian's support, then Simpson;
         the outer Gauss-Hermite runs over mu and over z, y = mu + Sigma^(1/2) z, n = 12 nodes per axis); a
         singular Sigma (one direction of y known from x) is a sum over the roots on that line.  Accuracy:
         the inner integral to ~1e-6, but yhat changes steeply with Sigma_B (the posterior moves between the
         arms of the level curve), so the outer quadrature converges slowly: W to ~3e-3 (n = 12 vs 20 differ
         by 2e-3; against an independent dense route, tests); see optimal_ml_flow for the consequence;
  M >= 3: refused (unsupported:M).
The ODE limit is self-averaging by assumption (VC97 work 'in the thermodynamic limit'): physics-level, not final.

_PLANT: private planted-error hook (tests only): "w" multiplies W; "dqr" multiplies the R-term of D_ii; "hess"
multiplies H (0.5: H read as de_g/dQ instead of its second variation 2 de_g/dQ).
"""
import math
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfinv

from ..core import Refusal

OPTIMALITY_CLASS = (
    "VC97's locally optimal modulation: the functions F_k of the update J_k <- J_k + (F_k / N) S (no weight decay, "
    "no separate learning rate) that may read every student field x_1..x_K (nonlocal across the hidden units), the "
    "teacher's output Sigma_B (not its hidden fields), and the exact order parameters (Q, R) and input "
    "distribution (unknown to a practical algorithm, as VC97 stress); 'optimal' = the stationary point of "
    "de_g/dalpha at the current state, i.e. greedy (the largest decay per example, VC97's 'local optimisation'; not "
    "the best trajectory: Saad & Rattray's global optimisation can do better); a maximum of the decay only where "
    "the functional Hessian H is positive definite, a saddle where H has a negative eigenvalue (early learning, "
    "VC97 Figs. 1-2 insets)")
_PLANT = {"w": 1.0, "dqr": 1.0, "hess": 1.0}
SQ2 = math.sqrt(2.0)
Plateau = namedtuple("Plateau", "r q eps eigs")


def eps_vc(Q, R):
    """e_g, VC97 Eq. (3) (M_nm = delta_nm)."""
    Q, R = np.asarray(Q, float), np.asarray(R, float)
    d = np.sqrt(1 + np.diag(Q))
    return float(np.arcsin(Q / np.outer(d, d)).sum() / math.pi
                 - 2 / math.pi * np.arcsin(R / (SQ2 * d[:, None])).sum() + R.shape[1] / 6)


def gradients(Q, R):
    """(D, G): D_ij = de_g/dQ_ij over ordered pairs (symmetric), G_kn = -de_g/dR_kn, both of Eq. (3)."""
    Q, R = np.asarray(Q, float), np.asarray(R, float)
    q = np.diag(Q)
    d = np.sqrt(1 + q)
    a = Q / np.outer(d, d)
    s = 1 / np.sqrt(1 - a * a)
    D = s / np.outer(d, d) / math.pi
    b = R / (SQ2 * d[:, None])
    sb = 1 / np.sqrt(1 - b * b)
    off = ~np.eye(len(q), dtype=bool)
    diag = (1 / ((1 + q) * np.sqrt(1 + 2 * q)) - np.where(off, s * a, 0).sum(1) / (1 + q)) / math.pi
    diag = diag + _PLANT["dqr"] / math.pi * (sb * b).sum(1) / (1 + q)
    D[np.diag_indices(len(q))] = diag
    return D, 2 / math.pi * sb / (SQ2 * d[:, None])


def hessian(Q, R):
    return 2 * _PLANT["hess"] * gradients(Q, R)[0]


def modulation_matrix(Q, R):
    """A = H^-1 G (K x M), the coefficients of F* = A yhat - x."""
    G = gradients(Q, R)[1]
    H = hessian(Q, R)
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-300 or np.linalg.cond(H) > 1e14:
        raise Refusal("no-solution:singular-hessian", "the functional Hessian H is singular here: the variational "
                      "equation (VC97 Eq. (4)) has no unique solution")
    return np.linalg.solve(H, G)


def posterior(Q, R):
    """(Bm, Sigma): mu = Bm x = <y | x>, Sigma = Cov(y | x) = 1 - R^T Q^+ R (eigenvalues clipped to [0, 1])."""
    Q, R = np.asarray(Q, float), np.asarray(R, float)
    Bm = (np.linalg.pinv(Q, rcond=1e-13) @ R).T
    S = np.eye(R.shape[1]) - Bm @ R
    w, U = np.linalg.eigh((S + S.T) / 2)
    return Bm, (U * np.clip(w, 0, 1)) @ U.T


g = lambda u: erf(u / SQ2)
ginv = lambda v: SQ2 * erfinv(np.clip(v, -1 + 1e-16, 1 - 1e-16))


def _grid2(n, C):
    """Gauss-Hermite nodes and weights of N(0, C), C 2 x 2 (a null direction gets a single node)."""
    w, U = np.linalg.eigh(C)
    x, wt = np.polynomial.hermite_e.hermegauss(n)
    wt = wt / math.sqrt(2 * math.pi)
    live = [k for k in range(2) if w[k] > 1e-14]
    if not live:
        return np.zeros((1, 2)), np.ones(1)
    if len(live) == 1:
        return np.outer(x, U[:, live[0]] * math.sqrt(w[live[0]])), wt
    A = U * np.sqrt(w)
    X = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    return X @ A.T, np.outer(wt, wt).ravel()


def _branch(m1, m2, sd1, sd2, s, Si, nc, nf, passes):
    """Log-weights and points of one branch of g(t) + g(o) = s, s >= 0, t from c = g^-1(s/2) up (|do/dt| <= 1):
    the window is shrunk `passes` times to where the Gaussian form is within e^-40 of its minimum, then Simpson."""
    lo = np.maximum(ginv(s / 2), m1 - 12 * sd1)
    hi = np.minimum(ginv(np.minimum(1, s + 1)), m1 + 12 * sd1)
    lo = np.maximum(lo, ginv(s - g(m2 + 12 * sd2)))
    hi = np.minimum(hi, ginv(s - g(m2 - 12 * sd2)))
    empty = hi <= lo
    hi = np.where(empty, lo + 1e-12, hi)

    def form(t):
        o = ginv(s[..., None] - g(t))
        d1, d2 = t - m1[..., None], o - m2[..., None]
        return Si[0, 0] * d1 * d1 + 2 * Si[0, 1] * d1 * d2 + Si[1, 1] * d2 * d2, o
    k = (np.arange(nc) + 0.5) / nc
    for _ in range(passes):
        h = (hi - lo) / nc
        q, _ = form(lo[..., None] + h[..., None] * (k * nc))
        ok = q <= q.min(-1, keepdims=True) + 80.0
        first = np.where(ok, np.arange(nc), nc).min(-1)
        last = np.where(ok, np.arange(nc), -1).max(-1)
        lo, hi = np.maximum(lo, lo + (first - 0.5) * h), np.minimum(hi, lo + (last + 1.5) * h)
    t = lo[..., None] + (hi - lo)[..., None] * np.linspace(0, 1, nf)
    q, o = form(t)
    lw = -0.5 * q + 0.5 * o * o                                  # the Gaussian over the Jacobian g'(o)
    wts = np.full(nf, 2.0)
    wts[1::2], wts[0], wts[-1] = 4.0, 1.0, 1.0
    lw = np.where(np.isfinite(lw) & ~empty[..., None], lw + np.log(wts / 3 * ((hi - lo) / (nf - 1))[..., None]), -np.inf)
    return lw, t, o


def yhat2(mu, s, Sig, nc=32, nf=33, passes=3):
    """<y | y ~ N(mu, Sig), g(y1) + g(y2) = s> for M = 2 (Sig nonsingular); mu (..., 2), s (...)."""
    sg = np.where(s < 0, -1.0, 1.0)                              # y -> -y maps s < 0 to s > 0
    mu, s = mu * sg[..., None], s * sg
    Si, sd = np.linalg.inv(Sig), np.sqrt(np.diag(Sig))
    la, ta, oa = _branch(mu[..., 0], mu[..., 1], sd[0], sd[1], s, Si, nc, nf, passes)
    lb, tb, ob = _branch(mu[..., 1], mu[..., 0], sd[1], sd[0], s, Si[::-1, ::-1], nc, nf, passes)
    top = np.maximum(la.max(-1), lb.max(-1))[..., None]
    wa, wb = np.exp(la - top), np.exp(lb - top)
    Z = wa.sum(-1) + wb.sum(-1)
    y = np.stack([((wa * ta).sum(-1) + (wb * ob).sum(-1)) / Z, ((wa * oa).sum(-1) + (wb * tb).sum(-1)) / Z], -1)
    return y * sg[..., None]


def _info_singular(Sig, lam, v, n):
    """Sigma = lam v v^T: y = mu + sqrt(lam) tau v; yhat sums over the roots tau' of g(y1) + g(y2) = s, located
    by sign changes on a grid of step 0.1 in tau' (|tau'| <= 14) and 40 bisections."""
    mu, wm = _grid2(n, np.eye(2) - Sig)
    tau, wt = np.polynomial.hermite_e.hermegauss(3 * n)
    wt = wt / math.sqrt(2 * math.pi)
    sv = math.sqrt(lam) * v
    m = np.repeat(mu, len(tau), axis=0)                          # (m n_tau, 2) outer nodes
    S = lambda t, k: g(m[k, 0] + t * sv[0]) + g(m[k, 1] + t * sv[1])
    every = np.arange(len(m))
    s = S(np.tile(tau, len(mu)), every)
    grid = np.linspace(-14, 14, 281)
    f = S(grid[None, :], every[:, None]) - s[:, None]
    k, j = np.nonzero(np.sign(f[:, :-1]) * np.sign(f[:, 1:]) < 0)   # one root per bracket
    lo, hi, flo = grid[j], grid[j + 1], f[k, j]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fm = S(mid, k) - s[k]
        left = np.sign(fm) * np.sign(flo) <= 0
        hi, lo, flo = np.where(left, mid, hi), np.where(left, lo, mid), np.where(left, flo, fm)
    t = 0.5 * (lo + hi)
    y1, y2 = m[k, 0] + t * sv[0], m[k, 1] + t * sv[1]
    dS = np.abs(math.sqrt(2 / math.pi) * (np.exp(-y1 * y1 / 2) * sv[0] + np.exp(-y2 * y2 / 2) * sv[1]))
    w = np.exp(-t * t / 2) / np.maximum(dS, 1e-300)
    Z = np.bincount(k, w, len(m))
    yh = np.stack([np.bincount(k, w * y1, len(m)), np.bincount(k, w * y2, len(m))], -1) / Z[:, None]
    return np.einsum("m,ma,mb->ab", np.outer(wm, wt).ravel(), yh, yh)


def info_matrix(Sig, M, n=12):
    """W = <yhat yhat^T> as a function of Sigma = Cov(y | x) (module docstring)."""
    if M == 1:
        return _PLANT["w"] * np.ones((1, 1))
    if M != 2:
        raise Refusal("unsupported:M", f"M={M}: the teacher-output posterior is implemented for M = 1, 2")
    Sig = (np.asarray(Sig, float) + np.asarray(Sig, float).T) / 2
    lam, U = np.linalg.eigh(Sig)
    if lam[1] < 1e-12:
        return _PLANT["w"] * (np.eye(2) - Sig)                     # x fixes y: yhat = y
    if lam[0] < 1e-10:
        return _PLANT["w"] * _info_singular(Sig, lam[1], U[:, 1], n)
    mu, wm = _grid2(n, np.eye(2) - Sig)
    z, wz = _grid2(n, Sig)
    y = mu[:, None, :] + z[None]
    yh = yhat2(np.broadcast_to(mu[:, None, :], y.shape), g(y[..., 0]) + g(y[..., 1]), Sig)
    return _PLANT["w"] * np.einsum("m,z,mza,mzb->ab", wm, wz, yh, yh)


@dataclass(frozen=True)
class OptimalFlow:
    K: int
    M: int

    def pack(self, Q, R):
        return np.concatenate([np.asarray(Q, float)[np.triu_indices(self.K)], np.asarray(R, float).ravel()])

    def unpack(self, y):
        iu = np.triu_indices(self.K)
        Q = np.zeros((self.K, self.K))
        Q[iu] = y[:len(iu[0])]
        Q[iu[1], iu[0]] = y[:len(iu[0])]
        return Q, np.asarray(y[len(iu[0]):], float).reshape(self.K, self.M)

    def rates(self, Q, R):
        """(dQ/dalpha, dR/dalpha) under F* (module docstring)."""
        A = modulation_matrix(Q, R)
        W = info_matrix(posterior(Q, R)[1], self.M)
        return A @ W @ A.T - Q, A @ W - R

    def vector_rate(self, y):
        return self.pack(*self.rates(*self.unpack(y)))

    def eps(self, Q, R):
        return eps_vc(Q, R)

    def decay(self, Q, R):
        """de_g/dalpha under F* (the greedy objective)."""
        D, G = gradients(Q, R)
        dQ, dR = self.rates(Q, R)
        return float((D * dQ).sum() - (G * dR).sum())


def optimal_flow(K, M):
    bad = lambda v: isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 4
    if bad(K):
        raise Refusal("precondition:K", f"K={K!r}: need an int 1..4 (student hidden units)")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise Refusal("precondition:M", f"M={M!r}: need an int >= 1 (teacher hidden units)")
    if M > 2:
        raise Refusal("unsupported:M", f"M={M}: the teacher-output posterior is implemented for M = 1, 2")
    return OptimalFlow(K, M)


_PLATEAU = {}


def plateau_k2m2():
    """K = M = 2, identical students J_1 = J_2 (Q = q 1 1^T, R = r 1 1^T): W > the information in x alone unless
    Sigma is singular, so the fixed point has r^2 / q = 1/2; it is the root of dR/dalpha (computed by the flow,
    W by the singular route) on q = 2 r^2."""
    key = tuple(sorted(_PLANT.items()))
    if key not in _PLATEAU:
        _PLATEAU[key] = _plateau()
    return _PLATEAU[key]


def _plateau():
    from scipy.optimize import brentq
    f = optimal_flow(2, 2)
    at = lambda r: (2 * r * r * np.ones((2, 2)), r * np.ones((2, 2)))
    r = brentq(lambda r: f.rates(*at(r))[1][0, 0], 0.2, 0.8, xtol=1e-13)
    Q, R = at(r)
    return Plateau(r, 2 * r * r, eps_vc(Q, R), tuple(np.linalg.eigvalsh(hessian(Q, R))))


__all__ = ["OPTIMALITY_CLASS", "OptimalFlow", "Plateau", "eps_vc", "gradients", "hessian", "info_matrix",
           "modulation_matrix", "optimal_flow", "plateau_k2m2", "posterior", "yhat2"]
