"""Soft committee machine (SCM), online gradient descent: the Saad-Solla order-parameter ODEs (O5).

Convention (Saad & Solla, PRL 74, 4337 (1995); PRE 52, 4225 (1995); as restated by the same authors in
NIPS 8, 302 (1996), Eqs. (1)-(4), and in the MANNA proceedings (1997), Eqs. (1)-(6), which were read):
  K student hidden units J_i, M teacher hidden units B_n (SS's labels: i, j, k, l for the student,
  n, m for the teacher; the O5 brief names them the other way round), inputs xi i.i.d. N(0, 1),
  g(x) = erf(x / sqrt 2), sigma = sum_i g(J_i . xi), tau = sum_n g(B_n . xi), error
  eps(J, xi) = (sigma - tau)^2 / 2, update J_i <- J_i + (eta / N) delta_i xi with
  delta_i = g'(x_i) [sum_n g(y_n) - sum_j g(x_j)], alpha = examples / N.
  Order parameters Q_ik = J_i . J_k, R_in = J_i . B_n, T_nm = B_n . B_m (fixed).
As N -> oo (self-averaging ASSUMED, as SS do; no concentration theorem fetched):
  dR_in/dalpha = eta <delta_i y_n>,
  dQ_ik/dalpha = eta <delta_i x_k> + eta <delta_k x_i> + eta^2 <delta_i delta_k>,
which, with the units z = (x_1..x_K, y_1..y_M) and signs s = (-1 (students), +1 (teachers)), is
  dR_in = eta sum_u s_u I3(i, n, u),
  dQ_ik = eta sum_u s_u [I3(i, k, u) + I3(k, i, u)] + eta^2 sum_uv s_u s_v I4(i, k, u, v),
  eps_g = (1/2) sum_uv s_u s_v I2(u, v)                                           (SS Eq. (4), NIPS Eq. (2))
with I2 = <g(u) g(v)>, I3 = <g'(u) v g(w)>, I4 = <g'(u) g'(v) g(w) g(z)> over the Gaussian with the
projected covariance C (closed forms below: SS MANNA Eqs. (5)-(6); I2 from NIPS Eq. (2)):
  I2 = (2/pi) arcsin(C12 / sqrt((1 + C11)(1 + C22))),
  I3 = (2/pi) (C23 (1 + C11) - C12 C13) / ((1 + C11) sqrt(L3)),   L3 = (1 + C11)(1 + C33) - C13^2,
  I4 = (4/pi^2) arcsin(L0 / sqrt(L1 L2)) / sqrt(L4),   L4 = (1 + C11)(1 + C22) - C12^2,
  L0 = L4 C34 - C23 C24 (1 + C11) - C13 C14 (1 + C22) + C12 C13 C24 + C12 C14 C23,
  L1 = L4 (1 + C33) - C23^2 (1 + C11) - C13^2 (1 + C22) + 2 C12 C13 C23,   L2 likewise with 4.
`quadrature` is an independent route (tensor Gauss-Hermite in the needed dimension).

Biehl & Schwarze (J. Phys. A 28, 643 (1995)) write the same model with Q_BS = |J| (their Q^2 = J . J) and,
for K = 2, a teacher of one unit (M = 1) and an output gain beta; beta = 1 is this model with M = 1.
Biehl, Riegler & Woehler (J. Phys. A 29, 4769 (1996)) use exactly SS's convention (their Eqs. (1)-(6)).

_PLANT: private planted-error hook (tests only): "eta2" multiplies the eta^2 (I4) term; "unit" replaces
the 1 in every 1 + C_aa (0.5 is the erf(x) instead of erf(x / sqrt 2) normalisation); "pref" multiplies
the 2/pi of I2 and I3 and the 4/pi^2 of I4.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import Refusal

_PLANT = {"eta2": 1.0, "unit": 1.0, "pref": 1.0}


def i2(C):
    """<g(u) g(v)>, C (..., 2, 2)."""
    C = np.asarray(C, dtype=float)
    u = _PLANT["unit"]
    a, b, c = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]
    return _PLANT["pref"] * 2 / math.pi * np.arcsin(c / np.sqrt((u + a) * (u + b)))


def i3(C):
    """<g'(u) v g(w)>, C (..., 3, 3) of (u, v, w)."""
    C = np.asarray(C, dtype=float)
    u = _PLANT["unit"]
    c11, c12, c13, c23, c33 = C[..., 0, 0], C[..., 0, 1], C[..., 0, 2], C[..., 1, 2], C[..., 2, 2]
    l3 = (u + c11) * (u + c33) - c13 * c13
    return _PLANT["pref"] * 2 / math.pi * (c23 * (u + c11) - c12 * c13) / ((u + c11) * np.sqrt(l3))


def i4(C):
    """<g'(u) g'(v) g(w) g(z)>, C (..., 4, 4) of (u, v, w, z)."""
    C = np.asarray(C, dtype=float)
    u = _PLANT["unit"]
    c = lambda a, b: C[..., a, b]
    a1, a2 = u + c(0, 0), u + c(1, 1)
    l4 = a1 * a2 - c(0, 1) ** 2
    l0 = (l4 * c(2, 3) - c(1, 2) * c(1, 3) * a1 - c(0, 2) * c(0, 3) * a2 + c(0, 1) * c(0, 2) * c(1, 3)
          + c(0, 1) * c(0, 3) * c(1, 2))
    l1 = l4 * (u + c(2, 2)) - c(1, 2) ** 2 * a1 - c(0, 2) ** 2 * a2 + 2 * c(0, 1) * c(0, 2) * c(1, 2)
    l2 = l4 * (u + c(3, 3)) - c(1, 3) ** 2 * a1 - c(0, 3) ** 2 * a2 + 2 * c(0, 1) * c(0, 3) * c(1, 3)
    return _PLANT["pref"] * 4 / math.pi ** 2 * np.arcsin(l0 / np.sqrt(l1 * l2)) / np.sqrt(l4)


_KINDS = {2: ("g", "g"), 3: ("dg", "id", "g"), 4: ("dg", "dg", "g", "g")}


def quadrature(C, n=24):
    """The average I2, I3 or I4 (by the size of C) by tensor Gauss-Hermite: z = A u, u ~ N(0, 1)^d, A A^T = C
    (symmetric square root, so singular C is allowed).  Independent of i2/i3/i4 (no closed form used)."""
    from scipy.special import erf
    C = np.asarray(C, dtype=float)
    d = C.shape[0]
    w, V = np.linalg.eigh(C)
    A = V * np.sqrt(np.clip(w, 0, None))
    x, wt = np.polynomial.hermite_e.hermegauss(n)
    wt = wt / math.sqrt(2 * math.pi)
    U = np.stack(np.meshgrid(*([x] * d), indexing="ij"), axis=-1).reshape(-1, d)
    W = np.prod(np.stack(np.meshgrid(*([wt] * d), indexing="ij"), axis=-1).reshape(-1, d), axis=1)
    Z = U @ A.T
    f = {"g": lambda z: erf(z / math.sqrt(2)), "dg": lambda z: math.sqrt(2 / math.pi) * np.exp(-z * z / 2),
         "id": lambda z: z}
    out = W.copy()
    for k, kind in enumerate(_KINDS[d]):
        out = out * f[kind](Z[:, k])
    return float(out.sum())


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def teacher(kind, M):
    """T_nm: 'isotropic' (delta_nm, SS / BRW) or 'graded' (n delta_nm, BRW's graded teacher)."""
    if kind == "isotropic":
        return np.eye(M)
    if kind == "graded":
        return np.diag(np.arange(1.0, M + 1))
    raise Refusal("unsupported:teacher", f"teacher {kind!r}: isotropic | graded")


@dataclass(frozen=True, eq=False)
class ScmFlow:
    K: int
    M: int
    eta: float
    T: np.ndarray
    truncated: bool = False               # SS's small-eta equations: the eta^2 term dropped
    _idx: tuple = field(default=None, repr=False)

    @property
    def size(self):
        return self.K * (self.K + 1) // 2 + self.K * self.M

    def _index(self):
        if self._idx is None:
            K, n = self.K, self.K + self.M
            grids = [np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1)
                     for shape in ((K, n, n), (K, K, n, n), (n, n))]
            object.__setattr__(self, "_idx", tuple((g[..., :, None], g[..., None, :]) for g in grids)
                               + (np.concatenate([-np.ones(K), np.ones(self.M)]), np.triu_indices(K)))
        return self._idx

    def unpack(self, y):
        K, M = self.K, self.M
        iu = self._index()[4]
        m = len(iu[0])
        Q = np.empty((K, K))
        Q[iu] = y[:m]
        Q[iu[1], iu[0]] = y[:m]
        return Q, np.asarray(y[m:], dtype=float).reshape(K, M)

    def pack(self, Q, R):
        return np.concatenate([np.asarray(Q, dtype=float)[self._index()[4]], np.asarray(R, dtype=float).ravel()])

    def cov(self, Q, R):
        K = self.K
        C = np.empty((K + self.M, K + self.M))
        C[:K, :K], C[:K, K:], C[K:, :K], C[K:, K:] = Q, R, np.asarray(R).T, self.T
        return C

    def rates(self, Q, R):
        """(dQ/dalpha, dR/dalpha): the documented ODE."""
        K = self.K
        C = self.cov(Q, R)
        ix3, ix4, _, s, _ = self._index()
        A3 = i3(C[ix3])                                              # (K, n, n): I3(i, b, u)
        dR = self.eta * A3[:, K:, :] @ s
        D = A3[:, :K, :] @ s
        dQ = self.eta * (D + D.T)
        if not self.truncated:
            A4 = i4(C[ix4])                                          # (K, K, n, n): I4(i, k, u, v)
            dQ = dQ + _PLANT["eta2"] * self.eta ** 2 * ((A4 @ s) @ s)
        return dQ, dR

    def vector_rate(self, y):
        Q, R = self.unpack(y)
        dQ, dR = self.rates(Q, R)
        return self.pack(dQ, dR)

    def eps(self, Q, R):
        """eps_g (SS NIPS Eq. (2))."""
        _, _, ix2, s, _ = self._index()
        return 0.5 * float(s @ i2(self.cov(Q, R)[ix2]) @ s)


def scm_flow(K, M, eta, teacher_kind="isotropic", truncated=False):
    bad = lambda v: isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 16
    if bad(K):
        raise Refusal("precondition:K", f"K={K!r}: need an int 1..16 (student hidden units)")
    if bad(M):
        raise Refusal("precondition:M", f"M={M!r}: need an int 1..16 (teacher hidden units)")
    if not (_real(eta) and eta > 0):
        raise Refusal("precondition:eta", f"eta={eta!r}: need a finite learning rate > 0")
    return ScmFlow(K, M, float(eta), teacher(teacher_kind, M), bool(truncated))


__all__ = ["ScmFlow", "i2", "i3", "i4", "quadrature", "scm_flow", "teacher"]
