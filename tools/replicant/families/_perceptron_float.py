"""float64 versions of the perceptron saddle functions, used ONLY to bracket
roots fast; every number returned by the public API is then polished and
audited in mpmath (perceptron.py, teacher_student.py).

Quadrature: composite 48-point Gauss-Legendre on the breakpoints, clipped to
|t| <= 12 (Gaussian tail < 1e-32).  H, E = phi/H and ln H switch to their
asymptotic (Mills-ratio) series for x >= 25, where erfc underflows; the
series is truncated after 1/x^8 (relative error < 1e-11 there).
"""
import math

import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)
_ERFC = np.frompyfunc(math.erfc, 1, 1)
_L = 12.0
_X_ASY = 25.0
_SQ2PI = math.sqrt(2 * math.pi)


def gauss(f, pts):
    """E_t f(t), t ~ N(0, 1); f vectorised over numpy arrays."""
    cuts = sorted({min(max(float(p), -_L), _L) for p in pts} | {-_L, _L})
    total = 0.0
    for a, b in zip(cuts, cuts[1:]):
        if b - a <= 0:
            continue
        t = (b - a) / 2 * _NODES + (a + b) / 2
        total += (b - a) / 2 * np.dot(_WEIGHTS, f(t) * np.exp(-t * t / 2) / _SQ2PI)
    return total


def _mills_series(x):          # H(x)/phi(x) * x for large x
    y = 1 / (x * x)
    return 1 - y + 3 * y**2 - 15 * y**3 + 105 * y**4


def lnH(x):
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, _X_ASY)
    direct = np.log(_ERFC(small / math.sqrt(2)).astype(float) / 2)
    big = np.maximum(x, _X_ASY)
    asy = -big * big / 2 - np.log(big * _SQ2PI) + np.log(_mills_series(big))
    return np.where(x < _X_ASY, direct, asy)


def E(x):
    """phi(x)/H(x)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, _X_ASY)
    direct = np.exp(-small * small / 2) / _SQ2PI / (_ERFC(small / math.sqrt(2)).astype(float) / 2)
    big = np.maximum(x, _X_ASY)
    return np.where(x < _X_ASY, direct, big / _mills_series(big))


def _t_pts(q, kappa):
    s, w = math.sqrt(q), math.sqrt(1 - q) / math.sqrt(q)
    t0 = kappa / s
    return [t0 - 30 * w, t0 - 3 * w, t0, t0 + 3 * w, t0 + 30 * w, 0.0]


def eh2(q, one_m_q, kappa):
    s, r = math.sqrt(q), math.sqrt(one_m_q)
    return gauss(lambda t: E((kappa - s * t) / r)**2, _t_pts(q, kappa))


def elnh(q, one_m_q, kappa):
    s, r = math.sqrt(q), math.sqrt(one_m_q)
    return gauss(lambda t: lnH((kappa - s * t) / r), _t_pts(q, kappa))


def spherical_alpha(q, kappa):
    return q / ((1 - q) * eh2(q, 1 - q, kappa))


def ising_q(qhat):
    g = math.sqrt(qhat)
    zp = [-30 / g, -3 / g, 0.0, 3 / g, 30 / g]
    sech2 = lambda y: 4 * np.exp(-2 * np.abs(y)) / (1 + np.exp(-2 * np.abs(y)))**2
    one_m_q = gauss(lambda z: sech2(g * z), zp)
    q = 1 - one_m_q if one_m_q < 0.5 else gauss(lambda z: np.tanh(g * z)**2, zp)
    return q, one_m_q, zp


def ising_alpha(qhat, kappa):
    q, one_m_q, _ = ising_q(qhat)
    return qhat * one_m_q / eh2(q, one_m_q, kappa)


def ising_entropy(qhat, kappa):
    g = math.sqrt(qhat)
    q, one_m_q, zp = ising_q(qhat)
    a = qhat * one_m_q / eh2(q, one_m_q, kappa)
    lncosh = gauss(lambda z: np.abs(g * z) + np.log1p(np.exp(-2 * np.abs(g * z))), zp)
    return -qhat * one_m_q / 2 + lncosh + a * elnh(q, one_m_q, kappa)


def gibbs_residual(u, alpha):
    """ln(R/sqrt(1-R)) - ln((alpha/pi) E_t[exp(-R t^2/2)/H(-sqrt(R) t)]), 1 - R = e^u."""
    d = math.exp(u)
    R = 1 - d
    s = math.sqrt(R)
    I = _SQ2PI * gauss(lambda t: E(-s * t), [-3.0, 0.0, 3.0])
    return math.log(R / math.sqrt(d)) - math.log(alpha / math.pi * I)
