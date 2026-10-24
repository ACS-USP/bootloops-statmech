"""float64 numerics for the cavity route: Gaussian averages and root finders.

Written for the cavity route alone (it does not reuse solve.py or the replica
families' quadratures, so the cavity-vs-replica comparison also compares two
independent numerical implementations).

  nodes(cuts)        -> (z, w): E_z f(z) ~ w @ f(z), z ~ N(0, 1)
  kink_cuts(z0, w)   -> breakpoints around a step of width w at z0
  illinois(f, lo, hi)-> bracketed root (regula falsi, Illinois variant)
  newton2(F, x, ok)  -> 2-d Newton with a central-difference Jacobian

Quadrature: composite 20-point Gauss-Legendre on |z| <= 9 (Gaussian tail mass
2 Phi(-9) ~ 2e-19), panels no longer than 0.5 between the breakpoints.  The
tests compare against the replica side's mpmath quadratures at 1e-12 and better.
"""
import math

import numpy as np

_L = 9.0
_N = 20
_HMAX = 0.5
_X, _WX = np.polynomial.legendre.leggauss(_N)
_C = 1 / math.sqrt(2 * math.pi)
_K = (0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24)


def nodes(cuts=(), L=_L):
    pts = np.array(sorted({-L, L} | {float(c) for c in cuts if -L < c < L}))
    a, b = pts[:-1], pts[1:]
    k = np.maximum(1, np.ceil((b - a) / _HMAX)).astype(int)
    lo = np.concatenate([a[i] + (b[i] - a[i]) * np.arange(k[i]) / k[i] for i in range(len(a))])
    h = np.concatenate([np.full(k[i], (b[i] - a[i]) / k[i]) for i in range(len(a))])
    z = (lo[:, None] + h[:, None] * (_X[None, :] + 1) / 2).ravel()
    w = (h[:, None] / 2 * _WX[None, :]).ravel() * _C * np.exp(-z * z / 2)
    return z, w


def kink_cuts(z0, w):
    """Breakpoints z0 +- k w (k in _K) for an integrand that steps at z0 on scale w."""
    if not (math.isfinite(z0) and math.isfinite(w) and w > 0):
        return ()
    return tuple(z0 + s * k * w for k in _K for s in (1, -1))


def illinois(f, lo, hi, *, xtol=1e-15, maxit=300):
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if (flo > 0) == (fhi > 0):
        raise ValueError("no sign change")
    side = 0
    for _ in range(maxit):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        if not lo < x < hi:
            x = (lo + hi) / 2
        fx = f(x)
        if fx == 0:
            return x
        if (fx > 0) == (fhi > 0):
            hi, fhi = x, fx
            if side == -1:
                flo /= 2
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi /= 2
            side = 1
        if hi - lo < xtol * (1 + abs(lo)):
            return (lo + hi) / 2
    raise ValueError("no convergence")


def jacobian(F, x, eps=1e-6):
    x = np.asarray(x, float)
    J = np.empty((len(x), len(x)))
    for k in range(len(x)):
        d = np.zeros(len(x))
        d[k] = eps * max(1.0, abs(x[k]))
        J[:, k] = (np.asarray(F(x + d)) - np.asarray(F(x - d))) / (2 * d[k])
    return J


def newton2(F, x, ok, *, tol=1e-14, maxit=50):
    """Newton for F(x) = 0, step-halving to stay where ok(x); None if it fails."""
    x = np.asarray(x, float)
    for _ in range(maxit):
        try:
            g = np.asarray(F(x))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if not np.all(np.isfinite(g)):
            return None
        if np.abs(g).max() < tol:
            return x
        try:
            step = np.linalg.solve(jacobian(F, x), -g)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, OverflowError):
            return None
        lam = 1.0
        while not ok(x + lam * step):
            lam /= 2
            if lam < 1e-8:
                return None
        x = x + lam * step
    return x if np.abs(np.asarray(F(x))).max() < 1e2 * tol else None
