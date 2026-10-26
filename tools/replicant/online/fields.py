"""Gaussian averages over the teacher and student local fields of an online perceptron.

Normalised fields: x = J . xi / |J| (student) and y = B . xi (teacher, |B| = 1) are standard
Gaussians with correlation rho = cos(theta).  The observed label b is sign(y) flipped with
probability `noise` (output noise, Copelli et al. Europhys. Lett. 37, 427 (1997) Eq. (7)).  For a
modulation f(x, b) the flow needs four averages (Averages):

  ay = <f b y>,  ax = <f b x>,  a2 = <f^2>,  drive = <f b (y - rho x)> = ay - rho ax

(`drive` is returned on its own so 1 - rho is never lost to cancellation near rho = 1).  The
overlap is passed as the angle theta = arccos(rho) for the same reason.

closed(rule, theta, noise, noise_est)   the named rules, whose f depends on Delta = b x only:
  hebb g = 1, perceptron g = Theta(-Delta), adatron g = -Delta Theta(-Delta): elementary
  quadrant moments of the bivariate normal (theta = arccos rho, s = sin theta), e.g.
  E[y; x < 0, y > 0] = (1 - rho)/(2 sqrt(2 pi)), E[x^2; x > 0, y > 0] = (pi - theta + rho s)/(2 pi),
  E[x y; x > 0, y > 0] = (rho (pi - theta) + s)/(2 pi), mixed over the label flip;
  optimal g = W*(Delta; rho, noise_est): the teacher field integrated analytically given Delta
  (y | x ~ N(rho x, s^2)), leaving a 1D integral over z = Delta / s done by Gauss-Legendre panels
  (geometric towards z -> -oo, where a mis-estimated noise makes the integrand wide), in log space.
quadrature(f, theta, noise)             any f(x, b) (vectorised): polar Gauss-Legendre over
  (x, w), y = rho x + s w, with the angular sectors cut where x = 0 and y = 0 (where a sign-
  dependent f jumps) and the label summed explicitly.  It shares no formula with closed().

_PLANT: private planted-error hook (tests only): "norm" is the sqrt(2 pi) of the Gaussian
normalisation in the elementary closed forms.
"""
import math
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr

from ..core import Refusal

Averages = namedtuple("Averages", "ay ax a2 drive")
CLOSED_RULES = ("hebb", "perceptron", "adatron", "optimal")
_PLANT = {"norm": math.sqrt(2 * math.pi)}


@lru_cache(maxsize=None)
def _leggauss(n):
    return np.polynomial.legendre.leggauss(n)


def _check(theta, noise, noise_est=None):
    if not (isinstance(theta, (int, float)) and 0 <= theta <= math.pi):
        raise Refusal("precondition:overlap", f"theta={theta!r}: need 0 <= theta <= pi")
    for name, v in (("noise", noise), ("noise_estimate", noise_est)):
        if v is not None and not (isinstance(v, (int, float)) and 0 <= v < 0.5):
            raise Refusal("precondition:noise", f"{name}={v!r}: a flip probability in [0, 1/2)")


def _x_minus_sin(x):
    """x - sin x without cancellation for small x."""
    if x > 1e-2:
        return x - math.sin(x)
    x2 = x * x
    return x * x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72)))


def closed(rule, theta, noise=0.0, noise_est=None):
    """Averages of a named rule's modulation per unit scale (see the module docstring)."""
    _check(theta, noise, noise_est)
    lam = float(noise)
    rho, s = math.cos(theta), math.sin(theta)
    one_m_rho = 2 * math.sin(theta / 2) ** 2
    c0 = 1 / _PLANT["norm"]
    if rule == "hebb":
        k = 2 * c0 * (1 - 2 * lam)                  # <b y> = (1 - 2 lam) E|y|
        return Averages(k, k * rho, 1.0, k * s * s)
    if rule == "perceptron":
        ax = -(1 - (1 - 2 * lam) * rho) * c0
        a2 = (1 - lam) * theta / math.pi + lam * (1 - theta / math.pi)
        return Averages((one_m_rho - 2 * lam) * c0, ax, a2, (1 - 2 * lam) * s * s * c0)
    if rule == "adatron":
        m_minus = _x_minus_sin(2 * theta) / 2       # theta - rho s = 2 pi E[x^2; x < 0, y > 0]
        m_plus = math.pi - m_minus                  # 2 pi E[x^2; x > 0, y > 0]
        ax = -((1 - lam) * m_minus + lam * m_plus) / math.pi
        drive = (1 - 2 * lam) * s ** 3 / math.pi
        return Averages(drive + rho * ax, ax, -ax, drive)
    if rule == "optimal":
        return _optimal(theta, lam, lam if noise_est is None else float(noise_est))
    raise Refusal("unsupported:rule", f"rule {rule!r}: closed forms exist for "
                  f"{', '.join(CLOSED_RULES)} (a callable rule uses fields.quadrature)")


def log_ghat(u, lam):
    """log(lam + (1 - 2 lam) Phi(u)) (vectorised), stable for u -> -oo."""
    lp = log_ndtr(u)
    if lam == 0:
        return lp
    return np.logaddexp(math.log(lam), math.log1p(-2 * lam) + lp)


def _panels(s, wide):
    """Gauss-Legendre nodes and weights in z on [-4 2^k, 12].  wide (est < lam): 4 2^k >= 12 / s,
    beyond which even the widest integrand (est = 0 < lam: ~ e^{-s^2 z^2 / 2}) is negligible.
    Otherwise 1 / G_est <= max(1/est, |u| + 1/|u|)-like growth at most, every integrand decays
    like e^{-z^2/2} times a polynomial and [-16, 12] suffices (k = 2)."""
    return _panels_k(max(0, math.ceil(math.log2(3.0 / s))) if wide else 2)


@lru_cache(maxsize=None)
def _panels_k(k):
    edges = [12.0, 4.0, 0.0] + [-4.0 * 2 ** j for j in range(k + 1)]
    x, w = _leggauss(24)
    a, b = np.array(edges[:-1]), np.array(edges[1:])
    mid, half = (a + b) / 2, (a - b) / 2
    z = (mid[:, None] - half[:, None] * x[None, :]).ravel()
    return z, (half[:, None] * w[None, :]).ravel() / (2 * math.pi)


def _optimal(theta, lam, est):
    """W*(Delta) = (s/rho)(1 - 2 est) phi(u) / G_est(u), u = rho Delta / s, G_l = l + (1 - 2 l) Phi.
    Given Delta: p(Delta) = 2 phi(Delta) G_lam(u), E[b y | Delta] = rho Delta +
    (1 - 2 lam) s phi(u) / G_lam(u); with Delta = s z, phi(Delta) phi(rho z) = e^{-z^2/2} / (2 pi)."""
    rho, s = math.cos(theta), math.sin(theta)
    if rho <= 0:
        raise Refusal("precondition:overlap", f"rho = {rho:.3g}: the optimal modulation is "
                      "defined for rho > 0 (it carries 1/rho)")
    if s == 0:
        return Averages(0.0, 0.0, 0.0, 0.0)
    z, w = _panels(s, est < lam)
    u = rho * z
    lgauss = -z * z / 2                              # the e^{-z^2/2}, kept in the exponents
    lphi = -u * u / 2 - 0.5 * math.log(2 * math.pi)
    g_lam, g_est = log_ghat(u, lam), log_ghat(u, est)
    k = 2 * s ** 3 / rho * (1 - 2 * est)
    if est == lam:      # matched: p(Delta) W* is even in Delta, so ax = 0 and drive = rho a2
        a2 = k * (1 - 2 * est) / rho * float(np.sum(w * np.exp(lgauss + lphi - g_est)))
        return Averages(rho * a2, 0.0, a2, rho * a2)
    ax = k * float(np.sum(w * z * np.exp(lgauss + g_lam - g_est)))
    drive = k * (1 - 2 * lam) * float(np.sum(w * np.exp(lgauss + lphi - g_est)))
    a2 = k * (1 - 2 * est) / rho * float(np.sum(w * np.exp(lgauss + lphi + g_lam - 2 * g_est)))
    return Averages(drive + rho * ax, ax, a2, drive)


@lru_cache(maxsize=64)
def _polar(theta, n_r, n_phi):
    rho, s = math.cos(theta), math.sin(theta)
    cuts = sorted({math.pi / 2, 3 * math.pi / 2} |
                  {(math.atan2(-rho, s) + k * math.pi) % (2 * math.pi) for k in (0, 1)})
    cuts = cuts + [cuts[0] + 2 * math.pi]
    xp, wp = _leggauss(n_phi)
    phis, wphi = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a > 1e-15:
            phis.append((a + b) / 2 + (b - a) / 2 * xp)
            wphi.append((b - a) / 2 * wp)
    phi, wphi = np.concatenate(phis), np.concatenate(wphi)
    xr, wr = _leggauss(n_r)
    r, wr = 7.0 * (xr + 1), 7.0 * wr                      # r in [0, 14]
    wr = wr * r * np.exp(-r * r / 2) / (2 * math.pi)
    x = np.outer(r, np.cos(phi)).ravel()
    w = np.outer(r, np.sin(phi)).ravel()
    return x, w, rho * x + s * w, np.outer(wr, wphi).ravel()


def quadrature(f, theta, noise=0.0, n_r=48, n_phi=32):
    """Averages of f(x, b) (vectorised over arrays x, b) by polar Gauss-Legendre."""
    _check(theta, noise)
    x, w, y, wt = _polar(float(theta), n_r, n_phi)
    s = math.sin(theta)
    out = np.zeros(4)
    for b, p in ((np.sign(y), 1 - noise), (-np.sign(y), noise)):
        if p == 0:
            continue
        fv = np.asarray(f(x, b), dtype=float) * np.ones_like(x)
        q = wt * p
        out += [np.sum(q * fv * b * y), np.sum(q * fv * b * x), np.sum(q * fv * fv),
                np.sum(q * fv * b * s * w)]
    return Averages(*(float(v) for v in out))


__all__ = ["Averages", "CLOSED_RULES", "closed", "log_ghat", "quadrature"]
