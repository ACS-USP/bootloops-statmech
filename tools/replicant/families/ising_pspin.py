"""Family (b): Ising p-spin glass, 1RSB (Gardner, Nucl. Phys. B 257, 747 (1985)).

Convention (as Montanari & Ricci-Tersenghi, EPJB 33, 339 (2003), Eq. (2.1); Rizzo, PRE 88,
032135 (2013), Eq. (27)).  H = -sum_{i1<..<ip} J_{i1..ip} s_i1..s_ip, s_i = +-1,
J ~ N(0, p!/(2 N^(p-1))) (J = 1), zero field, so q0 = 0 (MRT Sec. 2).

  ising_pspin_1rsb(p, beta, m=None)         -> IsingPSpin1RSB (m=None: the static m)
  ising_pspin_critical_temperature(p, kind)  T_s (static, m_s = 1) or T_d (dynamic)
  minus_beta_f(p, beta, q, m)               the 1RSB functional below
  audit_residuals(p, beta, q, m)             the two equations by adaptive mpmath quadrature

1RSB functional (MRT Eq. (2.2) times -beta; q in blocks of size m, q0 = 0, n -> 0):
  -beta f = beta^2/4 [1 + (p-1)(1-m) q^p - p q^(p-1)] + ln 2 + (1/m) ln I,
  I = int Dz cosh^m(a z),   a^2 = beta^2 p q^(p-1) / 2   (the conjugate field variance).
Stationarity in q (MRT Eq. (2.3)): q = E_m[tanh^2(a z)], E_m the cosh^m-weighted Gaussian
average.  In m (Yeo & Moore, PRE 101, 032127 (2020), Eq. (16)):
  G(m) = beta^2 (p-1) q^p m^2 / 4 + ln I - m E_m[ln cosh(a z)] = 0.
Energy (MRT Eq. (2.5)): e = -(beta/2) [1 - (1-m) q^p].
At m = 1 the functional is the paramagnet beta^2/4 + ln 2 for every q.

Branches.  At fixed (beta, m) the q equation has q = 0 and, below the spinodal, two
nonzero roots; the larger is the 1RSB solution (Rizzo 2013, Sec. V: "only the one with a
larger value of q is important"); the smaller is an unstable fixed point of the map
q -> E_m[tanh^2] (its slope is > 1, reported in `slopes`).  All roots are returned.  The
static m is the root of G in (0, 1) on the larger branch; more than one is refused.
T_d is where the m = 1 nonzero roots appear (fold of the q equation); T_s where m_s = 1.

Numerics: float64 composite Gauss-Legendre (200 nodes on [0, m a + 14], the integrands
are even in z), scipy brentq; about 1e-12 absolute on the equations.  audit_residuals
re-evaluates both equations by mpmath.quad (an independent quadrature path).
Honest label: replica n -> 0 and 1RSB, physics-level; 1RSB stability toward further
RSB is families/gardner.py's.
"""
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

from ..core import Refusal

_X, _W = np.polynomial.legendre.leggauss(200)
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


@dataclass(frozen=True)
class IsingPSpin1RSB:
    p: int
    beta: object
    q0: object
    q1: object
    m: object
    f: object
    e: object
    q_roots: tuple     # every nonzero root of the q equation at this (beta, m)
    slopes: tuple      # d E_m[tanh^2]/dq at each root (> 1: unstable fixed point)
    note: str = ""


def _check(p, beta):
    if not (isinstance(p, int) and p >= 3):
        raise Refusal("precondition:p-at-least-3",
                      f"p = {p!r}: the Ising p-spin 1RSB entry needs an integer p >= 3")
    b = float(mp.mpf(beta))
    if not b > 0:
        raise Refusal("precondition:positive-beta", f"beta = {beta} must be > 0")
    return b


def _lcosh(x):
    return x + np.log1p(np.exp(-2 * x)) - np.log(2)


def block_average(a, m, g):
    """(ln I, E_m[g(t, x)]) with x = a z, t = tanh x, I = int Dz cosh^m(a z)."""
    L = abs(m) * a + 14.0
    z, w = (_X + 1) * L / 2, _W * L          # 2 * int_0^L (even integrands)
    lw = -z * z / 2 + m * _lcosh(a * z)
    c = lw.max()
    ww = w * np.exp(lw - c)
    s = ww.sum()
    x = a * z
    return np.log(s) + c - _LOG_SQRT_2PI, float((ww * g(np.tanh(x), x)).sum() / s)


def field(p, beta, q):
    return beta * np.sqrt(p * q**(p - 1) / 2)


def _t2(t, x):
    return t * t


def _sech2(t, x):
    e = np.exp(-2 * x)                 # x >= 0 on the nodes: no cancellation near t = 1
    return 4 * e / (1 + e)**2


def _qres(q, p, b, m):
    """E_m[tanh^2] - q, written (1 - q) - E_m[sech^2] (exact near q = 1)."""
    return (1 - q) - block_average(field(p, b, q), m, _sech2)[1]


def _q_peak(p, b, m):
    """(q*, residual(q*)): the maximum of E_m[tanh^2] - q (one hump: below the
    spinodal it is positive and the two nonzero roots sit on either side)."""
    r = minimize_scalar(lambda q: -_qres(q, p, b, m), bounds=(1e-4, 1 - 1e-10),
                        method="bounded", options={"xatol": 1e-13})
    return r.x, -r.fun


def _q_roots(p, b, m):
    qs, top = _q_peak(p, b, m)
    if not top > 0:
        return ()
    lo = brentq(_qres, 1e-4, qs, args=(p, b, m), xtol=1e-15) if _qres(1e-4, p, b, m) < 0 else None
    # the larger root in u = 1 - q (it may sit within 1e-14 of q = 1 at low T)
    g = lambda v: _qres(1 - np.exp(v), p, b, m)          # v = ln(1 - q)
    vmin = np.log(1e-200)
    if not g(vmin) < 0:
        raise Refusal("no-solution:bracket", f"beta = {b}, m = {m}: the larger q root is within "
                      "1e-200 of q = 1 (not resolved in float64)")
    v = brentq(g, vmin, np.log(1 - qs), xtol=1e-14)
    return tuple(x for x in (lo, 1 - np.exp(v)) if x is not None)


def _slope(q, p, b, m, h=1e-6):
    return (_qres(q + h, p, b, m) - _qres(q - h, p, b, m)) / (2 * h) + 1


def _gm(m, p, b, q):
    logI, lc = block_average(field(p, b, q), m, lambda t, x: _lcosh(x))
    return b * b * (p - 1) * q**p * m * m / 4 + logI - m * lc


def _gm_branch(m, p, b):
    rs = _q_roots(p, b, m)
    return _gm(m, p, b, max(rs)) if rs else float("nan")


def _f(p, b, q, m):
    logI = block_average(field(p, b, q), m, _t2)[0]
    return b * b / 4 * (1 + (p - 1) * (1 - m) * q**p - p * q**(p - 1)) + np.log(2) + logI / m


def minus_beta_f(p, beta, q, m):
    return mp.mpf(_f(p, float(mp.mpf(beta)), float(mp.mpf(q)), float(mp.mpf(m))))


def _static_m(p, b):
    grid = np.linspace(1.0, 0.01, 21)
    vals = [(m, _gm_branch(m, p, b)) for m in grid]
    vals = [(m, v) for m, v in vals if v == v]
    roots = [brentq(_gm_branch, m1, m0, args=(p, b), xtol=1e-14)
             for (m0, v0), (m1, v1) in zip(vals, vals[1:]) if v0 * v1 < 0]
    if len(roots) > 1:
        raise Refusal("ambiguous:several-saddles",
                      f"beta = {b}: the m equation has {len(roots)} roots in (0, 1) on the "
                      f"larger-q branch: {roots}")
    if not roots:
        raise Refusal("no-solution:bracket", f"beta = {b}: no root of the m equation found "
                      "on a 21-point grid in [0.01, 1] (larger-q branch)")
    return roots[0]


def static_point(p, b, seed):
    """(q, m) of the static 1RSB by Newton (scipy hybr) from `seed` = (q, m): the fast
    path for scans; None unless converged to |residual| < 1e-12 with 0 < m < 1 and q on
    the larger branch (slope of the q map < 1).  Callers fall back to the full solve."""
    F = lambda x: [_qres(x[0], p, b, x[1]), _gm(x[1], p, b, x[0])]
    r = root(F, list(seed), method="hybr", options={"xtol": 1e-14})
    q, m = r.x
    if not (0 < q < 1 and 0 < m < 1 and max(map(abs, F(r.x))) < 1e-12):
        return None
    return (q, m) if _slope(q, p, b, m) < 1 else None


def _fold(p, b):
    """max_q of the m = 1 q residual (> 0 iff nonzero roots exist)."""
    return _q_peak(p, b, 1.0)[1]


@lru_cache(maxsize=None)
def _tcrit(p, kind):
    lo, hi = 0.05, 3.0
    bd = brentq(lambda b: _fold(p, b), 1 / hi, 1 / lo, xtol=1e-14)
    if kind == "dynamic":
        return 1 / bd
    td = 1 / bd
    g = lambda T: _gm(1.0, p, 1 / T, max(_q_roots(p, 1 / T, 1.0)))
    return brentq(g, 0.5 * td, td * (1 - 1e-7), xtol=1e-14)


def ising_pspin_critical_temperature(p, kind="static"):
    """T_s (m_s = 1) or T_d (the m = 1 nonzero q roots appear), J = 1."""
    _check(p, 1)
    if kind not in ("static", "dynamic"):
        raise Refusal("precondition:known-transition",
                      f"kind = {kind!r}: use 'static' (T_s) or 'dynamic' (T_d)")
    return mp.mpf(_tcrit(p, kind))


def ising_pspin_1rsb(p, beta, m=None) -> IsingPSpin1RSB:
    b = _check(p, beta)
    if m is None:
        ts = _tcrit(p, "static")
        if 1 / b >= ts:
            raise Refusal("no-solution", f"T = {1 / b:.8g} >= T_s = {ts:.8g}: paramagnet "
                          "(q = 0, -beta f = beta^2/4 + ln 2); the static 1RSB has m >= 1")
        mm, how = _static_m(p, b), "static m (root of the m equation)"
    else:
        mm, how = float(mp.mpf(m)), "m given"
        if not 0 < mm <= 1:
            raise Refusal("precondition:m-range", f"m = {m}: need 0 < m <= 1")
    rs = _q_roots(p, b, mm)
    if not rs:
        raise Refusal("no-solution", f"beta = {b:.8g}, m = {mm:.8g}: no nonzero root of the "
                      "q equation (beyond the spinodal)")
    q = max(rs)
    e = -(b / 2) * (1 - (1 - mm) * q**p)
    return IsingPSpin1RSB(p, mp.mpf(b), mp.mpf(0), mp.mpf(q), mp.mpf(mm),
                          mp.mpf(-_f(p, b, q, mm) / b), mp.mpf(e), tuple(map(mp.mpf, rs)),
                          tuple(mp.mpf(_slope(r, p, b, mm)) for r in rs),
                          f"{how}; larger of {len(rs)} nonzero q roots (Rizzo 2013, Sec. V)")


def audit_residuals(p, beta, q, m):
    """(E_m[tanh^2] - q, G(m)) by mpmath.quad at 25 digits: an evaluation path that shares
    no code with block_average."""
    with mp.workdps(25):
        b, q, m = mp.mpf(beta), mp.mpf(q), mp.mpf(m)
        a = b * mp.sqrt(p * q**(p - 1) / 2)
        lw = lambda z: -z * z / 2 + m * mp.log(mp.cosh(a * z))
        pts = [0, m * a, m * a + 6, m * a + 40]
        I = lambda g: 2 * mp.quad(lambda z: mp.exp(lw(z)) * g(z), pts)
        i0 = I(lambda z: 1)
        t2 = I(lambda z: mp.tanh(a * z)**2) / i0
        lc = I(lambda z: mp.log(mp.cosh(a * z))) / i0
        g = b * b * (p - 1) * q**p * m * m / 4 + mp.log(i0 / mp.sqrt(2 * mp.pi)) - m * lc
        return (t2 - q, g)
