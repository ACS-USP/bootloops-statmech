"""Cardy's crossing formula for a rectangle (P2), two routes that share no code.

Cardy, arXiv:hep-th/9111026 (J. Phys. A 25, L201 (1992)), Eq. (8): the probability that a cluster joins
the boundary arcs (x1, x2) and (x3, x4) of the upper half plane is
    pi = 3 Gamma(2/3) / Gamma(1/3)^2  eta^(1/3) 2F1(1/3, 2/3; 4/3; eta),
    eta = (x4 - x3)(x2 - x1) / ((x3 - x1)(x4 - x2)),
and for a rectangle (his p. 8, the points at (-1/k, -1, 1, 1/k)) eta = ((1 - k)/(1 + k))^2 with
r = K(1 - k^2) / (2 K(k^2)), K(u) the complete elliptic integral in the parameter convention.
Geometry (derived here from the Schwarz-Christoffel map w = int dz / sqrt((1 - z^2)(1 - k^2 z^2))):
(-1, 1) maps to a side of length 2K(k^2), (1, 1/k) and (-1/k, -1) to the two sides of length
K(1 - k^2) at distance 2K(k^2).  So Cardy's r is (length of the two sides joined) / (their distance),
and pi(r) increases with r.  For the HORIZONTAL crossing (left side to right side, open cluster) of a
rectangle of width w and height h this is r = h / w; the vertical crossing is 1 - pi_h at criticality
(pi_h(r) + pi_h(1/r) = 1, the two forms of Eq. (8)).  Boundary conditions: free (open) on the whole
boundary; the event is a crossing by an occupied cluster.  The scaling limit; at finite size the
lattice crossing differs by O(L^-eps) (Smirnov, arXiv:0909.4499, Remark 3).

  cardy_ball(aspect)    certified (arb): 2r = K'/K is the modulus tau = 2 r i of the elliptic lambda
                        function, k^2 = lambda(2 r i) (acb.modular_lambda), then Eq. (8)'s first form
                        with arb's 2F1 (its second form, in 1 - eta = 4k/(1 + k)^2, when eta >= 1/2:
                        2F1 near z = 1 loses its enclosure); aspect = w / h (r = 1 / aspect)
  cardy_float(aspect)   second route (mpmath floats, no arb): k^2 from findroot on
                        K(1 - m) / (2 K(m)) = r, then the regularised incomplete beta I_eta(1/3, 1/3)
                        (= Eq. (8): (1/3) int_0^eta t^(-2/3) (1 - t)^(-2/3) dt / B(1/3, 1/3), the
                        Schwarz-Christoffel map of the half plane onto an equilateral triangle, i.e.
                        Smirnov's Corollary 3 in Carleson's form; identity derived here)
Aspect ratios outside [1/64, 64] are refused precondition:aspect-ratio.
_HOOKS (tests): "invert_aspect" False | True (planted: r = aspect instead of 1 / aspect).
"""
from fractions import Fraction

import mpmath as mp

from ..core import Refusal
from ..lattice.ball import arb_q, ball_of, baller

_HOOKS = {"invert_aspect": False}
LIMITS = (Fraction(1, 64), Fraction(64))


def aspect_ratio(a):
    if isinstance(a, bool) or a is None:
        raise Refusal("precondition:aspect-ratio", f"aspect = {a!r}: need a positive number")
    try:
        q = Fraction(a.strip()) if isinstance(a, str) else Fraction(a)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise Refusal("precondition:aspect-ratio", f"aspect = {a!r}: need a finite rational")
    if not LIMITS[0] <= q <= LIMITS[1]:
        raise Refusal("precondition:aspect-ratio", f"aspect = {a}: need 1/64 <= width/height <= 64")
    return q


def _r(aspect):
    q = aspect_ratio(aspect)
    return q if _HOOKS["invert_aspect"] else 1 / q


def cardy_ball(aspect, prec=256):
    """Horizontal crossing probability of a width/height = aspect rectangle (Cardy Eq. (8)), a Ball."""
    r = _r(aspect)
    baller()
    from flint import acb, arb
    with baller().hygiene.ctx_guard(prec=prec):
        m = acb(0, 2 * arb_q(r)).modular_lambda()
        if not abs(m.imag) < arb(10) ** -(prec // 5):
            raise Refusal("no-solution:certification", f"lambda(2 r i) = {m} is not real")
        k = m.real.sqrt()
        third = arb(1) / 3
        pref = 3 * (2 * third).gamma() / third.gamma() ** 2
        F = lambda z: pref * z ** third * z.hypgeom_2f1(third, 2 * third, 4 * third)
        eta = ((1 - k) / (1 + k)) ** 2
        if eta < arb(1) / 2:                    # Eq. (8), first form
            return ball_of(F(eta))
        return ball_of(1 - F(4 * k / (1 + k) ** 2))     # second form, 1 - eta = 4k/(1 + k)^2


def eta_of(aspect, dps=30):
    """Cardy's cross-ratio by the float route: m = k^2 = 1/(1 + e^-s) from findroot in s on
    K(1 - m) / (2 K(m)) = r, at a working precision that resolves m (or 1 - m) ~ 16 e^(-2 pi r)."""
    r = _r(aspect)
    with mp.workdps(dps + 10 + int(3 * max(r, 1 / r))):
        rr = mp.mpf(r.numerator) / r.denominator
        def f(s):
            e = mp.exp(-s)
            return mp.ellipk(e / (1 + e)) / (2 * mp.ellipk(1 / (1 + e))) - rr
        s0 = mp.log(16) - 2 * mp.pi * rr if rr >= 1 else mp.pi / (2 * rr) - mp.log(16)
        s = mp.findroot(f, s0, tol=mp.mpf(10) ** (-2 * dps - 10), verify=False)
        if not abs(f(s)) < mp.mpf(10) ** (-dps - 3):
            raise Refusal("no-solution:certification", f"K(1 - m)/(2 K(m)) = r not solved: |f| = {abs(f(s))}")
        k = mp.sqrt(1 / (1 + mp.exp(-s)))
        return ((1 - k) / (1 + k)) ** 2


def cardy_float(aspect, dps=30):
    """Second route: I_eta(1/3, 1/3) with eta from eta_of (mpmath)."""
    r = _r(aspect)
    with mp.workdps(dps + 10 + int(3 * max(r, 1 / r))):
        eta = eta_of(aspect, dps)
        out = mp.betainc(mp.mpf(1) / 3, mp.mpf(1) / 3, 0, eta, regularized=True)
    with mp.workdps(dps):
        return +out


__all__ = ["aspect_ratio", "cardy_ball", "cardy_float", "eta_of"]
