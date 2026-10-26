"""Certified Gaussian expectations E g(a z + b), z ~ N(0, 1), in ball arithmetic (E10).

Every function here returns a python-flint arb enclosing the exact expectation at the current flint
precision (callers run it under baller.run / hygiene.ctx_guard: bounds/guerra.py, bounds/krsb.py).
The integral is truncated to |z| <= Z and the truncated part is enclosed by acb.integral (Arb's
rigorous Gauss-Legendre integrator; acb_calc, Johansson, ICMS 2018, not fetched; baller's battery
leg L15); the tail |z| > Z is added as a radius from an analytic bound of g:
  ln cosh x       0 <= g <= |x|:   tail in [0, 2 a phi(Z) + |b| P(|z| > Z)]
  tanh^2 x        0 <= g <= 1:     tail in [0, P(|z| > Z)]
  (tanh^2)'' x    -2/3 <= g <= 2:  tail in [-(2/3) P, 2 P]    ((tanh^2)'' = 2 s^2 - 4 s (1 - s),
                                                               s = sech^2 x in (0, 1])
with phi the standard normal density and P(|z| > Z) = erfc(Z / sqrt 2).  The cutoff Z grows with
the precision so that the tail is below 2^-(prec + 16) (any Z is rigorous; Z only sets tightness).
ln cosh and tanh^2 have their singularities at Im(a z + b) = pi/2 (mod pi): the integrand checks
analyticity (log with analytic=True; tanh has poles, which Arb turns into non-finite balls).
At h = 0 (b exactly zero) the integrand is even and [0, Z] is integrated twice.

  expect(kind, a, b) -> arb      kind "lncosh" | "tanh2" | "d2tanh2"; a >= 0 (an arb, may be a
                                 ball, e.g. beta sqrt(Q) over an interval Q), b an arb
  phi(z), tail_prob(Z), abs_mgf(t) = E e^{t |z|} = 2 e^{t^2/2} Phi(t), cutoff(extra)
"""
import math
from fractions import Fraction

from ..lattice.ball import baller

_TAIL_BITS = 16


def _flint():
    baller()                      # verified once; refuses unsupported:python-flint without flint
    import flint
    return flint


def phi(z):
    """Standard normal density at z (arb or acb)."""
    arb = _flint().arb
    return (-z * z / 2).exp() / (2 * arb.pi()).sqrt()


def tail_prob(Z):
    """P(|z| > Z) = erfc(Z / sqrt 2)."""
    arb = _flint().arb
    return (arb(Z) / arb(2).sqrt()).erfc()


def upper_tail(t):
    """P(z > t) = erfc(t / sqrt 2) / 2."""
    arb = _flint().arb
    return (arb(t) / arb(2).sqrt()).erfc() / 2


def abs_mgf(t):
    """E e^{t |z|} = 2 e^{t^2/2} P(z > -t)."""
    return 2 * (t * t / 2).exp() * upper_tail(-t)


def cutoff(extra=0.0):
    """A rational Z with phi(Z) and P(|z| > Z) below ~2^-(prec + 16), shifted by `extra` >= 0
    (e.g. m s for a tilted measure).  Exact rational, so the integration path is exact."""
    prec = _flint().ctx.prec
    z = math.sqrt(2 * (prec + _TAIL_BITS) * math.log(2)) + float(extra) + 1
    return Fraction(math.ceil(4 * z), 4)


def _q(x):
    from flint import arb, fmpq
    x = Fraction(x)
    return arb(fmpq(x.numerator, x.denominator))


def _integrand(kind, a, b):
    from flint import acb
    c = 1 / (2 * _flint().arb.pi()).sqrt()

    def f(z, analytic):
        w = a * z + b
        if kind == "lncosh":
            g = w.cosh().log(analytic=analytic)
        else:
            t = w.tanh()
            if kind == "tanh2":
                g = t * t
            else:
                s = 1 - t * t
                g = 2 * s * s - 4 * s * (1 - s)
        return c * (-z * z / 2).exp() * g
    return f


def _point(kind, x):
    if kind == "lncosh":
        return x.cosh().log()
    t = x.tanh()
    if kind == "tanh2":
        return t * t
    s = 1 - t * t
    return 2 * s * s - 4 * s * (1 - s)


def expect(kind, a, b):
    """E g(a z + b) for the kinds above, enclosed (truncated acb.integral + rigorous tail)."""
    fl = _flint()
    arb, acb = fl.arb, fl.acb
    if kind not in ("lncosh", "tanh2", "d2tanh2"):
        raise ValueError(f"kind {kind!r}")
    if a.is_zero():
        return _point(kind, b)
    Z = cutoff()
    Zb = _q(Z)
    f = _integrand(kind, a, b)
    if b.is_zero():
        val = 2 * acb.integral(f, 0, Zb).real
    else:
        val = acb.integral(f, -Zb, Zb).real
    P = tail_prob(Zb)
    if kind == "lncosh":
        hi, lo = (2 * a * phi(Zb) + abs(b) * P), arb(0)
    elif kind == "tanh2":
        hi, lo = P, arb(0)
    else:
        hi, lo = 2 * P, -2 * P / 3
    return val + _interval(lo, hi)


def _interval(lo, hi):
    """An arb containing [lo, hi] (lo <= hi, arbs)."""
    arb = _flint().arb
    l, h = arb(lo.lower()), arb(hi.upper())
    return (l + h) / 2 + arb(0, ((h - l) / 2).upper())


__all__ = ["abs_mgf", "cutoff", "expect", "phi", "tail_prob", "upper_tail"]
