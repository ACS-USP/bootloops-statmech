"""Certified values for lattice/: the Ball type and the bridge to tools/baller.

Every certified number here is a python-flint `arb` ball produced through baller's documented
front door: `certify(thunk)` runs `thunk()` (a 0-argument function building an arb from exact
inputs) under `baller.run(thunk, dps=d)` and lets `baller.solve` escalate d until `digits`
decimal digits are certified (baller's 1-ulp display criterion).  Multi-output computations
(lattice/bethe.py, lattice/kaufman.py) set the working precision themselves through
`baller.hygiene.ctx_guard`, which restores flint's process-global ctx.prec.  `baller.verify()`
(the fail-closed integrity check of baller's vendored files) runs once before the first use.

  Ball(arb, digits)  .mid() / .rad() (mpf), .value_and_error(), .contains(x), .overlaps(other),
                     .render(digits), -ball;  rad() is an upper bound of the radius
  certify(thunk, digits=DIGITS) -> Ball
  rational(x, name) -> Fraction   (int, Fraction, decimal string; refuses bool / non-finite)
  arb_q(Fraction) -> arb          (exact)

A missing python-flint is the refusal unsupported:python-flint (INSTALL.md: the project venv); a
failed baller.verify() is unsupported:baller-integrity.  Nothing falls back to mpmath.
"""
import os
import sys
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

from ..core import Refusal

DIGITS = 40
_BALLER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "baller")
_STATE = {}


def baller():
    """The verified baller module (imported and verified once per process)."""
    if "baller" not in _STATE:
        try:
            import flint  # noqa: F401
        except ImportError as exc:
            raise Refusal("unsupported:python-flint", f"lattice/ needs python-flint (Arb) for its "
                          f"certified values; install it (INSTALL.md: the project venv) ({exc})")
        if _BALLER_DIR not in sys.path:
            sys.path.insert(0, _BALLER_DIR)
        import baller as b
        try:
            v = b.verify()
        except Exception as exc:              # VendorTamperError: fail closed
            raise Refusal("unsupported:baller-integrity", f"baller.verify() failed: {exc}")
        if v.get("vendor_bad") or v.get("vendor_unpinned"):
            raise Refusal("unsupported:baller-integrity", f"baller.verify(): {v}")
        _STATE["baller"] = b
    return _STATE["baller"]


def _arb():
    baller()
    from flint import arb
    return arb


def arb_q(q):
    from flint import fmpq
    q = Fraction(q)
    return _arb()(fmpq(q.numerator, q.denominator))


def _mpf(a, dps):
    """The arb's midpoint as an mpf at `dps` digits (via its decimal string)."""
    with mp.workdps(dps):
        return mp.mpf(a.mid().str(dps + 5, radius=False))


@dataclass(frozen=True, eq=False)
class Ball:
    arb: object            # flint.arb: a rigorous enclosure
    digits: int            # certified decimal digits (baller.certified_digits)

    def mid(self, dps=None):
        return _mpf(self.arb, dps or self.digits + 10)

    def rad(self):
        """An upper bound of the radius, as an mpf (rounded up)."""
        with mp.workdps(20):
            return mp.mpf(self.arb.rad().str(15, radius=False)) * (1 + mp.mpf("1e-12"))

    def value_and_error(self, dps=40):
        """(v, e): v an mpf at `dps` digits and e >= |v - x| for every x in the ball (the radius
        plus the rounding of the midpoint to v), rounded up."""
        v = self.mid(dps)
        with mp.workdps(dps + 10), baller().hygiene.ctx_guard(prec=max(self._prec(), 4 * dps + 64)):
            d = (self.arb - _arb()(mp.nstr(v, dps + 5))).abs_upper()
            e = mp.mpf(d.str(15, radius=False)) * (1 + mp.mpf("1e-12")) + abs(v) * mp.mpf(10)**(-dps - 3)
        with mp.workdps(20):
            return v, +e

    def contains(self, x):
        """x (Fraction, int, mpf or decimal string) inside the ball?  An mpf/str is taken as the
        exact binary/decimal number it denotes; x is converted at the ball's own precision."""
        with baller().hygiene.ctx_guard(prec=self._prec()):
            return bool(self.arb.contains(_to_arb(x)))

    def overlaps(self, other):
        with baller().hygiene.ctx_guard(prec=self._prec()):
            return bool(self.arb.overlaps(other.arb if isinstance(other, Ball) else _to_arb(other)))

    def _prec(self):
        return 4 * self.digits + 128

    def __neg__(self):
        return Ball(-self.arb, self.digits)

    def render(self, digits=None):
        return baller().render(self.arb, digits=digits)

    def __repr__(self):
        return f"Ball({self.render()} +/- {mp.nstr(self.rad(), 3)})"


def _to_arb(x):
    arb = _arb()
    if isinstance(x, (int, Fraction)):
        return arb_q(x)
    if isinstance(x, mp.mpf):
        with mp.workdps(max(mp.mp.dps, 60)):
            return arb(mp.nstr(x, mp.mp.dps + 5))
    return arb(str(x))


def ball_of(a):
    """Ball of an arb computed at the current precision (an exact ball counts as DIGITS + 100)."""
    import math
    d = baller().certified_digits(a)
    return Ball(a, DIGITS + 100 if math.isinf(d) else int(d))


def certify(thunk, digits=DIGITS):
    """Ball of thunk() with `digits` certified digits (baller.solve over baller.run)."""
    b = baller()
    try:
        res = b.solve(lambda dps: b.run(thunk, dps=dps), digits)
    except b.SolveRefused as exc:
        raise Refusal("no-solution:certification", f"baller.solve could not certify {digits} "
                      f"digits: {exc}")
    return Ball(res.ball, res.certified_digits)


def rational(x, name="beta", positive=True):
    """Fraction from an int / Fraction / decimal string / float (a float is the exact binary
    number it stores); bools, non-finite and non-numbers are refused precondition:<name>."""
    if isinstance(x, bool) or x is None:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a number")
    try:
        q = Fraction(x) if not isinstance(x, str) else Fraction(x.strip())
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a finite rational number "
                      "(int, Fraction or decimal string)")
    if positive and not q > 0:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need {name} > 0")
    return q


__all__ = ["Ball", "DIGITS", "arb_q", "ball_of", "baller", "certify", "rational"]
