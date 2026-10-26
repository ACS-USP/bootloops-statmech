"""Kaufman's exact partition function of the Lx x Ly torus, in ball arithmetic, and its exact
integer polynomial recovered from balls (the comparison with the transfer matrix, transfer.py).

Formula (Kaufman, Phys. Rev. 76, 1232 (1949), not fetched; as printed by Izmailian & Hu, Phys.
Rev. E 65 (2002), arXiv:cond-mat/0009024v2, Eqs. (10)-(16), and Kastner, arXiv:cond-mat/0605568,
Eqs. (6)-(8)), with M = Lx inside the hyperbolic functions and N = Ly factors:
  Z = (1/2) (2 sinh 2K)^{MN/2} (Z1 + Z2 + Z3 + Z4),
  Z1 = prod_{r=0}^{N-1} 2 cosh(M g_{2r+1}/2),  Z2 = prod 2 sinh(M g_{2r+1}/2),
  Z3 = prod_{r=0}^{N-1} 2 cosh(M g_{2r}/2),    Z4 = prod 2 sinh(M g_{2r}/2),
  cosh g_r = cosh^2 2K / sinh 2K - cos(r pi / N)  (g_r > 0, 0 < r < 2N),
  g_0 = 2K + ln tanh K  (signed: Kastner Eq. (8b), e^{g_0} = e^{K'} tanh(K'/2) with K' = 2K).
Sizes 2 are the torus multigraph (two bonds between the same pair), as in the transfer matrix.

Exactness (route (b) of the task notes).  P(x) = x^N Z with x = e^{-2K} and N = Lx Ly is, by the
definition of Z, an integer polynomial of degree <= 2N (c_j = number of configurations with j
unsatisfied bonds).  kaufman_polynomial evaluates Kaufman's formula in balls at the 2N + 1
rational nodes x_i = i/(2N + 2), solves the Vandermonde system in arb (arb_mat.solve: a rigorous
enclosure), and raises the precision until every coefficient ball has radius < 1/4; each must then
contain exactly one integer (else the refusal no-solution:not-an-integer-polynomial).  Given that
Kaufman's expression is a polynomial of degree <= 2N in x (its theorem, Z), equality of the
recovered integers with the transfer matrix's is an identity of the two routes; the tests also
compare Kaufman's ball with the exact polynomial at held-out rational points.

  kaufman_partition_function(Lx, Ly, beta, J=1) -> Ball   Z (beta rational, BETA_C, or XPoint)
  kaufman_free_energy(Lx, Ly, beta, J=1) -> Ball          ln Z / (Lx Ly)
  kaufman_polynomial(Lx, Ly, require_integers=True) -> Recovery(coefficients, max_radius, prec)
  beta_of_x(x) -> XPoint                                   beta through x = e^{-2K}
_HOOKS["gamma0"] (K -> g_0): planted-error hook (tests plant the unsigned |g_0|).
"""
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

from ..core import Refusal
from .ball import Ball, arb_q, baller, certify
from .coupling import XPoint, coupling

MAX_SIDE = 12          # kaufman_polynomial (degree 2 Lx Ly interpolation)
MAX_TORUS = 512        # kaufman_partition_function
_MAX_PREC = 1 << 15


def _gamma0(K):
    return 2 * K + K.tanh().log()


_HOOKS = {"gamma0": _gamma0}


def _sizes(Lx, Ly, top=MAX_SIDE):
    for name, L in (("Lx", Lx), ("Ly", Ly)):
        if isinstance(L, bool) or not isinstance(L, int) or not 2 <= L <= top:
            raise Refusal("precondition:size", f"{name} = {L!r}: need an int in 2 .. {top}")
    return Lx, Ly


def beta_of_x(x):
    return XPoint(Fraction(x))


def _z(M, N, K):
    """Kaufman's Z at the arb K (current precision)."""
    from flint import arb, fmpq
    s2, c2 = (2 * K).sinh(), (2 * K).cosh()
    base = c2 * c2 / s2

    def g(r):
        return _HOOKS["gamma0"](K) if r == 0 else (base - arb.cos_pi_fmpq(fmpq(r, N))).acosh()
    odd = [M * g(2 * r + 1) / 2 for r in range(N)]
    even = [M * g(2 * r) / 2 for r in range(N)]
    terms = []
    for args, fn in ((odd, "cosh"), (odd, "sinh"), (even, "cosh"), (even, "sinh")):
        p = arb(1)
        for a in args:
            p *= 2 * getattr(a, fn)()
        terms.append(p)
    pref = (arb(M * N) / 2 * (2 * s2).log()).exp()
    return pref * (terms[0] + terms[1] + terms[2] + terms[3]) / 2


def kaufman_partition_function(Lx, Ly, beta, J=1) -> Ball:
    M, N = _sizes(Lx, Ly, MAX_TORUS)
    c = coupling(beta, J)
    return certify(lambda: _z(M, N, c.arb()))


def kaufman_free_energy(Lx, Ly, beta, J=1) -> Ball:
    """ln Z / N of the Lx x Ly torus from Kaufman's formula."""
    M, N = _sizes(Lx, Ly, MAX_TORUS)
    c = coupling(beta, J)
    return certify(lambda: _z(M, N, c.arb()).log() / (M * N))


@dataclass(frozen=True)
class Recovery:
    coefficients: tuple      # c_0 .. c_2N (ints)
    max_radius: object       # mpf: the largest coefficient-ball radius
    prec: int                # bits used


def kaufman_polynomial(Lx, Ly, require_integers=True) -> Recovery:
    from flint import arb_mat
    M, N = _sizes(Lx, Ly)
    n = 2 * M * N + 1
    nodes = [Fraction(i, n + 1) for i in range(1, n + 1)]
    prec = 64 + 4 * n
    while prec <= _MAX_PREC:
        with baller().hygiene.ctx_guard(prec=prec):
            xs = [arb_q(x) for x in nodes]
            ys = [_z(M, N, -x.log() / 2) * x ** (M * N) for x in xs]
            V = arb_mat([[x ** j for j in range(n)] for x in xs])
            c = V.solve(arb_mat([[y] for y in ys]))
            balls = [c[j, 0] for j in range(n)]
            if all(b.is_finite() and b.rad() < 0.25 for b in balls):
                rad = max(b.rad() for b in balls)
                if require_integers:
                    ints = [b.unique_fmpz() for b in balls]
                    if any(i is None for i in ints):
                        bad = [j for j, i in enumerate(ints) if i is None]
                        raise Refusal("no-solution:not-an-integer-polynomial",
                                      f"{M} x {N} torus: coefficient balls {bad} contain no integer "
                                      f"(radius < 1/4 at {prec} bits)")
                    coeffs = tuple(int(i) for i in ints)
                else:
                    # the nearest integer to each midpoint (planted-error tests only)
                    coeffs = tuple(int((b.mid() + 0.5).floor().unique_fmpz()) for b in balls)
                with mp.workdps(20):
                    return Recovery(coeffs, mp.mpf(rad.str(10, radius=False)), prec)
        prec *= 2
    raise Refusal("no-solution:certification", f"{M} x {N} torus: coefficient radii not < 1/4 at "
                  f"{_MAX_PREC} bits")


__all__ = ["MAX_SIDE", "MAX_TORUS", "Recovery", "beta_of_x", "kaufman_free_energy",
           "kaufman_partition_function", "kaufman_polynomial"]
