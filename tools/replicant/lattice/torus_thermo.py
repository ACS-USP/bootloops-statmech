"""Energy and specific heat of the finite Lx x Ly torus: K-derivatives of the exact partition
function, two ways, in ball arithmetic (Taylor coefficients by python-flint arb_series).

  (a) the exact integer polynomial (transfer.py: Z = e^{2NK} P(x), x = e^{-2K}, N = Lx Ly), and
  (b) Kaufman's formula (kaufman.py's _z, rewritten on arb_series: sinh, cosh, tanh by exp,
      acosh y = ln(y + sqrt(y^2 - 1)); g_0 = 2K + ln tanh K signed).
Both give ln Z(K0 + eps) as a series; with J = 1, u_L = -(1/N) [eps^1], c_L = K0^2 (1/N) 2! [eps^2].
(a) is exact for any size the transfer matrix reaches (Ly <= 7, Lx <= 16); (b) reaches sides <=
512.  Kaufman's Z equals the polynomial (kaufman.py, tests), so (a) and (b) must overlap; the
L -> oo sequence approaches Onsager's u, c off beta_c (a finite-size sequence: no bound on the gap
is claimed), and at beta_c c_L grows like A ln L (Onsager Eq. (127): the same slope 0.4945 for his
finite-width crystal; for the torus this is a consistency check of the amplitude, labelled so).

  torus_thermo_polynomial(Lx, Ly, beta) -> (Ball u_L, Ball c_L)
  torus_thermo_kaufman(Lx, Ly, beta) -> (Ball u_L, Ball c_L)      (beta rational or BETA_C)
  torus_thermo_moments(Lx, Ly, beta) -> (Ball u_L, Ball c_L)      <E>/N and K^2 Var(E)/N over the
      integer coefficients: no derivative extraction, so it checks _thermo (and its hook)
_HOOKS["derivative_order"] (2): the order whose Taylor coefficient makes c_L (planted error: 1).
"""
from math import factorial

from ..core import Refusal
from .ball import Ball, baller, ball_of
from .coupling import coupling
from .kaufman import MAX_TORUS
from .transfer import MAX_LX, MAX_LY, torus_partition_polynomial

_HOOKS = {"derivative_order": 2}
_LEN = 4


def _series(K0):
    from flint import arb_series
    return arb_series([K0, 1], prec=_LEN)


def _cosh(s):
    return (s.exp() + (-s).exp()) / 2


def _sinh(s):
    return (s.exp() - (-s).exp()) / 2


def _thermo(lnz, K0, N):
    """(u, c) per site from the series ln Z(K0 + eps)."""
    co = lnz.coeffs()
    o = _HOOKS["derivative_order"]
    return -co[1] / N, K0 ** o * factorial(o) * co[o] / N


def _lnz_polynomial(cs, K):
    x = (-2 * K).exp()
    p = 0 * x
    for c in reversed(cs):
        p = p * x + c
    N = (len(cs) - 1) // 2
    return 2 * N * K + p.log()


def _lnz_kaufman(M, N, K):
    from flint import arb, arb_series, fmpq
    s2, c2 = _sinh(2 * K), _cosh(2 * K)
    base = c2 * c2 / s2
    e2 = (2 * K).exp()
    g0 = 2 * K + ((e2 - 1) / (e2 + 1)).log()            # 2K + ln tanh K (signed)

    def g(r):
        if r == 0:
            return g0
        y = base - arb.cos_pi_fmpq(fmpq(r, N))
        return (y + (y * y - 1).sqrt()).log()
    terms = []
    for par in (1, 0):
        args = [M * g(2 * r + par) / 2 for r in range(N)]
        for fn in (_cosh, _sinh):
            p = arb_series([1], prec=_LEN)
            for a in args:
                p = p * (2 * fn(a))
            terms.append(p)
    tot = terms[0] + terms[1] + terms[2] + terms[3]
    return (M * N) * (2 * s2).log() / 2 + (tot / 2).log()


def _run(beta, build, N, prec):
    c = coupling(beta)
    with baller().hygiene.ctx_guard(prec=prec):
        K0 = c.arb()
        u, cl = _thermo(build(_series(K0)), K0, N)
        return ball_of(u), ball_of(cl)


def _side(name, L, top):
    if isinstance(L, bool) or not isinstance(L, int) or not 2 <= L <= top:
        raise Refusal("precondition:size", f"{name} = {L!r}: need an int in 2 .. {top}")
    return L


def torus_thermo_polynomial(Lx, Ly, beta, prec=320):
    Lx, Ly = _side("Lx", Lx, MAX_LX), _side("Ly", Ly, MAX_LY)
    cs = torus_partition_polynomial(Lx, Ly)
    return _run(beta, lambda K: _lnz_polynomial(cs, K), Lx * Ly, prec)


def torus_thermo_moments(Lx, Ly, beta, prec=320):
    """(u_L, c_L) as energy moments over the integer coefficients, with no K-derivative and no
    shared code with _thermo: weight w_j = c_j x^j of the configurations with j unsatisfied bonds,
    energy E_j = -(2N - j) + j = 2j - 2N (J = 1), u_L = <E>/N, c_L = K^2 <(E - <E>)^2>/N."""
    Lx, Ly = _side("Lx", Lx, MAX_LX), _side("Ly", Ly, MAX_LY)
    cs = torus_partition_polynomial(Lx, Ly)
    N = Lx * Ly
    c = coupling(beta)
    with baller().hygiene.ctx_guard(prec=prec):
        K0 = c.arb()
        x = (-2 * K0).exp()
        w = [cj * x ** j for j, cj in enumerate(cs) if cj]
        E = [2 * j - 2 * N for j, cj in enumerate(cs) if cj]
        Z = sum(w[1:], w[0])
        mean = sum((wi * e for wi, e in zip(w, E)), 0 * x) / Z
        var = sum((wi * (e - mean) * (e - mean) for wi, e in zip(w, E)), 0 * x) / Z
        return ball_of(mean / N), ball_of(K0 * K0 * var / N)


def torus_thermo_kaufman(Lx, Ly, beta, prec=None):
    M, N = _side("Lx", Lx, MAX_TORUS), _side("Ly", Ly, MAX_TORUS)
    return _run(beta, lambda K: _lnz_kaufman(M, N, K), M * N, prec or 256 + 2 * M)


__all__ = ["torus_thermo_kaufman", "torus_thermo_moments", "torus_thermo_polynomial"]
