"""The square-lattice Ising model in zero field: Onsager's free energy and Yang's magnetization,
certified in ball arithmetic (python-flint arb through tools/baller; lattice/ball.py).

Conventions (Duminil-Copin, arXiv:2208.00864 (2022), Sec. 3; J = 1 there): K = beta J, values are
-beta f = lim ln Z / N.
  Onsager (Duminil-Copin Eq. (3.6)):
    -beta f = ln 2 + (1/(8 pi^2)) int_0^2pi int_0^2pi ln[cosh^2 2K - sinh 2K (cos t1 + cos t2)] dt1 dt2.
  The inner angle in closed form (int_0^2pi ln(a - b cos t) dt = 2 pi ln((a + sqrt(a^2 - b^2))/2),
  a >= |b|) gives the single integral this module certifies,
    -beta f = ln 2 + (1/(2 pi)) int_0^pi ln[(a(t) + sqrt(a(t)^2 - S^2))/2] dt,
    a(t) = C^2 - S cos t,  C = cosh 2K,  S = sinh 2K;
  a - S >= C^2 - 2S = (1 - S)^2, so off beta_c (S != 1) a^2 - S^2 > 0 on [0, pi]: the integrand
  is analytic on a neighbourhood of the path and Arb's rigorous integrator (acb.integral, acb_calc:
  Johansson, "Numerical integration in arbitrary-precision ball arithmetic", ICMS 2018, not
  fetched; baller's battery leg L15) returns an enclosure.  At beta_c (the sentinel BETA_C,
  K_c = (1/2) ln(1 + sqrt 2)) the
  value is the closed form (1/2) ln 2 + 2G/pi (G Catalan's constant; OEIS A242743, the bank).
  Yang (Duminil-Copin Eq. (3.7)): m*(beta) = (1 - sinh(2K)^-4)^(1/8) for beta >= beta_c.
  yang_magnetization refuses beta <= beta_c (precondition:ordered-phase: there m* = 0), decided by a
  certified comparison sinh 2K > 1 (a rational K is never K_c, so precision escalation decides).

  critical_beta(J=1) -> Ball                    (1/2) ln(1 + sqrt 2) / J
  onsager_free_energy(beta, J=1) -> Ball        -beta f (beta: rational or BETA_C)
  onsager_free_energy_mp(beta, J=1, dps=50, method="single" | "trapezoid", nodes=None) -> mpf
      independent mpmath evaluation: tanh-sinh on the single integral, or the periodic
      trapezoid rule on the double integral itself (exponentially convergent off beta_c)
  yang_magnetization(beta, J=1) -> Ball
  yang_loglog_slope(d1, d2) -> Ball             d ln m* / d ln(beta - beta_c) between beta_c + d1
      and beta_c + d2: a consistency check of the implemented formula (exponent 1/8 + O(d))
_HOOKS["catalan_coefficient"] (2): planted-error hook of the closed form; _HOOKS["naive_integrand"]
(False): the cancelling integrand, whose radius stalls near beta_c (tests: refused
no-solution:radius-stall, see lattice/ball.certify).
"""
from fractions import Fraction

import mpmath as mp

from ..core import Refusal
from .ball import DIGITS, Ball, arb_q, baller, certify, rational
from .coupling import BETA_C, coupling

_HOOKS = {"catalan_coefficient": 2, "naive_integrand": False}
_MAX_PREC = 1 << 13


def critical_beta(J=1):
    Jq = rational(J, "J")
    return certify(lambda: _kc() / arb_q(Jq))


def _kc():
    from flint import arb
    return (1 + arb(2).sqrt()).log() / 2


def _critical_closed_form():
    from flint import arb
    return arb(2).log() / 2 + _HOOKS["catalan_coefficient"] * arb.const_catalan() / arb.pi()


def _single_integral(K):
    """ln 2 + (1/(2 pi)) int_0^pi ln[(a + sqrt(a^2 - S^2))/2] dt, enclosed by acb.integral.

    Cancellation-free form: a - S = (1 - S)^2 + 2 S sin^2(t/2), a + S = (a - S) + 2S, so
    a^2 - S^2 = (a - S)(a + S) is never formed as a difference.  The naive a*a - S*S, evaluated on
    the complex boxes of the integrator, loses everything near beta_c (radius stuck at 4e-6 at
    beta = 0.4407, the integrator at its limits); this form certifies 40 digits in < 0.06 s from
    beta = 0.3 to within 1e-22 of beta_c (measured, tests/test_lattice.py)."""
    from flint import acb, arb
    S = (2 * K).sinh()
    d2 = (1 - S) * (1 - S)

    def f(t, analytic):
        if _HOOKS["naive_integrand"]:                  # planted: the cancelling form
            a = (2 * K).cosh() ** 2 - S * t.cos()
            return ((a + (a * a - S * S).sqrt(analytic=analytic)) / 2).log(analytic=analytic)
        s = (t / 2).sin()
        am = d2 + 2 * S * s * s
        root = (am * (am + 2 * S)).sqrt(analytic=analytic)
        return ((am + S + root) / 2).log(analytic=analytic)

    integral = acb.integral(f, 0, arb.pi())
    return arb(2).log() + integral.real / (2 * arb.pi())


def onsager_free_energy(beta, J=1) -> Ball:
    c = coupling(beta, J)
    if c.kind == "critical":
        return certify(_critical_closed_form)
    return certify(lambda: _single_integral(c.arb()))


def onsager_free_energy_mp(beta, J=1, dps=50, method="single", nodes=None):
    """-beta f by mpmath (no arb, no code shared with the ball routes)."""
    c = coupling(beta, J)
    with mp.workdps(dps + 10):
        K = mp.log(1 + mp.sqrt(2)) / 2 if c.kind == "critical" else \
            mp.mpf(c.K.numerator) / c.K.denominator
        C2, S = mp.cosh(2 * K)**2, mp.sinh(2 * K)
        if method == "single":
            def g(t):
                a = C2 - S * mp.cos(t)
                return mp.log((a + mp.sqrt(a * a - S * S)) / 2)
            val = mp.log(2) + mp.quad(g, [0, mp.pi / 2, mp.pi]) / (2 * mp.pi)
        elif method == "trapezoid":
            n = int(nodes or 120)
            cs = [mp.cos(2 * mp.pi * i / n) for i in range(n)]
            tot = mp.fsum(mp.log(C2 - S * (ci + cj)) for ci in cs for cj in cs)
            val = mp.log(2) + tot / (2 * n * n)
        else:
            raise Refusal("precondition:method", f"method = {method!r}: 'single' or 'trapezoid'")
    with mp.workdps(dps):
        return +val


def _ordered(K):
    """True iff sinh(2K) > 1 (beta > beta_c), decided by a certified comparison."""
    from flint import arb
    hyg = baller().hygiene
    prec = 128
    while prec <= _MAX_PREC:
        with hyg.ctx_guard(prec=prec):
            d = (2 * K()).sinh() - 1
            if d > 0:
                return True
            if d < 0:
                return False
        prec *= 2
    raise Refusal("no-solution:certification", "sinh(2K) - 1 undecided at 8192 bits")


def _yang(K):
    s = (2 * K).sinh()
    s2 = s * s
    return (1 - 1 / (s2 * s2)).root(8)


def yang_magnetization(beta, J=1) -> Ball:
    c = coupling(beta, J)
    if c.kind == "critical" or not _ordered(c.arb):
        raise Refusal("precondition:ordered-phase", f"beta = {beta!r} (J = {J}) is at or below beta_c "
                      "= (1/2) ln(1 + sqrt 2) / J: m* = 0 there (Duminil-Copin Eq. (3.7) holds for "
                      "beta >= beta_c; the paramagnet has no spontaneous magnetization)")
    return certify(lambda: _yang(c.arb()))


def yang_loglog_slope(d1, d2, digits=30) -> Ball:
    """[ln m*(b2) - ln m*(b1)] / [ln(b2 - beta_c) - ln(b1 - beta_c)], b_i = q + d_i with q a rational
    within 1e-60 of beta_c (so b_i are exact rationals and b_i - beta_c is a ball)."""
    d = [rational(x, "d") for x in (d1, d2)]
    with baller().hygiene.ctx_guard(prec=400):
        q = Fraction(_kc().mid().str(70, radius=False))
    bs = [q + x for x in d]

    def thunk():
        kc = _kc()
        ms = [_yang(arb_q(b)).log() for b in bs]
        ds = [(arb_q(b) - kc).log() for b in bs]
        return (ms[1] - ms[0]) / (ds[1] - ds[0])
    return certify(thunk, digits)


__all__ = ["BETA_C", "critical_beta", "onsager_free_energy", "onsager_free_energy_mp",
           "yang_loglog_slope", "yang_magnetization", "DIGITS"]
