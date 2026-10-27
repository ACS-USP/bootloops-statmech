"""Square-lattice Ising thermodynamics in zero field: Onsager's energy and specific heat, certified
off beta_c, the logarithmic divergence at beta_c, and independent routes.

Source read: Onsager, Phys. Rev. 65, 117 (1944) (scan of the journal pages, pp. 140-143).  J = 1
unless given, K = beta J (Onsager's H), per-site values u = U/N (units of J), c = C/(N k).
  k1 = 2 sinh 2K / cosh^2 2K,  k1'' = 2 tanh^2 2K - 1,  |k1''| = k1' = (1 - k1^2)^{1/2}    Eq. (114)
  u = -J coth 2K [1 + (2/pi) k1'' K1]                                                Eq. (116)
  c = (K coth 2K)^2 (2/pi) [2 K1 - 2 E1 - (1 - k1'') (pi/2 + k1'' K1)]               Eq. (117)
  K1 = K(k1), E1 = E(k1), modulus k1 (integrands (1 - k1^2 sin^2)^{-+1/2})          Eq. (118a)
  at beta_c (K_c = (1/2) ln cot(pi/8)): -u_c/J = 2^{1/2}                              Eq. (121)
  near beta_c: K1 ~ ln(4/k1') ~ ln(2^{1/2}/|K - K_c|),
               c ~ (2/pi) (ln cot(pi/8))^2 (K1 - 1 - pi/4)                           Eq. (120)
so c = -A ln|K - K_c| + O(1) on both sides with A = (2/pi) ln^2(1 + sqrt 2) = 0.4945 (cot(pi/8) =
1 + sqrt 2); Onsager's finite-width maximum, Eq. (127), 0.4945 log n + 0.1879, has the same slope.
K1 = pi / (2 agm(1, k1')) with k1' = |k1''| taken from Eq. (114) directly (no 1 - k1^2 cancellation);
E1 by Arb's acb.elliptic_e (parameter m = k1^2).

Independent route in this file (derived here, not from the source): u = -d(-beta f)/dK and
c = K^2 d^2(-beta f)/dK^2 of Onsager's free energy (lattice/onsager.py's single integral), with
-beta f = ln 2 + (1/2 pi) int_0^pi [acosh w + ln(S/2)] dt, w = (C^2 - S cos t)/S = g(K) - cos t,
g = S + 1/S (C = cosh 2K, S = sinh 2K): differentiating under the integral,
  d/dK  = C/S + (1/2 pi) int g' / sqrt(w^2 - 1),
  d2/dK2 = -2/S^2 + (1/2 pi) int [g'' / sqrt(w^2 - 1) - w g'^2 / (w^2 - 1)^{3/2}],
  g' = 2C (1 - 1/S^2),  g'' = 4S - 4/S + 8 C^2/S^3,
w^2 - 1 = (w - 1)(w + 1), w - 1 = [(1 - S)^2 + 2 S sin^2(t/2)]/S (cancellation-free), enclosed by
acb.integral (analytic off beta_c).

  onsager_energy(beta, J=1) -> Ball                (beta rational or BETA_C: -sqrt 2 J)
  onsager_specific_heat(beta, J=1) -> Ball         refused precondition:off-critical at BETA_C
  specific_heat_amplitude() -> Ball                A = (2/pi) ln^2(1 + sqrt 2)
  onsager_energy_integral(beta), onsager_specific_heat_integral(beta) -> Ball   the derivative route
  specific_heat_log_slope(d1, d2) -> Ball          [c(b2) - c(b1)] / ln(d1/d2), b_i = beta_c + d_i
                                                   (signed d_i, rational, |d_i| small): -> A
_HOOKS["amplitude_coefficient"] (2): planted-error hook of A's (coefficient / pi).
"""
from fractions import Fraction

from ..core import Refusal
from .ball import Ball, arb_q, baller, certify, rational
from .coupling import coupling

_HOOKS = {"amplitude_coefficient": 2}


def _k1(K):
    C, S = (2 * K).cosh(), (2 * K).sinh()
    T = S / C
    return 2 * S / (C * C), 2 * T * T - 1


def _ellk(kp):
    """K(k) from the complementary modulus k' > 0: pi / (2 agm(1, k'))."""
    from flint import arb
    return arb.pi() / (2 * arb(1).agm(kp))


def _u(K):
    from flint import arb
    _, kpp = _k1(K)
    return -(2 * K).cosh() / (2 * K).sinh() * (1 + 2 / arb.pi() * kpp * _ellk(abs(kpp)))


def _c(K):
    from flint import acb, arb
    k1, kpp = _k1(K)
    K1 = _ellk(abs(kpp))
    E1 = acb(k1 * k1).elliptic_e().real
    coth = (2 * K).cosh() / (2 * K).sinh()
    return (K * coth) ** 2 * 2 / arb.pi() * (2 * K1 - 2 * E1 - (1 - kpp) * (arb.pi() / 2 + kpp * K1))


def _amplitude():
    from flint import arb
    return _HOOKS["amplitude_coefficient"] / arb.pi() * (1 + arb(2).sqrt()).log() ** 2


def onsager_energy(beta, J=1) -> Ball:
    c = coupling(beta, J)
    Jq = rational(J, "J")
    if c.kind == "critical":
        from flint import arb
        return certify(lambda: -arb(2).sqrt())                  # Eq. (121)
    return certify(lambda: arb_q(Jq) * _u(c.arb()))


def onsager_specific_heat(beta, J=1) -> Ball:
    c = coupling(beta, J)
    if c.kind == "critical":
        raise Refusal("precondition:off-critical", "c diverges logarithmically at beta_c "
                      "(Onsager 1944, Eq. (120)): no value there; its amplitude is "
                      "specific_heat_amplitude()")
    return certify(lambda: _c(c.arb()))


def specific_heat_amplitude() -> Ball:
    return certify(_amplitude)


# ---- the derivative route --------------------------------------------------------------------
def _derivs(K, order):
    """(d/dK)^order of -beta f (order 1 or 2) by acb.integral of the differentiated integrand."""
    from flint import acb, arb
    C, S = (2 * K).cosh(), (2 * K).sinh()
    d2 = (1 - S) * (1 - S)
    g1 = 2 * C * (1 - 1 / (S * S))
    g2 = 4 * S - 4 / S + 8 * C * C / (S * S * S)

    def f(t, analytic):
        s = (t / 2).sin()
        wm = (d2 + 2 * S * s * s) / S                 # w - 1
        w = wm + 1
        q = wm * (wm + 2)                             # w^2 - 1
        r = q.sqrt(analytic=analytic)
        if order == 1:
            return g1 / r
        return g2 / r - w * g1 * g1 / (q * r)

    integral = acb.integral(f, 0, arb.pi()).real / (2 * arb.pi())
    return (C / S if order == 1 else -2 / (S * S)) + integral


def onsager_energy_integral(beta, J=1) -> Ball:
    c = _off(beta, J)
    Jq = rational(J, "J")
    return certify(lambda: -arb_q(Jq) * _derivs(c.arb(), 1))


def onsager_specific_heat_integral(beta, J=1) -> Ball:
    c = _off(beta, J)
    return certify(lambda: c.arb() ** 2 * _derivs(c.arb(), 2))


def _off(beta, J):
    c = coupling(beta, J)
    if c.kind == "critical":
        raise Refusal("precondition:off-critical", "the differentiated integrand is singular at "
                      "beta_c (w = 1 at t = 0)")
    return c


def specific_heat_log_slope(d1, d2, digits=30) -> Ball:
    """[c(q + d2) - c(q + d1)] / ln(d1/d2), q a rational within 1e-60 of beta_c (d1, d2 of the
    same sign): the coefficient A of -ln|beta - beta_c|, up to O(d ln d)."""
    d = [rational(x, "d", positive=False) for x in (d1, d2)]
    if not (d[0] * d[1] > 0 and abs(d[0]) != abs(d[1])):
        raise Refusal("precondition:d", f"d1 = {d1}, d2 = {d2}: need distinct nonzero d of one sign")
    from flint import arb
    with baller().hygiene.ctx_guard(prec=400):
        q = Fraction(((1 + arb(2).sqrt()).log() / 2).mid().str(70, radius=False))
    bs = [q + x for x in d]

    def thunk():
        cs = [_c(arb_q(b)) for b in bs]
        return (cs[1] - cs[0]) / (arb_q(d[0] / d[1])).log()
    return certify(thunk, digits)


def series_energy_specific_heat(beta, order=200, dps=50):
    """(u, c, last) from E8's exact high-temperature series (series/ising2d.py) truncated at v^order,
    v = tanh K: ln Z/N = ln 2 + 2 ln cosh K + g(v), dv/dK = 1 - v^2, so
      u = -[2v + (1 - v^2) g'(v)],  c = K^2 (1 - v^2) [2 - 2v g'(v) + (1 - v^2) g''(v)];
    `last` is the size of the order-th terms of g', g'' (|n a_n v^(n-1)| + |n^2 a_n v^(n-2)|), an
    indication of the truncation, not a bound.  mpf values (not certified); converges for
    v < sqrt 2 - 1 (beta < beta_c)."""
    import mpmath as mp
    from ..series.ising2d import ising2d_free_energy_series
    K = rational(beta, "beta")
    a = ising2d_free_energy_series(order)
    with mp.workdps(dps + 10):
        k = mp.mpf(K.numerator) / K.denominator
        v = mp.tanh(k)
        if not v < mp.sqrt(2) - 1:
            raise Refusal("precondition:series-convergence", f"beta = {beta}: v = tanh beta >= "
                          "sqrt 2 - 1, outside the series' disc of convergence")
        q = lambda x: mp.mpf(x.numerator) / x.denominator
        g1 = mp.fsum(n * q(a[n]) * v ** (n - 1) for n in range(1, order + 1) if a[n])
        g2 = mp.fsum(n * (n - 1) * q(a[n]) * v ** (n - 2) for n in range(2, order + 1) if a[n])
        w = 1 - v * v
        u = -(2 * v + w * g1)
        c = k * k * w * (2 - 2 * v * g1 + w * g2)
        n = order if a[order] else order - 1
        last = abs(n * q(a[n]) * v ** (n - 1)) + abs(n * n * q(a[n]) * v ** (n - 2))
    with mp.workdps(dps):
        return +u, +c, +last


__all__ = ["onsager_energy", "onsager_energy_integral", "onsager_specific_heat",
           "onsager_specific_heat_integral", "series_energy_specific_heat",
           "specific_heat_amplitude", "specific_heat_log_slope"]
