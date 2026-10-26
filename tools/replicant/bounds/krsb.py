"""Guerra's K-step broken-replica bound P_K(beta, h; q, m) on the SK pressure, certified (E10).

Conventions (families/sk.py, families/parisi.py: J_ij ~ N(0, 1/N), field h, P = -beta f).  Order
parameter (Guerra, Comm. Math. Phys. 233, 1 (2003), arXiv:cond-mat/0205123, Eqs. (7)-(8), in the
package's labels): 0 <= q_0 <= q_1 <= ... <= q_K <= 1 and 0 < m_1 <= ... <= m_K <= 1, i.e.
x(q) = 0 on [0, q_0), m_k on [q_{k-1}, q_k), 1 on [q_K, 1].  Guerra's Eq. (12) (the recursion of
the antiparabolic equation (10)-(11), with y measured in units of the field and f(1, y) = ln cosh
beta y) in the package's variables Phi_k(y) = f(q_k, y / beta):
  Phi_K(y) = ln cosh y,  Phi_{k-1}(y) = (1/m_k) ln E_z exp(m_k Phi_k(y + s_k z)),
  s_k = beta sqrt(q_k - q_{k-1}),
and Eq. (14) with identity (26) (the segment [q_K, 1], x = 1, closes to beta^2 (1 - q_K)/2):
  P_K = ln 2 + (beta^2/4) [(1 - q_K)^2 - sum_k m_k (q_k^2 - q_{k-1}^2)]
        + E_z Phi_0(beta h + beta sqrt(q_0) z).
The bracket is families/sk.py's quadratic_term (tests: the sympy forms are equal); K = 0 is RS.
Guerra's Theorem 3: (1/N) E ln Z_N <= P_K for every N and every admissible (q, m).

Certification (nested acb.integral).  Write Z_k = exp(Phi_k); every level k < K is
Z_k(x)^{m_{k+1}} = E_z Z_{k+1}(x + s_{k+1} z)^{m_{k+1}}, truncated to |z| <= Y with the tail
(0 <= tail <= e^{m (|x| + C_{k+1})} 2 e^{m^2 s^2/2} P(z > Y - m s)) as a radius, where
1 <= Z_k(x) <= e^{|x| + C_k}, C_K = 0, C_{k-1} = C_k + (1/m_k) ln E e^{m_k s_k |z|}.  At a node
(a real point ball) the integrand is the accurate inner integral (recursively); on a complex box
(acb.integral's analyticity and magnitude bounds) it is a cheap rigorous enclosure: for w = x + iv,
|v| <= v0 < pi/2, Z_k(w) = Z_k(x) zeta with |arg zeta| <= v0 and c_k <= |zeta| <= 1, c_K = cos v0,
c_{k-1} = c_k cos(m_k v0)^{1/m_k} (cosh(x + iv) = cosh x (cos v + i tanh x sin v); a positive
average of points in the convex cone |arg| <= m v0 < pi/2 stays in it).  So Z_k is holomorphic
and zero-free on the strip, and the enclosure is Re in [c_k^m cos(m v0), e^{m(|x|max + C_k)}],
|Im| <= e^{m(|x|max + C_k)} sin(m v0); a box reaching |v| >= pi/2 gets a non-finite value
(acb.integral subdivides).  Outer level: 0 <= Phi_0(y) <= |y| + C_0 gives the tail.  Cost grows
as (nodes)^(K+1) (beta = 2, measured): K = 1 at 8 / 12 / 20 / 40 digits 0.10 / 0.26 / 0.54 / 3.3 s;
K = 2 at 12 / 20 digits 37 / 116 s.

  guerra_krsb_bound(beta, qs, ms, h=0, digits=20) -> Ball     qs = (q_0..q_K), ms = (m_1..m_K)
  guerra_1rsb_bound(beta, q0, q1, m, h=0, digits=20) -> Ball
  quadratic_term(qs, ms) -> sympy expression of the bracket (symbolic qs, ms)
  order_parameter(qs, ms) -> (tuple[Fraction], tuple[Fraction])  validated, exact
Parameters are taken as the exact rationals they denote (an mpf or a float is its exact binary
value).  _HOOKS["inner_exponent_scale"] (1): planted-error hook (the innermost cosh^{m} becomes
cosh^{scale m}); tests check the chain's functional comparison catches it.
"""
from fractions import Fraction
from functools import lru_cache

import mpmath as mp
import sympy as sp

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, certify, rational
from .gauss import abs_mgf, cutoff, phi, tail_prob, upper_tail

_HOOKS = {"inner_exponent_scale": 1}


def exact(x, name):
    """Fraction of x (int, Fraction, decimal string, float, mpf); refusals precondition:<name>."""
    if isinstance(x, mp.mpf):
        if not mp.isfinite(x):
            raise Refusal(f"precondition:{name}", f"{name} = {x}: need a finite number")
        sign, man, e, _ = x._mpf_
        v = Fraction(int(man)) * (Fraction(2) ** e)
        return -v if sign else v
    return rational(x, name, positive=False)


_DIGITS_MAX = 200


def check_digits(digits):
    """An int 1 .. 200 (certified decimal digits), else Refusal precondition:digits."""
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= _DIGITS_MAX:
        raise Refusal("precondition:digits", f"digits = {digits!r}: need an int 1 .. {_DIGITS_MAX} "
                      "(certified decimal digits)")
    return digits


def order_parameter(qs, ms):
    qs = tuple(exact(q, "order-parameter") for q in qs)
    ms = tuple(exact(m, "order-parameter") for m in ms)
    if len(qs) != len(ms) + 1:
        raise Refusal("precondition:levels", f"{len(qs)} overlaps and {len(ms)} Parisi parameters: "
                      "need q_0 .. q_K and m_1 .. m_K")
    if not (0 <= qs[0] and all(a <= b for a, b in zip(qs, qs[1:])) and qs[-1] <= 1):
        raise Refusal("precondition:order-parameter", f"q = {[str(q) for q in qs]}: Guerra's bound "
                      "needs 0 <= q_0 <= ... <= q_K <= 1 (Eq. (7))")
    if ms and not (0 < ms[0] and all(a <= b for a, b in zip(ms, ms[1:])) and ms[-1] <= 1):
        raise Refusal("precondition:order-parameter", f"m = {[str(m) for m in ms]}: need 0 < m_1 <= "
                      "... <= m_K <= 1 (x(q) nondecreasing, Eq. (7); m = 0 is not implemented)")
    return qs, ms


def quadratic_term(qs, ms):
    """(1 - q_K)^2 - sum_k m_k (q_k^2 - q_{k-1}^2), from Guerra's Eq. (14) and (26) (sympy)."""
    qs, ms = [sp.sympify(q) for q in qs], [sp.sympify(m) for m in ms]
    xq = sum(m * (qs[k + 1] ** 2 - qs[k] ** 2) / 2 for k, m in enumerate(ms)) + (1 - qs[-1] ** 2) / 2
    # ln 2 + f(0) - (beta^2/2) int q x dq, f(0) = E Phi_0 + beta^2 (1 - q_K)/2: bracket * beta^2/4
    return sp.expand(2 * (1 - qs[-1]) - 2 * xq)


def _quad_arb(qs, ms):
    q = [arb_q(x) for x in qs]
    m = [arb_q(x) for x in ms]
    return (1 - q[-1]) ** 2 - sum((m[k] * (q[k + 1] ** 2 - q[k] ** 2) for k in range(len(m))),
                                  start=0 * q[0])


class _Nested:
    """Phi_0 and its enclosures for one order parameter at the current flint precision."""

    def __init__(self, beta, h, qs, ms):
        from flint import arb, ctx
        self.K = len(ms)
        self.b, self.hb = arb_q(beta), arb_q(beta) * arb_q(h)
        self.m = [arb_q(m) for m in ms]
        self.s = [None] + [self.b * (arb_q(qs[k]) - arb_q(qs[k - 1])).sqrt()
                           for k in range(1, self.K + 1)]
        self.zero_s = [None] + [qs[k] == qs[k - 1] for k in range(1, self.K + 1)]
        self.a0 = self.b * arb_q(qs[0]).sqrt()
        self.zero_a0 = qs[0] == 0
        C = [arb(0)] * (self.K + 1)
        for k in range(self.K, 0, -1):
            C[k - 1] = C[k] + abs_mgf(self.m[k - 1] * self.s[k]).log() / self.m[k - 1]
        self.C = [arb(c.upper()) for c in C]
        self.halfpi = arb.pi() / 2
        self.node_rad = arb(2) ** (-(ctx.prec // 2))
        self.scale = arb_q(_HOOKS["inner_exponent_scale"])

    # ---- enclosures on complex boxes ------------------------------------------------------
    def _box_power(self, k, w, m):
        """Enclosure of Z_k(w)^m over the box w (k < K), or None off the strip."""
        from flint import acb, arb
        v0 = arb(abs(w.imag).upper())
        if not v0 < self.halfpi:
            return None
        xm = arb(abs(w.real).upper())
        c = v0.cos()
        for j in range(self.K, k, -1):
            c = c * (self.m[j - 1] * v0).cos() ** (1 / self.m[j - 1])
        mv = m * v0
        hi = (m * (xm + self.C[k])).exp()
        lo = c ** m * mv.cos()
        return acb(arb(lo.lower()).union(arb(hi.upper())), arb(0, (hi * mv.sin()).upper()))

    def _is_node(self, z):
        return z.imag.is_zero() and z.real.rad() < self.node_rad

    # ---- accurate values at real point balls ----------------------------------------------
    def log_Z(self, k, x):
        """Phi_k(x) for a real (narrow) ball x."""
        if k == self.K:
            return x.cosh().log()
        m = self.m[k]
        return self.mean_power(k, x).log() / m

    def mean_power(self, k, x):
        """E_z Z_{k+1}(x + s_{k+1} z)^{m_{k+1}} (= Z_k(x)^{m_{k+1}}), enclosed."""
        from flint import acb, arb
        m, s = self.m[k], self.s[k + 1]
        if self.zero_s[k + 1]:
            return (m * self.log_Z(k + 1, x)).exp()
        Y = cutoff(float((m * s).mid()))
        Yb = arb_q(Y)
        c = 1 / (2 * arb.pi()).sqrt()
        if k + 1 == self.K:
            me = m * self.scale

            def g(z, analytic):
                return c * (me * (x + s * z).cosh().log(analytic=True) - z * z / 2).exp()
        else:
            def g(z, analytic):
                w = x + s * z
                if self._is_node(z):
                    return acb(c * (m * self.log_Z(k + 1, w.real) - z.real * z.real / 2).exp())
                e = self._box_power(k + 1, w, m)
                if e is None:
                    return acb("nan")
                return c * (-z * z / 2).exp() * e
        val = acb.integral(g, -Yb, Yb).real
        T = (m * (abs(x) + self.C[k + 1])).exp() * 2 * (m * m * s * s / 2).exp() * upper_tail(Yb - m * s)
        t = arb(T.upper()) / 2
        return val + t + arb(0, t.upper())

    def outer(self):
        """E_z Phi_0(beta h + a_0 z), enclosed."""
        from flint import acb, arb
        if self.zero_a0:
            return self.log_Z(0, self.hb)
        a0, hb = self.a0, self.hb
        c = 1 / (2 * arb.pi()).sqrt()

        def f(z, analytic):
            w = a0 * z + hb
            if self._is_node(z):
                return acb(c * (-z.real * z.real / 2).exp() * self.log_Z(0, w.real))
            e = self._box_power(0, w, arb(1))
            if e is None:
                return acb("nan")
            return c * (-z * z / 2).exp() * e.log(analytic=True)
        Zb = arb_q(cutoff())
        if hb.is_zero():
            val = 2 * acb.integral(f, 0, Zb).real
        else:
            val = acb.integral(f, -Zb, Zb).real
        T = 2 * a0 * phi(Zb) + (abs(hb) + self.C[0]) * tail_prob(Zb)
        t = arb(T.upper()) / 2
        return val + t + arb(0, t.upper())


def krsb_pressure_arb(beta, h, qs, ms):
    """P_K at the current precision (an arb); RS when ms is empty."""
    from flint import arb
    nest = _Nested(beta, h, qs, ms)
    return arb(2).log() + nest.b ** 2 * _quad_arb(qs, ms) / 4 + nest.outer()


def guerra_krsb_bound(beta, qs, ms, h=0, digits=20) -> Ball:
    b = rational(beta, "beta")
    hq = exact(h, "h")
    qs, ms = order_parameter(qs, ms)
    return _cached(b, hq, qs, ms, check_digits(digits), Fraction(_HOOKS["inner_exponent_scale"]))


@lru_cache(maxsize=128)
def _cached(b, hq, qs, ms, digits, scale):
    """Memoised per exact (beta, h, q, m, digits) and the hook value (a planted run is its own key)."""
    return certify(lambda: krsb_pressure_arb(b, hq, qs, ms), digits, start_dps=digits + 8)


def guerra_1rsb_bound(beta, q0, q1, m, h=0, digits=20) -> Ball:
    return guerra_krsb_bound(beta, (q0, q1), (m,), h=h, digits=digits)


__all__ = ["check_digits", "exact", "guerra_1rsb_bound", "guerra_krsb_bound", "krsb_pressure_arb",
           "order_parameter", "quadratic_term"]
