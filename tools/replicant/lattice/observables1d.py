"""1D Ising observables in a field, certified, and the exact finite ring with its N -> oo limit.

Conventions (Baxter, Exactly Solved Models in Statistical Mechanics (1982), Ch. 2, read: the ANU
PDF, pp. 32-38): K = beta J, H = beta h (Baxter's h), J > 0; per-site values of the infinite chain.
  m      = e^K sinh H / [e^{2K} sinh^2 H + e^{-2K}]^{1/2}                     Baxter Eq. (2.1.18)
  chi    = dm/dh = beta e^{-K} cosh H / [e^{2K} sinh^2 H + e^{-2K}]^{3/2}     (1.7.17) applied to (2.1.18):
           the derivative is taken here (h = 0: beta e^{2K}, arXiv:2511.15399 Eq. (15))
  g(r)   = <s_0 s_r> - m^2 = sin^2(2 phi) (lambda_2/lambda_1)^r,  cot 2 phi = e^{2K} sinh H   (2.2.9), (2.2.14)
  xi     = 1 / ln(lambda_1 / lambda_2)                                        (2.2.15)
  chi_fl = beta sum_{r in Z} g(|r|) = beta sin^2(2 phi) (1 + rho)/(1 - rho)   (1.7.30) with (2.2.14),
           rho = lambda_2/lambda_1: a second closed form of chi (the geometric sum is done here)
lambda_{1,2} = e^K cosh H +- (e^{2K} sinh^2 H + e^{-2K})^{1/2} (2.1.17); transfer.py's _lambdas.

The ring (independent: only the definition Z_N = Tr V^N, V(s, s') = exp[K s s' + H (s + s')/2],
Baxter (2.1.7)-(2.1.10), (2.2.7)).  With a = e^K and b = e^H the ring weights are a^{s s'} b^{s}, so at
rational (a, b) every ring quantity is an exact rational: d/dH enters through truncated series in
eps (V(H + eps): e^{+-eps} = 1 +- eps + eps^2/2), giving
  m_N = (1/N) d ln Z_N/dH,  chi_N = beta (1/N) d^2 ln Z_N/dH^2,  G_N(r) = <s_0 s_r>_N - m_N^2.
The same code runs on arb balls at the run's rational beta, h (a = e^K, b = e^H as balls).

The N -> oo limit, explicit (derived here from Z_N = lambda_1^N + lambda_2^N, Baxter (2.1.15), and the
eigenvector form (2.2.11); l = ln(lambda_2/lambda_1) as a function of H, q = rho^N):
  m_N - m       = q l' / (1 + q)                                  |.| <= q |l'|
  (chi_N - chi)/beta = N q l'^2/(1 + q)^2 + q l''/(1 + q)         |.| <= q (N l'^2 + |l''|)
  G_N(r) - g(r) = (1 - m^2) rho^{N-r} (1 - rho^{2r})/(1 + q) + (m - m_N)(m + m_N)
                                                                 |.| <= rho^{N-r} + 2 q |l'|
l', l'' and the m inside the corrections are Taylor coefficients of ln lambda_{1,2}(H) by arb_series
(never the closed forms above), so a planted closed form cannot move its own check.

  ising1d_magnetization(beta, h, J=1), ising1d_susceptibility(...), ising1d_susceptibility_fluctuation(...),
  ising1d_correlation(beta, h, r, J=1), ising1d_correlation_length(beta, h, J=1) -> Ball
  ring_observables(N, a, b, r) -> RingObs (exact Fractions at a = e^K, b = e^H rational)
  ring_brute(N, a, b, r) -> RingObs       (N <= 12: sums over the 2^N configurations)
  ring_check(N, beta, h, r, J=1) -> RingCheck   balls at rational beta, h: ring, closed, correction, bound
_HOOKS["magnetization" | "susceptibility" | "correlation" | "correlation_length"]: the closed forms
(arb K, H [, beta | r]) -> arb, planted-error hooks (tests).
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from ..core import Refusal
from .ball import DIGITS, Ball, arb_q, baller, ball_of, rational
from .ball import certify as _certify
from .transfer import _lambdas


def _d(K, H):
    sh = H.sinh()
    return (2 * K).exp() * sh * sh + (-2 * K).exp()


def _m(K, H):
    return K.exp() * H.sinh() / _d(K, H).sqrt()


def _chi(K, H, beta):
    D = _d(K, H)
    return beta * (-K).exp() * H.cosh() / (D * D.sqrt())


def _sin2(K, H):
    """sin^2(2 phi) from cot 2 phi = e^{2K} sinh H (Baxter (2.2.9))."""
    c = (2 * K).exp() * H.sinh()
    return 1 / (1 + c * c)


def _rho(K, H):
    lp, lm = _lambdas(K, H)
    return lm / lp


def _corr(K, H, r):
    return _sin2(K, H) * _rho(K, H) ** r


def _xi(K, H):
    lp, lm = _lambdas(K, H)
    return 1 / (lp / lm).log()


def _chi_fluct(K, H, beta):
    rho = _rho(K, H)
    return beta * _sin2(K, H) * (1 + rho) / (1 - rho)


_HOOKS = {"magnetization": _m, "susceptibility": _chi, "correlation": _corr,
          "correlation_length": _xi}


def certify(thunk):
    """ball.certify; an exact ball (m = 0 at h = 0: infinitely many certified digits) is given
    DIGITS + 100 digits, as ball.ball_of does, so its printing precision stays finite."""
    b = _certify(thunk)
    return b if b.digits != float("inf") else Ball(b.arb, DIGITS + 100)


def _kh(beta, h, J):
    b = rational(beta, "beta")
    return b, b * rational(J, "J"), b * rational(h, "h", positive=False)


def _r(r, top=10000):
    if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r <= top:
        raise Refusal("precondition:distance", f"r = {r!r}: need an int in 0 .. {top}")
    return r


def ising1d_magnetization(beta, h, J=1) -> Ball:
    _, K, H = _kh(beta, h, J)
    return certify(lambda: _HOOKS["magnetization"](arb_q(K), arb_q(H)))


def ising1d_susceptibility(beta, h, J=1) -> Ball:
    b, K, H = _kh(beta, h, J)
    return certify(lambda: _HOOKS["susceptibility"](arb_q(K), arb_q(H), arb_q(b)))


def ising1d_susceptibility_fluctuation(beta, h, J=1) -> Ball:
    b, K, H = _kh(beta, h, J)
    return certify(lambda: _chi_fluct(arb_q(K), arb_q(H), arb_q(b)))


def ising1d_correlation(beta, h, r, J=1) -> Ball:
    _, K, H = _kh(beta, h, J)
    r = _r(r)
    return certify(lambda: _HOOKS["correlation"](arb_q(K), arb_q(H), r))


def ising1d_correlation_length(beta, h, J=1) -> Ball:
    _, K, H = _kh(beta, h, J)
    return certify(lambda: _HOOKS["correlation_length"](arb_q(K), arb_q(H)))


# ---- the ring ------------------------------------------------------------------------------
@dataclass(frozen=True)
class RingObs:
    Z: object               # Tr V^N
    m: object               # (1/N) d ln Z / dH
    chi_over_beta: object   # (1/N) d^2 ln Z / dH^2
    corr: object            # <s_0 s_r> - m^2
    corr_next: object       # <s_0 s_{r+1}> - m^2


def _smul(p, q):            # truncated series (degree 2) product
    return (p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[0] * q[2] + p[1] * q[1] + p[2] * q[0])


def _sadd(p, q):
    return tuple(x + y for x, y in zip(p, q))


def _mmul(A, B, mul, add):
    return [[add(mul(A[i][0], B[0][j]), mul(A[i][1], B[1][j])) for j in range(2)] for i in range(2)]


def _mpow(A, n, mul, add):
    R, P = None, A
    while n:
        if n & 1:
            R = P if R is None else _mmul(R, P, mul, add)
        n >>= 1
        if n:
            P = _mmul(P, P, mul, add)
    return R


def _ring(N, a, b, r, one):
    """RingObs of the N-ring at a = e^K, b = e^H (Fractions, or arbs with one = arb(1))."""
    zero, half = one - one, one / 2
    up, dn, off = a * b, a / b, one / a
    T = [[(up, up, up * half), (off, zero, zero)], [(off, zero, zero), (dn, -dn, dn * half)]]
    P = _mpow(T, N, _smul, _sadd)
    z0, z1, z2 = _sadd(P[0][0], P[1][1])
    w = z1 / z0
    m = w / N
    chi = (2 * z2 / z0 - w * w) / N          # w * w, not w ** 2: arb's pow of a ball at 0 is nan
    V = [[up, off], [off, dn]]
    mul, add = (lambda x, y: x * y), (lambda x, y: x + y)

    def two_point(s):
        if s == 0:
            return one
        A = _mpow(V, s, mul, add)
        B = _mpow(V, N - s, mul, add)
        # Tr S V^s S V^{N-s}, S = diag(1, -1)
        return (A[0][0] * B[0][0] - A[0][1] * B[1][0] - A[1][0] * B[0][1] + A[1][1] * B[1][1]) / z0
    return RingObs(z0, m, chi, two_point(r) - m * m, two_point(r + 1) - m * m)


def _ring_args(N, r):
    if isinstance(N, bool) or not isinstance(N, int) or not 2 <= N <= 1000:
        raise Refusal("precondition:size", f"N = {N!r}: need an int in 2 .. 1000")
    _r(r, N - 2)
    return N, r


def ring_observables(N, a, b, r=1) -> RingObs:
    N, r = _ring_args(N, r)
    a, b = rational(a, "a"), rational(b, "b")
    return _ring(N, a, b, r, Fraction(1))


def ring_brute(N, a, b, r=1) -> RingObs:
    """The same quantities by summing over the 2^N configurations (N <= 12): weights
    prod a^{s_i s_{i+1}} b^{s_i}; m, chi/beta from <M>, <M^2> - <M>^2 (M = sum s_i)."""
    if isinstance(N, bool) or not isinstance(N, int) or not 3 <= N <= 12:
        raise Refusal("precondition:size", f"N = {N!r}: need an int in 3 .. 12")
    _r(r, N - 2)
    a, b = rational(a, "a"), rational(b, "b")
    Z = M1 = M2 = Fraction(0)
    C = [Fraction(0), Fraction(0)]
    for s in product((1, -1), repeat=N):
        w = Fraction(1)
        for i in range(N):
            w *= a ** (s[i] * s[(i + 1) % N]) * b ** s[i]
        M = sum(s)
        Z, M1, M2 = Z + w, M1 + w * M, M2 + w * M * M
        C[0] += w * s[0] * s[r]
        C[1] += w * s[0] * s[r + 1]
    m = M1 / (N * Z)
    return RingObs(Z, m, (M2 / Z - (M1 / Z) ** 2) / N, C[0] / Z - m * m, C[1] / Z - m * m)


# ---- the N -> oo limit in balls ----------------------------------------------------------------
@dataclass(frozen=True)
class Observable:
    ring: Ball           # the N-ring's value
    closed: Ball         # the infinite chain's closed form (hooked)
    correction: Ball     # ring - closed, exact finite-N formula (from ln lambda_{1,2} only)
    bound: Ball          # an upper bound of |ring - closed|
    identity: bool       # ring overlaps closed + correction
    within_bound: bool   # |ring - closed| <= bound (ball comparison not refuted)
    limit_tested: bool = True   # False when the bound is infinite or >= |closed| (and nonzero):
                                # within_bound then says nothing (small N, rho near 1)


@dataclass(frozen=True)
class RingCheck:
    N: int
    r: int
    rho: Ball
    observables: dict    # name -> Observable: magnetization, susceptibility, correlation, correlation_length

    @property
    def ok(self):
        return all(o.identity and o.within_bound for o in self.observables.values())

    @property
    def limit_untested(self):
        """Observables whose N -> oo leg is uninformative at this N (see Observable.limit_tested)."""
        return tuple(k for k, o in self.observables.items() if not o.limit_tested)


def _tested(bound, closed):
    """The limit leg tests something iff the bound is finite and below |closed| (a zero bound,
    e.g. m at h = 0, is an exact statement and counts as tested)."""
    if not bound.is_finite():
        return False
    return bool(bound == 0) or bool(bound < closed.abs_lower())


def _log_lambda_series(K, H):
    """(ln lambda_1, ln lambda_2) as arb_series in H (length 3): Taylor coefficients in H."""
    from flint import arb_series
    X = arb_series([H, 1], prec=3)
    e, em = X.exp(), (-X).exp()
    sh, ch = (e - em) / 2, (e + em) / 2
    root = ((2 * K).exp() * sh * sh + (-2 * K).exp()).sqrt()
    return (K.exp() * ch + root).log(), (K.exp() * ch - root).log()


def _coef(s, i):
    return s.coeffs()[i]


def ring_check(N, beta, h, r=1, J=1, prec=None) -> RingCheck:
    N, r = _ring_args(N, r)
    bq, K, H = _kh(beta, h, J)
    from flint import arb
    prec = prec or 4 * 40 + 64 + 4 * N
    with baller().hygiene.ctx_guard(prec=prec):
        k, x, be = arb_q(K), arb_q(H), arb_q(bq)
        ring = _ring(N, k.exp(), x.exp(), r, arb(1))
        lp, lm = _log_lambda_series(k, x)
        ell = lm - lp
        l1, l2 = _coef(ell, 1), 2 * _coef(ell, 2)
        m0 = _coef(lp, 1)                                    # m = d ln lambda_1 / dH
        rho = (_coef(ell, 0)).exp()
        q = rho ** N
        corr = {"magnetization": (q * l1 / (1 + q), q * l1.abs_upper()),
                "susceptibility": (be * (N * q * l1 * l1 / (1 + q) ** 2 + q * l2 / (1 + q)),
                                   be * q * (N * l1 * l1 + l2.abs_upper()))}
        mN = ring.m
        g_fs = (lambda s: (1 - m0 * m0) * rho ** (N - s) * (1 - rho ** (2 * s)) / (1 + q)
                + (m0 - mN) * (m0 + mN))
        bnd = lambda s: rho ** (N - s) + 2 * q * l1.abs_upper()
        corr["correlation"] = (g_fs(r), bnd(r))
        closed = {"magnetization": _HOOKS["magnetization"](k, x),
                  "susceptibility": _HOOKS["susceptibility"](k, x, be),
                  "correlation": _HOOKS["correlation"](k, x, r)}
        rings = {"magnetization": mN, "susceptibility": be * ring.chi_over_beta,
                 "correlation": ring.corr}
        # xi: rho from the ring's G_N(r + 1)/G_N(r), finite-size corrected (identity) or
        # widened by the bounds (limit); xi = -1/ln rho
        g0, g1 = ring.corr - g_fs(r), ring.corr_next - g_fs(r + 1)
        xi_ring = -1 / (ring.corr_next / ring.corr).log()
        xi_corr = -1 / (g1 / g0).log()
        wide = lambda g, b: g + arb(0, b.abs_upper())
        lo_hi = (wide(ring.corr_next, bnd(r + 1)) / wide(ring.corr, bnd(r)))
        xi_lim = -1 / lo_hi.log() if lo_hi > 0 and lo_hi < 1 else arb(0, float("inf"))
        out = {}
        for name in ("magnetization", "susceptibility", "correlation"):
            c, b = corr[name]
            diff = rings[name] - closed[name]
            bu = arb(b.abs_upper())
            out[name] = Observable(ball_of(rings[name]), ball_of(closed[name]), ball_of(c),
                                   ball_of(bu), bool(diff.overlaps(c)),
                                   not bool(diff.abs_lower() > b.abs_upper()),
                                   _tested(bu, closed[name]))
        cxi = _HOOKS["correlation_length"](k, x)
        bxi = arb(xi_lim.rad())
        out["correlation_length"] = Observable(
            ball_of(xi_ring), ball_of(cxi), ball_of(xi_ring - xi_corr), ball_of(bxi),
            bool(xi_corr.overlaps(cxi)), bool(xi_lim.overlaps(cxi)), _tested(bxi, cxi))
        return RingCheck(N, r, ball_of(rho), out)


__all__ = ["Observable", "RingCheck", "RingObs", "ising1d_correlation", "ising1d_correlation_length",
           "ising1d_magnetization", "ising1d_susceptibility", "ising1d_susceptibility_fluctuation",
           "ring_brute", "ring_check", "ring_observables"]
