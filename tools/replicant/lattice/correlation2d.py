"""The square-lattice Ising correlation length in zero field, certified, and its strip spectrum.

Which xi: the exponential (true) correlation length along a lattice axis, xi^-1 = lim -(1/r) ln
<s_0 s_r>_connected, i.e. the gap of the row-to-row transfer matrix; J = 1, K = beta, isotropic.

Source read: Onsager, Phys. Rev. 65, 117 (1944) (journal scan, pp. 117 and 137-138).  With the dual
coupling K* (Onsager's H*, Eq. (17): sinh 2K sinh 2K* = 1, i.e. e^{-2K*} = tanh K) and, for a strip
of circumference n (Onsager's crystal, periodic across), gamma_r (r = 0 .. 2n-1) from Eq. (89a),
  cosh gamma_r = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(r pi / n) = cosh^2 2K / sinh 2K - cos(r pi/n),
the two sectors of the spectrum are
  ln lambda_max - (n/2) ln(2 sinh 2K) = (1/2) sum_{r odd} gamma_r                 Eqs. (95), (97)
  ln lambda_max^(-) - (n/2) ln(2 sinh 2K) = (1/2)(gamma_0 sgn(K - K*) + sum_{r even, r > 0} gamma_r),
  gamma_0 = 2 |K - K*|                                                             Eqs. (99), (100)
and the next level of the principal sector is ln lambda_max - 2 gamma_1                 Eq. (102)
(gamma_1 = gamma_{2n-1}: the pair with momenta +-pi/n, total momentum zero).  So on the strip
  gap_odd  = ln(lambda_0 / lambda_1) = (1/2)(sum_odd gamma_r - sum_{even > 0} gamma_r) - (K - K*),
  gap_even = ln(lambda_0 / lambda_2) = 2 gamma_1.
The limits (Onsager (101) and the paragraph after it: "the range ... is inversely proportional to
log(lambda_max/lambda)"):
  T > T_c (K < K*): xi^-1 = lim gap_odd = 2 (K* - K)       [Onsager (101) prints exp(2H* - 2H') for
      the ratio lambda^(-)/lambda; with (99), (100) it is exp(2H' - 2H*) < 1: a sign slip there]
  T < T_c (K > K*): lambda_0, lambda_1 are asymptotically degenerate (tunnelling: gap_odd -> 0
      exponentially in n, Onsager (101), (103)); the connected correlation decays with the
      two-particle gap, xi^-1 = lim gap_even = 2 gamma(0) = 4 (K - K*).
  The two-particle identification below T_c is taken here from the form-factor structure of
  Lyberg & McCoy, arXiv:math-ph/0612051, Eqs. (5), (6), (13), (20), (21), (31), (32) (read): the
  row correlation's odd terms (T > T_c) start with one z^N integral, the even ones (T < T_c) with
  two, the branch points at z = alpha_2^{+-1}, alpha_2 = e^{-2K} coth K = e^{2(K* - K)} giving the
  decay alpha_2^{-N} above and alpha_2^{2N} below.  Near T_c both agree with Dohm, arXiv:1907.00859
  (read), Eq. (30): xi_0+ = (4 beta_c J)^-1 = [2 ln(1 + sqrt 2)]^-1, and Eqs. (41), (46):
  xi_0+ / xi_0- = 2 (exponential correlation lengths, xi = xi_0+- |t|^-1, t = T/T_c - 1).

  dual_coupling(beta) -> Ball K*
  correlation_length(beta) -> Ball xi;  inverse_correlation_length(beta) -> Ball xi^-1
      (refused precondition:off-critical at BETA_C: xi = oo)
  strip_gaps(n, beta) -> StripGaps(odd, even) Balls: Onsager's spectrum on the width-n strip
  xi_log_slope(d1, d2) -> Ball   [ln xi(b2) - ln xi(b1)] / [ln|b2 - K_c| - ln|b1 - K_c|] -> -nu
  xi_amplitude(d) -> Ball        xi(b) |K_c / b - 1|, b = K_c + d (rational): -> xi_0+ (d < 0) or
                                 xi_0- (d > 0) (a consistency value)
  xi_amplitudes() -> (Ball, Ball) the closed forms xi_0+ = 1/(4 K_c), xi_0- = 1/(8 K_c)
  two_particle_gap(beta) -> Ball  4 |K - K*| = lim 2 gamma_1 (Onsager Eq. (102))
  critical_gap_scaled(n) -> Ball n gap_odd(n) at beta_c: -> 2 pi x_sigma = pi eta (Cardy)
_HOOKS: "below_factor" (4: xi^-1 = 4(K - K*) below T_c; planted 2 = the T > T_c branch),
"gamma0_signed" (True; planted False = the wrong sector boundary condition |gamma_0| in the
odd sector), "nu" (1) and "eta" (1/4): the exponents the consistency checks compare with.
"""
from dataclasses import dataclass
from fractions import Fraction

from ..core import Refusal
from .ball import Ball, arb_q, baller, certify
from .coupling import coupling
from .onsager import _ordered

_HOOKS = {"below_factor": 4, "gamma0_signed": True, "nu": 1, "eta": Fraction(1, 4)}
MAX_STRIP = 4096


def _kc():
    from flint import arb
    return (1 + arb(2).sqrt()).log() / 2


def _kstar(K):
    return -K.tanh().log() / 2


def _xi_inv(K, below):
    d = K - _kstar(K)
    return _HOOKS["below_factor"] * d if below else -2 * d


def _phase(beta):
    c = coupling(beta)
    if c.kind == "critical":
        raise Refusal("precondition:off-critical", "xi diverges at beta_c (xi ~ xi_0 |t|^-nu, nu = 1): "
                      "no value there; the critical strip gap is critical_gap_scaled(n)")
    return c, _ordered(c.arb)


def dual_coupling(beta) -> Ball:
    c = coupling(beta)
    return certify(lambda: _kstar(c.arb()))


def inverse_correlation_length(beta) -> Ball:
    c, below = _phase(beta)
    return certify(lambda: _xi_inv(c.arb(), below))


def correlation_length(beta) -> Ball:
    c, below = _phase(beta)
    return certify(lambda: 1 / _xi_inv(c.arb(), below))


# ---- Onsager's strip spectrum ----------------------------------------------------------------
def _gamma(K, n, r):
    """gamma_r, r >= 1, cancellation-free: y - 1 = (1 - S)^2 / S + 2 sin^2(r pi / 2n), y = cosh gamma."""
    from flint import arb, fmpq
    S = (2 * K).sinh()
    s = arb.sin_pi_fmpq(fmpq(r, 2 * n))
    ym = (1 - S) * (1 - S) / S + 2 * s * s
    return (1 + ym + (ym * (ym + 2)).sqrt()).log()


def _gap_odd(K, n):
    g0 = 2 * (K - _kstar(K))
    if not _HOOKS["gamma0_signed"]:
        g0 = abs(g0)
    odd = sum((_gamma(K, n, r) for r in range(1, 2 * n, 2)), 0 * K)
    even = sum((_gamma(K, n, r) for r in range(2, 2 * n, 2)), 0 * K)
    return (odd - even - g0) / 2


def _gap_even(K, n):
    return 2 * _gamma(K, n, 1)


@dataclass(frozen=True)
class StripGaps:
    n: int
    odd: Ball          # ln(lambda_0 / lambda_1): principal vs odd sector
    even: Ball         # ln(lambda_0 / lambda_2): principal vs its two-particle level


def _width(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= MAX_STRIP:
        raise Refusal("precondition:size", f"n = {n!r}: need an int in 2 .. {MAX_STRIP}")
    return n


def strip_gaps(n, beta) -> StripGaps:
    n, c = _width(n), coupling(beta)
    return StripGaps(n, certify(lambda: _gap_odd(c.arb(), n)), certify(lambda: _gap_even(c.arb(), n)))


def critical_gap_scaled(n) -> Ball:
    n = _width(n)
    return certify(lambda: n * _gap_odd(_kc(), n))


# ---- exponents and amplitudes (consistency checks of the closed forms) --------------------------
def _near_kc():
    with baller().hygiene.ctx_guard(prec=400):
        return Fraction(_kc().mid().str(70, radius=False))


def _signed(d, name="d"):
    from .ball import rational
    q = rational(d, name, positive=False)
    if q == 0:
        raise Refusal(f"precondition:{name}", f"{name} = 0: need a nonzero offset from beta_c")
    return q


def xi_log_slope(d1, d2, digits=30) -> Ball:
    d = [_signed(x) for x in (d1, d2)]
    if not (d[0] * d[1] > 0 and abs(d[0]) != abs(d[1])):
        raise Refusal("precondition:d", f"d1 = {d1}, d2 = {d2}: need distinct nonzero d of one sign")
    bs = [_near_kc() + x for x in d]
    below = d[0] > 0

    def thunk():
        kc = _kc()
        lx = [-_xi_inv(arb_q(b), below).log() for b in bs]
        ld = [abs(arb_q(b) - kc).log() for b in bs]
        return (lx[1] - lx[0]) / (ld[1] - ld[0])
    return certify(thunk, digits)


def two_particle_gap(beta) -> Ball:
    """lim 2 gamma_1 = 2 gamma(0) = 4 |K - K*| (Onsager Eqs. (100), (102)): the principal sector's
    two-particle gap, in both phases (below T_c it is 1/xi, the identification derived here)."""
    c = coupling(beta)
    return certify(lambda: 4 * abs(c.arb() - _kstar(c.arb())))


def xi_amplitudes():
    """(xi_0+, xi_0-) in closed form, xi = xi_0+- |t|^-1, t = T/T_c - 1 = K_c/K - 1: at K_c,
    dK*/dK = -1/sinh 2K_c = -1, so K* - K = 2 K_c t + O(t^2) above and K - K* = 2 K_c |t| + O(t^2)
    below; with 1/xi = 2(K* - K), 4(K - K*): xi_0+ = 1/(4 K_c) = [2 ln(1 + sqrt 2)]^-1 (Dohm Eq.
    (30)), xi_0- = 1/(8 K_c) (ratio 2, Dohm Eq. (46))."""
    return certify(lambda: 1 / (4 * _kc())), certify(lambda: 1 / (8 * _kc()))


def xi_amplitude(d, digits=30) -> Ball:
    b = _near_kc() + _signed(d)

    def thunk():
        K = arb_q(b)
        return abs(_kc() / K - 1) / _xi_inv(K, b > _near_kc())
    return certify(thunk, digits)


@dataclass(frozen=True)
class StripLimit:
    """Strip -> bulk, a finite-size consistency check (no proven bound): the estimate of xi^-1 from
    Onsager's strip gaps at widths `widths`, its band (3 x the last change of the estimate, floored at
    1e-30 relative: the midpoints' precision; e.g. at beta = 40 the sequence is constant) and
    whether the band resolves the factor-2 contrast between the two branches (band < xi^-1 / 10)."""
    pair: str          # which eigenvalue pair
    widths: tuple
    sequence: tuple    # mpf gaps
    estimate: object   # mpf
    band: object       # mpf
    resolved: bool


def strip_limit(beta, widths=(32, 64, 128, 256)) -> StripLimit:
    """T > T_c: gap_odd (lambda_0/lambda_1) converges exponentially in n: the estimate is the last
    width's gap.  T < T_c: gap_even (lambda_0/lambda_2; lambda_0/lambda_1 is the tunnelling pair, -> 0)
    converges like 1/n^2 (the pair's momenta +-pi/n): Richardson in 1/n^2 on the last two pairs of
    widths (each width double the previous)."""
    import mpmath as mp
    c, below = _phase(beta)
    gap = _gap_even if below else _gap_odd
    seq = tuple(certify(lambda n=n: gap(c.arb(), _width(n))).mid() for n in widths)
    xi_inv = inverse_correlation_length(beta).mid()
    with mp.workdps(40):
        if below:
            r = [(4 * b - a) / 3 for a, b in zip(seq, seq[1:])]
            est, inc = r[-1], abs(r[-1] - r[-2])
        else:
            est, inc = seq[-1], abs(seq[-1] - seq[-2])
        band = 3 * inc + mp.mpf("1e-30") * max(1, abs(xi_inv))   # floor: the 40-digit balls' midpoints
        return StripLimit("lambda_0/lambda_2 (principal sector, momenta +-pi/n; lambda_1 tunnels)"
                          if below else "lambda_0/lambda_1 (principal vs odd sector)", tuple(widths),
                          seq, est, band, bool(band < xi_inv / 10))


__all__ = ["MAX_STRIP", "StripGaps", "StripLimit", "correlation_length", "critical_gap_scaled",
           "dual_coupling", "inverse_correlation_length", "strip_gaps", "strip_limit",
           "two_particle_gap", "xi_amplitude", "xi_amplitudes", "xi_log_slope"]
