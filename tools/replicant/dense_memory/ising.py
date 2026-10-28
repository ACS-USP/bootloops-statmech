"""Ising neurons, +-1 patterns, exponential interaction F(x) = exp(lam x): the model of Demircigil, Heusel,
Lowe, Upgang & Vermet (J. Stat. Phys. 168, 288 (2017), arXiv:1702.01929v2, PDF read), whose zero-temperature
single-flip dynamics is

    T_i(sigma) = sgn[ sum_mu F(xi_i^mu + sum_{j != i} xi_j^mu sigma_j) - F(-xi_i^mu + sum_{j != i} ... ) ]

(their Eq. (1)); with F = exp(lam .) it is the zero-temperature descent of E(x) = -(1/lam) log sum_mu
exp(lam x.xi^mu) on the hypercube.

PUBLISHED (rigorous, lam = 1):
  Thm 1.3   M = e^{alpha N} + 1 patterns, 0 < alpha < log(2)/2, a start xi~ drawn uniformly on the Hamming
            sphere S(xi^mu, rho N), rho in [0, 1/2): if alpha < I(1 - 2 rho)/2 then P(exists mu, i : T_i(xi~)
            != xi_i^mu) -> 0, I(x) = ((1+x) log(1+x) + (1-x) log(1-x))/2
  Rem. 1.4  rho = 0: all patterns are fixed points with probability -> 1 for alpha < I(1)/2 = log(2)/2

  fixed_point_exponent(rho)    I(1 - 2 rho)/2

DERIVED HERE (not in either paper), from the REM noise of rem_noise.py with the binary ensemble:
  Mechanism.  alpha = log(2)/2 is the BIRTHDAY threshold: two of the e^{alpha N} patterns differ in at most
  one site with probability ~ M^2 N / 2^N.  A pair at distance 1 cancels exactly in one coordinate of the decision sum, and that coordinate is then decided by
  the remaining noise (a tie broken at random); below the threshold the noise never matters.  expected_close_pairs gives the
  exact finite-N count.  Observed over every sampled instance (tests, sim.ising_flip_trial, N = 16 and 20): NO pair within
  Hamming distance 1 implies that every pattern is a fixed point; with such a pair some pattern usually (not always) is not.
  Typical pattern at a typical site.  The decision sum at site i is sum_mu xi_i^mu xi_i^1 exp(lam m_mu^(i)):
  the signal exp(lam (N-1)) against a sum of terms with INDEPENDENT FAIR SIGNS (xi_i^mu is independent of the
  overlap with the other sites), whose size is the square root of sum_mu exp(2 lam m_mu), that is the REM noise
  at 2 lam: exp(lam N phi_binary(alpha, 2 lam)).  The pattern is stable when phi_binary(alpha, 2 lam) < 1:

  alpha_1_flip(lam)   = alpha_1('binary', 2 lam) = 2 lam - log cosh(2 lam)     (rem_noise, thresholds)

  Sampled in sim.sign_noise_mc, which converges to phi(2 lam) and not to phi(lam).  This is the
  typical-pattern threshold; at lam = 1, the only lam with a theorem, it is 0.675 against the all-patterns
  exponent log(2)/2 = 0.347, because a union bound over the M^2 pairs halves the exponent: log 2 / 2 is
  exactly half of the lam -> inf limit log 2 of alpha_1_flip.
  alpha_1_energy(lam) = alpha_1('binary', lam) = lam - log cosh(lam): the paper's Eq. (7) criterion
  phi < 1 transplanted unchanged (the energy of xi^1 below the noise floor); it ignores the sign
  cancellation and is smaller than alpha_1_flip.
"""
import math

import mpmath as mp

from ..core import Refusal
from .ensembles import _prec, binary_rate, num
from .thresholds import alpha_1, alpha_1_closed

BANKED_LAMBDA = 1        # Demircigil et al. prove their theorems for F(x) = e^x only


def fixed_point_exponent(rho=0):
    """I(1 - 2 rho)/2: the largest alpha of Thm 1.3 (Remark 1.4 at rho = 0: log(2)/2).  rho in [0, 1/2)."""
    with _prec():
        r = num(rho, "rho")
        if not 0 <= r < mp.mpf(1) / 2:
            raise Refusal("precondition:known-parameter",
                          f"rho = {rho!r}: Demircigil et al. Thm 1.3 needs rho in [0, 1/2)")
        return binary_rate(1 - 2 * r) / 2


def alpha_1_flip(lam, method="closed"):
    """Typical-pattern single-flip stability threshold (derived here, module docstring):
    alpha_1('binary', 2 lam)."""
    with _prec():
        l = 2 * num(lam, "lambda", positive=True)
        return alpha_1_closed("binary", l) if method == "closed" else alpha_1("binary", l)


def alpha_1_energy(lam):
    """The paper's criterion phi_{alpha,1}(lam) < 1 for binary patterns: lam - log cosh(lam)."""
    return alpha_1_closed("binary", num(lam, "lambda", positive=True))


def expected_close_pairs(N, P, distance=1):
    """Exact expected number of unordered pairs of the P patterns, i.i.d. uniform on {-1,+1}^N, at Hamming
    distance <= `distance`: C(P,2) sum_{k <= d} C(N,k)/2^N.  For distance = 1 and M^2 N / 2^N << 1 the
    probability that no such pair exists is exp(-expected_close_pairs) (a Poisson count)."""
    if not (isinstance(N, int) and isinstance(P, int) and N >= 1 and P >= 1 and distance >= 0):
        raise Refusal("precondition:size", f"N = {N!r}, P = {P!r}, distance = {distance!r}")
    return math.comb(P, 2) * sum(math.comb(N, k) for k in range(distance + 1)) / 2.0**N


def log_expected_close_pairs(N, alpha, distance=1):
    """log of the expected number of pairs within Hamming distance <= `distance` among P = e^{alpha N} i.i.d.
    uniform patterns, for real alpha (no integer P): log(P (P - 1)/2) + log sum_{k<=d} C(N,k) - N log 2.  Its
    limit divided by N is 2 alpha - log 2 for any fixed distance: the pairs appear at alpha = log(2)/2."""
    from scipy.special import gammaln
    if not (isinstance(N, int) and N >= 1 and distance >= 0):
        raise Refusal("precondition:size", f"N = {N!r}, distance = {distance!r}")
    lp = float(alpha) * N
    log_pairs = lp + (lp + math.log(-math.expm1(-lp))) - math.log(2)  # log(P (P - 1)/2), P = e^lp, no overflow
    ks = range(distance + 1)
    log_ball = float(max(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) for k in ks)) + math.log(distance + 1)
    return log_pairs + log_ball - N * math.log(2)


__all__ = ["BANKED_LAMBDA", "log_expected_close_pairs", "alpha_1_energy", "alpha_1_flip", "expected_close_pairs",
           "fixed_point_exponent"]
