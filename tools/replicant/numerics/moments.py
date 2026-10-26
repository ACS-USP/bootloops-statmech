"""Exact annealed moments of the SK partition function at finite N, and the quenched-minus-
annealed gap they predict (E11).

  log_moments(N, beta, law="pm_one", dps=40) -> (ln E Z, L2, L3),  L_k = ln E Z^k - k ln E Z
  quenched_gap(N, beta, order, law="pm_one") -> predicted E ln Z - ln E Z
  pm_one_gap_limit(beta) -> the N -> oo limit of the +-J order-2 closure (beta < 1)

Model: H = -sum_{i<j} J_ij s_i s_j, h = 0, M = N(N-1)/2 pairs.  Exact moments (derived here):
  E Z^k = sum over k replicas of prod_{i<j} E exp(beta J_ij sum_a s_i^a s_j^a).
  +-J (J = +-1/sqrt N, b = beta/sqrt N): E e^{b J x} = cosh(b x).
    k = 1: E Z = 2^N cosh(b)^M.
    k = 2: gauge s^2 = sigma s^1; a pair has x = +-2 when sigma_i sigma_j = 1, else 0, so
           E Z^2 = 2^N sum_n binom(N, n) cosh(2b)^(C(n,2) + C(N-n,2)).
    k = 3: gauge (s^2, s^3) = (sigma, tau) s^1; |x| = 3 when sites i, j have the same type
           (sigma, tau) and 1 otherwise, so E Z^3 = 2^N sum over the type counts n_1..n_4 of
           multinomial(N; n) cosh(3b)^S cosh(b)^(M - S), S = sum_t C(n_t, 2).
  Gaussian (J ~ N(0, 1/N), c = beta^2 / (2N)): E e^{beta J x} = exp(c x^2), the same sums with
           cosh(m b) replaced by exp(c m^2).
Prediction (a cumulant closure, derived here): with X = ln Z - ln E Z, L_k = ln E e^{kX}
exactly.  Truncating the cumulant series of X after kappa_2 (order 2) gives, from k = 1, 2,
    E X = kappa_1 = -L2 / 2      (the log-normal closure);
after kappa_3 (order 3), k = 1, 2, 3 give three linear equations for kappa_1..kappa_3.  The
closure is exact if ln Z is Gaussian; its error is not bounded here, and callers use
|order 2 - order 3| as the stated truncation estimate.  Background: for Gaussian J and beta < 1
the fluctuations of ln Z - ln E Z are O(1) (Aizenman, Lebowitz & Ruelle, Commun. Math. Phys.
112, 3 (1987); not fetched, quoted via Guerra 1992 arXiv:1212.2905), so the per-spin gap is
O(1/N); for +-J, J^2 = 1/N is not random and the O(1) gap is smaller.

+-J asymptote (a cross-check, derived here, not used by any assertion on data): for beta < 1
the cycle expansion of ln Z - ln E Z has variance sum_{k >= 3} beta^{2k} / (2k) = -(1/2) ln(1 -
beta^2) - beta^2/2 - beta^4/4 (the Gaussian law's sum starts at k = 1; with J^2 = 1/N fixed the
k = 1 term, the edge-variance term, vanishes, and the k = 2 term is cancelled by the cosh
normalisation), so the log-normal closure tends to
    -1/4 [-ln(1 - beta^2) - beta^2 - beta^4 / 2].
tests/test_enumeration.py checks that the exact order-2 closure approaches it as N grows
(N = 400, 1600: differences 2.6e-4, 6.5e-5 at beta = 0.7, i.e. O(1/N)).
"""
import mpmath as mp

from ..core import Refusal


def _pair_factor(law, b, m):
    if law == "pm_one":
        return mp.cosh(m * b)
    return mp.exp((m * b)**2 / 2)          # Gaussian g ~ N(0, 1): E e^{m b g} = e^{m^2 b^2 / 2}


def log_moments(N, beta, law="pm_one", dps=40):
    if law not in ("pm_one", "gaussian"):
        raise Refusal("precondition:supported-disorder-law", f"law {law!r}: pm_one or gaussian")
    if not (isinstance(N, int) and 2 <= N <= 200):
        raise Refusal("precondition:sizes", f"N = {N!r}: need an int in [2, 200]")
    with mp.workdps(dps):
        b, M = mp.mpf(beta) / mp.sqrt(N), N * (N - 1) // 2
        f1, f2, f3 = (_pair_factor(law, b, m) for m in (1, 2, 3))
        L1 = N * mp.log(2) + M * mp.log(f1)
        Z2 = mp.fsum(mp.binomial(N, n) * f2**(n * (n - 1) // 2 + (N - n) * (N - n - 1) // 2)
                     for n in range(N + 1))
        terms = []
        for n1 in range(N + 1):
            for n2 in range(N + 1 - n1):
                for n3 in range(N + 1 - n1 - n2):
                    n4 = N - n1 - n2 - n3
                    S = sum(n * (n - 1) // 2 for n in (n1, n2, n3, n4))
                    multi = mp.factorial(N) / (mp.factorial(n1) * mp.factorial(n2) *
                                               mp.factorial(n3) * mp.factorial(n4))
                    terms.append(multi * f3**S * f1**(M - S))
        Z3 = mp.fsum(terms)
        # ln E Z^k = N ln 2 + ln(sum); minus k ln E Z
        L2 = N * mp.log(2) + mp.log(Z2) - 2 * L1
        L3 = N * mp.log(2) + mp.log(Z3) - 3 * L1
        return +L1, +L2, +L3


def _L2(N, beta, law, dps=40):
    """L2 alone (the k = 2 sum, O(N) terms): for the order-2 closure at large N."""
    with mp.workdps(dps):
        b, M = mp.mpf(beta) / mp.sqrt(N), N * (N - 1) // 2
        f1, f2 = _pair_factor(law, b, 1), _pair_factor(law, b, 2)
        Z2 = mp.fsum(mp.binomial(N, n) * f2**(n * (n - 1) // 2 + (N - n) * (N - n - 1) // 2)
                     for n in range(N + 1))
        return N * mp.log(2) + mp.log(Z2) - 2 * (N * mp.log(2) + M * mp.log(f1))


def pm_one_gap_limit(beta):
    b = mp.mpf(beta)
    if not 0 <= b < 1:
        raise Refusal("precondition:betas", f"beta = {beta!r}: the limit is for 0 <= beta < 1")
    return float(-(-mp.log(1 - b**2) - b**2 - b**4 / 2) / 4)


def quenched_gap(N, beta, order, law="pm_one"):
    if order not in (2, 3):
        raise Refusal("precondition:order", f"order {order!r}: 2 or 3")
    if order == 2:
        if law not in ("pm_one", "gaussian") or not (isinstance(N, int) and N >= 2):
            log_moments(N, beta, law)                 # the same refusals
        return float(-_L2(N, beta, law) / 2)
    with mp.workdps(40):
        _, L2, L3 = log_moments(N, beta, law)
        A = mp.matrix([[1, mp.mpf(1) / 2, mp.mpf(1) / 6], [2, 2, mp.mpf(4) / 3],
                       [3, mp.mpf(9) / 2, mp.mpf(9) / 2]])
        return float(mp.lu_solve(A, mp.matrix([0, L2, L3]))[0])
