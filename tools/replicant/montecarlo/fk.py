"""The Fortuin-Kasteleyn coupling of the cluster algorithms: inverse temperatures, the bond probability,
and the exact critical points of the square-lattice Potts models.

Bond probability.  Ising (q = 2, H = -J sum s_i s_j, J = 1): p = 1 - e^{-2 beta J}, a bond between
equal spins (Liu, Polkovnikov & Sandvik, arXiv:1310.6327v3 (2014), PDF read,
Sec. III B: "P = 1 - e^{-2J/T}"; the same bond rule for Wolff, Sec. III C).  q-state Potts
(H = -beta sum (delta(c_i, c_j) - 1)): W = prod_b (1 - p + p delta_b) / Z with p = 1 - e^{-beta}, the
Edwards-Sokal joint weight prod_b [(1 - p) delta(n_b, 0) + p delta_b delta(n_b, 1)] whose bond marginal is
the Fortuin-Kasteleyn random-cluster model p^{open} (1 - p)^{closed} q^{C}, and the Swendsen-Wang
conditional updates (bonds between equal spins open with probability p; each cluster a uniform new
colour) (Salas & Sokal, arXiv:hep-lat/9605018v2 (1997), PDF read, Sec. 2.1,
Eqs. (2.1)-(2.5) and p. 5; Edwards & Sokal, Phys. Rev. D 38, 2009 (1988), their ref. [26], not
fetched).  With K the coupling of exp(K sum delta) (K = 2 beta at q = 2, beta at q >= 3) both read
p = 1 - e^{-K}.  Random-cluster measure also as Beffara & Duminil-Copin, arXiv:1006.5073v2
(2010-2013, PDF read), Eq. (1.1).
Critical points.  The self-dual point p_sd(q) = sqrt q / (1 + sqrt q) is critical for every q >= 1 (BDC
Theorem 1); the Potts critical inverse temperature is beta_c(q) = log(1 + sqrt q), q >= 2 (BDC Theorem 3);
at q = 2 in the Ising convention beta_c = (1/2) log(1 + sqrt 2) (lattice/coupling.BETA_C).
  critical_beta(q) -> (mpf beta_c, detail): log(1 + sqrt q) [halved at q = 2], and a second route,
      -log(1 - p_sd(q)) [halved], which must agree to 30 digits (else no-solution:route-disagreement)
  beta_value(q, beta) -> mpf: "beta_c" -> critical_beta(q); a rational beta (int, Fraction, decimal
      string) > 0, else precondition:beta
  bond_probability(q, beta) -> float p = 1 - exp(-K)
_HOOKS["coupling_factor"] (tests): K = factor x beta x (2 at q = 2); planted 1/2 (p = 1 - e^{-beta} for
Ising: the missing factor 2).
"""
import math

import mpmath as mp

from ..core import Refusal
from ..lattice.ball import rational

_HOOKS = {"coupling_factor": 1}


def check_q(q):
    if isinstance(q, bool) or not isinstance(q, int) or not 2 <= q <= 64:
        raise Refusal("precondition:potts-q", f"q = {q!r}: need an int in 2 .. 64 (q = 2: Ising)")
    return q


def critical_beta(q):
    check_q(q)
    with mp.workdps(40):
        half = mp.mpf(1) / 2 if q == 2 else mp.mpf(1)
        a = half * mp.log(1 + mp.sqrt(q))
        psd = mp.sqrt(q) / (1 + mp.sqrt(q))
        b = -half * mp.log(1 - psd)
        if abs(a - b) > mp.mpf(10) ** -30:
            raise Refusal("no-solution:route-disagreement", f"q = {q}: log(1 + sqrt q) {a} vs -log(1 - p_sd) {b}")
        return +a, (f"beta_c = {'(1/2) ' if q == 2 else ''}log(1 + sqrt {q}) = {mp.nstr(a, 15)} (BDC Theorem 3) = "
                    f"-{'(1/2) ' if q == 2 else ''}log(1 - p_sd), p_sd = sqrt q / (1 + sqrt q) = {mp.nstr(psd, 15)} "
                    "(BDC Theorem 1), agreeing to 1e-30")


def beta_value(q, beta):
    if isinstance(beta, str) and beta == "beta_c":
        return critical_beta(q)[0]
    b = rational(beta, "beta")
    with mp.workdps(40):
        return mp.mpf(b.numerator) / b.denominator


def coupling_K(q, beta):
    return _HOOKS["coupling_factor"] * (2 if q == 2 else 1) * float(beta)


def bond_probability(q, beta):
    return -math.expm1(-coupling_K(q, beta_value(q, beta)))


__all__ = ["beta_value", "bond_probability", "check_q", "coupling_K", "critical_beta"]
