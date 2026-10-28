"""The dictionary between the dense associative memory and an InfoNCE retrieval problem (see
MAPPING_INFONCE.md for what it does and does not say).

InfoNCE with n unit document vectors v_x in R^d and a unit query u,

    l_tau(u) = log sum_x exp(<u, v_x>/tau) - (<u, v_1> + <u, v_2>)/(2 tau),

has the softmax denominator of the dense associative memory with N = d neurons, x = sqrt(d) u, xi^x = sqrt(d) v_x
(so |x|^2 = |xi|^2 = N, rho = 1, spherical patterns for unit documents drawn uniformly on the sphere) and

    exp(<u, v>/tau) = exp(lam x.xi)   with   lam = 1/(tau d),      n = P = e^{alpha d}.

The dictionary is exact; what the memory's REM analysis then says is only about a query u that is INDEPENDENT of
the documents it is compared with (the paper's condition for Eq. (6)), with i.i.d. random documents:

  noise      log sum_{x >= 3} exp(<u,v_x>/tau) = (1/tau) phi_{alpha,1}(lam)  [spherical ensemble]
  signal     <u, v_a>/tau with <u, v_a> = level: 1 for u = v_a (one positive; the paper's Eq. (7)), and
             sqrt((1 + c)/2) for u = (v_1 + v_2)/|v_1 + v_2|, c = <v_1, v_2> (two positives as a mixture state;
             1/sqrt(2) at c = 0)
  criterion  the signal wins when phi_{alpha,1}(lam) < level: n < exp(d alpha_1(lam; level)).

`level` != 1 is the EXTENSION of Eq. (7) in thresholds.py (not in the paper); the mixture state is not treated by
the paper.  Nothing here concerns an OPTIMISED query, optimised documents, the ranking of a pair (a top-2 event) or
a union over all pairs: those are exactly the ingredients of the critical n(d, tau) and the per-query threshold
c(d) tau of the free-embedding problem, and the REM analysis does not contain them.

  lam_from_tau(tau, d)              1/(tau d)
  tau_from_lam(lam, d)              1/(lam d)
  mixture_level(c)                  sqrt((1 + c)/2)
  alpha_threshold(d, tau, ...)      the load alpha at which phi_{alpha,1}(1/(tau d)) = level
  n_threshold(d, tau, ...)          exp(d alpha): the largest n for which the (non-adapted) query's signal beats the
                                    REM noise of n i.i.d. documents
  tau_threshold(n, d, ...)          the largest tau with that property (Refused no-solution:no-retrieval when even
                                    tau -> 0 fails: n above the hard-max cap)
  n_cap(level, d)                   the tau -> 0 limit (1 - level^2)^(-d/2) = exp(d s(level)): 2^(d/2) at 1/sqrt(2)
"""
import mpmath as mp

from ..core import Refusal
from .ensembles import _prec, num, rate
from .thresholds import alpha_1, lambda_1


def lam_from_tau(tau, d):
    """lam = 1/(tau d): exp(<u,v>/tau) = exp(lam x.xi) for x = sqrt(d) u, xi = sqrt(d) v."""
    with _prec():
        return 1 / (num(tau, "tau", positive=True) * num(d, "d", positive=True))


def tau_from_lam(lam, d):
    with _prec():
        return 1 / (num(lam, "lambda", positive=True) * num(d, "d", positive=True))


def mixture_level(c=0):
    """<u, v_1> = <u, v_2> = sqrt((1 + c)/2) for u = (v_1 + v_2)/|v_1 + v_2| and c = <v_1, v_2> in (-1, 1)."""
    with _prec():
        c = num(c, "c")
        if not -1 < c < 1:
            raise Refusal("precondition:known-parameter", f"c = {c!r}: need -1 < c < 1")
        return mp.sqrt((1 + c) / 2)


def alpha_threshold(d, tau, level=1, ensemble="spherical"):
    """alpha with phi_{alpha,1}(1/(tau d)) = level (thresholds.alpha_1 at lam = 1/(tau d))."""
    return alpha_1(ensemble, lam_from_tau(tau, d), level)


def n_threshold(d, tau, level=1, ensemble="spherical"):
    """exp(d alpha_threshold): the exponential-order count, with the sub-exponential factor undetermined."""
    with _prec():
        return mp.exp(num(d, "d", positive=True) * alpha_threshold(d, tau, level, ensemble))


def tau_threshold(n, d, level=1, ensemble="spherical"):
    """The largest tau such that n documents are below the threshold: tau = 1/(d lambda_1(alpha)), alpha = log n/d
    (phi decreases in lam, so a smaller tau is safer).  Refused no-solution:no-retrieval if alpha is above the
    hard-max cap s(level)."""
    with _prec():
        d_, n_ = num(d, "d", positive=True), num(n, "n", positive=True)
        return tau_from_lam(lambda_1(ensemble, mp.log(n_) / d_, level), d_)


def n_cap(level, d, ensemble="spherical"):
    """exp(d s(level)): the tau -> 0 limit of n_threshold, the largest n at which the best possible random document
    still has overlap below the level (the cap-covering bound)."""
    with _prec():
        return mp.exp(num(d, "d", positive=True) * rate(ensemble, level, 1))


__all__ = ["alpha_threshold", "lam_from_tau", "mixture_level", "n_cap", "n_threshold", "tau_from_lam",
           "tau_threshold"]
