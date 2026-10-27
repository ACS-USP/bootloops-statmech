"""Closed forms that do not go through the generating-function code (percolation/laws.py, cavity.py,
clusters.py are not imported): the independent route of the cavity entries' values, in arb balls or
exact rationals.  Each says where it comes from.

Bethe lattice, coordination k, occupation t (Malthe-Sorenssen, Percolation Theory Using Python,
Lecture Notes in Physics 1029 (2024), Ch. 3 (MS)):
  u for k = 3: the root (1 - t)/t of t u^2 - u + 1 - t = 0 (MS Eqs. (3.7)-(3.10), there for site
      occupation Q; the same equation for bond, CNSW Eq. (5) with G1 = u^2);
  u for k = 4: t u^3 - u + 1 - t = (u - 1)(t u^2 + t u - (1 - t)), u = (sqrt(t^2 + 4 t (1 - t)) - t)/(2t)
      (derived here);
  mean cluster size below t_c: (1 + t)/(1 - (k - 1) t) (MS Eq. (3.20); its middle form prints
      "1 - p p_c" for 1 - p/p_c, a slip recorded in the bank cite);
  P_s = s g_s t^(s-1) (1 - t)^((k-2)s + 2) for an occupied vertex (MS Eq. (3.27): n(s, p) =
      g_s p^s (1 - p)^((Z-2)s+2), and P_s = s n_s / p), with the animal count
      g_s = k ((k - 1) s)! / (s! ((k - 2) s + 2)!)   (derived here by Lagrange inversion; MS leave g_s
      open, "In general, we do not know g_s"; the tests count subtrees of the tree directly).
Erdos-Renyi, mean degree c, bond occupation t (lambda = c t; site occupation q: P_inf = q S(c q)):
  S solves S = 1 - exp(-lambda S) (NSW Eq. (37)): S = 1 + W0(-lambda e^-lambda)/lambda for lambda > 1
      (principal Lambert branch; derived here);
  <s> = 1/(1 - lambda + lambda S) (NSW Eq. (38));
  P_s = (lambda s)^(s-1) e^(-lambda s) / s!  (the Borel law; derived here by Lagrange inversion).
"""
from fractions import Fraction
from math import factorial

from ..core import Refusal
from ..lattice.ball import arb_q, baller


def bethe_u(k, t):
    """The nontrivial fixed point for k = 3, 4 (an arb; None for other k or below threshold)."""
    t = Fraction(t)
    if t * (k - 1) <= 1 or k not in (3, 4):
        return None
    if k == 3:
        return arb_q((1 - t) / t)
    T = arb_q(t)
    return ((T * T + 4 * T * (1 - T)).sqrt() - T) / (2 * T)


def bethe_mean_below(k, t):
    t = Fraction(t)
    if not t * (k - 1) < 1:
        raise Refusal("precondition:below-threshold", "MS Eq. (3.20) holds below p_c only")
    return (1 + t) / (1 - (k - 1) * t)


def animals(k, s):
    """g_s: clusters of s sites containing a given site of the Bethe lattice, per site / s."""
    return Fraction(k * factorial((k - 1) * s), factorial(s) * factorial((k - 2) * s + 2))


def bethe_ps(k, t, s):
    t = Fraction(t)
    return s * animals(k, s) * t ** (s - 1) * (1 - t) ** ((k - 2) * s + 2)


def er_giant(lam):
    """S(lambda) by Lambert W (an arb; 0 for lambda <= 1)."""
    lam = Fraction(lam)
    if lam <= 1:
        return arb_q(0)
    L = arb_q(lam)
    return 1 + (-L * (-L).exp()).lambertw() / L


def er_mean(lam, S):
    return 1 / (1 - arb_q(lam) + arb_q(lam) * S)


def borel_ps(lam, s):
    L = arb_q(Fraction(lam))
    return (L * s) ** (s - 1) * (-L * s).exp() / factorial(s)


def prec_guard(prec=256):
    return baller().hygiene.ctx_guard(prec=prec)


__all__ = ["animals", "bethe_mean_below", "bethe_ps", "bethe_u", "borel_ps", "er_giant", "er_mean",
           "prec_guard"]
