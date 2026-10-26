"""The coupling K = beta J of the lattice functions, kept exact until it enters a ball.

  BETA_C            the sentinel "beta_c": K = K_c = (1/2) ln(1 + sqrt 2) (square lattice)
  XPoint(x)         beta given through x = e^{-2 beta J} (a rational 0 < x < 1): K = -(1/2) ln x
  coupling(beta, J) -> Coupling: .kind ("critical" | "rational" | "x"), .K (Fraction or None),
                       .arb() (K as an arb at the current flint precision)
A rational beta (int, Fraction, decimal string) with a rational J > 0 gives an exact rational K;
beta = 0, negative, bool or non-finite values are refused precondition:beta, J <= 0 precondition:J
(the lattice routines are for the ferromagnet).
"""
from dataclasses import dataclass
from fractions import Fraction

from ..core import Refusal
from .ball import arb_q, rational

BETA_C = "beta_c"


@dataclass(frozen=True)
class XPoint:
    x: Fraction


@dataclass(frozen=True)
class Coupling:
    kind: str
    K: Fraction | None = None
    x: Fraction | None = None

    def arb(self):
        from flint import arb
        if self.kind == "critical":
            return (1 + arb(2).sqrt()).log() / 2
        if self.kind == "x":
            return -arb_q(self.x).log() / 2
        return arb_q(self.K)


def coupling(beta, J=1):
    Jq = rational(J, "J")
    if isinstance(beta, str) and beta == BETA_C:
        if Jq != 1:
            raise Refusal("precondition:J", "beta = 'beta_c' is the sentinel for K = K_c with J = 1")
        return Coupling("critical")
    if isinstance(beta, XPoint):
        x = Fraction(beta.x)
        if not 0 < x < 1 or Jq != 1:
            raise Refusal("precondition:beta", f"XPoint x = {x}: need 0 < x < 1 (and J = 1)")
        return Coupling("x", x=x)
    return Coupling("rational", K=rational(beta, "beta") * Jq)


__all__ = ["BETA_C", "Coupling", "XPoint", "coupling"]
