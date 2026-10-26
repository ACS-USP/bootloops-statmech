"""Exact series and holonomic recognition (E8): the 2D Ising (Onsager) free energy.

  ising2d.py     the high-temperature series in v = tanh K from the Onsager integral (exact), the
                 independent L x L torus route
  recognize.py   recognize_holonomic (tools/annihilator), Operator, Certificate, singularities
  local.py       exact local exponents / logarithms at a point of Q(sqrt d), critical_exponent
  elliptic.py    the order-2 factor as the pulled-back hypergeometric (K(k)) operator, exactly
  susceptibility.py  the banked OEIS A002906 susceptibility coefficients (the stretch check)
"""
from .elliptic import EllipticCheck, elliptic_factor_check, theta_g_identity
from .ising2d import (MAX_ORDER, VARIABLES, ising2d_free_energy_series, onsager_moment,
                      torus_free_energy_series, torus_polynomial, torus_window)
from .local import LocalExponents, QuadNumber, critical_exponent, local_exponents
from .recognize import (PRIMES, Certificate, NoOperatorWithinBounds, Operator, Recognition,
                        Singularity, recognize_holonomic, singularities)
from .susceptibility import susceptibility_series

__all__ = ["Certificate", "EllipticCheck", "LocalExponents", "MAX_ORDER", "NoOperatorWithinBounds",
           "Operator", "PRIMES", "QuadNumber", "Recognition", "Singularity", "VARIABLES",
           "critical_exponent", "elliptic_factor_check", "ising2d_free_energy_series",
           "local_exponents", "onsager_moment", "recognize_holonomic", "singularities",
           "susceptibility_series", "theta_g_identity", "torus_free_energy_series", "torus_polynomial", "torus_window"]
