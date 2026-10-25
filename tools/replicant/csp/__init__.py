"""Constraint-satisfaction problems on sparse random hypergraphs (E5).

  xorsat.py         random K-XORSAT: the leaf-removal / zero-temperature cavity closed forms
                    (dynamical threshold alpha_d, satisfiability threshold alpha_s, 2-core size,
                    complexity), through a degree-law hook (DEGREE_LAWS)
  xorsat_finite.py  finite N, independent of xorsat.py: sampled instances, exact Gaussian
                    elimination over GF(2) (numpy bit-packed), leaf removal on the sampled
                    hypergraph, the satisfiable fraction over named seeds with a 99%
                    Clopper-Pearson band
  ksat.py           random K-SAT: instances, exhaustive counts (N <= 22), the annealed
                    (first-moment) bound, belief propagation on an instance
  sp.py             survey propagation: the update hook SP_UPDATES and its enumeration audit,
                    SP and its complexity on an instance, the SP population dynamics (1RSB,
                    y -> oo) and the root of its complexity Sigma(alpha)
  _checks.py        shared input checks (named seeds)
"""
from .xorsat import (DEGREE_LAWS, PoissonLaw, XorsatCore, XorsatThresholds, xorsat_bp_threshold,
                     xorsat_complexity, xorsat_core_size, xorsat_sat_threshold, xorsat_thresholds)
from .xorsat_finite import (SatProbability, gf2_solve, leaf_removal_core, xorsat_gf2_satisfiable,
                            xorsat_instance, xorsat_sampled_core, xorsat_sat_probability)

__all__ = ["DEGREE_LAWS", "PoissonLaw", "SatProbability", "XorsatCore", "XorsatThresholds",
           "gf2_solve", "leaf_removal_core", "xorsat_bp_threshold", "xorsat_complexity",
           "xorsat_core_size", "xorsat_gf2_satisfiable", "xorsat_instance", "xorsat_sampled_core",
           "xorsat_sat_probability", "xorsat_sat_threshold", "xorsat_thresholds"]
