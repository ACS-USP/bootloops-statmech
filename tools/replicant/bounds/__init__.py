"""Rigorous bounds on the SK pressure, certified in ball arithmetic (E10).

  gauss.py   certified Gaussian expectations E g(a z + b) (acb.integral + analytic tails)
  krsb.py    Guerra's K-step broken-replica bound P_K(q, m), nested certified integrals
  guerra.py  the RS bound P_RS(q), all its stationary points, its certified minimum over q;
             an independent mpmath route
  chain.py   check_bound_chain: the package's RS / 1RSB / K-RSB optimiser outputs against the
             certified bounds (consistency with the proved ordering of the infima)
"""
from .chain import ChainReport, LevelCheck, Link, PackageValue, check_bound_chain, package_values
from .guerra import (BestBound, Stationary, guerra_rs_best_bound, guerra_rs_bound, rs_bound_mp,
                     rs_stationary_q_mp)
from .krsb import guerra_1rsb_bound, guerra_krsb_bound

__all__ = ["BestBound", "ChainReport", "LevelCheck", "Link", "PackageValue", "Stationary",
           "check_bound_chain", "guerra_1rsb_bound", "guerra_krsb_bound", "guerra_rs_best_bound",
           "guerra_rs_bound", "package_values", "rs_bound_mp", "rs_stationary_q_mp"]
