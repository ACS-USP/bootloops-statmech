"""Solvable lattice models, certified in ball arithmetic (E9).

  ball.py       Ball (a python-flint arb with its certified digits) and the bridge to tools/baller
                (baller.verify / solve / run / render / certified_digits, hygiene.ctx_guard)
  coupling.py   K = beta J kept exact: rational beta, the sentinel BETA_C, XPoint (x = e^{-2K})
  onsager.py    Onsager's free energy (certified single integral, closed form at beta_c), Yang's
                magnetization, an independent mpmath evaluation
  kaufman.py    Kaufman's finite-torus partition function and its exact integer polynomial
  transfer.py   the exact torus transfer matrix (integers), the 1D Ising ring
  bethe.py      the Bethe lattice / random regular graph: all BP fixed points, certified
  observables1d.py  1D magnetization, susceptibility, correlations, correlation length; the exact
                ring and its explicit N -> oo limit
  onsager_thermo.py 2D energy and specific heat (Onsager 1944), the log divergence's amplitude
  torus_thermo.py   finite-torus energy and specific heat (Kaufman's Z, the exact polynomial)
"""
from . import bethe, kaufman, observables1d, onsager, onsager_thermo, torus_thermo, transfer
from .ball import DIGITS, Ball, certify
from .bethe import BetheFixedPoint, BetheResult, bethe_critical_beta, bethe_ising_fixed_points, \
    bethe_ising_free_energy
from .coupling import BETA_C, XPoint
from .kaufman import Recovery, beta_of_x, kaufman_free_energy, kaufman_partition_function, \
    kaufman_polynomial
from .onsager import critical_beta, onsager_free_energy, yang_magnetization
from .transfer import ising1d_free_energy, ising_transfer_matrix_free_energy, torus_partition_polynomial

__all__ = ["BETA_C", "Ball", "BetheFixedPoint", "BetheResult", "DIGITS", "Recovery", "XPoint",
           "bethe", "bethe_critical_beta", "bethe_ising_fixed_points", "bethe_ising_free_energy",
           "beta_of_x", "certify", "critical_beta", "ising1d_free_energy",
           "ising_transfer_matrix_free_energy", "kaufman", "kaufman_free_energy",
           "kaufman_partition_function", "kaufman_polynomial", "observables1d", "onsager",
           "onsager_free_energy", "onsager_thermo", "torus_thermo",
           "torus_partition_polynomial", "transfer", "yang_magnetization"]
