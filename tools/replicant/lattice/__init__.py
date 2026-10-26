"""Solvable lattice models, certified in ball arithmetic (E9).

  ball.py       Ball (a python-flint arb with its certified digits) and the bridge to tools/baller
                (baller.verify / solve / run / render / certified_digits, hygiene.ctx_guard)
  coupling.py   K = beta J kept exact: rational beta, the sentinel BETA_C, XPoint (x = e^{-2K})
  onsager.py    Onsager's free energy (certified single integral, closed form at beta_c), Yang's
                magnetization, an independent mpmath evaluation
  kaufman.py    Kaufman's finite-torus partition function and its exact integer polynomial
  transfer.py   the exact torus transfer matrix (integers), the 1D Ising ring
  bethe.py      the Bethe lattice / random regular graph: all BP fixed points, certified
"""
from . import bethe, kaufman, onsager, transfer
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
           "kaufman_partition_function", "kaufman_polynomial", "onsager", "onsager_free_energy",
           "torus_partition_polynomial", "transfer", "yang_magnetization"]
