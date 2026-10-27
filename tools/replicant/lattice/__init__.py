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
  correlation2d.py  2D correlation length above / below T_c, Onsager's strip spectrum, the nu and
                eta consistency checks
  strip_tm.py   the strip transfer matrix built and diagonalised directly (numpy, symmetry sectors)
  critical2d.py critical correlations: nearest neighbour, diagonal (product, Toeplitz, asymptote)
  anisotropic.py  the anisotropic square lattice: free energy (Kac-Ward, Baxter's diagonal form),
                critical line, duality; certified bisection
  tri_honey.py  triangular and honeycomb: free energies, critical points, star-triangle and duality
                maps, the triangular antiferromagnet's ground-state entropy
  geom_tori.py  exact finite tori and cylinders of the square, triangular and honeycomb lattices;
                the star-triangle and four-sector duality identities
"""
from . import anisotropic, bethe, correlation2d, critical2d, geom_tori, kaufman, observables1d, \
    onsager, onsager_thermo, strip_tm, torus_thermo, transfer, tri_honey
from .ball import DIGITS, Ball, certify
from .bethe import BetheFixedPoint, BetheResult, bethe_critical_beta, bethe_ising_fixed_points, \
    bethe_ising_free_energy
from .coupling import BETA_C, XPoint
from .kaufman import Recovery, beta_of_x, kaufman_free_energy, kaufman_partition_function, \
    kaufman_polynomial
from .onsager import critical_beta, onsager_free_energy, yang_magnetization
from .transfer import ising1d_free_energy, ising_transfer_matrix_free_energy, torus_partition_polynomial

__all__ = ["BETA_C", "Ball", "BetheFixedPoint", "anisotropic", "geom_tori", "tri_honey", "BetheResult", "DIGITS", "Recovery", "XPoint",
           "bethe", "bethe_critical_beta", "bethe_ising_fixed_points", "bethe_ising_free_energy",
           "beta_of_x", "correlation2d", "critical2d", "certify", "critical_beta", "ising1d_free_energy",
           "ising_transfer_matrix_free_energy", "kaufman", "kaufman_free_energy",
           "kaufman_partition_function", "kaufman_polynomial", "observables1d", "onsager",
           "onsager_free_energy", "onsager_thermo", "strip_tm", "torus_thermo",
           "torus_partition_polynomial", "transfer", "yang_magnetization"]
