"""Dense associative memories with exponentially many patterns (P = e^{alpha N}).

  ensembles.py    zeta (cumulant generating function) and the rate function s of the overlap for Gaussian and
                  spherical patterns (Lucibello & Mezard 2024) and, DERIVED HERE, binary patterns
  rem_noise.py    the REM noise phi_{alpha,rho}(lam), its condensation line and Parisi parameter, by four routes
  thresholds.py   alpha_1(lam) (Eq. (7)), lambda_1(alpha), the all-patterns bound, basins, the Ramsauer bound
  ising.py        Ising neurons, +-1 patterns (Demircigil et al. 2017): the fixed-point exponent and the derived
                  single-flip thresholds
  sim.py          finite-N checks sharing no formula with the above: the exact-law REM sampler, the gradient
                  descent of the paper, the single-flip dynamics, Gray-code enumeration
  infonce.py      the dictionary lam = 1/(tau d) to an InfoNCE retrieval problem (read MAPPING_INFONCE.md)

Catalog entries `dense_memory` and `dense_memory_ising` (catalog_dense_memory.py); the label-replica derivation in
derive_dense_memory.py reuses families/rem.py and ansatz.py.
"""
from . import ensembles, infonce, ising, rem_noise, sim, thresholds

__all__ = ["ensembles", "infonce", "ising", "rem_noise", "sim", "thresholds"]
