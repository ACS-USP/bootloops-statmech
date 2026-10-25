"""Inference and codes (E6).

  ldpc.py          regular (l, r) LDPC ensembles on the binary erasure channel: density
                   evolution, the BP threshold (tangency of eps(x)) and the MAP threshold
                   (Maxwell construction: root of the trial entropy, and the area theorem by
                   quadrature), through the degree-polynomial hook DEGREE_PAIRS
  ldpc_finite.py   finite N, independent of ldpc.py: configuration-model graphs, the peeling
                   decoder, success fractions over named seeds (99% Clopper-Pearson)
  spiked.py        rank-one spiked Wigner Y = (lambda/N) x x^T + W: Bayes-optimal state evolution
                   through the PRIORS denoiser hook (gaussian | rademacher), the RS potential, the
                   threshold lambda_c, and finite-N Bayes AMP with spectral start (Student-t bands)
  sbm.py           two-group stochastic block model: the Kesten-Stigum line, an exact sampler, BP
                   with the external field (DKMZ), the BP overlap over named seeds
"""
from .ldpc import (DEGREE_PAIRS, BecThresholds, DEResult, RegularPair, bec_bp_threshold,
                   bec_density_evolution, bec_map_threshold, bec_map_threshold_area,
                   bec_thresholds, trial_entropy)
from .ldpc_finite import (LdpcGraph, PeelResult, PeelStats, ldpc_ensemble_graph,
                          ldpc_peeling_decode, ldpc_peeling_success)

__all__ = ["BecThresholds", "DEGREE_PAIRS", "DEResult", "LdpcGraph", "PeelResult", "PeelStats",
           "RegularPair", "bec_bp_threshold", "bec_density_evolution", "bec_map_threshold",
           "bec_map_threshold_area", "bec_thresholds", "ldpc_ensemble_graph",
           "ldpc_peeling_decode", "ldpc_peeling_success", "trial_entropy"]
