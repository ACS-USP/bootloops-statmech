"""AMP, state evolution and TAP for SK, and GAMP for the teacher-student perceptron (finite
N): a route independent of the replica and cavity code (this subpackage imports only numpy,
mpmath, scipy.special and replicant.core).

  sk_matrix(N, seed, law)                         -> J (gaussian | pm_one)
  sk_amp(J, beta, *, iters, seed, damping, h, reaction, init) -> AMPResult
  sk_amp_ensemble(N, beta, h, *, seeds, iters, law, reaction) -> AMPEnsemble (99% t band)
  sk_state_evolution(beta, h, *, q0, iters)       -> SEResult(q, trajectory, fixed_points, ...)
  sk_two_time_state_evolution(beta, h, *, iters)  -> TwoTimeSE(Q, D, increment)
  sk_tap_free_energy(m, J, beta, h, *, onsager)   -> mpf phi_TAP = -beta F_TAP / N
  sk_tap_gradient(m, J, beta, h, *, onsager)      -> N d phi_TAP / d m
  tap_fixed_point(J, beta, h, *, seed, ...)       -> TAPResult
  perceptron_instance(N, alpha, prior, seed)      -> (Phi, y, w*) (prior gaussian | ising)
  gamp_perceptron(X, y, prior, *, iters, seed, teacher) -> GAMPResult
  gamp_ensemble(N, alpha, prior, *, seeds, iters) -> GAMPEnsemble (99% t bands)
  perceptron_state_evolution(alpha, prior, teacher=True, *, iters) -> gamp_se.SEResult
  ising_thresholds()                              -> Thresholds(alpha_it, alpha_amp)
See amp/gamp.py and amp/gamp_se.py for the perceptron (Bayes-optimal only on the
Nishimori line; GAMP SE for GLMs is a claim, Barbier et al. 2019 Claim 1), and amp/sk.py
for the SK conventions and validity statement (SE is a theorem for Gaussian J at each
fixed t; convergence of the iterates only on the RS-stable side of the AT line,
beta^2 E cosh^-4 <= 1, with h > 0).
"""
from ._stats import Band, student_band
from .sk import AMPEnsemble, AMPResult, check_matrix, sk_amp, sk_amp_ensemble, sk_matrix
from .state_evolution import (FixedPoint, SEResult, TwoTimeSE, random_init_q,
                              sk_state_evolution, sk_two_time_state_evolution)
from .tap import TAPResult, sk_tap_free_energy, sk_tap_gradient, tap_fixed_point
from .gamp import GAMPEnsemble, GAMPResult, gamp_ensemble, gamp_perceptron, perceptron_instance
from .gamp_se import Thresholds, ising_thresholds, perceptron_state_evolution

__all__ = ["GAMPEnsemble", "GAMPResult", "Thresholds", "gamp_ensemble", "gamp_perceptron",
           "ising_thresholds", "perceptron_instance", "perceptron_state_evolution",
           "AMPEnsemble", "AMPResult", "Band", "FixedPoint", "SEResult", "TAPResult",
           "TwoTimeSE", "check_matrix", "random_init_q", "sk_amp", "sk_amp_ensemble",
           "sk_matrix", "sk_state_evolution", "sk_tap_free_energy", "sk_tap_gradient",
           "sk_two_time_state_evolution", "student_band", "tap_fixed_point"]
