"""Monte Carlo and exact enumeration as independent checks (E11).

  enumeration.py  exact +-J SK spectra (Gray-code blocks), exact SK ground states (N <= 26),
                  exact Ising perceptron solution counts (N <= 30)
  pt.py           parallel tempering for SK couplings (numpy-vectorised over samples and
                  temperatures), equilibration diagnostic
  finite_size.py  SK ground-state energy from PT minima at several N, extrapolated in N^-omega
"""
from .enumeration import (PmOneSpectrum, enumerate_pm_one_sk, exact_ground_energy, gray_states,
                          ising_perceptron_capacity, ising_perceptron_counts,
                          ising_perceptron_log_count, random_pm_one_J)
from .finite_size import FSResult, sk_ground_energy_finite_size
from .pt import Equilibration, PTResult, equilibration, geometric_betas, pt_sk

__all__ = ["Equilibration", "FSResult", "PTResult", "PmOneSpectrum", "enumerate_pm_one_sk",
           "equilibration", "exact_ground_energy", "geometric_betas", "gray_states",
           "ising_perceptron_capacity", "ising_perceptron_counts", "ising_perceptron_log_count",
           "pt_sk", "random_pm_one_J", "sk_ground_energy_finite_size"]
