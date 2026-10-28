"""Advanced Monte Carlo (M1 cluster algorithms; M2, M3 reuse stats.py).

  stats.py         shared statistics: tau_int (self-consistent window), binning, jackknife, Student-t
                   bands over named seeds, the equilibration diagnostic, chi^2 goodness of fit, z fits
  fk.py            Fortuin-Kasteleyn coupling: bond probability, exact Potts critical points (two routes)
  exact_small.py   exact Boltzmann distributions of small Potts / Ising tori by enumeration
  cluster.py       Swendsen-Wang and Wolff on the shared union-find (the reference implementation)
  cluster_fast.py  vectorised Swendsen-Wang for L >= 16, checked against cluster.py trajectory-for-trajectory
  metropolis.py    checkerboard single-spin-flip Metropolis for the 2D Ising model (the local baseline)
  run.py           runs over named seeds and their statistics
  fk_percolation.py  the FK machinery at q = 1 is bond percolation: P2's exact crossing polynomial
  dos_exact.py     M2: the exact g(E) of the Ising torus (Kaufman's integer polynomial, L <= 16), canonical averages
  flat_walk.py     M2: the single-spin-flip walk in energy shared by Wang-Landau and the multicanonical run
  wang_landau.py   M2: Wang-Landau (original and 1/t schedules), the ln g error profile and the error-against-time slope
  multicanonical.py  M2: multicanonical runs with WL weights, reweighting to beta, the fixed-weight chi^2 tests
  reweight.py      M2: Ferrenberg-Swendsen single and multiple histogram reweighting, the overlap window
  sk_metropolis.py   M3: sequential single-spin-flip Metropolis for SK / +-J couplings on a population, exact ln Z
  population.py    M3: population annealing, the free-energy estimator, its finite-R bias, family diagnostics
"""
