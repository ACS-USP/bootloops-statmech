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
"""
