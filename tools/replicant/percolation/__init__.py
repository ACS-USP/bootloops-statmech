"""Percolation (P1: trees and random graphs; P2: 2D lattices).

  unionfind.py    disjoint sets with path compression and union by size (shared with P2 and the
                  cluster Monte Carlo algorithms; interface in its docstring)
  laws.py         degree laws: G0, G1 and their derivatives in arb, exact moments, Molloy-Reed Q
  cavity.py       every fixed point of u = 1 - t + t G1(u), certified, with its stability; the giant,
                  the mean finite-cluster size, the threshold and the critical amplitudes
  clusters.py     the cluster-size distribution P_s (series iteration and Lagrange inversion)
  closed_forms.py the Bethe lattice and Erdos-Renyi closed forms (no generating-function code)
  sampling.py     finite random graphs over named seeds, percolated, labelled by union-find
  window.py       the Erdos-Renyi critical window: |C1| ~ N^(2/3), Nachmias-Peres Theorems 1, 2
  exact2d.py      2D thresholds (closed forms, a second route, star-triangle and duality identities)
                  and exponents (scaling relations), certified
  cardy.py        Cardy's crossing formula for rectangles, two routes
  enumerate2d.py  exhaustive enumeration of small open lattices: crossing polynomials, duality
                  identities, an exact cross-check of the Newman-Ziff sweep
  lattice2d.py    Newman-Ziff sweeps on square / triangular / honeycomb lattices (union-find)
  fss2d.py        finite-size scaling: crossings of R_L, a collapse for 1/nu, bootstrap(-resampling) bands
  kcore.py        (P3) the k-core equation of random graphs / K-uniform hypergraphs: every fixed point,
                  certified, the largest selected; the core's size
  kcore_threshold.py  the k-core's birth point (a certified global minimum), the jump, and the
                  Balogh-Pittel closed forms of bootstrap percolation (the infection dynamics)
  kcore_audit.py  the selection's audits: the exclusion scan above it and the peeling bracket
  kcore_dgm.py    the second route in Dorogovtsev-Goltsev-Mendes' own form, mpmath
  kcore_sample.py exact peeling and bootstrap spreading on sampled graphs over named seeds
"""
from .cavity import FixedPoint, GiantResult, percolation_fixed_points, percolation_giant
from .clusters import cluster_distribution
from .laws import parse_law
from .sampling import sample_percolation
from .unionfind import UnionFind

__all__ = ["FixedPoint", "GiantResult", "UnionFind", "cluster_distribution", "parse_law",
           "percolation_fixed_points", "percolation_giant", "sample_percolation"]
