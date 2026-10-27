"""Percolation (P1: trees and random graphs).

  unionfind.py    disjoint sets with path compression and union by size (shared with P2 and the
                  cluster Monte Carlo algorithms; interface in its docstring)
  laws.py         degree laws: G0, G1 and their derivatives in arb, exact moments, Molloy-Reed Q
  cavity.py       every fixed point of u = 1 - t + t G1(u), certified, with its stability; the giant,
                  the mean finite-cluster size, the threshold and the critical amplitudes
  clusters.py     the cluster-size distribution P_s (series iteration and Lagrange inversion)
  closed_forms.py the Bethe lattice and Erdos-Renyi closed forms (no generating-function code)
  sampling.py     finite random graphs over named seeds, percolated, labelled by union-find
  window.py       the Erdos-Renyi critical window: |C1| ~ N^(2/3), Nachmias-Peres Theorems 1, 2
"""
from .cavity import FixedPoint, GiantResult, percolation_fixed_points, percolation_giant
from .clusters import cluster_distribution
from .laws import parse_law
from .sampling import sample_percolation
from .unionfind import UnionFind

__all__ = ["FixedPoint", "GiantResult", "UnionFind", "cluster_distribution", "parse_law",
           "percolation_fixed_points", "percolation_giant", "sample_percolation"]
