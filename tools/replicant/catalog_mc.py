"""The advanced-Monte-Carlo catalog entries (ruling A-R1): one function the catalog calls, one module per topic.
  catalog_mc_cluster.py   cluster_mc (M1: Swendsen-Wang, Wolff, Metropolis baseline; route sample)
M2 and M3 add their entries to mc_entries() and their LaTeX modules to latex_mc.MODULES.
"""


def mc_entries():
    from .catalog_mc_cluster import cluster_mc
    return [cluster_mc()]


__all__ = ["mc_entries"]
