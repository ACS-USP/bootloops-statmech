"""The advanced-Monte-Carlo catalog entries (ruling A-R1): one function the catalog calls, one module per topic.
  catalog_mc_cluster.py   cluster_mc (M1: Swendsen-Wang, Wolff, Metropolis baseline; route sample)
  catalog_mc_wl.py        wang_landau_mc (M2: Wang-Landau 1/t density of states, multicanonical; route sample)
  catalog_mc_fs.py        histogram_reweighting_mc (M2: Ferrenberg-Swendsen single / multiple histogram; route sample)
  catalog_mc_pa.py        population_annealing_mc (M3: population annealing, free energy and families; route sample)
M2 and M3 add their entries to mc_entries() and their LaTeX modules to latex_mc.MODULES.
"""


def mc_entries():
    from .catalog_mc_cluster import cluster_mc
    from .catalog_mc_fs import histogram_reweighting_mc
    from .catalog_mc_pa import population_annealing_mc
    from .catalog_mc_wl import wang_landau_mc
    return [cluster_mc(), wang_landau_mc(), histogram_reweighting_mc(), population_annealing_mc()]


__all__ = ["mc_entries"]
