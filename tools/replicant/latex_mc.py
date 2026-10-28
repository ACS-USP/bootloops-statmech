"""LaTeX registry of the advanced-Monte-Carlo entries (ruling A-R1: latex_entries.py is at its line guard, so
it reads this map once): entry name -> the latex_<module>.py whose function `<entry>(route)` returns its
blocks.  M2 and M3 added theirs."""
MODULES = {"cluster_mc": "mc_cluster", "wang_landau_mc": "mc_flat", "histogram_reweighting_mc": "mc_flat",
           "population_annealing_mc": "mc_pa"}

__all__ = ["MODULES"]
