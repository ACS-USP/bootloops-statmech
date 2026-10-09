# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""geomorder — geometric Laporta ordering of arXiv:2511.15381 for Kira.

Modules:
  baikov     loop-by-loop maximal-cut Baikov representation per sector (irreducible divisors P_i, exponents (a_i,b_i))
  iota       the map iota-hat: Feynman integral I_nu -> element of the form space A_omega (sum of omega_mu[Q])
  integers   r (residue count), o (pole order), w = n + r, |mu| of a form omega_mu[Q]
Conventions follow tools/maxcut/decisive_helpers.gram_baikov where they overlap.
"""
