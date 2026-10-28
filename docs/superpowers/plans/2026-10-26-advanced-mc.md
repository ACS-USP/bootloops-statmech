# Advanced Monte Carlo in `replicant`: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** add advanced Monte Carlo methods to `tools/replicant/numerics/`. E11 already has parallel tempering, Gray-code enumeration and finite-size extrapolation. The new methods are:
- cluster algorithms: Swendsen–Wang and Wolff;
- flat-histogram and generalized-ensemble methods: Wang–Landau, multicanonical, and population annealing;
- Ferrenberg–Swendsen histogram reweighting.

Each method is validated against exact results the package already has.

**Spec:** the user's request of 2026-10-26, "adicione métodos avançados de Monte Carlo como Swendsen–Wang". The method list beyond SW is the controller's call [call] and is open to override.

**Execution order:** after the percolation plan (P1–P3), because SW and Wolff share the Fortuin–Kasteleyn clusters and the union–find from P2. Then the online task O7, whose docs cover these methods too.

## Global Constraints

- Every method gets an independent exact target from existing code:
  - the E9 Kaufman/transfer-matrix torus polynomials, which give the exact density of states g(E) for the 2D Ising model on L×L tori;
  - the Onsager and Yang closed forms;
  - E11 enumeration for SK and ±J;
  - the I1 exact 1D observables.
- No method is checked only against another Monte Carlo method.
- Detailed balance or ergodicity is tested on small systems against exact Boltzmann weights, using χ² goodness-of-fit with a stated α.
- Results are statistical. Each one carries a stderr from binning or jackknife, with the integrated autocorrelation time τ_int reported, a 99% Student-t band over named seeds, and an equilibration diagnostic. Values are never final, and an unequilibrated run is labelled undetermined.
- Every check ships a planted control: a wrong bond probability, a wrong acceptance rule, or a wrong flatness criterion, each of which must fail.
- Literature values, such as the dynamic exponent z of SW versus Metropolis, come from fetched papers with their error bars, never from memory.
- Budget: ≲ 1 s cold for the whole plan in the default tier. Heavy runs are `@slow`, ≲ 2 min each.
- Commit only your own paths; others commit to main. Nothing under `.superpowers` is committed.

## Task M1: Cluster algorithms (Swendsen–Wang, Wolff)

- SW (1987) and Wolff (1989) for the Ising model, and for q-state Potts through the Fortuin–Kasteleyn representation (bond probability p = 1 − e^{−2βJ} for Ising; state the Potts form). Disordered couplings get a note: SW is valid but loses its efficiency on frustrated systems.
- Checks:
  1. Exact distribution on small tori (L ≤ 4) against Boltzmann weights from the exact polynomial, by χ².
  2. ⟨e⟩, ⟨|m|⟩ and C at L = 8–32 against the exact Kaufman torus values, within bands.
  3. Critical slowing down: measure τ_int at β_c against L for Metropolis (E11-style single flip) and for SW/Wolff, and fit z with bands. Compare with fetched literature values. If the band cannot separate SW's z from 0 or from a log, print the result and do not assert it.
  4. FK bond percolation at β_c ties to P2.

## Task M2: Wang–Landau, multicanonical, histogram reweighting

- **Wang–Landau (2001), 2D Ising:** ln g(E) estimated with the 1/t refinement (Belardinelli–Pereyra), checked against the exact torus g(E) from the E9 polynomial at L ≤ 16. The error profile is reported against the exact values.
- **Multicanonical (Berg–Neuhaus):** weights from the WL estimate. Canonical averages at any β are checked against exact values.
- **Ferrenberg–Swendsen single and multiple histogram reweighting:** a run at β0 is reweighted to β near β0 and checked against exact values. The reliable window, set by histogram overlap, is stated. A reweight outside that window is refused.

## Task M3: Population annealing

- **Population annealing (Hukushima–Iba, Machta):** a free-energy estimate from the weight normalisations, checked against exact ln Z for the 2D Ising tori and the E11 enumeration of SK and ±J at small N.
- **Families and diagnostics:** report the family-size diagnostics (ρ_t, the effective number of families), and label a run undetermined when its diagnostics fail.
- **Planted controls:** a wrong reweighting factor and a missing resampling step.

## As built

From the ledger (`.superpowers/sdd/2026-10-26-advanced-mc/progress.md`, not committed) and the task reports;
commits c23c6d1 .. dea3fe5; documented in the online plan's task O7.

- **Built.** A new package `tools/replicant/montecarlo/` (not `numerics/`, ruling A-R1) with entries `cluster_mc`
  (M1: Swendsen-Wang, Wolff and a checkerboard Metropolis baseline for the Ising / q-state Potts torus),
  `wang_landau_mc` and `histogram_reweighting_mc` (M2: Wang-Landau with the 1/t refinement against the exact torus
  g(E), multicanonical reweighting, Ferrenberg-Swendsen single / multiple histogram) and
  `population_annealing_mc` (M3: the 2D Ising torus and one SK / +-J matrix at small N), all route `sample`,
  statistical, and one shared statistics module (ruling A-R4).
- **Deviations.** The dynamic exponents are printed, not asserted, for SW and Wolff (SW cannot be separated from a
  logarithm at L <= 64; Wolff's 0.382 +- 0.066 over L = 8 .. 32 excludes LPS's quench-scaling 0.30(2)); the
  Metropolis z band is compared with Nightingale-Blote's heat-bath value under an assumed universality of local
  dynamics. <|m|> has no Kaufman torus form (ruling A-R6): exact only by enumeration on small tori. The FK tie to
  P2 is the q = 1 crossing polynomial (ruling A-R7). WL is checked against the exact g(E) up to L = 16 (ruling
  A-R3; L = 13 .. 16 by the same certified interpolation, slow only); the energy-range discovery rules are this
  package's; at L = 8 the 1/t error slope does not reproduce BP's -0.488(8) (the discovery transient, printed).
  Swendsen-Wang 1987, Wolff 1989, the Ferrenberg-Swendsen PRLs and Hukushima-Iba 2003 were not fetched (cited
  through Salas-Sokal and Liu-Polkovnikov-Sandvik, Kumar et al. 1992, Machta and Wang-Machta-Katzgraber).
  M3 added the nominal-R normalisation (WMK's actual-R~ form biases e^{ln Z~}, measured) for Machta's weighted
  average.
- **Rulings that changed scope.** A-R2: a csgraph SW for L >= 16, checked trajectory-for-trajectory against the
  union-find SW; an undetermined run never feeds an asserted band (M1), except named estimator-property tests
  (M3 ruling); the default chi^2 states its measured power; the run path labels an out-of-window default beta
  undetermined and refuses only an explicitly requested one (M2). A-R5: <= 1 s default-tier growth for the plan;
  measured at O7 by the default-selftest growth between 6e4f9ca and HEAD (three or more cold runs each): +0.5 s
  (the new test functions alone 0.7 s in-suite; cold isolated 1.3 s, imports included).
