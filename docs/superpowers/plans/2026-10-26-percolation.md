# Percolation in `replicant`: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** add percolation as new certified, cavity and statistical entries:
- tree-like and random-graph percolation (exact via cavity / generating functions);
- 2D lattice percolation with exact thresholds and exponents, checked by Monte Carlo with finite-size scaling;
- k-core / bootstrap percolation, linked to the E5 XORSAT 2-core.

**Spec:** the user's request of 2026-10-26, "inclua problemas de percolação". The scope below is the controller's call [call], open to override. It fits the existing patterns: cavity, certified balls, finite-N bands, and banked cites.

**Execution order:** after the Ising plan (I1–I3), and before online task O7, so that O7's documentation pass covers it.

## Global Constraints

The constraints are the same as for E5–E11 and the online plan.
- **Sources.** Thresholds and exponents come from fetched sources, never from memory. These include Kesten 1980; Sykes–Essam 1964; Wierman for the triangular and honeycomb bond thresholds; den Nijs and Nienhuis for the exponents; Cardy 1992 for the crossing formula; Newman–Ziff 2000/2001 for the algorithm and numerical thresholds; and Dorogovtsev–Goltsev–Mendes and Pittel–Spencer–Wormald for k-cores.
- **Numerical thresholds.** A threshold with no closed form, such as square-site p_c ≈ 0.5927…, is banked from a fetched paper with that paper's error bar.
- **Labels.**
  - Exact thresholds that rest on proofs are `exact-by-literature`, citing the theorem.
  - Exponents for 2D percolation are proved for triangular site percolation (Smirnov–Werner). Elsewhere they are physics-level.
  - Monte Carlo results are statistical, with Student-t bands over named seeds, and are never final.
- **Finite-size scaling.** It states the band and the exponent systematic. A contrast the data cannot resolve is printed, not asserted.
- **Controls and budget.**
  - Every check has a planted control.
  - The default tier costs ≲ 1 s cold for the whole plan; heavy Monte Carlo goes in `@slow`, at most about 2 min per test.
- **Commits.** Commit only your own paths; others commit to main. Nothing under `.superpowers` is committed.

## Task P1: Percolation on trees and random graphs (cavity)

- **Bethe lattice with coordination k:**
  - p_c = 1/(k−1);
  - P∞(p) from the cavity fixed point;
  - the mean finite-cluster size and its divergence;
  - the cluster-size distribution from generating functions.
- **Erdős–Rényi and configuration-model graphs.** The giant component S solves S = 1 − G0(1 − S·…) (state the exact form). The threshold is ⟨k²⟩/⟨k⟩ = 2 (Molloy–Reed). Site and bond dilution are included.
- **Certification.** Fixed points are certified with balls, as E9's bethe_ising does: return every root, refuse an ambiguous one, and certify stability from the computed derivative.
- **Checks.**
  - Exact sampling on finite graphs with union–find, over named seeds with bands.
  - Finite-size scaling of the critical window at N^{−1/3} for ER, with a stated band.
  - Planted errors: a wrong G0/G1, and a dropped root.

## Task P2: 2D lattice percolation

- **Exact thresholds:**
  - square-lattice bond percolation, 1/2 (Kesten);
  - triangular-lattice site percolation, 1/2;
  - triangular bond, 2 sin(π/18);
  - honeycomb bond, 1 − 2 sin(π/18), the last two related by star–triangle duality.
- **Exact exponents:** β = 5/36, ν = 4/3, γ = 43/18, and Cardy's crossing formula for rectangles.
- **Monte Carlo.** A Newman–Ziff union–find sweep in its own code. It measures:
  - spanning probability;
  - the largest cluster;
  - mean cluster size.
- **Finite-size scaling:**
  - spanning-probability crossings for p_c;
  - a collapse for 1/ν, with bootstrap bands.
  - Validation: the analysis must recover the exact p_c and ν on the exactly solved lattices before it is applied to square-site percolation, which is banked numerically.
  - Cardy's formula is checked against the Monte Carlo crossing probability at a few aspect ratios.
- **Exact identities.** The star–triangle and duality relations on the threshold polynomials are checked as exact identities.
- **Planted errors:** a wrong neighbourhood, and a wrong boundary condition in the spanning test.

## Task P3: k-core and bootstrap percolation

- **k-core on random graphs.** The cavity / generating-function equations give the discontinuous k ≥ 3 transition and the core size at the jump. For k = 2 on hypergraphs, the result must reproduce the E5 XORSAT 2-core, as a cross-check against an existing entry.
- **Bootstrap percolation on random regular graphs**, where a fetched closed form exists.
- **Checks.** The exact peeling (pruning) algorithm on sampled graphs, with bands. Planted errors: a wrong threshold k, and a missing hyperedge term.

## As built

From the ledger (`.superpowers/sdd/2026-10-26-percolation/progress.md`, not committed) and the task reports;
commits cefa87f .. 6e4f9ca; documented in the online plan's task O7.

- **Built.** P1: package `percolation/` with one union-find (ruling P-R2, reused by P2 and M1), entries
  `percolation_bethe` and `percolation_graph` (certified cavity fixed points, P_inf, mean cluster size,
  Molloy-Reed, cluster-size distributions; a new composer route `sample` for finite graphs), the ER critical
  window in the slow tier. P2: `percolation_lattice` (square, triangular, honeycomb; certified thresholds, the
  Smirnov-Werner exponents, Cardy's formula by two routes, exact crossing polynomials with their duality
  identities; Newman-Ziff sweeps checked against enumeration; finite-size scaling validated on the four solved
  lattices before square site). P3: `percolation_kcore`, `percolation_bootstrap` (certified k-core and bootstrap
  fixed points and birth points; the E5 2-core consumed as a cross-check; exact peeling and spreading).
- **Deviations.** Kesten 1980, Wierman 1981, Sykes-Essam 1964, den Nijs, Nienhuis, Fisher-Essam 1961 and Balogh &
  Pittel 2007 were not fetched: thresholds and exponents rest on Bollobas-Riordan, Grimmett-Manolescu and
  Smirnov-Werner, the Bethe formulas on Malthe-Sorenssen and Abraham-Delmas, bootstrap percolation on Janson and
  Bartha-Pete. 2 sin(pi/18) and Cardy's aspect convention are derived here. Pittel-Spencer-Wormald was fetched in
  P3's fix round (c4 ~ 5.14; Janson-Luczak's 5.15 banked beside it). The FSS validation states its achieved
  resolution (the collapse rejects p_c +- 0.005, the crossing estimator +- 0.01 only); the systematic's window/10
  floor was chosen after seeing the data (disclosed); the slow statistical RefChecks use the bootstrap standard
  deviation as a Student-t stderr, so their "99%" has no exact coverage. K >= 3 hypergraph cores are derived
  here; further first-order core transitions above the birth point are not enumerated.
- **Rulings that changed scope.** P-R1 (new package, per-topic files, guarded files <= 380); P-R2 (one shared
  union-find); a correct sampler is never refused at runtime, statistical sampler checks are diagnostics; the
  Bethe p_c is exact-by-literature through the Bienayme-Galton-Watson extinction theorem; `crossing_exact` keeps
  the "exact by construction" convention; both c_k prints banked with a tolerance covering both; an auxiliary
  route never blocks a certified run; a parameter at a certified threshold is refused `ambiguous:critical-point`.
  P-R3: <= 1 s default-tier growth for the plan. Measured at O7 by the default-selftest growth between 2091a08 and
  6e4f9ca (three cold runs each): +1.1 s, slightly above the ruling (the task reports' accounting 0.30 + 0.30 +
  0.28 s; the new test functions alone 0.5 s in-suite; P2's cold isolated 0.38-0.40 s was parked for O7: the
  plan's test modules together take 1.3 s cold isolated, imports included). `catalog_percolation_kcore.py` (402
  lines) was split at O7, the bootstrap entry moving to `catalog_percolation_bootstrap.py`.
