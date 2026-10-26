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
