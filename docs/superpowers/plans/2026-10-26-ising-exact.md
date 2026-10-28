# Exact Ising solutions in 1D and 2D: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend the certified lattice entries from E9 (`ising1d`, `ising2d`, in `tools/replicant/lattice/`) with the classical exact results beyond the free energy:
- 1D observables;
- 2D energy and specific heat;
- 2D correlation length and critical correlations;
- anisotropic square, triangular and honeycomb lattices.

**Spec:** the user's request of 2026-10-26: "inclua as soluções para modelo de ising em 1d e 2d". The scope was chosen through AskUserQuestion: all four groups. The E9 design is `docs/superpowers/specs/2026-10-24-statmech-extensions-design.md`, item E9.

**Execution order:** after online task O6 and before O7, so that O7's documentation pass covers these entries too. This order is recorded in the online ledger.

## Global Constraints

These are the same as E9:
- Certified values are ball enclosures via python-flint and tools/baller, with their radii in `errors`, printed as 40-digit enclosures.
- `rigor: certified` applies only with a true enclosure. Anything else is labelled as it is.
- Closed forms come from fetched sources, never from memory: Onsager 1944, Yang 1952, Kaufman 1949, Baxter's book, Houtappel 1950 or Wannier 1950 for the triangular and honeycomb lattices, McCoy–Wu, and the Montroll–Potts–Ward or Wu correlation results. Where only a secondary source can be read, it is named.
- Every closed form has an independent exact route:
  - exact finite rings or tori via the transfer matrix, the Kaufman polynomials, or derivatives of the exact polynomials;
  - the E8 high-temperature series;
  - an independent mpmath quadrature.
- Every check has a planted control.
- The default-tier cost of the whole plan is ≲ 1 s cold; heavy work goes in `@slow`.
- Commit only your own paths; others commit to main. Nothing under `.superpowers` is committed. The commit trailer is as in the harness.

## Task I1: 1D observables and 2D thermodynamics

- **1D:**
  - magnetisation m(β, h), susceptibility χ(β, h), correlation ⟨s0 sr⟩ − m², and correlation length ξ = 1/ln(λ+/λ−), all certified;
  - each checked against exact finite rings (exact rational derivatives of ln Z_N) with the N → ∞ limit made explicit.
- **2D:**
  - the energy u(β) and specific heat c(β) from Onsager, both certified away from β_c;
  - the logarithmic divergence at β_c, with its amplitude banked from a fetched source.
  - Checked against derivatives of the exact Kaufman torus polynomials, as a finite-size sequence, and against the E8 series where it converges.
  - Planted errors: a wrong amplitude, and a wrong derivative order.

## Task I2: 2D correlation length and critical correlations

- **Correlation length:** the exact ξ(T) above and below T_c, from a fetched source such as McCoy–Wu or Baxter. Checked by exact transfer-matrix strips: the ratio of the two largest eigenvalues at widths L ≤ 10, extrapolated in L, with a band.
- **Critical exponents:** ν = 1 and η = 1/4, checked as consistency tests on the exact data, with a stated resolution.
- **Critical correlation:** the spin–spin correlation at T_c, if a fetched closed form exists, for example the nearest-diagonal correlations. Otherwise reported as not done.

## Task I3: Anisotropic square, triangular and honeycomb lattices

- **Anisotropic square lattice:** the critical line sinh 2K1 sinh 2K2 = 1, and the anisotropic Onsager free energy.
- **Triangular and honeycomb lattices:** critical points and free energies (Houtappel/Wannier, Baxter).
- **Independent routes:** exact transfer matrices on small periodic lattices of each geometry, and the star–triangle (triangular ↔ honeycomb) and duality relations checked as exact identities.
- **Planted errors:** a wrong lattice weight, and a wrong duality map.

## As built

From the ledger (`.superpowers/sdd/2026-10-26-ising-exact/progress.md`, not committed) and the task reports;
commits 8ff6aec .. 2091a08; documented in the online plan's task O7.

- **Built.** I1: the 1D magnetization, susceptibility, connected correlation and correlation length, certified,
  against exact finite rings with the N -> oo bound; Onsager's 2D energy and specific heat (Eqs. (116), (117),
  the log amplitude of (120)) against Kaufman-torus derivatives, integer moments and the E8 series. I2: the 2D
  correlation length above and below T_c from Onsager's strip spectrum, the strip -> bulk sequence and a direct
  strip matrix, nu / eta consistency checks, the critical nearest-neighbour and diagonal correlations (Perk &
  Au-Yang, Deift-Its-Krasovsky). I3: three new entries `ising2d_anisotropic`, `ising_triangular`,
  `ising_honeycomb` (route `exact`, certified) rather than parameters of `ising2d`, which stays unchanged.
- **Deviations.** Houtappel 1950, Wannier 1950, McCoy-Wu and Wu-McCoy-Tracy-Barouch were not fetched: the
  triangular free energy comes from Athanasopoulos & Ueltschi's Kac-Ward integral, the honeycomb from Baxter's
  star-triangle and duality maps and Pierre, Bernu & Messio, xi_0+ from Dohm. Two errors in the controller's I2
  dispatch were corrected from the sources (the critical strip gap is pi eta = pi/4, Cardy Eq. (19); 1/xi =
  2(K* - K) above T_c, 4(K - K*) below). Below T_c `correlation_length` is `not-computed` (its identification with
  Onsager's two-particle gap is derived here), `two_particle_gap` exact-by-literature. The 1D chi(h) is
  exact-by-literature "derivative taken here". The direct strip matrix is double precision, not certified. The
  I2 fix restored `ising2d` at large beta (a strip overflow guard had refused beta > ~33); the I3 fixes added a
  high-temperature torus identity that catches a wrong lattice weight and reports the signed torus under a gauge
  that does not close on odd tori.
- **Rulings that changed scope.** L-R1: per-topic files (`catalog_lattice_<x>.py`, `latex_lattice_<x>.py`,
  `lattice/<x>.py`), guarded files <= 380 lines. L-R2: <= 1 s default-tier growth for the plan. Measured at O7
  by the default-selftest growth between 99acf67 and 2091a08 (three cold runs each): +1.6 s, above the ruling;
  the task reports' direct accounting gave about 0.25 + 0.28 + 0.35 s, the new test functions alone cost 0.7 s
  in-suite, the rest is the growth of the shared catalog-RefCheck and LaTeX tests with three new entries.
