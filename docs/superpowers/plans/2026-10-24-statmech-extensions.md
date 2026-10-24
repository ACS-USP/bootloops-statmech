# Statistical-mechanics extensions to `replicant` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eleven statistical-mechanics techniques to `tools/replicant/`, in the order E1…E11, each with banked references, independent checks that can fail, per-value finality and documentation.

**Architecture:** Extend the existing framework (typed steps, registries `derive.FAMILIES/SITES/LAWS`, catalog + composer + LaTeX writer, refdata bank). New numerical engines live in new subpackages; new catalog entries follow the Hopfield/SK wiring (grep `hopfield` across `tools/replicant` to find every touch point: catalog entry, parameter set, finality, LaTeX entry, reference JSON, GUIDE row, tests).

**Tech Stack:** Python 3.14, sympy, mpmath, numpy, scipy 1.18; `tools/annihilator`, `tools/ratfit`, `tools/baller` for E8–E10. No new third-party dependency without a recorded ruling.

**Spec:** `docs/superpowers/specs/2026-10-24-statmech-extensions-design.md` (parent: `2026-10-23-replica-machinery-design.md`).

## Global Constraints

- Banked values come from papers fetched during the build, never from memory; each bank key has `value`, `tol`, `cite`, `independent_route` (a route that does not use the paper's number); tolerance covers the spread of `sources_disagree`; secondary sources are named as such.
- A banked number is a target, never an input of the computation it checks.
- Every check ships with a positive control (a deliberately wrong variant that must fail); planted errors go into the hook/engine being tested, registries restored in `try/finally` with identity asserts.
- Refusal over invention: named `Refusal(reason, detail)`; no silent fallback; unknown run parameters are refused (`precondition:known-parameter`); declared parameter set per catalog entry/route.
- Per-value finality for every catalog value: `stable` (computed), `unstable`, `not-computed`, `exact-by-literature` (requires `cite`); `final: true` means "stability computed positive or a theorem in the cited paper", never "proven here". Statistical outputs carry stderr and Student-t bands with named seeds; certified outputs carry ball radii.
- mpmath precision only through `mp.workdps`; tests never leave global precision changed.
- Default battery budget: the whole extension adds ≲ 8 s to the default tier (current ≈ 17 s); each task's default tests ≲ 0.7 s; everything heavy is `@slow` (run only with `--full`).
- Docs in English; new files ≲ 400 lines each; module split by responsibility; commit messages end with the attribution trailer given by the harness; nothing under `.superpowers` is committed.
- Do not change v1 numerical algorithms, bank values or tolerances, except where a task says finality labels become computed.

## Review Focus

1. A "derived" quantity that is secretly typed in or copied from the family code (agreement by construction): each derivation task must show a hook-level planted error is detected.
2. Finite-size/statistical checks whose band is so wide they cannot detect the error they claim to test: each statistical check states the band and what it would exclude.
3. Finality labels that overclaim (stable/exact) without a computed quantity or a citation; and labels that become stale when a computed replicon changes a previous literature-based claim (E1, E4 must update the catalog consistently, including GUIDE and LaTeX).
4. New models whose catalog wiring misses a touch point (parameter set, finality, LaTeX derivation, bank claim/UNCLAIMED, GUIDE row, `list` output).
5. Budget creep in the default tier.

---

### Task 1: E1 — Replicon at level (a) for Gaussian-integral energies

**Files:** modify `derive_replicon.py` (+ new `derive_replicon_gauss.py` if needed), `solve_derived.py`, `derive_families.py`, `catalog_entries.py` (finality of perceptron/Hopfield derived values), tests `test_replicon_derived.py`.

**Interfaces:** Consumes `derived_rs_equations`, `LnGaussProd`/`GaussE` (gauss.py), `Family` hooks, `families.perceptron.gardner_*` replicon, `families.hopfield` replicon. Produces `derived_replicon(spec, rs_solution) -> Replicon(value, route, note)` (value = AT eigenvalue in the replicon direction, produced by rules: second variation of the RS functional with respect to an off-diagonal perturbation q_ab → q + δ ε_ab for the RS-reduced `LnGaussProd`; no typed-in formula); `solve_derived_rs` returns `stable` True/False (not None) for spherical perceptron, Ising perceptron and Hopfield; the "replicon not derived for this family" refusal text disappears for these families and remains for unsupported ones.

- [ ] Tests (names): `test_derived_replicon_equals_family_replicon` (spherical perceptron at α=1.0, κ=0; Ising perceptron at α=0.5 and α=1.2 (AT-unstable ≈1.0147); Hopfield retrieval and spin-glass points of Task 9's cross-check table: −11.22, −1.033, −0.0281, +0.969, +0.800), each ≥ 3 points, tolerance 1e-8; `test_planted_hook_error_changes_the_replicon` (hook-level, registry restored); `test_replicon_sign_flips_at_AT_line` (Ising perceptron zero at α≈1.0147 located by root-finding on the derived replicon; reference value only as target); `test_composed_ising_perceptron_stability_is_computed` (compose result `stable` is not None; finality `unstable` above the AT point, `stable` below with the entropy positive); `test_unsupported_family_still_refuses`.
- [ ] Update catalog finality (perceptron and Hopfield derived values) and the GUIDE hazard text ("replicon derived only for the SK site" becomes the true list); LaTeX derivation of the replicon appears in the documents of these entries.

### Task 2: E2 — 1RSB at level (a)

**Files:** new `derive_rsb.py`, `solve_derived_rsb.py`; modify `compose.py` (route/param `ansatz=1rsb` for composed specs), `catalog_entries.py` as needed; tests `test_derive_rsb.py`.

**Interfaces:** `derived_1rsb_functional(spec) -> RSBFunctional` (block overlap matrix from `ansatz.rsb_matrix`, `sum_offdiag_power` closed forms, n→0 by the existing rules; unknowns q0, q1, m (+ q̂0, q̂1 conjugates)); `solve_derived_1rsb(spec, **params) -> DerivedRSBSolution(values, free_energy, m, q0, q1, residual, roots, stable=None|bool, trace, caveat)` with the same branch/ambiguity/independent-audit discipline as `solve_derived_rs`; frozen limit q1=1 supported (zero of S for discrete variables).

- [ ] Tests: `test_sk_derived_1rsb_matches_sk_1rsb` (T=0 energy −0.76526 to 1e-7 and finite-T at β=2 vs `families.sk.sk_1rsb` to 1e-8, comparison target only); `test_ising_perceptron_frozen_1rsb_zero_of_entropy` (S_RS(α)=0 at the banked 0.8330786 within 1e-6 via root-finding on the derived functional: target, not input); `test_ising_perceptron_full_1rsb_capacity_above_rs` (α_c(1RSB) from the bank's Krauth–Mézard value if banked, else report); `test_hopfield_1rsb` (only if a literature value is banked; else the test checks the m=1 reduction to RS and the T→0 limit sanity and the finality note says "no independent target"); `test_planted_error_in_1rsb_hook_is_caught`; `test_m_equal_one_reduces_to_rs` (derived 1RSB at m=1 equals derived RS).
- [ ] Composer: composed Ising perceptron with `ansatz=1rsb` returns the frozen/full 1RSB capacity with finality `not-computed` (1RSB stability is Task 5's); report which steps were reused/adapted relative to the RS derivation.

### Task 3: E3a — AMP, state evolution and TAP for SK

**Files:** new `amp/__init__.py`, `amp/sk.py`, `amp/state_evolution.py`; reference JSON if values are banked; tests `test_amp_sk.py`.

**Interfaces:** `sk_amp(J, beta, *, iters, seed, damping=None) -> AMPResult(m, q_trace, converged)` (Bolthausen AMP with Onsager term for symmetric Gaussian/±J matrices, zero field and with field); `sk_state_evolution(beta, h=0) -> SEResult(q, trajectory)`; `sk_tap_free_energy(m, J, beta, h=0) -> mpf` (Plefka/TAP with the Onsager reaction term) and `tap_fixed_point(J, beta, ...)`.

- [ ] Tests: `test_state_evolution_fixed_point_equals_rs_q` (q* equals `sk_rs` and the derived RS q at β=1.5: comparison targets); `test_amp_finite_N_matches_state_evolution` (N=1500, seeds 0–5, Student-t band; the band half-width is reported and the naive-mean-field control (no Onsager term) must fall outside it: positive control); `test_tap_free_energy_stationary_at_amp_fixed_point` (gradient ≈ 0, free energy per spin close to the RS −βf within the finite-N band); `test_pm_one_matrix_universality` (AMP on ±J matrix: same q within band); `test_paramagnet_trivial_below_tc`; refusals.
- [ ] Mark heavy tests `@slow`; default tier: state evolution (cheap) and refusals only.

### Task 4: E3b — GAMP for the perceptron and teacher–student; catalog route `amp`; held-out strengthening

**Files:** new `amp/gamp.py`; modify `catalog*.py` (route `amp` for `sk`, `teacher_student`), `compose.py` (route handling), `tests/test_heldout.py` (new independent check), LaTeX entries; tests `test_gamp.py`.

**Interfaces:** `gamp_perceptron(X, y, prior, *, iters, seed) -> GAMPResult`, `perceptron_state_evolution(alpha, prior="gaussian"|"ising", teacher=True) -> SEResult(overlap, mse)`; catalog routes: `sk` (`replica`, `cavity`, `amp`), `teacher_student` (`replica`, `amp`).

- [ ] Tests: `test_gamp_state_evolution_matches_teacher_student_bayes_error` (Bayes-optimal GAMP/SE generalization error vs `teacher_student_error(alpha, "bayes")` to 1e-6: comparison target); `test_gamp_finite_N_matches_state_evolution` (N=800, α∈{1,2}, seeds, Student-t band; band-excludes control); `test_heldout_pm_one_sk_q_vs_amp_on_pm_one_matrix` (the composed ±J SK q compared with AMP run on a sampled ±J matrix: a new independent route added to the held-out test, replacing no existing check); `test_amp_route_in_catalog_and_cli` (`run sk --route amp`, finality statistical for finite-N outputs, state-evolution outputs `exact-by-literature` only if the cited paper proves SE for that setting (Bayati–Montanari 2011 for AMP on Gaussian matrices) else `not-computed`).

### Task 5: E4 — Gardner transition (1RSB stability)

**Files:** new `families/gardner.py`; modify `families/pspin.py` finality hooks, `catalog_entries.py`, GUIDE; reference JSON (`gardner_transition`); tests `test_gardner_transition.py`.

**Interfaces:** `onersb_replicon(family, **params) -> mpf` (replicon eigenvalue of the 1RSB solution in the replicon direction; for the spherical p-spin the closed form from Crisanti–Leuzzi/Castellani–Cavagna; derive it from the functional by rules where Task 2's machinery allows, else implement the cited formula and say so); `pspin_gardner_temperature(p) -> mpf` (zero of the 1RSB replicon, found by root-finding); `ising_perceptron_1rsb_replicon(alpha)`.

- [ ] Tests: `test_gardner_temperature_matches_bank` (banked T_G(p=3) from a cited source, e.g. Gardner 1985 / Montanari–Ricci-Tersenghi 2003 / Crisanti–Leuzzi; read from the paper at build time); `test_replicon_sign_scan` (computed replicon positive above T_G and negative below on a grid: independent of the root-finder); `test_planted_error_in_replicon_formula_is_caught`; `test_pspin_finality_is_conditional` (catalog: p-spin 1RSB values are final for T > T_G and `unstable` below; replaces the blanket exact-by-literature claim, GUIDE text updated); `test_ising_perceptron_1rsb_stability` (reports sign; if literature gives a statement bank it).

### Task 6: E5a — XORSAT

**Files:** new `csp/__init__.py`, `csp/xorsat.py`; reference JSON `xorsat.json`; catalog entry `xorsat`; tests `test_xorsat.py`.

**Interfaces:** `xorsat_bp_threshold(K) -> mpf` (dynamical/clustering threshold α_d from the BP/leaf-removal fixed point), `xorsat_sat_threshold(K) -> mpf` (α_s from the 1RSB/entropy condition), `xorsat_core_size(K, alpha)`, `xorsat_gf2_satisfiable(K, N, alpha, seed)` (exact: Gaussian elimination over GF(2), numpy bit-packed), `xorsat_sat_probability(K, N, alpha, seeds)`.

- [ ] Tests: `test_k3_thresholds_match_bank` (α_d = 0.818469…, α_s = 0.917935… from Mézard–Ricci-Tersenghi–Zecchina 2003: bank from the paper; independent route = the leaf-removal closed form for α_d and the entropy-vanishing condition for α_s computed by the code, the paper's numbers only as targets); `test_gf2_crossing_near_alpha_s` (exact elimination at N=400..2000, seeds named: fraction satisfiable crosses 1/2 near α_s within the finite-size band; the band is stated and a shifted-threshold control (e.g. α_s+0.05) falls outside); `test_core_size_jumps_at_alpha_d` (leaf-removal on sampled hypergraphs vs closed form); `test_positive_control_wrong_degree_distribution`; refusals; finality `exact-by-literature` for the thresholds with cite (Dubois–Mandler 2002 / Pittel–Sorkin 2016 proved α_s for K≥3: cite what the fetched paper states).

### Task 7: E5b — K-SAT: BP and survey propagation

**Files:** new `csp/ksat.py`, `csp/sp.py`; reference JSON `ksat.json`; catalog entry `ksat`; tests `test_ksat.py`.

**Interfaces:** `ksat_bp(formula, *, iters, damping, seed) -> BPSATResult`, `ksat_sp(formula, *, iters, seed) -> SPResult(complexity)`, `random_ksat(K, N, alpha, seed)`, `ksat_population_dynamics_sp(K, alpha, *, pop, sweeps, seed) -> PDResult` (complexity Σ(α) from the SP population, statistical), `ksat_annealed_bound(K, alpha) -> mpf`.

- [ ] Tests: `test_sp_complexity_vanishes_at_banked_threshold` (3-SAT α_s ≈ 4.2667 from Mézard–Parisi–Zecchina 2002 / Krzakala et al. 2007, bank from the paper: target; population-dynamics complexity crossing located by root-finding within a statistical band); `test_bp_converges_below_alpha_d_and_not_above` (finite-N, seeds named); `test_small_N_exact_counts_below_annealed_bound` (N ≤ 18 exhaustive satisfiability and solution counts: independent; Jensen bound ln⟨Z⟩ ≥ ⟨ln Z⟩ asserted exactly); `test_planted_error_in_sp_update_is_caught`; refusals; finality: complexity values statistical `not-computed`; thresholds `not-computed` unless the cited paper proves them (K-SAT thresholds proven only for large K: Ding–Sly–Sun 2015: cite).
- [ ] Stretch (only if it fits at quality, else report not done): 3-coloring 1RSB threshold c_s ≈ 4.69 (Krzakala et al. 2007).

### Task 8: E6a — LDPC codes on the BEC

**Files:** new `inference/__init__.py`, `inference/ldpc.py`; reference JSON `ldpc.json`; catalog entry `ldpc_bec`; tests `test_ldpc.py`.

**Interfaces:** `bec_density_evolution(l, r, eps) -> DEResult(erasure_prob, converged)`, `bec_bp_threshold(l, r) -> mpf`, `bec_map_threshold(l, r) -> mpf` (Maxwell construction/area theorem), `ldpc_peeling_decode(l, r, N, eps, seed) -> PeelResult`, `ldpc_ensemble_graph(l, r, N, seed)`.

- [ ] Tests: `test_bp_threshold_36_matches_bank` ((3,6): ε_BP = 0.4294, ε_MAP = 0.4881 from Richardson–Urbanke / Mézard–Montanari: bank from the source; independent route: closed-form solution of the DE fixed-point equation by the code); `test_peeling_finite_N_crossing` (N=20000 peeling success vs ε: crossing near ε_BP within the band, seeds named; a shifted-threshold control falls outside); `test_density_evolution_vs_peeling_residual_erasure` (below threshold the residual erasure fraction → 0); positive control wrong edge-perspective degree polynomials; refusals; finality `exact-by-literature` with cite for the DE thresholds (Luby et al. / Richardson–Urbanke).

### Task 9: E6b — Spiked Wigner and the stochastic block model

**Files:** new `inference/spiked.py`, `inference/sbm.py`; reference JSON `spiked.json`, `sbm.json`; catalog entries `spiked_wigner`, `sbm`; tests `test_spiked.py`, `test_sbm.py`.

**Interfaces:** `spiked_state_evolution(snr, prior="gaussian"|"rademacher") -> SEResult(mmse, overlap)`, `spiked_amp(Y, prior, *, iters, seed)`, `spiked_threshold(prior) -> mpf` (λ_c = 1), `sbm_kesten_stigum(a, b, q=2) -> bool/margin`, `sbm_bp(graph, a, b, *, iters, seed) -> overlap`, `sbm_sample(N, a, b, seed)`.

- [ ] Tests: `test_spiked_se_mmse_matches_bank_formula` (Gaussian prior closed form MMSE(λ) = 1 − overlap², overlap² = 1 − 1/λ² for λ>1 (Lelarge–Miolane / Barbier et al.: bank from the paper; independent route: the AMP finite-N simulation at N=2000, seeds named, Student-t band, and the code's own state-evolution fixed point)); `test_spiked_rademacher_se_is_not_closed_form_but_below_gaussian` (ordering MMSE_Rademacher ≤ MMSE_Gaussian); `test_sbm_kesten_stigum_threshold` ((a−b)² = 2(a+b): below threshold BP overlap → 0, above positive on simulated graphs N=5000; margins stated; shifted-threshold control); planted errors; refusals; finality: SE/AMP finite-N `statistical`, thresholds `exact-by-literature` with cite.

### Task 10: E7 — Ridge regression and double descent

**Files:** new `learning/__init__.py`, `learning/ridge.py`; reference JSON `ridge.json`; catalog entry `ridge`; tests `test_ridge.py`.

**Interfaces:** `ridge_test_error(gamma, lam, sigma2=0) -> mpf` (Marchenko–Pastur Stieltjes-transform closed form for isotropic Gaussian design and isotropic teacher), `ridgeless_test_error(gamma, sigma2)` (σ²γ/(1−γ) for γ<1; (1−1/γ)+σ²/(γ−1) for γ>1: verify against the cited paper), `ridge_replica_derivation(...)` through the existing formal layer if feasible (else the replica route is the closed form with a cited derivation and the entry says so), `ridge_simulate(n, p, lam, sigma2, seed) -> float`.

- [ ] Tests: `test_ridgeless_peak_at_gamma_one_and_matches_bank` (Hastie–Montanari–Rosset–Tibshirani 2022 / Advani–Saxe: bank at several γ); `test_simulation_matches_closed_form` (n=400..1000, 20 seeds per γ, Student-t band; wrong-formula control (e.g. missing the (1−1/γ) bias term) falls outside); `test_optimal_lambda_removes_the_peak`; refusals; finality `exact-by-literature` with cite.

### Task 11: E8 — Exact series and holonomic recognition

**Files:** new `series/__init__.py`, `series/ising2d.py`, `series/recognize.py`; reference JSON `series.json`; catalog entry `series_ising2d`; tests `test_series.py`.

**Interfaces:** `ising2d_free_energy_series(order) -> list[Fraction]` (exact rational high-temperature series of the Onsager free energy per site in the expansion variable chosen and documented, produced by exact expansion of the closed-form Onsager integral, with an independent exact route: linked-cluster/polygon counting on finite strips or the transfer-matrix/Kac–Ward exact partition function at small sizes), `recognize_holonomic(series, max_order, max_degree) -> Recognition(operator, certificate, verified_terms)` driving `tools/annihilator` through its documented interface (read tools/annihilator/GUIDE.md), `singularities(operator) -> list[...]`, `critical_exponent(operator, point) -> mpf`.

- [ ] Tests: `test_onsager_free_energy_series_is_d_finite_order_two` (the recovered operator has order 2 (elliptic) and annihilates ≥ 30 coefficients beyond the fitting window: held-out coefficients); `test_singularity_at_critical_point_and_log_exponent` (t_c = √2 − 1 in the chosen variable; α = 0 with logarithmic singularity: banked exponent statement from Onsager 1944); `test_series_matches_exact_finite_lattice` (independent route: exact transfer-matrix partition function on strips L ≤ 8 reproduces the series coefficients up to the order the strip width supports); `test_recognition_refuses_when_series_is_not_holonomic_at_bounds` (a known non-D-finite series or a corrupted coefficient: the recogniser says no with the certificate of the bound used); finality `exact-by-literature`/`stable` as applicable; stretch (report if not done): impossibility certificate for the 2D Ising susceptibility series using banked coefficients (Nickel / Orrick–Nickel–Guttmann–Perk 2001, cite) at the largest order/degree that fits the time limit.

### Task 12: E9 — Solvable lattice models, certified

**Files:** new `lattice/__init__.py`, `lattice/onsager.py`, `lattice/transfer.py`, `lattice/bethe.py`; reference JSON `lattice.json`; catalog entries `ising2d`, `ising1d`, `bethe_ising`; tests `test_lattice.py`.

**Interfaces:** `onsager_free_energy(beta, J=1) -> Ball` (certified via `tools/baller` ball arithmetic, with radius), `yang_magnetization(beta) -> Ball`, `ising_transfer_matrix_free_energy(L, beta, ...)` (exact rationals/algebraic for small widths), `kaufman_partition_function(Lx, Ly, beta)` (exact finite-torus formula by determinants), `ising1d_free_energy(beta, h)`, `bethe_ising_free_energy(k, beta, h)` (belief propagation fixed point, certified).

- [ ] Tests: `test_onsager_certified_value_matches_bank` (Onsager critical −βf at β_c: ln√2 + 2G/π ≈ 0.929695…: closed forms checked with ball radii < 1e-30 vs mpmath at higher precision (independent) and the bank); `test_kaufman_equals_transfer_matrix_exact` (torus partition function for several sizes: exact rational/algebraic equality of two independent routes); `test_yang_magnetization_exponent_one_eighth` (log-log slope within stated band from the certified values); `test_bethe_ising_matches_viana_bray_regular` (the earlier sparse routes as comparison targets); refusals; certified values report radii in `errors`; finality `exact-by-literature` (Onsager, Yang) with cites.

### Task 13: E10 — Guerra and Aizenman–Sims–Starr bounds, certified

**Files:** new `bounds/__init__.py`, `bounds/guerra.py`; reference JSON `guerra.json`; catalog entry `sk_bounds`; tests `test_guerra.py`.

**Interfaces:** `guerra_rs_bound(beta, h, q) -> Ball` (P_RS(q) = ln 2 + β²(1−q)²/4 + E ln cosh(β√q z + βh), certified Gaussian expectation via ball arithmetic), `guerra_rs_best_bound(beta, h) -> Ball` (minimum over q, certified minimiser interval), `guerra_1rsb_bound(beta, q0, q1, m)`, `check_bound_chain(beta) -> ChainReport` (P_Parisi,K ≤ … ≤ P_1RSB ≤ P_RS with the package's own numbers as inputs to the chain; each inequality verified with enclosures).

- [ ] Tests: `test_rs_bound_certified_and_minimised_at_rs_q` (best bound equals the package's RS pressure at the stationary q within the radius; independent: mpmath at 40 digits); `test_chain_respected_by_package_values` (the K-RSB staircase values from `parisi.py` at β=2 and the 1RSB/RS pressures satisfy the chain with enclosures); `test_planted_violation_is_detected` (a perturbed 1RSB pressure above the RS bound makes `check_bound_chain` report the violation); `test_high_temperature_bound_is_tight` (β<1: best bound equals ln 2 + β²/4 exactly); refusals; finality `exact-by-literature` for the bounds (Guerra 2003 proved; cite), certified radii reported.

### Task 14: E11 — Monte Carlo and exact enumeration as independent checks

**Files:** new `numerics/__init__.py`, `numerics/pt.py`, `numerics/enumeration.py`; modify `tests/test_heldout.py` (strengthen), reference JSON `mc.json`; tests `test_pt.py`, `test_enumeration.py`.

**Interfaces:** `pt_sk(J, betas, sweeps, seed) -> PTResult(energy_per_spin_trace, ...)` (numpy-vectorised parallel tempering with exchange), `sk_ground_energy_finite_size(Ns, samples, seed) -> FSResult(e_inf, band)` (N^{−2/3} extrapolation), `enumerate_pm_one_sk(J) -> exact spectrum / ln Z` (Gray code, N ≤ 26 vectorised), `ising_perceptron_log_count(alpha, N, seed) -> float` (exact counting by meet-in-the-middle or DP up to N≈30).

- [ ] Tests: `test_pt_sk_energy_extrapolates_to_parisi` (e∞ within the stated band of the K-RSB/banked −0.7632; band reported and the RS value −0.7979 and 1RSB −0.7653 must lie outside it: positive control); `test_pm_one_sk_finite_T_energy_vs_composed_rs_and_1rsb` (β=2, ±J, N=256 PT over seeds vs composed ±J energy: the 1RSB value is closer than RS; the RS-vs-PT difference exceeds the band); `test_enumeration_logZ_vs_annealed_and_quenched_expansion` (N ≤ 22, many samples: quenched ln Z/N below the annealed bound by the amount the second-order expansion predicts: this check can tell quenched from annealed where the old one could not); `test_ising_perceptron_log_count_vs_composed_entropy` (N up to 26–30, α ∈ grid: finite-size extrapolation in 1/N versus the composed entropy and zero near 0.8331; band stated; spherical-site control fails); the new checks are added to `test_heldout.py` as additional routes (the v1 checks stay).

### Task 15: Documentation, registration, references, final cleanup

**Files:** `tools/replicant/GUIDE.md`, `README.md`, `tools/README.md` row text, `toolkit/ours/README.md`, `REFERENCES.md`, `CHANGELOG.md` (Unreleased), spec as-built notes, `reference/*.json` consistency.

- [ ] Update the catalog table, refusal table, routes and finality notes in GUIDE for every new entry/route; update hazards (the list of limits changes: stability now computed for perceptrons/Hopfield; p-spin finality conditional on T_G; held-out evidence strengthened by AMP/PT/enumeration: restate precisely what each check can and cannot exclude); REFERENCES section for the new keys (no verification date claimed); CHANGELOG entry; run default tier and `python3 run_selftests.py replicant`; report default-tier time and the slow-tier time.
- [ ] Final whole-branch review on the most capable model, one fix wave, one scoped re-review.

---

## Self-review notes

- Spec coverage: E1→T1, E2→T2, E3→T3–T4, E4→T5, E5→T6–T7, E6→T8–T9, E7→T10, E8→T11, E9→T12, E10→T13, E11→T14, docs→T15.
- Order follows the user's enumeration; T3/T4 and T6/T7 and T8/T9 split one item in two tasks each.
- Budget: each task's default tests ≲ 0.7 s → ≲ 8 s total with T1–T14; slow tier grows by minutes.
