# Replica Machinery (`replicant`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new BootLoops package, `tools/replicant/`, that holds published replica and cavity calculations as recorded, typed steps, repeats them, and recombines them for models outside the catalog.

**Architecture:** One step/trace core. Level (a) derives equations symbolically with sympy; level (b) holds per-family pipelines with tested solvers; the two must agree. A cavity route (dense RS, sparse BP/population dynamics, sparse 1RSB) runs beside the replica route and must agree where both exist. A composer matches a model spec to steps and reports reused, adapted, and missing steps.

**Tech Stack:** Python 3.14, sympy 1.14, mpmath 1.3, numpy; `tools/baller` for certified reference values where roots/integrals can be enclosed. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-23-replica-machinery-design.md`

## Global Constraints

- Package `tools/replicant/`, working name `replicant`; package docs in English.
- Layout like neighbours: `GUIDE.md`, `README.md`, code, selftest battery; a row in `tools/README.md`, a line in `tools/BATTERIES.json`; default battery about 20 s, heavier tiers behind flags.
- Python with sympy and mpmath; certified numerics through `baller` where it applies, optional elsewhere.
- Reference values come from cited papers fetched during the build, never from memory; each is also pinned by a route that does not use the paper's number.
- Where sources disagree, record both with citations; the tolerance covers the spread.
- Refusal over invention: failed precondition or missing step stops with a named reason. No silent fallback to RS when stability says RS is unstable.
- n→0 continuation and RSB ansätze are physics-level, not proven; outputs say so. Population-dynamics numbers are labelled statistical, not certified.
- A number that fed a fit never certifies it.
- Out of scope: TAP, DMFT/dynamics, quantum models, automatic RSB-ansatz discovery, rigorous proofs, non-Gaussian disorder beyond what catalog entries and the two held-out models need.
- Every run writes a machine-readable result and a rendered step trace.
- **[RV, 2026-10-23]** The system can write LaTeX that details each calculation: every step with its equations, the citations, the banked-vs-computed check table, and the physics-level / statistical labels (Task 15).
- Full RSB (Parisi PDE) sits behind a flag with its own slower battery.
- Do not commit anything touching `docs/` or the package until the task that owns it says so (the spec and this plan are committed in Task 1).

## Review Focus

1. **n→0 prefactors.** Sums over `a≠b` vs `a<b` and the `1/n` scale flip signs silently. Expect: derivative at n=0 matches brute-force polynomial identities at integer n (Task 4).
2. **No nontrivial solution.** T above T_c, α above capacity, or no fixed point: expect a reported `no_solution`/trivial branch, never a converged wrong value (Task 5).
3. **Wrong fixed-point branch.** Hopfield has mixture states and spurious branches: expect the branch named in the result, selectable, and checked (Task 10).
4. **Field vs no field.** AT line is meaningful only at h≠0; at h=0 the replicon criterion is the T_c test. Expect both, with the right one chosen per call (Task 6).
5. **Unknown model given to the composer.** Expect a refusal naming the missing step, never a partial result presented as a full one (Task 14). Related: seeds and population size for population dynamics: expect identical results for identical seed, and an error bar that shrinks with population size (Task 13).

---

### Task 1: Core, registry, trace, battery runner

**Files:**
- Create: `tools/replicant/__init__.py`, `core.py`, `testing.py`, `selftest.py`, `tests/test_core.py`, `README.md` (stub)
- Modify: `tools/README.md` (index row, class `selftest`), `tools/BATTERIES.json` (`"replicant": {"cmd": "python3 tools/replicant/selftest.py", "cwd": "root"}`)
- Add to git: the spec and this plan.

**Interfaces:**
- Produces: `class Refusal(Exception)` with `.reason: str` (kebab-case named reason); `@dataclass(frozen=True) class Step(name: str, requires: Callable[[State], list[str]], apply: Callable[[State], State], cite: str | None)`; `class State` (immutable mapping; `.with_(**kw) -> State`); `class TraceEntry(step: str, cite: str|None, summary: str)`; `class Trace` with `.render() -> str`, `.to_json() -> dict`; `run_pipeline(steps: Sequence[Step], state: State) -> tuple[State, Trace]`; `@slow` decorator in `testing.py`; `selftest.py [--full]` discovers `tests/test_*.py` functions without pytest, skips `@slow` unless `--full`, exits 0/1.

- [ ] **Step 1: Write the failing test** `tests/test_core.py`: `test_pipeline_records_trace_in_order` (two steps → `trace.to_json()["steps"] == ["a","b"]`); `test_failed_precondition_refuses_with_named_reason` (step requiring `"gaussian_disorder"` absent from state → `Refusal` with `.reason == "precondition:gaussian_disorder"`, and later steps not run); `test_render_contains_citations`.
- [ ] **Step 2: Run** `python3 tools/replicant/selftest.py` → Expected: FAIL (import error).
- [ ] **Step 3: Implement** the signatures above in `core.py`/`testing.py`/`selftest.py`. `requires` returns the list of missing preconditions; non-empty → `Refusal`.
- [ ] **Step 4: Run** `python3 tools/replicant/selftest.py` and `python3 run_selftests.py replicant` → Expected: pass.
- [ ] **Step 5: Commit** `replicant: core, trace, selftest runner (+ design spec and plan)`.

### Task 2: Model specification and reference-data store

**Files:**
- Create: `tools/replicant/spec.py`, `refdata.py`, `reference/_schema.md`, `tests/test_refdata.py`

**Interfaces:**
- Consumes: `Refusal` (Task 1).
- Produces: `@dataclass(frozen=True) class ModelSpec(family: str, variables: str, disorder: str, constraint: str | None, prior: str | None, params: dict)` (`variables` in `{"ising","spherical"}`, `disorder` in `{"gaussian","pm_one","none"}` and open to extension); `@dataclass(frozen=True) class RefValue(key, value: str, tol: str, cite: str, independent_route: str, sources_disagree: tuple = ())`; `load(entry: str) -> dict[str, RefValue]`; `check(ref: RefValue, got, *, label: str) -> CheckResult(ok, got, want, tol)`.

- [ ] **Step 1: Write the failing tests:** `test_load_refuses_entry_without_cite` (JSON lacking `cite` → `Refusal.reason == "unbanked:no-citation"`); `test_load_refuses_entry_without_independent_route` (`"unbanked:no-independent-route"`); `test_check_uses_tolerance` (value `0.7632`, tol `1e-4`: `0.76325` ok, `0.7640` not ok); `test_disagreeing_sources_widen_tolerance` (two cited values 0.7633/0.7632 are recorded and `check` accepts both).
- [ ] **Step 2: Run** selftest → Expected: FAIL.
- [ ] **Step 3: Implement** the signatures; values are decimal strings parsed with mpmath.
- [ ] **Step 4: Run** selftest → Expected: pass.
- [ ] **Step 5: Commit** `replicant: ModelSpec and banked reference store`.

### Task 3: Level (a) symbolic derivation core

**Files:**
- Create: `tools/replicant/derive.py`, `tests/test_derive.py`

**Interfaces:**
- Consumes: `ModelSpec`, `Step`, `Refusal`.
- Produces: `replicate(spec: ModelSpec, n: sympy.Symbol) -> Step` and `average_disorder() -> Step` and `introduce_order_parameters() -> Step` and `saddle_equations() -> Step` (each a `Step` over `State` keys `Zn`, `avg`, `functional`, `equations`); `annealed_log_Z(spec, N) -> sympy.Expr` (the n=1 average, exact at finite N).

- [ ] **Step 1: Write the failing tests** with exact values: `test_sk_gaussian_annealed_exact` — for SK, J_ij~N(0,1/N), `annealed_log_Z(spec, N) == N*log(2) + beta**2*(N-1)/4`, and a brute-force enumeration over all 2^N states with the closed-form Gaussian moment agrees for N≤10; `test_sk_pm_one_annealed_exact` — `annealed_log_Z == N*log(2) + (N*(N-1)/2)*log(cosh(beta/sqrt(N)))`; `test_unsupported_disorder_refuses` — `disorder="lognormal"` → `Refusal.reason == "precondition:supported-disorder-law"`; `test_functional_is_symmetric_in_replica_indices`.
- [ ] **Step 2: Run** selftest → Expected: FAIL.
- [ ] **Step 3: Implement.** Gaussian average via the exact Gaussian moment; `pm_one` via exact cosh product then expansion in 1/N (leading cumulant, stating the dropped order in the trace summary); introduce `q_ab` with delta-function conjugates; saddle equations by sympy differentiation.
- [ ] **Step 4: Run** selftest → Expected: pass.
- [ ] **Step 5: Commit** `replicant: symbolic replicate/average/order-parameter/saddle steps`.

### Task 4: Ansätze and the n→0 rules

**Files:**
- Create: `tools/replicant/ansatz.py`, `tests/test_ansatz.py`

**Interfaces:**
- Consumes: sympy.
- Produces: `rs_matrix(n, q) -> Matrix`; `rsb_matrix(ms: Sequence, qs: Sequence, n) -> Matrix` (1RSB when one `m`); `sum_offdiag_power(kind: str, ms, qs, k, n) -> Expr` (closed-form polynomial in n and the `m`'s for `Σ_{a≠b} q_ab^k`); `n_to_zero(expr, n) -> Expr` (limit of `expr/n`); `replicon_eigenvalue(kind, ...)` is defined in Task 6.

- [ ] **Step 1: Write the failing tests:** `test_rs_offdiag_sum_matches_bruteforce` — for n in 2..6, k in 1..4, `sum_offdiag_power("rs",...) == explicit double sum` of the matrix; `test_1rsb_offdiag_sum_matches_bruteforce` — for (n,m) with m | n, n≤8, same; `test_n_to_zero_rs` — `n_to_zero(sum_offdiag_power("rs",[],[q],k,n), n) == -q**k`; `test_n_to_zero_1rsb_k2` — equals `(m-1)*q1**2 - m*(q1**2 - q0**2)*... ` expressed as the polynomial the brute-force identity fixes (assert via the polynomial identity at 5 integer (n,m) points rather than a hand-typed form).
- [ ] **Step 2: Run** selftest → Expected: FAIL.
- [ ] **Step 3: Implement.** Sum is a polynomial in (n, m); fit/derive symbolically, validate by the brute force in tests; continuation is substituting the polynomial.
- [ ] **Step 4: Run** selftest → Expected: pass.
- [ ] **Step 5: Commit** `replicant: RS/1RSB overlap matrices and n->0 rules`.

### Task 5: Numerical solvers

**Files:**
- Create: `tools/replicant/solve.py`, `tests/test_solve.py`

**Interfaces:**
- Produces: `gauss_expect(f: Callable[[mpf], mpf], *, dps: int = 30) -> mpf` (E over z~N(0,1)); `fixed_point(F: Callable, x0, *, dps: int = 30, tol: str = "1e-20", branch: str | None = None) -> FixedPoint(value, residual, converged: bool, branch: str)`; raises `Refusal("no-solution")` when no branch converges.

- [ ] **Step 1: Write the failing tests:** `test_gauss_expect_known` (`E[z**2]==1`, `E[cos z]==exp(-1/2)` to 25 digits); `test_fixed_point_sk_q_below_tc_is_trivial` — `F(q)=gauss_expect(lambda z: tanh(beta*sqrt(q)*z)**2)`, beta=0.8: returned branch `"trivial"`, value 0; `test_fixed_point_sk_q_above_tc_nontrivial` — beta=1.5: branch `"nontrivial"`, value>0, residual<1e-20; `test_no_convergence_refuses`.
- [ ] **Step 2: Run** selftest → Expected: FAIL.
- [ ] **Step 3: Implement** with Gauss–Hermite/mpmath quad and damped iteration seeded from several starts to find distinct branches; use `baller` enclosures for the banked reference values only (Task 6 onward).
- [ ] **Step 4: Run** selftest → Expected: pass.
- [ ] **Step 5: Commit** `replicant: Gaussian expectations and branch-aware fixed-point solver`.

### Task 6: Family (b) — Sherrington–Kirkpatrick (RS, stability, 1RSB)

**Files:**
- Create: `tools/replicant/families/__init__.py`, `families/sk.py`, `reference/sk.json`, `tests/test_sk.py`

**Interfaces:**
- Consumes: `solve.fixed_point`, `ansatz`, `refdata.load/check`.
- Produces: `sk_rs(beta, h=0, *, require_stable=False) -> RSSolution(q, f, e, stable: bool)`; `sk_replicon(beta, h=0) -> mpf` (negative = unstable); `sk_1rsb(beta, h=0) -> RSBSolution(q0, q1, m, f, e)`; `sk_ground_energy(ansatz: str) -> mpf`.

- [ ] **Step 1: Fetch and bank** SK references from the original papers (Sherrington–Kirkpatrick 1975, de Almeida–Thouless 1978, Parisi 1979/80, 1RSB ground-state energy from a cited source) into `reference/sk.json` with `cite` and `independent_route` (closed form where it exists: RS ground energy `-sqrt(2/pi)`).
- [ ] **Step 2: Write the failing tests:** `test_tc_is_one` (q>0 iff beta>1); `test_rs_zero_temperature_energy` (`sk_ground_energy("rs") == -sqrt(2/pi)` to 20 digits); `test_rs_unstable_below_tc_at_zero_field` (`sk_replicon(2.0) < 0`); `test_require_stable_refuses` (`sk_rs(2.0, require_stable=True)` → `Refusal.reason == "unstable:rs"`, no fallback); `test_field_uses_at_line` (at h=0.3 the replicon sign changes at a beta consistent with the banked AT value); `test_1rsb_energy_matches_reference`; `test_1rsb_lower_than_rs`.
- [ ] **Step 3: Run** selftest → Expected: FAIL.
- [ ] **Step 4: Implement** the signatures; free energies by the standard RS/1RSB functionals.
- [ ] **Step 5: Run** selftest → Expected: pass. **Commit** `replicant: SK family (RS, AT, 1RSB) with banked references`.

### Task 7: Family (b) — REM and spherical p-spin

**Files:**
- Create: `families/rem.py`, `families/pspin.py`, `reference/rem.json`, `reference/pspin.json`, `tests/test_rem_pspin.py`

**Interfaces:**
- Produces: `rem_free_energy(beta, N_scale) -> mpf` and `rem_critical_beta() -> mpf` (closed form `sqrt(2 ln 2)`); `pspin_spherical_1rsb(p, beta) -> RSBSolution`; `pspin_critical_temperature(p) -> mpf`.

- [ ] **Step 1: Bank** Derrida's REM results and the spherical p-spin 1RSB references (Crisanti–Sommers 1992, Gardner 1985), cited.
- [ ] **Step 2: Failing tests:** `test_rem_critical_beta_closed_form`; `test_rem_free_energy_continuous_at_critical_beta`; `test_pspin_tc_matches_reference_p3` (value read from `reference/pspin.json`).
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** → pass; **Commit** `replicant: REM and spherical p-spin`.

### Task 8: Family (b) — perceptrons and teacher–student

**Files:**
- Create: `families/perceptron.py`, `reference/perceptron.json`, `tests/test_perceptron.py`

**Interfaces:**
- Produces: `gardner_capacity(variables: str, kappa=0) -> Capacity(alpha_c: mpf, stable: bool, ansatz: str)` (`"spherical"` at κ=0 → 2; `"ising"` → RS value with `stable` flag, 1RSB value from the cited source when `ansatz="1rsb"`); `gardner_capacity_curve(variables, kappas) -> list[Capacity]` (spherical: `α_c(κ) = [∫_{-κ}^{∞} Dt (t+κ)²]^{-1}`); `gardner_entropy(alpha, variables, kappa=0) -> mpf`; `teacher_student_error(alpha, learning: str) -> mpf` (`"gibbs"`, `"bayes"`); `asymptotic_constant(learning) -> mpf`.

- [ ] **Step 1: Bank** Gardner 1988, Gardner–Derrida 1988, Krauth–Mézard 1989 (Ising), Györgyi 1990 / Seung–Sompolinsky–Tishby 1992 (learning curves), cited; the independent route for Ising is exact enumeration at small N plus the `alpha→0` limit `ln 2`.
- [ ] **Step 2: Failing tests:** `test_spherical_capacity_is_two` (κ=0, 20 digits, with `Capacity.stable` True); `test_spherical_capacity_curve_monotone_and_matches_integral` (decreasing in κ; equals the integral formula via `gauss_expect`; κ<0 gives α_c>2); `test_ising_capacity_rs_flagged_unstable_and_1rsb_matches_reference`; `test_capacity_is_where_entropy_vanishes` (spherical: `gardner_entropy` → −∞ at α_c, finite below); `test_ising_entropy_to_ln2_at_alpha_zero`; `test_ising_rs_flags_instability` (the RS Ising solution near capacity reports `stable=False`); `test_gibbs_error_asymptotic_constant_matches_reference` (`alpha*error → constant` within tolerance); `test_bayes_better_than_gibbs`.
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** → pass; **Commit** `replicant: perceptron capacity and teacher-student curves`.

### Task 9: Family (b) — Hopfield

**Files:**
- Create: `families/hopfield.py`, `reference/hopfield.json`, `tests/test_hopfield.py`

**Interfaces:**
- Produces: `hopfield_rs(alpha, beta, *, branch: str = "retrieval") -> HopfieldSolution(m, q, r, branch)` (branches `"retrieval"`, `"spinglass"`, `"paramagnet"`); `hopfield_alpha_c(beta=None) -> mpf` (zero-temperature AGS critical load).

- [ ] **Step 1: Bank** Amit–Gutfreund–Sompolinsky 1985/87, cited; independent route: low-α limit `m→1` and the paramagnetic transition at `T = 1+√α`.
- [ ] **Step 2: Failing tests:** `test_retrieval_at_low_alpha_has_m_near_one`; `test_paramagnet_boundary` (`T = 1 + sqrt(alpha)`); `test_alpha_c_matches_reference`; `test_branch_is_named_and_selectable` (spin-glass and retrieval branches both reachable at α=0.1, β large, with different `m`).
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** → pass; **Commit** `replicant: Hopfield RS with branch control`.

### Task 10: Level agreement (a) ↔ (b)

**Files:**
- Create: `tools/replicant/agree.py`, `tests/test_agree.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 6, 8, 9.
- Produces: `derived_rs_equations(spec: ModelSpec) -> list[sympy.Eq]` and `pipeline_rs_equations(family: str) -> list[sympy.Eq]`; `equations_agree(a, b) -> bool` (symbolic equality after sympy simplification, fixed-point residual check at 3 random points as backup).

- [ ] **Step 1: Failing tests:** `test_sk_derived_equals_pipeline`, `test_spherical_perceptron_derived_equals_pipeline`, `test_hopfield_derived_equals_pipeline`; `test_agreement_detects_a_planted_sign_error` (flip a sign in one side → `equations_agree` False).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass; **Commit** `replicant: level (a)/(b) agreement battery`.

### Task 11: Cavity — RS on fully connected models

**Files:**
- Create: `tools/replicant/cavity/__init__.py`, `cavity/dense.py`, `tests/test_cavity_dense.py`

**Interfaces:**
- Consumes: Tasks 5, 6, 8, 9.
- Produces: cavity `Step`s `remove_site`, `cavity_field_law`, `add_site`, `self_consistency` (typed like Task 1 steps); `cavity_sk_rs(beta, h=0) -> RSSolution`; `cavity_hopfield_rs(alpha, beta, branch) -> HopfieldSolution`; `cavity_perceptron_rs(alpha, variables) -> ...`.

- [ ] **Step 1: Failing tests:** `test_cavity_sk_equals_replica_sk` (q and e agree to 1e-12 at β=1.5, h=0 and β=0.9, h=0.3); `test_cavity_hopfield_equals_replica_hopfield` (m, q); `test_cavity_perceptron_equals_replica_gardner_spherical` (entropy at α=1, and the capacity α_c(κ) at κ=0 and κ=0.5); `test_cavity_trace_names_cavity_steps`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with the same solver and catalog entry format, `route="cavity"` in the trace. **Step 4: Run** → pass; **Commit** `replicant: dense RS cavity route`.

### Task 12: Cavity — sparse graphs, BP and population dynamics

**Files:**
- Create: `cavity/sparse.py`, `reference/viana_bray.json`, `tests/test_cavity_sparse.py`

**Interfaces:**
- Produces: `bp_fixed_point(graph_degree_law, beta, J_law, *, seed: int) -> BPResult(messages, converged)`; `population_dynamics(degree_law, J_law, beta, *, pop: int, sweeps: int, seed: int) -> PDResult(value, stderr, label="statistical")`; `viana_bray_tc(c: int, J_law) -> mpf` (`tanh(β_c)` solves `c·tanh²(β_c)... = 1` for ±J; formula from the cited source).

- [ ] **Step 1: Bank** Viana–Bray 1985, Mézard–Parisi 2001 (Bethe lattice), cited; independent route: exact BP on a tree (finite depth) vs enumeration.
- [ ] **Step 2: Failing tests:** `test_bp_on_tree_matches_enumeration` (small trees, exact to 1e-12); `test_viana_bray_tc_closed_form_regular_graph`; `test_pd_same_seed_same_result`; `test_pd_stderr_shrinks_with_population` (pop 1e3 → 1e4 reduces stderr by ≥2×); `test_pd_result_is_labelled_statistical`.
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** Seeds via `numpy.random.Generator`. **Step 5: Run** → pass; **Commit** `replicant: sparse cavity (BP, population dynamics)`.

### Task 13: Cavity — sparse 1RSB (stageable)

**Files:**
- Create: `cavity/rsb_sparse.py`, `tests/test_cavity_rsb_sparse.py`

**Interfaces:**
- Produces: `pop_of_pops_1rsb(degree_law, J_law, beta, m, *, pop, inner, sweeps, seed) -> PDResult` labelled statistical; `rsb_sparse_energy(...) -> PDResult`.

- [ ] **Step 1: Failing tests:** `test_m_equal_one_reduces_to_rs` (m=1 result equals Task 12 RS within combined error); `test_result_is_labelled_statistical`; `test_reproducible_with_seed`; `@slow test_energy_matches_banked_reference` (against a cited sparse 1RSB energy added to `reference/viana_bray.json`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `--full` → pass; **Commit** `replicant: sparse 1RSB cavity`.
- Note: if this task proves too large, the plan stages it as the second milestone (spec allows), without leaving v1.

### Task 14: Composer and CLI

**Files:**
- Create: `tools/replicant/compose.py`, `__main__.py`, `catalog.py`, `tests/test_compose.py`

**Interfaces:**
- Consumes: all families, derive steps, cavity steps.
- Produces: `catalog() -> dict[str, CatalogEntry]` (entry: `name, spec: ModelSpec, steps: list[Step], cite, routes: set[str], refs`); `compose(spec: ModelSpec, route: str = "replica") -> ComposedCalculation(steps, report: CompositionReport(reused, adapted, missing))`; `ComposedCalculation.run() -> Result(values, trace, rigor="physics-level", errors)`; CLI `python3 -m replicant run <entry|spec.json> [--route replica|cavity] [--out result.json]` and `python3 -m replicant list`.

- [ ] **Step 1: Failing tests:** `test_catalog_entry_reproduces_its_banked_numbers` (loop over all entries: every banked key passes `check`); `test_compose_known_spec_reports_all_reused`; `test_compose_unknown_family_refuses_naming_missing_step` (`Refusal.reason` starts `"missing-step:"`, no partial `Result`); `test_result_declares_physics_level`; `test_cli_refusal_prints_REFUSED_and_exits_2`; `test_cli_writes_json_and_rendered_trace`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Matching is by `(variables, disorder, constraint, prior)` against step preconditions; `adapted` means a step reused with a changed parameter or law. **Step 4: Run** → pass; **Commit** `replicant: catalog, composer, CLI`.

### Task 15: LaTeX derivation writer

**Files:**
- Create: `tools/replicant/latex.py`, `tests/test_latex.py`
- Modify: `core.py` (optional `Step.tex`; `TraceEntry` carries its `TexBlock`s; a `Refusal` carries the partial trace), `catalog.py` (`CatalogEntry.derivation`), `__main__.py` (`--latex PATH`, `--pdf`), `derive.py` (tex hooks on the level (a) steps)

**Interfaces:**
- Consumes: `Trace`, `Result`, `CatalogEntry`, `refdata.RefValue`, the family functionals (SK, REM, p-spin, perceptron, Hopfield, cavity).
- Produces: `@dataclass(frozen=True) class TexBlock(kind: str, title: str, body: str, expr: sympy.Expr | None = None)` (`kind` in `{"equation","align","text"}`; `expr` set when the block displays an equation the code can evaluate); `Step.tex: Callable[[State, State], list[TexBlock]] | None`; `render_latex(result: Result, *, title: str | None = None, standalone: bool = True) -> str` (preamble with amsmath; one section per trace step, in order, with its blocks; a "Reference check" table of key / computed / banked / tolerance / verdict from `refdata`; a bibliography built from the `cite` strings; the physics-level notice; the route label `replica` or `cavity`; population-dynamics quantities marked statistical); `write_latex(result, path, *, pdf: bool = False) -> Path` (with `pdf=True` compiles with `tectonic` or `pdflatex` if on PATH, otherwise refuses with `precondition:tex-engine`). CLI `python3 -m replicant run <entry> --latex out.tex [--pdf]`.
- Design rules: level (a) steps render the sympy objects they hold with `sympy.latex`; each level (b) and cavity catalog entry supplies `derivation` blocks, each equation given as a sympy expression whenever the family code has the functional, otherwise as a LaTeX string with its paper citation; the writer never invents an equation the code does not hold.

- [ ] **Step 1: Failing tests:** `test_sk_latex_contains_steps_in_order` (replicate, average, order-parameter, saddle appear in trace order, each with an `align` or `equation` block); `test_displayed_equations_match_implementation` (every `TexBlock` with an `expr` is lambdified and agrees with the family code to 1e-12 at three parameter points: e.g. the SK RS free-energy functional against `sk_rs`, the Gardner entropy against `gardner_entropy`); `test_every_catalog_entry_has_derivation` (at least one equation block, one citation, and a reference table row for each banked key); `test_latex_environments_balanced_and_specials_escaped` (every `\begin{X}` has its `\end{X}`; `_`, `%`, `&`, `#` in text blocks and cites are escaped); `test_refusal_renders_partial_derivation` (a refused composition renders the steps applied before the refusal and the refusal reason; closes the Task 1 deferred minor); `test_statistical_and_physics_labels_present` (cavity population-dynamics documents say statistical; every document carries the physics-level notice); `test_cli_latex_flag_writes_tex`; `@slow test_pdf_compiles` (skipped with a message when neither `tectonic` nor `pdflatex` is on PATH; compiles SK and Hopfield documents and asserts a non-empty PDF).
- [ ] **Step 2: Run** selftest → Expected: FAIL.
- [ ] **Step 3: Implement.** Add the tex hooks to the level (a) steps and `derivation` to every catalog entry created so far; keep each family's equations next to its code so the display cannot drift from the implementation (the consistency test is the guard).
- [ ] **Step 4: Run** selftest and `--full` → Expected: pass; default-tier cost of this task's tests under ~1.5 s.
- [ ] **Step 5: Commit** `replicant: LaTeX derivation writer`.

### Task 16: Held-out models and the positive control

**Files:**
- Create: `tests/test_heldout.py`, `reference/heldout.json`

**Interfaces:**
- Consumes: Task 14.
- Produces: no new API; two composed calculations and a rejection.

- [ ] **Step 1: Failing tests:**
  - `test_sk_pm_one_composed_matches_independent_routes` — SK with `disorder="pm_one"`, composed (not in the catalog): n=1 annealed free energy equals exact enumeration for N≤12 (Task 3 formula), and the quenched large-N T_c and RS ground energy equal the Gaussian ones (universality, the independent limit), with `report.adapted` naming the disorder step.
  - `test_ising_perceptron_composed_matches_independent_routes` — perceptron with `variables="ising"` composed from the spherical perceptron steps: `alpha→0` entropy `ln 2`, exact enumeration at N≤20 consistent with the annealed bound and the RS curve within finite-size scaling; `report.adapted` names the variable-measure step.
  - `test_wrong_ansatz_is_rejected` (positive control) — force the RS ansatz where the replicon is negative (SK at β=2): the held-out check fails with `Refusal("unstable:rs")`; and a deliberately sign-flipped saddle equation fails the independent route.
  - `test_heldout_models_not_in_catalog` — asserts neither spec is a catalog entry.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** any missing adapter in the existing steps (no special-casing the two models). **Step 4: Run** → pass; **Commit** `replicant: held-out reuse tests and positive control`.

### Task 17: Full RSB (Parisi PDE), behind a flag

**Files:**
- Create: `families/parisi.py`, `tests/test_parisi.py` (all `@slow`)

**Interfaces:**
- Produces: `parisi_ground_energy(*, grid: int, dps: int = 20) -> PDResult(value, error_estimate, label="numerical")`; `sk_frsb(beta) -> RSBSolution` (enabled only with `--full`).

- [ ] **Step 1: Failing tests (`@slow`):** `test_parisi_energy_within_banked_range` (value inside the cited FRSB interval, noting the sources' small disagreement); `test_frsb_below_1rsb`; `test_error_estimate_shrinks_with_grid`.
- [ ] **Step 2: Run** `python3 tools/replicant/selftest.py --full` → FAIL. **Step 3: Implement** a Crank–Nicolson (or spectral) solver for the Parisi PDE. **Step 4: Run** `--full` → pass, and the default battery still runs in about 20 s. **Step 5: Commit** `replicant: Parisi full-RSB (slow tier)`.

### Task 18: Documentation, registration, references

**Files:**
- Create: `tools/replicant/GUIDE.md`
- Modify: `tools/replicant/README.md`, `tools/README.md` (full row text), `toolkit/ours/README.md` (roster), `REFERENCES.md` (every banked paper), `CHANGELOG.md`

- [ ] **Step 1: Write `GUIDE.md`** in the house format (PURPOSE, USE-WHEN, NOT-FOR, INVOKE, INPUTS, OUTPUTS, acceptance gates, known hazards including "physics-level, not proven", the statistical label, the refusal reasons list). Include a worked CLI example per route.
- [ ] **Step 2: Run** `python3 run_selftests.py replicant` and `python3 run_selftests.py` → Expected: replicant passes in about 20 s; nothing else regresses.
- [ ] **Step 3: Check** every key cited in `GUIDE.md` resolves in `REFERENCES.md`.
- [ ] **Step 4: Commit** `replicant: guide, index row, references, changelog`.

---

## Self-review notes

- LaTeX derivations (user request): Task 15; held-out models (Task 16) also render a derivation.
- Spec coverage: step library (T1,3,4,5), catalog (T2,6–9), composer (T14), cavity dense/sparse/1RSB (T11–13), level agreement (T10), held-out + positive control (T15), FRSB flag (T16), docs/registration (T17), honest labelling (T14 result, T12–13 labels), refusal (T1,6,14).
- Staging if size bites: T13 is the designated second milestone; T16 is already behind a flag.
