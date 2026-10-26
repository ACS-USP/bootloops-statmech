# Online learning in `replicant`: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the 1990s statistical mechanics of online learning (Caticha et al., Saad–Solla, Biehl et al.) as a new `online` route, with one engine shared by five catalog entries. Every result carries a banked cited reference, an independent check that can fail, per-value finality and documentation.

**Architecture:**
- A new subpackage `tools/replicant/online/`. It holds:
  - the order-parameter ODEs in α, built from Gaussian averages over local fields;
  - integrators;
  - the variational optimal modulation;
  - finite-N simulators.
- New catalog modules `catalog_online*.py`, plus LaTeX blocks and bank files, wired like E7's `ridge` (commit bc6d548): grep `ridge` for every touch point.

**Tech Stack:** Python 3 with numpy, scipy, mpmath and sympy, through the project `.venv`. No new dependency without a recorded ruling.

**Spec:** `docs/superpowers/specs/2026-10-26-online-learning-design.md`

## Global Constraints

- Banked values come from papers fetched during the build, never from memory.
  - Each bank key has `value`, `tol`, `cite` (with equation, table or figure) and `independent_route`.
  - A value read off a figure says so, and its tolerance covers the reading error.
  - A paper that cannot be fetched is reported, and its numbers are not banked.
  - Numbers quoted in this plan are approximate pointers to be verified, never targets to type in.
- A banked number is a target, never an input of the computation it checks.
- Every check ships a positive control: a deliberately wrong variant that must fail. Planted errors go through module hooks, restored in `try/finally` with identity asserts.
- Refuse rather than invent:
  - a named `Refusal(reason, detail)`, with no silent fallback;
  - unknown run parameters are refused;
  - each entry and route declares its parameter set.
- Per-value finality, as in E1–E11.
  - ODE values are `exact-by-literature` only with a fetched theorem (the concentration of the online dynamics as N → ∞) and a cite. Otherwise they are `not-computed` with a note.
  - Simulation values are statistical: stderr plus a 99% Student-t band over named seeds, never final.
  - "Optimal" always names the class it is optimal in.
- Plateaus: a trajectory that has not left the symmetric plateau inside the integration window reports `undetermined`, never an extrapolated escape time.
- mpmath precision only through `mp.workdps`.
- Budget:
  - the whole online extension adds ≲ 4 s to the default tier (currently 21.4 s);
  - each task's default tests take ≲ 0.6 s cold;
  - heavy simulations are `@slow`, at most about 2 min per test.
- Docs are in English. New files ≲ 400 lines. The guarded files (catalog_entries.py, compose.py, latex.py, latex_entries.py) stay ≤ 380 lines.
- Commit messages end with the harness attribution trailer. Nothing under `.superpowers` is committed. Run with `/Users/renatovicente/Projects/bootloops/.venv/bin/python3`.
- Do not change existing entries' numbers or behaviour.

## Review Focus

1. An ODE checked only against itself. The finite-N simulation must share no code with the ODE right-hand side. A planted wrong ODE term, such as a dropped η² term, must fall outside the simulation band.
2. Gaussian averages: the closed form and the quadrature route must be truly independent. Erf integrals typed with a wrong normalisation are a classic bug.
3. Convention mismatches between papers: α = P/N versus time t = P/N, the learning rate η versus η/N, ε defined as P(error) versus a quadratic error, and the normalisation of the teacher. Each entry states its convention and maps the paper's numbers into it.
4. An "optimal" rule or learning rate claimed outside its class or regime.
5. Soft-committee symmetric plateaus: integrator tolerance or a small initial symmetry breaking can fake an escape. Record the dependence on the initial conditions.

---

### Task O1: Engine and online perceptron (papers 1, 11, 6)

**Files:**
- new `online/__init__.py`, `online/fields.py` (Gaussian averages over the teacher–student local fields: closed forms and quadrature), `online/odes.py` (dR/dα and dQ/dα for a perceptron student with modulation F, teacher noise, learning-rate schedule), `online/optimal.py` (the variational optimal modulation F*, Kinouchi–Caticha 1992), `online/simulate.py` (finite-N online perceptron over named seeds);
- `catalog_online.py` with entry `online_perceptron`, route `online`;
- `latex_online.py`, `reference/online_perceptron.json`, `tests/test_online_perceptron.py`.

**Interfaces:**
- `perceptron_flow(rule, eta=…, noise=…) -> Flow`, where `rule` is one of hebb, perceptron, adatron or optimal, or a callable F(h, b, …).
- `integrate(flow, alpha_max, init) -> Trajectory(alpha, R, Q, eps)`.
- `optimal_modulation(...) -> callable` with its class stated.
- `simulate_perceptron(rule, N, alpha_max, seeds) -> SimResult(eps_mean, band)`.

**Tests:**
- `test_hebb_closed_form`: the ODE ε(α) for Hebb equals the closed form, which is derived and checked against the paper.
- `test_rules_learning_curves_vs_banked` covers Hebb, perceptron, AdaTron and optimal. Asymptotic exponents and constants are compared with Biehl–Riegler and Kinouchi–Caticha. The optimal rule's prefactor is about 0.88/α (verify), compared with the batch Bayes result already in `teacher_student`.
- `test_ode_vs_simulation_band`: the simulation runs at N ≈ 1000 in `@slow`, and at a smaller N in the default tier. A planted ODE error must be excluded.
- `test_fields_closed_vs_quadrature`.
- `test_noise_robustness`: teacher output noise or weight noise, per paper 6 (fetch it to decide which); report if it is not reproducible.
- Refusals for an unknown rule, a non-positive η, and a missing state point.

### Task O2: Linear perceptron, online versus offline (paper 3)

**Files:** `online/linear.py`; entry `online_linear`; bank; tests `test_online_linear.py`.

**Tests:**
- The online ODE for the linear student with quadratic error, with an optimal schedule η(α).
- The online-optimal ε(α) is compared with the offline (batch) result of the same paper. Where `ridge` (E7) gives the same quantity, it must agree. The convention mapping is stated.
- The banked asymptotics from paper 3.
- A simulation band, and a planted error.

### Task O3: Drifting rules (papers 2, 8, 14)

**Files:** `online/drift.py`, with the teacher drift model: random-walk drift at rate D, and abrupt changes if a paper uses them. Entry `online_drift`; bank; tests `test_online_drift.py`.

**Tests:**
- `test_drift_stationary_error`: a stationary ε(D) for Hebb and the optimal rule.
- The Kinouchi–Caticha 1993 lower bound must hold for every rule the engine runs, as a certified-sign or numerical check with tolerance, with a planted violation.
- Vicente–Kinouchi–Caticha 1998: the variational modulation does not depend on the drift parameters. Test the claim exactly as the paper states it, and bank the reported curves.
- Biehl–Schwarze drift results: fetch and confirm the paper, or report it.
- A simulation band.

### Task O4: Tree committee machines (papers 4, 5)

**Files:** `online/tcm.py` (K non-overlapping receptive fields, the hidden-unit fields, the internal-representation average); entry `online_tcm`; bank; tests `test_online_tcm.py`.

**Tests:**
- The online learning curves for the committee machine (Copelli–Caticha 1995): optimal local learning, and the asymptotic constant against K.
- Copelli–Kinouchi–Caticha 1996: optimal TCM learning with K hidden units equals optimal noisy-perceptron learning at the mapped noise level. This is checked as an identity between two engine routes (TCM ODE versus perceptron ODE with noise). The routes must not share the mapping code, and a planted wrong mapping must fail.
- A simulation band at K = 3.

### Task O5: Soft committee machines, online gradient descent (papers 9, 10, 12, 13)

**Files:** `online/scm.py` (the Saad–Solla ODEs for Q_ik, R_in, T_nm with g = erf(x/√2), the closed-form erf averages I2/I3/I4, M student and K teacher hidden units), `online/plateau.py` (plateau detection: its height, length, and dependence on η and the initial overlaps); entry `online_scm`; bank; tests `test_online_scm.py`.

**Tests:**
- The Saad–Solla 1995 learning curves and the plateau height for the realisable case (K = M, e.g. 2 or 3).
- The optimal fixed learning rate and the asymptotic decay, compared with the banked values from the PRE paper.
- Biehl–Schwarze 1995: the K = 2 results.
- Biehl–Riegler–Wöhler 1996: the transient dynamics and the plateau-length scaling with the initial symmetry breaking, banked with the paper's definition.
- The erf closed forms against quadrature.
- The ODE against finite-N simulation. This is `@slow`, N ≈ 500–1000, and must include passage through the plateau.
- A planted error, for example a missing η² term, must change the plateau escape beyond the band.
- An `undetermined` plateau report where the window is too short.

### Task O6: Functional optimisation in multilayer networks (paper 7)

**Files:** `online/optimal_multilayer.py`; extend `online_tcm` and/or `online_scm` with `rule=optimal`; bank; tests `test_online_optimal_ml.py`.

**Tests:**
- The variationally optimal online algorithm of Vicente–Caticha 1997. Whichever architecture the paper treats, its learning curve beats gradient descent in the paper's stated regime, and its banked values are reproduced.
- Optimality within the stated class: a perturbed modulation must do worse, so the check can fail.
- A simulation band.

### Task O7: Documentation and cleanup

Update:
- GUIDE: catalog table, refusal table, the `online` route, finality notes, worked examples (run literally), hazards;
- README, REFERENCES (each paper: primary, secondary, or abstract only), CHANGELOG (Unreleased), and the tools/README row;
- the spec's As-built section.

Fix the deferred minors from the ledger. Run the default and full tiers and record the measured times. Then a final whole-branch review on the most capable model, one fix wave, and one scoped re-review.
