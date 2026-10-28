# Online learning in `replicant`: design

Status: draft for approval (2026-10-26).
Marks: **[RV]** = decided by Renato; **[call]** = my call, open to override.

## Intent

**[RV]** Bring into the system the statistical-mechanics papers on online learning by Caticha and co-workers from the 1990s, together with the results of Saad and Solla and the papers of Biehl et al.

Today the package has teacher–student learning only in batch form: replica, and GAMP for Bayes-optimal learning. Online learning is the other half of that literature. Each example is seen once, and in the limit N → ∞ the order parameters (R, Q, and for committee machines the matrices Q, R, T) follow deterministic ODEs in α = examples/N. The papers below derive those ODEs, solve them, and optimise the learning rule variationally.

Success means:
- the system reproduces the published learning curves, optimal rules and asymptotic constants, each with a recorded derivation and a banked, cited reference value;
- each is checked by a route that did not produce it: finite-N simulation over named seeds, closed forms where they exist, and an independent quadrature of the Gaussian averages;
- the same engine computes a nearby model the papers do not treat, for example a different activation, noise level or drift, and says what it reused.

## Literature in scope [RV]

The build fetches every paper and cites what it actually read. A paper that cannot be fetched is reported, and its numbers are not banked from memory.

Caticha and co-workers:
1. Kinouchi & Caticha, J. Phys. A 25, 6243 (1992), *Optimal generalization in perceptrons*: the variationally optimal online modulation function, with the asymptotic ε ≈ 0.88/α.
2. Kinouchi & Caticha, J. Phys. A 26, 6161 (1993), *Lower bounds on generalization errors for drifting rules*.
3. Kinouchi & Caticha, Phys. Rev. E 52, 2878 (1995), *On-line versus off-line learning in the linear perceptron*.
4. Copelli & Caticha, J. Phys. A 28, 1615 (1995), *On-line learning in the committee machine*.
5. Copelli, Kinouchi & Caticha, Phys. Rev. E 53, 6341 (1996), *Equivalence between learning in noisy perceptrons and tree committee machines*.
6. Copelli, Eichhorn, Kinouchi, Biehl, Simonetti, Riegler & Caticha, Europhys. Lett. 37, 427 (1996/97; check the year), *Noise robustness in multilayer neural networks*.
7. Vicente & Caticha, J. Phys. A 30, L599 (1997), *Functional optimization of online algorithms in multilayer neural networks*.
8. Vicente, Kinouchi & Caticha, Machine Learning 32, 179 (1998), *Statistical mechanics of online learning of drifting concepts: a variational approach*.

Saad & Solla:
9. Phys. Rev. Lett. 74, 4337 (1995), *Exact solution for on-line learning in multilayer neural networks*.
10. Phys. Rev. E 52, 4225 (1995), *On-line learning in soft committee machines*: the ODEs for Q, R, T, the symmetric plateau, the escape and the role of the learning rate.

Biehl et al. The build confirms the exact list and the venues, and reports what it found:
11. Biehl & Riegler, Europhys. Lett. (1994), *On-line learning with a perceptron*: Hebb, perceptron and AdaTron learning curves.
12. Biehl & Schwarze, J. Phys. A 28, 643 (1995), *Learning by on-line gradient descent*: the two-hidden-unit soft committee.
13. Biehl, Riegler & Wöhler, J. Phys. A 29, 4769 (1996), *Transient dynamics of on-line learning in two-layered neural networks*: plateaus and their escape.
14. Biehl & Schwarze, *on-line learning of drifting concepts* (1992/93; confirm).

## Shape [call]

- **A new module `tools/replicant/online/`** holds one engine. It takes a teacher architecture, a student architecture, a modulation function or learning rule, the noise, the drift and the learning-rate schedule. It returns the order-parameter ODEs, written as Gaussian averages over the local fields, integrates them in α, and gives the generalisation error.
  - The averages use closed forms where the papers give them (Saad–Solla's erf integrals, the perceptron's 2D Gaussian), and quadrature otherwise.
  - The variational step computes the optimal modulation function as a functional derivative (papers 1, 7, 8) and plugs it back into the ODEs.
- **A new catalog route `online`**, with entries grouped as:
  - `online_perceptron`: papers 1, 11, 6;
  - `online_linear`: paper 3, compared with the offline result;
  - `online_drift`: papers 2, 8, 14;
  - `online_tcm`: papers 4, 5;
  - `online_scm`: papers 9, 10, 12, 13, 7.
  Each entry gets a declared parameter set, per-value finality, a LaTeX derivation block, a bank file with cites, and a GUIDE row. `list` and `run --route online` work like the other routes.
- **The composer** can assemble a nearby online model from the engine's parts and say which parts it reused or adapted. One example is a soft committee with a different activation or a mismatched number of hidden units.

## Checks (each must be able to fail)

- **Closed forms.**
  - Hebb learning gives an exact ε(α).
  - Several asymptotic constants have analytic forms: the optimal rule's 0.88/α versus Bayes offline, AdaTron, and Saad–Solla's optimal learning rate.
  - The ODE integration must match these forms within its stated tolerance.
- **Finite-N simulation.** The online algorithm itself runs at N up to a few thousand, over named seeds, with Student-t bands. The ODE curve must fall within the band. A planted wrong ODE term, such as a dropped second-order η² term or a wrong Onsager-like correction, must fall outside it.
- **Independent quadrature.** The Gaussian averages are recomputed by a second rule (mpmath against the closed form) at several points.
- **Plateaus.** For the soft committee, the code reports the plateau's height and length versus the learning rate and the initial conditions. A plateau is labelled "undetermined" where the ODE has not left it within the integration window. No escape time is guessed.
- **Banked values are targets, never inputs.** Any value that is only read off a figure is banked with a figure-reading tolerance and labelled as such.

## Labels

- The ODE limit is exact as N → ∞ for these models in the physics literature. Mark it `exact-by-literature` only where a fetched paper proves the concentration (for example Goldt et al. 2019 for the soft committee, if fetched). Otherwise mark it physics-level.
- Variational optimality holds within the class of local modulation functions that the papers define. Say so.
- Simulation results are statistical and never final.

## Goals and non-goals

Goals:
- All 14 papers reproduced, or reported as not reproducible with the reason.
- One engine reused across all five entries.
- Default battery growth ≲ 4 s; heavy simulation goes in `@slow`.

Non-goals [call]:
- Restricted training sets (Coolen–Saad).
- Natural gradient (Rattray–Saad).
- Finite-N corrections as a theory: finite N appears only as a simulation check.
- Learning in recurrent networks.

Anti-goals:
- A banked number used as an input.
- A plateau escape time stated where the integration did not escape.
- A rule called optimal outside the class it was optimised in.

## Execution [call]

Same discipline as E1–E11: a plan with one task per paper group, subagent-driven, a review per task, and a final whole-branch review. Work goes directly on `main`, the user's standing choice, with no push until the user asks.

Order:
1. Engine and perceptron: 1, 11, 6.
2. Linear: 3.
3. Drift: 2, 8, 14.
4. Tree committee: 4, 5.
5. Soft committee: 9, 10, 12, 13.
6. Functional optimisation in multilayer networks: 7.
7. Docs.

## As built

Deviations of the implementation from the text above, taken from the build ledger
(`.superpowers/sdd/2026-10-26-online-learning/progress.md`, not committed), the task reports and the code.
Commits 490d90f (this design and its plan) .. 99acf67 (O6 fix round), then the documentation and cleanup task
O7, which also documents the Ising, percolation and advanced Monte Carlo plans built in between.

- **Shape.** One engine package `online/` and the route `online`, as designed, but split by model family rather
  than one generic teacher/student engine: the perceptron flow (`odes.py`, `fields.py`, `optimal.py`) is
  extended by composition for drift (`drift.py`), while the linear perceptron (`linear.py`), the tree committee
  machine (`tcm.py`, `tcm_flow.py`), the soft committee machine (`scm.py`, `scm_flow.py`, `plateau.py`) and
  VC97's optimal algorithm (`optimal_multilayer.py`, `optimal_ml_flow.py`) have their own flows; each has its own
  finite-N simulator that shares no code with the ODE. Ruling O-R1 put each entry in its own
  `catalog_online_<x>.py` / `latex_online_<x>.py`. The five entries are as designed; VC97 (paper 7) is
  `rule=optimal` of `online_scm`, not a sixth entry.
- **Composer.** NOT built for online models: a spec that is not a catalog entry is refused
  (`missing-step:family:<f>`) on the `online` route. Nearby models are reached through the engine's API instead
  (a callable rule F(h, b, Q), a learning-rate schedule, output noise, drift, a mismatched K and M in the soft
  committee machine), which the GUIDE documents.
- **Labels.** No concentration theorem was fetched (Goldt et al. was not), so every ODE value is `not-computed`;
  simulation values are statistical. "Optimal" names its class on every value (`OPTIMALITY_CLASS`). A plateau not
  left inside the window, an escape seeded below round-off (O5 fix) and VC97's K = M = 2 curve past min eig Sigma
  = 1e-3 (alpha ~ 17, O6) are `undetermined`.
- **Papers (goal: all 14 reproduced or reported).** Not fetched: paper 1 (Kinouchi-Caticha 1992; its 0.88 is
  banked from two secondary prints, the derived constant is 0.8834), papers 9 and 10 (Saad-Solla's PRL and PRE;
  read through their NIPS 8 and MANNA restatements by the same authors), paper 14 (Biehl-Schwarze 1992 / 1993:
  nothing banked, their weight-decay Hebb not implemented; the drift models as VKC 1998 and KC 1993 restate them).
  Paper 5 (Copelli-Kinouchi-Caticha 1996) was read from its abstract only; the noisy-perceptron mapping is derived
  here and checked between two independent routes. Read from preprints: papers 2-4, 6, 11-13; from arXiv: 7, 8.
  Source misprints recorded, not banked: Copelli et al. 1997 Eq. (8) (gives 1.325 for the 0.88 of the same group),
  KC93 Eq. (22) (a sign), VKC 1998 Table II (a determinant formula); VKC 1998 p. 15's perceptron exponents 5, 8 are
  UNCLAIMED (the ODE and the O3 reviewer give 3, 4).
- **Not implemented.** Seung-Sompolinsky-Tishby's offline saddle point (the offline linear values come from a
  derived zero-energy geometry and the ridge route), an optimal eta schedule (KC95 prints none; the greedy eta
  is derived here, not banked), BRW's eta_d / eta_s and parity details, BS95's eta_d; the non-goals stay out.
- **Checks.** As designed (closed forms, finite-N bands with planted ODE terms, independent quadrature); the
  default simulation bands use a few hundred inputs, the reference sizes (N = 1000) are in the slow tier.
- **Time standard.** Ruling O-R3: online default-tier growth <= 4 s, O7 adds no default tests beyond doc checks.
  Measured by the default-selftest growth at the plan boundaries (three cold runs each on clean exports):
  20.8 s at 9f97a28 -> 25.4 s at 99acf67, of which about 0.6 s is the foreign REM commit 157cff0, so +4.0 s for
  O1-O6 (the new test functions alone 2.7 s in-suite; cold isolated, the online test modules 3.6 s). The whole
  default tier is now 655 tests, 28.3-28.9 s; the full tier 859 tests, 34 min 39 s (cold, clean export of 40fc924).
