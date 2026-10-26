# Changelog

All notable changes to BootLoops are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (1.0.x = fixes and tool additions that change nothing existing, 1.x.0 = changes to existing tools).

## Unreleased

### Added
- `replicant`: published replica-method and cavity-method calculations of disordered systems as
  typed, recorded steps (each with its citation and named preconditions; a missing one is a named
  refusal, never a silent fallback), with per-value finality labels (stable, unstable,
  not-computed, exact-by-literature) and checks against banked, cited reference values. v1
  catalog: random energy model, Sherrington-Kirkpatrick (RS, AT stability, 1RSB, full RSB in the
  slow tier), spherical p-spin, spherical perceptron (Gardner capacity), teacher-student learning
  curves, Hopfield (AGS), sparse +-J spin glass (Viana-Bray / Bethe lattice, RS and 1RSB). Routes:
  replica (symbolic level (a) derivation and level (b) family pipelines, with agreement tests) and
  cavity (dense, sparse BP / population dynamics, sparse 1RSB). A composer adapts a catalogued
  derivation to a nearby model and reports what it reused, adapted or could not do; a LaTeX writer
  renders any run as a derivation document (`python3 -m replicant list|run ... --latex out.tex
  [--pdf]`). Held-out evidence: the composed +-J SK model and the composed Ising perceptron are
  checked by independent routes (enumeration, bank targets, an annealed bound) with positive
  controls; its limits are stated in the GUIDE (the Ising perceptron 4/pi RS endpoint is not
  checked, small-N enumeration is coarse, the +-J agreement tests the adapted step and not
  universality). Results are physics-level, not proofs; population-dynamics results are
  statistical. Battery: `python3 tools/replicant/selftest.py` (about 16 s; `--full` about 6 min).
- `replicant` (E1): the replicon at level (a) for Gaussian-integral energies. The derived RS
  solution of the spherical and Ising perceptron and of Hopfield now carries a computed replicon
  (Price covariance derivative, RS split, n -> 0, eigenvector projection; the spherical volume by an
  explicit-n projection; for Hopfield also the m-mode from the Hessian of the derived functional),
  so `solve_derived_rs` returns `stable` True/False for them and composed
  Ising-perceptron values get computed finality; `derived_replicon(spec, solution)`; the replicon
  derivation appears in the LaTeX documents. Checked against the families' replicons (perceptron,
  Hopfield) at 12 points to 1e-8 and at the Ising AT root; planted hook errors move it.
- `replicant` (E2): 1RSB at level (a). The 1RSB functional is derived by rules (block overlap sums,
  two-level Hubbard-Stratonovich, the 1RSB split and spectrum, nested expectations, frozen q1 = 1 for
  counting measures, a T = 0 limit for SK-type energies) and solved with an independent-rule audit
  (`solve_derived_1rsb`); composer route `ansatz=1rsb` (values never final). Checked against the SK
  family 1RSB (T = 0 and finite T, 1e-8) and the banked Krauth-Mezard frozen capacity of the Ising
  perceptron; each rule against brute force at integer n.
- `replicant` (E3a): AMP, state evolution and the TAP free energy for SK (`replicant.amp`), a finite-N
  route that imports none of the replica or cavity code: Bolthausen's iteration with the Onsager term
  for Gaussian and +-J matrices (`sk_amp`, `sk_amp_ensemble` with 99% Student-t bands over named
  seeds), the scalar and two-time state evolution (`sk_state_evolution`, all fixed points listed, the
  predicted AMP convergence from psi'(q)), the TAP free energy (averaged, Plefka and naive forms) and
  TAP fixed points. Checked: the SE fixed point equals the RS q of `families/sk.py` and of the derived
  RS equations (1e-10); psi'(q) equals 1 - lambda_AT (1e-5); at N = 1500 the AMP q lies in the band
  around the SE value for both coupling laws while naive mean field does not; the TAP functional is
  stationary at the AMP fixed point and its value lies in the band around the RS -beta f. Statistical,
  not certified; validity statements (SE theorem for Gaussian J, convergence only above the AT line,
  many TAP solutions at low temperature) are in the GUIDE. No catalog route yet.
- `replicant` (E3b): GAMP for the Bayes-optimal teacher-student perceptron (gaussian or +-1 prior,
  noiseless sign channel) and its state evolution (`amp/gamp.py`, `amp/gamp_se.py`), the RS potential
  and the binary-prior thresholds alpha_IT 1.244684 / alpha_AMP 1.492632 (banked targets from Barbier
  et al. 2019 and a secondary source, which disagree by 0.004 on alpha_IT); catalog route `amp` for
  `sk` (SE value exact-by-literature by Bayati-Montanari Thm 4; finite-N AMP statistical) and
  `teacher_student` (GAMP SE values not-computed: a claim, not a theorem); a new independent held-out
  check, the composed +-J SK q against AMP on sampled +-J matrices (slow).
- `replicant` (E4): the Gardner transition. The 1RSB replicon toward further RSB
  (`families/gardner.py`, the cited Crisanti-Leuzzi / Montanari-Ricci-Tersenghi formula, checked
  against an explicit integer-n replica functional) gives computed finality for the p-spin entries;
  new Ising p-spin family and catalog entry `ising_pspin` (T_s, T_d, T_G = 0.240260 at p = 3 against
  Rizzo 2013; state-point values stable only for T_G < T < T_s); the spherical pure p-spin has no
  Gardner transition (refused with the computed scan) and its 1RSB values are now computed stable;
  the frozen Ising-perceptron 1RSB stability is refused as not computed.
- `replicant` (E5): constraint satisfaction. Random K-XORSAT (`csp/xorsat.py`, entry `xorsat`:
  leaf-removal / zero-temperature cavity thresholds, 2-core, complexity; exact GF(2) elimination and
  leaf removal on sampled instances) and random K-SAT (`csp/ksat.py`, `csp/sp.py`, entry `ksat`:
  first-moment bound, BP, survey propagation on instances and as population dynamics, the root of the
  SP complexity 4.2623 +- 0.0108 against Mertens-Mezard-Zecchina's 4.26675; exhaustive counts for
  N <= 22). Only the proved values are final; the K-SAT thresholds are cavity conjectures. The
  3-coloring stretch was not done.
- `replicant` (E6): inference and codes. Regular LDPC codes on the erasure channel (`inference/ldpc.py`,
  entry `ldpc_bec`: density evolution, BP and MAP thresholds by tangency, trial entropy and the area
  theorem; the peeling decoder on configuration-model graphs); the rank-one spiked Wigner model
  (`inference/spiked.py`, entry `spiked_wigner`, route amp: Bayes-optimal state evolution, the
  threshold, Bayes AMP) and the two-group stochastic block model (`inference/sbm.py`, entry `sbm`: the
  Kesten-Stigum line, BP on sampled graphs).
- `replicant` (E7): ridge and ridgeless regression on isotropic Gaussian design (`learning/`, entry
  `ridge`): the Marchenko-Pastur closed forms of Hastie et al. 2022 and the optimal penalty, the RS
  saddle point of Loureiro et al. 2022 (a cited closed form, not a level-(a) derivation), SVD
  simulations over named seeds extrapolated in 1/n.
- `replicant` (E8): the exact high-temperature series of the 2D Ising free energy from the Onsager
  integral and its holonomic operator through `tools/annihilator` (`series/`, entry `series_ising2d`,
  route series): theta-order 3, degree 8, inhomogeneous, with the elliptic right factor proved
  exactly; the critical point 3 - 2 sqrt 2 and alpha = 0 from exact local exponents; an independent
  exact torus route; the susceptibility has no operator within (6, 40) (a bound, not a proof).
- `replicant` (E9): solvable lattice models certified in ball arithmetic through `tools/baller`
  (`lattice/`, entries `ising2d`, `ising1d` route exact, `bethe_ising` route cavity; `rigor`
  "certified" with radii): Onsager's free energy, Yang's magnetization, Kaufman's torus against the
  exact integer transfer matrix, the 1D ring, every BP fixed point of the Bethe ferromagnet. Needs
  python-flint (the project venv).
- `replicant` (E10): Guerra's RS / 1RSB / 2RSB upper bounds on the SK pressure, certified (`bounds/`,
  entry `sk_bounds`, route bound), with every stationary point of the RS bound certified and a
  consistency check of the package's own RS / 1RSB / K-RSB optimisers against the bound chain.
- `replicant` (E11): Monte Carlo and exact enumeration as independent checks of the held-out models
  (`numerics/`: Gray-code enumeration of +-J SK spectra up to N = 26 and of Ising-perceptron counts up
  to N = 30, numpy parallel tempering with an equilibration diagnostic, N^-2/3 finite-size
  extrapolation, exact finite-N moments). The held-out tests now exclude RS for the SK ground energy
  (-0.7610 +- 0.0124 contains Parisi), tell quenched from annealed for the +-J E ln Z, and exclude the
  annealed entropy and capacity of the Ising perceptron; what they do not resolve (1RSB against
  Parisi, RS against 1RSB at beta = 2) is printed, not asserted.

### Changed
- `replicant`: finality labels that became computed: the derived RS stability of the perceptrons and
  Hopfield (E1), the p-spin 1RSB values (E4). `tests/test_heldout.py` split by model
  (`test_heldout_sk.py`, `test_heldout_perceptron.py`, `test_heldout_mc.py`, helpers in
  `heldout_common.py`); no assertion changed. Default battery `python3 tools/replicant/selftest.py`
  424 tests, about 22 s (measured 21.6 s); `--full` 555 tests, about 18 minutes (measured 18 min 2 s). GUIDE, README, REFERENCES and the extension spec's
  as-built notes updated.

### Fixed
- `replicant`: `run sk|teacher_student --route amp --param N=... --param seeds=1` ended in
  `ERROR: TypeError` (exit 1), and an empty seed list fell back to the default; both are now refused
  `precondition:named-seeds`. `lattice.ball.certify` no longer treats an exact ball (radius 0 at two
  precisions) as a radius stall. The LDPC MAP-threshold cite names MMU Example 6 only for (3, 6); the
  spherical p-spin `no-transition:gardner` detail states its scan and resolution first.

### Known issues
- `replicant`: `families/sk.py` `_rsb_phi_e` (v1, float64 Gauss-Hermite) is inaccurate in the q0 = 0
  / large q1 - q0 sector (2.8e-9, 2.8e-7, 3.8e-6 at beta = 2, 2.5, 3 against the certified balls of
  `bounds/`); no v1 result is affected (that candidate loses the selection by 1.9e-4 or more). Left
  unchanged; recorded in the GUIDE hazards.

## 1.0.1 (2026-10-09)

### Added
- `cutbasis`: geometric master-basis selector on the maximal cut (arXiv:2511.15381
  Algorithms 1-3) writing Kira `preferred_masters` lists in three forms, with a certified
  reducer comparison (`evaluate`) and a self-test; worked example: the two-loop Møller
  double box with a massive loop.

### Fixed
- `run_selftests.py`: a crash in a data-gated package was reported as
  "REFUSED (by design)" and left the run green. A data-gated battery now
  signals a refusal only by exit code 4 (`DATA_GATED_REFUSAL_RC`) with a
  printed reason, kept in `selftest_results.json`; any other nonzero exit,
  an exit 4 without a reason, or one whose last line is an uncaught
  exception, is a FAIL. `ffcapital`, `frobenius-boundary` (was exit 3) and
  `galois` follow the contract; `galois` resolves its reference data before
  any check runs. Documented in `tools/README.md` (Verification classes)
  and `INSTALL.md`. Reported by @Aspect5 (BootLoops-ai/bootloops#3).
- `run_selftests.py`: a timeout in the Julia `Pkg.instantiate()` pre-step
  aborted the whole run and `selftest_results.json` was never written. The
  timeout now fails that one package (with the cause and the remedy in the
  record) and the run continues, exiting 1. Fix by @hzshen88
  (BootLoops-ai/bootloops#2), reported in BootLoops-ai/feedback#1.
- `nestor`: the farm's orphan guard compared each worker's parent pid with
  the pool parent's pid, which is wrong under the `forkserver` start method
  (Python 3.14's default on Linux): every worker exited at its first chunk
  and the farm self-test (P5) never finished. The guard now uses the parent
  pid seen from inside the worker and also checks that the pool parent is
  alive, so it behaves the same under `fork`, `forkserver` and `spawn`.
  Reported by @drdavidglew-afk (BootLoops-ai/feedback#2).
