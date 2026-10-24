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
