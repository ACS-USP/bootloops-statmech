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

- `replicant` (online learning, O1-O6): the N -> oo order-parameter ODEs of online learning in alpha = examples / N
  (`online/`, new route `online`), each checked against finite-N simulation over named seeds that shares no code with
  the ODE, closed forms and an independent quadrature, with planted ODE terms that must fall outside the band. Entries
  `online_perceptron` (Hebb, perceptron, AdaTron and the variationally optimal rule of Kinouchi-Caticha, output
  noise: Biehl-Riegler 1994, Copelli et al. 1997), `online_linear` (online versus offline, Kinouchi-Caticha 1995),
  `online_drift` (random and deterministic drift, Vicente-Kinouchi-Caticha 1998, Kinouchi-Caticha 1993),
  `online_tcm` (tree committee machine and its noisy-perceptron equivalence, Copelli-Caticha 1995), `online_scm`
  (soft committee machines, plateaus and their escape, critical learning rates: Saad-Solla read through their 1996 /
  1997 restatements, Biehl-Schwarze 1995, Biehl-Riegler-Woehler 1996; `rule=optimal`: Vicente-Caticha 1997's
  functional optimisation). Every ODE value is `not-computed` (no concentration theorem fetched); "optimal" names its
  class; a plateau not left inside the window is `undetermined`. Not fetched: Kinouchi-Caticha 1992 (0.88 read from
  two secondary prints), Saad-Solla's PRL / PRE, Biehl-Schwarze 1992 / 1993; Copelli-Kinouchi-Caticha 1996 abstract
  only. Default tier +4.0 s (ruling: <= 4 s).
- `replicant` (exact Ising, I1-I3): certified 1D magnetization, susceptibility, connected correlation and
  correlation length checked against exact finite rings with the N -> oo bound; Onsager's 2D energy and specific heat
  (the log amplitude (2/pi) ln^2(1 + sqrt 2)) against Kaufman-torus derivatives and the E8 series; the 2D correlation
  length above and below T_c from Onsager's strip spectrum (below T_c `not-computed`: the two-particle gap's
  identification with xi is derived here), nu / eta consistency checks, the critical nearest-neighbour and diagonal
  correlations (Perk-Au-Yang, Deift-Its-Krasovsky); new entries `ising2d_anisotropic`, `ising_triangular` and
  `ising_honeycomb` (route `exact`, certified: Athanasopoulos-Ueltschi's Kac-Ward integral, Baxter's maps, Pierre et
  al.; the triangular antiferromagnet's ground-state entropy), with star-triangle and four-sector duality identities on
  exact tori and a high-temperature torus identity. `ising2d` and `ising1d` keep their earlier values. Default tier
  +1.6 s (ruling: <= 1 s; the task reports' own direct accounting gave about 0.9 s, see Changed).
- `replicant` (percolation, P1-P3): new package `percolation/` with a shared union-find and entries
  `percolation_bethe`, `percolation_graph` (certified cavity fixed points of the generating-function equations, P_inf,
  mean cluster size, Molloy-Reed, cluster-size distributions; route `sample`: finite graphs over named seeds, the ER
  critical window in the slow tier), `percolation_lattice` (square, triangular, honeycomb; certified thresholds 1/2,
  2 sin(pi/18), 1 - 2 sin(pi/18), the Smirnov-Werner exponents, Cardy's formula by two routes, exact crossing
  polynomials; Newman-Ziff sweeps and finite-size scaling validated on the solved lattices), `percolation_kcore` and
  `percolation_bootstrap` (certified k-core and bootstrap fixed points and birth points of Dorogovtsev-Goltsev-Mendes
  and Janson, against csp/xorsat.py's 2-core and Balogh-Pittel's closed forms; exact peeling and spreading). The
  sample route never refuses a correct sampler. Default tier +1.1 s (ruling: <= 1 s).
- `replicant` (advanced Monte Carlo, M1-M3): new package `montecarlo/` and entries `cluster_mc` (Swendsen-Wang,
  Wolff and a Metropolis baseline for the Ising / q-state Potts torus; stationarity chi^2 of stated power; z fits
  printed against the literature), `wang_landau_mc` (Wang-Landau with the 1/t refinement against the exact torus
  g(E), multicanonical reweighting), `histogram_reweighting_mc` (Ferrenberg-Swendsen single and multiple histogram as
  printed by Kumar et al. 1992, with a stated overlap window and refusals outside it) and `population_annealing_mc`
  (Hukushima-Iba as described by Machta and Wang-Machta-Katzgraber; free energy against Kaufman and exact enumeration,
  family diagnostics, the finite-R bias and its corrections), all route `sample`, statistical, unequilibrated runs
  labelled `undetermined`; a shared statistics module (tau_int, binning, jackknife, Student-t bands, equilibration
  diagnostic, chi^2). Default tier +0.5 s (ruling: <= 1 s).
### Changed
- `replicant` (documentation and cleanup, O7): GUIDE (catalog, refusal table, routes, finality notes, hazards,
  every worked example re-run literally on a clean export), README, REFERENCES (every source of the four plans
  classified primary / secondary / abstract only) and the online spec's and three plans' as-built sections updated;
  `catalog_percolation_kcore.py` split (the `percolation_bootstrap` entry moved to `catalog_percolation_bootstrap.py`,
  outputs unchanged); no number or behaviour changed. Time standard, measured cold on clean `git archive` exports:
  default tier 655 tests, 28.3-28.9 s (425 tests, 20.3-21.1 s at the last push, 9f97a28: +7.8 s for the four plans,
  of which about 0.6 s is the REM level-(a) derivation committed between them); full tier (`--full`) 859 tests,
  34 min 39 s. Per-plan default-tier growth by default-selftest differences at the plan boundaries (the rulings'
  method; three cold runs each, run-to-run noise about +-0.5 s): online +4.0 s, Ising +1.6 s, percolation +1.1 s,
  Monte Carlo +0.5 s; the new test functions alone (in-suite) cost 2.7, 0.7, 0.5 and 0.7 s, the rest is the growth of
  the shared catalog-RefCheck and LaTeX tests with each new entry. Cold isolated, each plan's own test modules take
  3.6, 1.3, 1.3 and 1.3 s (including about 0.45 s of imports each).
- `replicant`: finality labels that became computed: the derived RS stability of the perceptrons and
  Hopfield (E1), the p-spin 1RSB values (E4). `tests/test_heldout.py` split by model
  (`test_heldout_sk.py`, `test_heldout_perceptron.py`, `test_heldout_mc.py`, helpers in
  `heldout_common.py`); no assertion changed. Default battery `python3 tools/replicant/selftest.py`
  425 tests, about 22 s (measured 21.4 s); `--full` 557 tests, about 18 minutes (measured 18 min 11 s). GUIDE, README, REFERENCES and the extension spec's
  as-built notes updated.

### Fixed
- `replicant` (CLI regression from E2): `--param ansatz` was taken by the level-(a) 1RSB composer for
  every spec, so the v1 `run sk --param ansatz=1rsb --param beta=2` no longer reached the family 1RSB
  pipeline and `run viana_bray --route cavity ... --param ansatz=1rsb` was refused. For a catalog
  entry `ansatz` is again the entry's own parameter; the composer switch is `ansatz` for specs that
  are not catalog entries and `--param level_a_ansatz=1rsb` for any spec.
- `replicant`: `run sk|teacher_student --route amp --param N=... --param seeds=1` ended in
  `ERROR: TypeError` (exit 1), and an empty seed list fell back to the default; both are now refused
  `precondition:named-seeds`. `lattice.ball.certify` no longer treats an exact ball (radius 0 at two
  precisions) as a radius stall. The LDPC MAP-threshold cite names MMU Example 6 only for (3, 6); the
  spherical p-spin `no-transition:gardner` detail states its scan and resolution first.
- `replicant` (final review of the online, exact-Ising, percolation and Monte Carlo plans): `ising1d` crashed
  (`ValueError`) on a NaN ring ball (N = 3, r = 1; beta = 50): a non-finite ball is now refused
  `no-solution:enclosure`, the ring check raises its precision until its balls are finite and omits the ring
  correlation length where the ring's reflection symmetry makes it undefined (2 r + 1 = N). The percolation FSS
  crashed (`IndexError`) or printed `p_c_crossing` NaN with stderr 0 when too few seeds or sweeps left the
  crossing unresolved: such values are omitted and labelled `undetermined:crossing` (also `:collapse`, `:width`).
  `histogram_reweighting_mc`'s fixed default beta / betas were refused at L >= 32 by its own window: for L > 8 they
  now move towards beta_c by 8/L (L <= 8 unchanged), and an implicit default outside the window is labelled
  undetermined, not refused. The online asymptote and finite-N rho-sign diagnostics label instead of discarding the
  run (`asymptote_status`, `sim_status`); online runs that compute no value say so. The Monte Carlo LaTeX documents
  no longer describe their route as union-find graph sampling. `population_annealing_mc` refuses parameters its
  model does not read and sweeps = 0; one s_max cap for both percolation routes; no "+ systematic 0" in RefCheck
  details. No banked number changed. Default tier 662 tests, 30.2 s cold on a clean export (three runs, 30.17-30.21 s;
  +1.4 s, mostly the FSS and online regression tests); full tier 868 tests.

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
