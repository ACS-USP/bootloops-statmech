# replicant — guide

Tool page: none yet (this GUIDE and `README.md` are the documentation).

KIND: package (`tools/replicant/`; run with `tools/` on the path: `cd tools && python3 -m replicant ...`).
Requirements: Python 3 with `sympy`, `mpmath`, `numpy`, `scipy`; `tectonic` or `pdflatex` only for
`--pdf`. Battery: `python3 tools/replicant/selftest.py [--full]`.

PURPOSE: published replica-method and cavity-method calculations of disordered systems (spin glasses,
neural-network capacity and learning) as TYPED, RECORDED STEPS. Each step names its preconditions and its
citation; a missing precondition is a named refusal, never a silent fallback; every run returns a trace a
reader can follow to the paper, a per-value finality label, and a comparison with banked, cited reference
values. Two levels: (a) a symbolic derivation (replicate, average the disorder, introduce order
parameters, RS ansatz, n -> 0, stationarity, solve) that works for models you describe, and (b) the
published pipelines of a catalog of 15 entries, each with a replica route and/or a cavity route (and an
`amp` route for `sk` and `teacher_student`; the constraint-satisfaction entries `xorsat` and `ksat`, the code entry `ldpc_bec` and the block model `sbm` are cavity-only; the spiked Wigner model `spiked_wigner` is amp-only; the ridge-regression entry `ridge` is replica-only, its RS saddle point a cited closed form, not a level-(a) derivation; the 2D Ising entry `series_ising2d` has the route `series` only: an exact series and its holonomic operator, no replica or cavity step). A composer
matches a model description against the catalog, reuses what applies, adapts what it can through
registered hooks, and says what it reused, adapted, or could not do. A LaTeX writer turns any run into a
derivation document. For theoretical physicists and agents who need a published statistical-mechanics
result reproduced with its derivation, or the same calculation for a nearby model.

USE-WHEN:
- You need a published replica/cavity number (a critical temperature, a ground-state energy, a Gardner
  capacity, a learning-curve constant, a Hopfield capacity, a sparse-graph transition) WITH the recorded
  derivation and a check against a banked reference value from the paper.
- You want the calculation for a model near a catalogued one (a different coupling law, a different
  variable type) and want the composer to say what it reused, what it adapted, and where it refuses.
- You need a LaTeX appendix with the steps, equations, citations, check table and honest labels.

NOT-FOR:
- Rigorous proofs. Every result is physics-level: the n -> 0 continuation and the RS/RSB ansaetze (or the
  cavity clustering assumptions) are not proved; the results say so (`rigor: "physics-level"`).
  Nothing here is "certified". The only theorem-level statement is a value flagged
  `exact-by-literature`, which rests on the cited paper, not on this package.
- Population-dynamics results are statistical (value +- stderr over independent runs), never certified.
- Dynamics beyond AMP, DMFT, or quantum models. AMP, state evolution and the TAP free energy exist for
  SK, and GAMP with its state evolution for the Bayes-optimal teacher-student perceptron (gaussian or
  +-1 prior), in the module `amp/`; the catalog route `amp` exists for `sk` and `teacher_student` only.
- Automatic discovery of the RSB structure: the ansatz is an input; RS stability is computed or reported
  as unknown, and there is no search for the right ansatz.
- A model with no level-(a) hook (families: `sk`, `perceptron`, `hopfield`; variables `ising`,
  `spherical`; disorder `gaussian`, `pm_one`): the composer refuses (`missing-step:*`).

INVOKE:
CLI (from `tools/`):
`python3 -m replicant list` prints the catalog (entry, routes, citation).
`python3 -m replicant run <entry|spec.json> [--route replica|cavity|amp] [--param k=v ...] [--out result.json]
[--latex out.tex [--pdf]]`
- JSON result to `--out` (default stdout); the rendered step trace to stderr. Exit 0 ok, 2 refusal
  (`REFUSED: <reason>: <detail>` on stderr; with `--latex` the partial derivation and the reason are still
  written), 1 any other error (`ERROR: <type>: <message>`). `--pdf` without `--latex` is refused
  (`precondition:latex-path`).
- `--param k=v`: integers parse as int; decimals stay DECIMAL STRINGS (mpmath converts them, so `0.1` is
  not rounded through a binary float); `true`/`false` booleans; a comma list is a tuple (a trailing comma
  makes a 1-list: `ys=0.3,`); anything else is a string (`learning=bayes`).
- Worked example, replica route (catalog entry, state point):
  `python3 -m replicant run sk --param beta=2 --out sk.json` -> RS `q`, `f`, `e`, `replicon`; here the
  RS saddle has a negative replicon, so every value is labelled `unstable` and `final: false`.
  `python3 -m replicant run perceptron --param alpha=1.5 --param kappa=0 --latex p.tex --pdf` -> the
  spherical Gardner entropy with the derivation as a PDF.
- Worked example, cavity route: `python3 -m replicant run sk --route cavity --param beta=2` (the dense
  cavity route needs a state point and reproduces the RS equations by a different derivation);
  `python3 -m replicant run viana_bray --route cavity` (the sparse entry is cavity-only, so `--route
  cavity` is required: the default route is `replica` and is refused; returns transition `beta_c`, exact
  T = 0 RS and factorized 1RSB for the 3-regular +-J graph, under 1 s); population dynamics,
  `python3 -m replicant run viana_bray --route cavity --param beta=1.25 --param k=5 --param runs=8`
  (RS; under 1 s measured at the default sizes) or `... --route cavity --param beta=1.25 --param k=5
  --param ansatz=1rsb --param m=0.24` (1RSB population of populations; about 13 s measured), are
  STATISTICAL (value +- stderr); reference-size runs are slower (minutes).
- Worked example, random K-XORSAT (`xorsat`, cavity-only like `viana_bray`; under 1 s each,
  measured): `python3 -m replicant run xorsat --route cavity` -> `alpha_d` 0.818469 (2-core / clustering
  threshold), `alpha_s` 0.917935 (SAT/UNSAT), `lambda_d` 1.256431, `lambda_s` 2.149126, the core jump
  `core_d` 0.357666, all `exact-by-literature` and final; `--param K=4` for K = 4.
  `... --route cavity --param alpha=0.9` adds the 2-core (`core_variables` 0.607076, `core_checks`
  0.594959) and `complexity` 0.008399; with `--param N=1000` (seeds 0-19) also `sat_fraction` 0.95
  (99% Clopper-Pearson [0.683, 1.000]; `errors` gives 0.267, that interval's larger half-width) from exact GF(2) elimination and the sampled core
  (`core_variables_sampled` 0.6134 +- 0.0052): statistical, `not-computed`, so `final` is false.
- Worked example, random K-SAT (`ksat`, cavity-only; measured times; the catalog's SP population
  is fixed at seed 0, 4 runs, burn-in 100: other seeds through `sp.ksat_population_dynamics_sp`;
  an `undetermined` population says so in its finality note): `python3 -m replicant run ksat
  --route cavity` (0.5 s) -> `alpha_1` 5.190893 (first-moment root, `exact-by-literature`: an upper
  bound on the threshold, not the threshold) and `alpha_t` 1.636938 (tangency of the zero-survey
  equation, `not-computed`), so `final` is false. `... --param alpha=3.0 --param N=16` (0.6 s):
  exhaustive counts of seeds 0-19, `sat_fraction` 1.0, `entropy_exact_mean` 0.2913 +- 0.0121,
  `ln_mean_z` 0.3125, `annealed_entropy` 0.292553. `... --param alpha=4.2 --param N=1000 --param
  seeds=0,1,2,3` (2.2 s): BP converged on 0/4 (`bp_converged_fraction` 0, 99% Clopper-Pearson upper
  edge 0.734), SP on 4/4 with `sp_complexity_instances` 0.0071 +- 0.0016. `... --param alpha=4.2
  --param pop=20000` (5 s): SP population `complexity` 0.0051 +- 0.0013, `mean_survey` 0.1779,
  `zero_survey_fraction` 0.00371 (Eq. (39) gives 0.003757), `sp_phase` nontrivial.
  `--param crossing=true` (about 40 s) adds `alpha_s_sp`, the root of the SP-population complexity
  (K = 3 only). Every value but `alpha_1` and `annealed_entropy` is statistical and `not-computed`.
- Worked example, LDPC codes on the erasure channel (`ldpc_bec`, cavity-only; measured times):
  `python3 -m replicant run ldpc_bec --route cavity` (0.6 s) -> (3, 6): `design_rate` 0.5, `eps_bp`
  0.429440 (`x_bp` 0.260571), `eps_map` 0.488151 (`x_map` 0.432263) and `eps_map_area` (area theorem,
  same digits), all `exact-by-literature` and final; `--param l=4 --param r=8` for (4, 8).
  `... --param eps=0.46` adds density evolution: `de_edge_erasure` 0.378887, `de_check_erasure`
  0.907562, `de_bit_erasure` 0.343864 (MMU Example 9: 0.3789, 0.9076, 1 - 0.6561). With `--param
  N=20000` (0.7 s; seeds 0-19) also the peeling decoder: `peel_success_fraction` 0.0 (99%
  Clopper-Pearson [0, 0.233]; `errors` gives 0.233), `peel_residual_mean` 0.3427 +- 0.0027: statistical,
  `not-computed`, so `final` is false. `--param l=2` is refused
  `precondition:variable-degree-at-least-3` (the thresholds need l >= 3).
- Worked example, spiked Wigner (`spiked_wigner`, amp-only; Y = (lambda/N) x x^T + W, W ~ GOE(N);
  measured times): `python3 -m replicant run spiked_wigner --route amp` (0.8 s) -> `lambda_c`
  1.000000 (`exact-by-literature`, final). `... --param snr=2` (0.7 s): `gamma_se` 3, `q_se` 0.75,
  `overlap_se` 0.866025, `mmse_se` 0.25, `matrix_mmse_se` 0.4375 (Lelarge-Miolane Eq. (10) in their
  lambda_LM = lambda^2), all final. `... --param snr=1.5 --param prior=rademacher`: `q_se` 0.692295,
  `matrix_mmse_se` 0.520728 (below the Gaussian 0.691358). `... --param snr=2 --param N=2000` (1.0 s;
  seeds 0-7, iters 30): Bayes AMP `q_amp` 0.7478 +- 0.0074 (stderr; 99% half-width in
  `q_amp_halfwidth99`), `matrix_mmse_amp` 0.4323 +- 0.0039: statistical, `not-computed`, so `final`
  is false. `--param snr=0.9 --param N=500` is refused `precondition:snr-above-one`.
- Worked example, stochastic block model (`sbm`, cavity-only; edge probability a/N inside a group,
  b/N across): `python3 -m replicant run sbm --route cavity --param c=3` (0.6 s) -> the Kesten-Stigum
  point at mean degree 3: `a_ks` 4.732051, `b_ks` 1.267949, `inverse_eps_ks` 3.732051 (DKMZ Fig. 1:
  3.73), final. `... --param a=5.5 --param b=0.5`: `ks_margin` 2.083333 (tau > 1: detectable, final).
  With `--param N=5000` (1.4 s; seeds 0-9): BP `bp_overlap` 0.8121 +- 0.0034, statistical; at `a=4
  b=2` (tau = 1/3) `bp_overlap` 0.0112 +- 0.0022 (a random guess has 0.0113) with every run on the
  factorized fixed point (`bp_factorized_fraction` 1).
- Worked example, ridge regression (`ridge`, replica-only; X with i.i.d. N(0, 1) entries, |beta|^2 =
  `signal`, gamma = p/n, ridge (X^T X + n lam I)^{-1} X^T y; values are the EXCESS risk |beta_hat -
  beta|^2, `test_mse*` add sigma^2): `python3 -m replicant run ridge --param gamma=2 --param lam=0.1`
  (0.5 s) -> `risk_ridgeless` 1.5 (= r^2 (1 - 1/gamma) + sigma^2/(gamma - 1) at the defaults sigma2 =
  signal = 1), `lambda_opt` 2, `risk_opt` 0.780776 (`test_mse_opt` 1.780776 = Dobriban-Wager Eq. (8)),
  `risk` 1.227707 (`bias` 0.506136, `variance` 0.721571), `risk_rs` equal to it (`rs_closed_form_gap` 0 at 30 digits), all final.
  `--param gamma=1` skips the ridgeless step (the peak: it diverges) and still gives `risk_opt`
  0.618034. With `--param n=200 --param gamma=0.5` (seeds 0-19): `risk_sim` 1.0150 +- 0.0037
  (stderr; 99% half-width 0.0105 in `risk_sim_halfwidth99`), statistical and `not-computed`: the
  finite-n mean is off the limit 1 by O(1/n) (exactly p/(n - p - 1) = 1.0101 in expectation), so a
  fixed-n band need not contain the limit. `python3 -m replicant run ridge` is refused
  `precondition:state-point` (gamma is required).
- Worked example, exact series (`series_ising2d`, route `series` only; square-lattice Ising, zero
  field, v = tanh K, u = v^2, g = ln Z/N - ln 2 - 2 ln cosh K):
  `python3 -m replicant run series_ising2d --route series` (0.9 s) -> `a4` 1, `a8` 4.5, `a20` 8174.2
  (exact rationals in `coefficients`, u^0 .. u^100), `torus_agrees_through` 6,
  `torus_even_agrees_through` 12 and `torus_excess_at_L` 2/7 (the exact 7 x 7 torus: a4 .. a12),
  `operator_order` 3, `operator_degree` 8 (fitted on 62 u-terms, `heldout_verified` 39),
  `elliptic_factor_order` 2 with `elliptic_identity` and `theta_g_identity` true, `u_c` 0.171573
  (`u_c_exact` 3 - 2*sqrt(2)), `v_c` 0.414214, `critical_exponent` 2, `alpha` 0. The coefficients,
  `u_c`, `v_c`, `critical_exponent` and `alpha` are final; the three operator values are
  `not-computed` (L g = P is proved exactly, but that is not a computed stability; minimality only
  within the bounds), so `final` is false. `--param max_order=2` is refused
  `no-operator-within-bounds` (53 equations, 30 unknowns, full column rank mod both primes); without
  `--route series` the run is refused `missing-step:route:replica`.
- Worked example, amp route (`sk`, `teacher_student`; under 1 s each, measured):
  `python3 -m replicant run sk --route amp --param beta=1.5` -> `q_se` 0.352602 (the SE fixed point,
  `exact-by-literature`: Bayati-Montanari Thm 4 proves SE for AMP on Gaussian J; it is the AMP limit,
  not a Gibbs-measure statement). With `--param h=0.8 --param N=1500` it also runs finite-N AMP over
  the seeds 0-5: `q_amp` 0.6091 +- 0.0076 (stderr, in `errors`), `q_amp_halfwidth99` 0.0307, SE at the
  same t `q_se_t` 0.613094; `q_amp` is statistical and `not-computed` (never final).
  `python3 -m replicant run teacher_student --route amp --param alpha=2 --param learning=bayes` ->
  `eps_se` 0.174534 (= the replica route's Bayes error to 1e-15), `q_se` 0.728298, all `not-computed`
  (GAMP SE for GLMs is a claim, Barbier et al. 2019 Claim 1); with `--param N=800`: `q_gamp` 0.7269 +-
  0.0369 (99%), `eps_gamp` 0.1726 (plug-in error of the GAMP estimate).
- Composing from a spec: `python3 -m replicant run spec.json --param beta=1.5 --param h=0`. spec.json:
  `{"family": "sk", "variables": "ising", "disorder": "pm_one", "constraint": null, "prior": null,
  "params": {"beta": null, "h": null}}`. Required keys: `family`, `variables`, `disorder`; optional:
  `constraint` (perceptron: `"margin kappa"`), `prior` (`"teacher"` for teacher-student), `params`. A
  param given as `null` is left SYMBOLIC (only `beta`, `h`, `alpha`, `kappa`: the derivation's own
  symbols); a number is a fixed run parameter. Known values: `variables` in {`ising`, `spherical`},
  `disorder` in {`gaussian`, `pm_one`, `none`}; the level-(a) hooks cover `sk` (gaussian, pm_one),
  `perceptron` (gaussian; `margin kappa`) and `hopfield` (pm_one patterns). A spec whose structural key
  (family, variables, disorder, constraint, prior) equals a catalog entry IS that entry (its pipeline is
  reused whole); any other spec is assembled from the seven level-(a) steps and the report lists, per
  step, what was REUSED and what was ADAPTED relative to the nearest entry, e.g.
  `disorder: gaussian -> pm_one; hook disorder:pm_one`. Cavity is available for catalog entries only
  (`missing-step:route:cavity` otherwise).
- Level-(a) 1RSB (`--param ansatz=1rsb`, any spec with level-(a) hooks, catalog specs included: their
  own pipelines are not used): the 1RSB functional derived by rules (`derive_rsb.py`) and solved
  (`solve_derived_rsb.py`); the report lists replicate / average-disorder / introduce-order-parameters
  as REUSED and the RS ansatz, n -> 0, stationarity and solve steps as ADAPTED, relative to the RS
  level-(a) derivation of the same spec. Knobs (also `--param`s): `frozen=true` (q1 = 1; discrete
  variables with a counting measure), `solve_for=<parameter>` (frozen threshold: x fixed to 1, the
  parameter solved for), `zero_temperature=true` (SK-type energies only), `branch`
  (`nontrivial-k` or `max-f`), `audit` (`independent` default, or `quadrature`). An unread parameter
  name is refused `precondition:known-parameter` on this route, composed and catalog specs alike. The frozen Ising-perceptron capacity:
  `python3 -m replicant run spec.json --param ansatz=1rsb --param frozen=true --param solve_for=alpha
  --param kappa=0` with spec.json `{"family": "perceptron", "variables": "ising", "disorder":
  "gaussian", "constraint": "margin kappa", "params": {"alpha": null}}` -> `alpha` 0.83307859974
  (about 1 s). Every 1RSB value is `not-computed` (the 1RSB replicon is not computed): never final.
Python API (put `tools/` on `sys.path`):
`from replicant.catalog import catalog` (dict of `CatalogEntry`);
`from replicant.compose import compose, nearest_entry`; `compose(spec, route).run(require_final=False,
**params) -> Result`; `from replicant.spec import ModelSpec`;
`from replicant.solve_derived import solve_derived_rs` (solve the DERIVED RS equations of a spec at a
state point: all roots found from seeded searches, an audit residual, a computed replicon where derived);
`from replicant.solve_derived_rsb import solve_derived_1rsb` (`solve_derived_1rsb(spec, frozen=,
zero_temperature=, branch=, seeds=, fixed=, solve_for=, **params)`: the derived 1RSB functional
solved, every audited root in `roots`, `stable` always None; `derive_rsb.derived_1rsb_functional(spec)`
is the functional itself);
`from replicant.derive_replicon import derived_replicon` (`derived_replicon(spec, solution_or_mapping)`:
the level-(a) replicon at an RS solution, a mapping first checked to be a root of the derived equations);
`from replicant.families import gardner` (E4, the 1RSB replicon toward further RSB):
`gardner.onersb_replicon(family, p=3, beta=..., m=None) -> mpf` (`family` `pspin` | `ising_pspin`;
`m` only for `ising_pspin`, default the static m), `gardner.onersb_stability(...)` (with the point and
the outer-block replicon), `gardner.pspin_gardner_temperature(p, variables="spherical" | "ising")`
(the Ising T_G 0.240260 at p = 3; spherical refused `no-transition:gardner`),
`gardner.replicon_formula(variables, p, beta, q, m)`, `gardner.explicit_n_replicon(variables, p, beta,
q, n, m)` (the independent integer-n check), `gardner.ising_perceptron_1rsb_replicon(alpha)` (refused
`not-implemented:frozen-1rsb-replicon`); `from replicant.families.ising_pspin import
ising_pspin_1rsb, ising_pspin_critical_temperature` (`ising_pspin_1rsb(3, 2.5)` -> q1 0.929223,
m 0.554371, both nonzero q roots in `q_roots`).
`from replicant import amp` (SK AMP / state evolution / TAP, an independent finite-N route that imports
none of the replica or cavity code; `amp/sk.py` states the conventions): `amp.sk_matrix(N, seed, law)`
(`gaussian` or `pm_one`, variance 1/N, zero diagonal); `amp.sk_amp(J, beta, *, iters, seed, h=0,
damping=None, reaction=True, init="random")` -> `AMPResult(m, q_trace, converged, increments, ...)`
(x^{t+1} = J m^t - beta (1 - Q_t) m^{t-1}, m = tanh(beta (h + x)); `reaction=False` is naive mean field);
`amp.sk_amp_ensemble(N, beta, h, *, seeds, iters, law)` (99% Student-t band over named seeds);
`amp.sk_state_evolution(beta, h=0, *, q0=1, iters=None)` -> `SEResult(q, trajectory, fixed_points,
corr_slope, amp_converges, ...)`; `amp.sk_two_time_state_evolution(beta, h, *, iters)` (predicted
increments between successive iterates); `amp.sk_tap_free_energy(m, J, beta, h=0, *,
onsager="averaged"|"plefka"|"none")` -> mpf phi_TAP = -beta F_TAP / N; `amp.sk_tap_gradient`;
`amp.tap_fixed_point(J, beta, h=0, *, seed)` -> `TAPResult` (AMP then Newton on the TAP equations).
Example: `J = amp.sk_matrix(1500, 0); r = amp.sk_amp(J, 1.5, h=0.8, iters=150, seed=0)` -> `r.converged`
True, `r.q_trace[-1]` 0.5895 (one seed; the seeds 0-5 band is 0.609 +- 0.031, SE q* 0.613094);
refusals `precondition:symmetric-matrix`, `:zero-diagonal`, `:coupling-variance`, `:beta-nonnegative`,
`:known-init`, `:positive-iters`, `:damping-range`, `:known-law`, `:known-onsager`,
`:magnetization-range`, `no-convergence:tap`.
GAMP (teacher-student perceptron, `amp/gamp.py`, `amp/gamp_se.py` state the conventions: teacher and
patterns i.i.d., `y = sign(Phi w*/sqrt N)`, rho = 1, Bayes-optimal): `amp.perceptron_instance(N, alpha,
prior, seed)` -> `(Phi, y, w*)` (`prior` `gaussian` | `ising`); `amp.gamp_perceptron(X, y, prior, *,
iters, seed, teacher=None)` -> `GAMPResult(x_hat, v, overlap_trace, norm_trace, converged, ...)`;
`amp.gamp_ensemble(N, alpha, prior, *, seeds, iters)` (99% Student-t bands of the overlap and of the
plug-in generalization error); `amp.perceptron_state_evolution(alpha, prior="gaussian", teacher=True,
*, iters=None)` -> `SEResult(overlap, mse, r, trajectory, converged, eps_bayes, eps_gibbs,
free_entropy, fixed_points, optimal_overlap, gamp_is_optimal, ...)`; `amp.ising_thresholds()` ->
`Thresholds(alpha_it, alpha_amp)` (1.244684, 1.492632). Refusals `precondition:known-prior`,
`:nonnegative-alpha`, `:positive-alpha`, `:pattern-matrix`, `:label-count`, `:sign-labels`,
`:teacher-length`, `:positive-iters`, `not-implemented:mismatched-teacher-student`.
`from replicant.csp import xorsat, xorsat_finite` (E5a, random K-XORSAT; `csp/xorsat.py` states the
conventions): `xorsat.xorsat_thresholds(K)` -> `XorsatThresholds(alpha_d, alpha_s, q_d, lambda_d,
core_d, q_s, lambda_s)`; `xorsat_bp_threshold(K)`, `xorsat_sat_threshold(K)` (mpf);
`xorsat_core_size(K, alpha)` -> `XorsatCore(q, q_hat, lam, variables, checks, complexity, trivial)`;
`xorsat.DEGREE_LAWS` (the degree-law hook the recursion reads; `PoissonLaw`);
`xorsat_finite.xorsat_instance(K, N, alpha, seed)`, `gf2_solve(idx, rhs, N)` -> (satisfiable, rank),
`xorsat_gf2_satisfiable(K, N, alpha, seed)`, `xorsat_sat_probability(K, N, alpha, seeds)` ->
`SatProbability(fraction, n_sat, n, stderr, lo, hi)` (99% Clopper-Pearson), `leaf_removal_core(idx,
N)`, `xorsat_sampled_core(K, N, alpha, seed)`. Example: `xorsat_sat_threshold(3)` -> 0.917935276658.
`from replicant.csp import ksat, sp` (E5b, random K-SAT; the module docstrings state the
conventions): `ksat.random_ksat(K, N, alpha, seed)` -> `KSatFormula(K, N, alpha, seed, var, sign)`
(`sign` +1 unnegated, -1 negated); `ksat_annealed_bound(K, alpha)`, `ksat_first_moment_threshold(K)`,
`ksat_sp_support_threshold(K)`, `ksat_sp_support(K, alpha)` -> (t, tau) (mpf);
`ksat_exact_count(f)`, `ksat_exact_marginals(f)` -> (Z, ones) (N <= 22), `ksat_count_reference(f)`;
`ksat_bp(f, *, iters, damping, seed, tol)` -> `BPSATResult(converged, iterations, residual, marginals,
entropy, messages, note)`; `ksat_bp_converged_fraction(K, N, alpha, seeds, *, iters, damping)` (99%
Clopper-Pearson); `sp.ksat_sp(f, *, iters, seed, tol, damping, rule)` -> `SPResult(converged,
iterations, residual, trivial, complexity, surveys, note)`; `sp.ksat_population_dynamics_sp(K, alpha,
*, pop, sweeps, seed, burn_in, runs, rule)` -> `PDResult` (values `complexity`, `complexity_mmz18`,
`mean_survey`, `zero_survey_fraction`, with stderr; `phase` trivial / nontrivial / undetermined);
`sp.ksat_sp_crossing(K, alphas, *, pop, sweeps, seed, burn_in, runs, rule, warm_burn_in)` ->
`SPCrossing(alpha_s, stderr, halfwidth99, roots, sigma, ...)`; `sp.SP_UPDATES` (the SP-factor hook),
`sp.audit_sp_rule()` (max deviation from the warning enumeration). Example:
`ksat_first_moment_threshold(3)` -> 5.19089306968.
`from replicant.inference import ldpc, ldpc_finite` (E6a, regular (l, r) LDPC codes on BEC(eps);
`inference/ldpc.py` states the conventions: edge-perspective lambda(x) = x^(l-1), rho(x) = x^(r-1),
DE x_{t+1} = eps lambda(1 - rho(1 - x_t)) from x_0 = eps): `ldpc.bec_density_evolution(l, r, eps)` ->
`DEResult(erasure_prob, converged, edge_erasure, check_erasure, iterations)` (binary64;
`erasure_prob` is the bit erasure eps Lambda(y)); `bec_thresholds(l, r)` -> `BecThresholds(design_rate,
eps_bp, x_bp, eps_map, x_map)`; `bec_bp_threshold(l, r)`, `bec_map_threshold(l, r)` (trial-entropy
root), `bec_map_threshold_area(l, r)` (area theorem by quadrature), `trial_entropy(l, r, x)`,
`eps_of_x(l, r, x)`, `de_map(l, r, eps, x)` (mpf); `ldpc.DEGREE_PAIRS` (the degree-polynomial hook;
`RegularPair`); `ldpc_finite.ldpc_ensemble_graph(l, r, N, seed)` -> `LdpcGraph(var_checks, M,
multi_edges)`, `ldpc_peeling_decode(l, r, N, eps, seed)` -> `PeelResult(success, residual,
residual_fraction, erased, multi_edges, erased_mask, residual_mask, ...)`, `ldpc_peeling_success(l, r,
N, eps, seeds)` -> `PeelStats(fraction, lo, hi, halfwidth99, residual_mean, residual_stderr, ...)` (99%
Clopper-Pearson). Example: `bec_map_threshold(3, 6)` -> 0.488150884192.
`from replicant.inference import spiked, sbm` (E6b). `spiked.spiked_state_evolution(snr, prior)`
(prior `gaussian`|`rademacher`) -> `SEResult(snr, prior, gamma, q, overlap, mmse, matrix_mmse,
fixed_points, bayes, potential, note)` (binary64; q = gamma*/lambda^2, overlap = sqrt(q) the cosine,
mmse = 1 - q the sign-resolved vector MSE, matrix_mmse = 1 - q^2; `bayes`: the root is the global
minimum of the RS potential); `spiked_threshold(prior)` -> mpf; `spiked_potential(snr, gamma,
prior)`; `scalar_mmse(prior, gamma)`; `spiked_matrix(N, snr, prior, seed)` -> (Y, x);
`spiked_amp(Y, prior, *, snr, iters, seed)` -> `AMPResult(xhat, x, gamma_trace, increments,
converged, ...)` (spectral start; `seed` seeds the eigensolver); `amp_metrics(xhat, x0)`;
`spiked_amp_ensemble(N, snr, prior, *, seeds, iters, reaction)` -> `AMPEnsemble(q, overlap, mse,
matrix_mse, converged, bands, ...)` (99% Student-t `Band`s); `spiked.PRIORS` (the denoiser hook;
`Prior(name, nodes, weights, F, dF, sample)`). `sbm.sbm_kesten_stigum(a, b, q=2)` -> `KSResult(a, b,
mean_degree, margin, detectable, impossibility_proved)`; `sbm_ks_line(c)` -> `KSLine(c, a, b,
ratio)`; `sbm_sample(N, a, b, seed)` -> `SBMGraph(N, a, b, seed, labels, edges)`; `sbm_bp(graph, a, b,
*, iters, seed, tol, damping)` -> `BPResult(overlap, converged, iterations, max_change, factorized,
marginals, messages, field)`; `sbm_bp_ensemble(N, a, b, *, seeds, iters)` -> `BPEnsemble(overlaps,
converged, factorized, iterations, band, ...)`; `dkmz_residual(graph, a, b, messages)`. Example:
`spiked_state_evolution(2).q` -> 0.75; `sbm_ks_line(3).ratio` -> 3.73205080757.
`from replicant.learning import ridge, ridge_rs, ridge_sim` (E7; gamma = p/n, lam in HMRT's
convention, `sigma2` the noise variance, `signal` = |beta|^2; mpf at 30 digits):
`ridge.ridge_test_error(gamma, lam, sigma2=0, signal=1)` (excess risk; lam = 0 is the ridgeless
limit), `ridge_bias_variance(...)`, `ridgeless_test_error(gamma, sigma2, signal=1)`,
`ridgeless_bias_variance(...)`, `optimal_lambda(gamma, sigma2, signal=1)` (= sigma2 gamma / signal),
`optimal_test_error(...)`, `argmin_lambda(...)` (golden section on the closed form), `stieltjes_mp(gamma,
lam)` -> (m(-lam), m'(-lam), sqrt D); `ridge._SCALE` (private planted-error hook: bias / variance
factors). `ridge_rs.ridge_replica_rs(gamma, lam, sigma2=0, signal=1)` (lam > 0) -> `RidgeRS(gamma,
lam, V, V_hat, q, q_hat, m, m_hat, risk)`. `ridge_sim.ridge_fit(n, p, lam, sigma2, seed, signal=1)`
-> `RidgeFit(beta, beta_hat, risk, train_residual, bias_x, variance_x, risk_x)`;
`ridge_simulate(...)` -> R_X (float); `ridge_ensemble(n, p, lam, sigma2, seeds, signal=1)` -> 99%
Student-t `Band`; `extrapolate_in_n({n: Band})` -> the band of R_inf in mean_n = R_inf + c/n (the
bands must use disjoint seeds); `limit_ensemble(ns, gamma, lam, sigma2, seeds, signal=1)` -> ({n: Band},
Band of R_inf), seeds shifted by k * 10^6 for the k-th size.
Example: `ridge_test_error("1", "0.5", "1", "1")` -> 0.666666666666667 (2/3).
`from replicant.series import ...` (E8; exact `Fraction` coefficients; needs `tools/` on the path for
`tools/annihilator`): `ising2d_free_energy_series(order, variable="v")` -> [a_0 .. a_order] in v =
tanh K (`variable="u"`: [b_0 .. b_order] in u = v^2; anything else `unsupported:variable`; order <=
`MAX_ORDER` = 600 in v), `onsager_moment(m)` (<(cos t1 + cos t2)^m>, exact), `ising2d._HOOKS`
(private planted-error hook: the moment), `ising2d.onsager_free_energy_numeric(v)` (quadrature, |v| <
sqrt 2 - 1), `torus_polynomial(L, order)` (even subgraphs of the L x L torus by edge count, 3 <= L <=
8, order <= 2L), `torus_free_energy_series(L, order)` ((1/L^2) ln P_L), `torus_window(L)` -> (all,
even): equal to the bulk for every k <= L - 1, and for even k <= 2L - 2 when L is odd (L - 2 when
L is even); `theta_g_identity(n)` (theta g = R + s F on exact coefficients through u^n);
`recognize_holonomic(series, max_order, max_degree, *, fit_terms=None, primes=PRIMES)` ->
`Recognition(operator, certificate, verified_terms, recurrence, fit_terms)`, `Operator(terms, rhs)`
(`terms` {(u power, theta power): Fraction}, `order` = theta-order, `degree`, `coefficient(k)`,
`apply(series)`; L f = rhs, the boundary polynomial), `Certificate(found, max_order, max_degree,
n_terms, equations, unknowns, primes, detail)`, `NoOperatorWithinBounds` (a `Refusal` with
`.certificate`); `elliptic_factor_check(op)` -> `EllipticCheck(right_factor_theta, factor_order,
kills_sF, maps_R_to_rhs, ok)`; `singularities(op)` -> [`Singularity(point, value, factor, multiplicity,
kind, exponents)`] (kind `singular` / `apparent` / `irregular` / `not-analysed`); `QuadNumber(a, b, d)`
(a + b sqrt d, exact); `local_exponents(op, x0)` -> `LocalExponents(point, regular, indicial,
integer_exponents, other_exponents, log_exponents)`; `critical_exponent(op, x0)` -> mpf;
`susceptibility_series(n)` (OEIS A002906, n <= 420). Example: `ising2d_free_energy_series(10)` ->
[0, 0, 0, 0, 1, 0, 2, 0, 9/2, 0, 12]; `critical_exponent(op, QuadNumber(3, -2, 2))` -> 2.0.
`from replicant.latex import render_latex, write_latex` (`write_latex(result, path, pdf=False)`; a
`Refusal` is also accepted and writes its partial derivation).

INPUTS: a catalog entry name or a spec.json (above); run parameters by entry (exactly the names each
route reads, `CatalogEntry.parameters[route]`; any other name, e.g. a typo `kapa`, is refused
`precondition:known-parameter` with the accepted names, never ignored):
`rem`: `beta`, `N_scale` (default 1/2, Derrida's normalization); `sk` replica: `beta`, `h`, `ansatz`
(`rs`|`1rsb`, at a `beta`), `fullrsb` (flag, slow tier); `sk` cavity: `beta`, `h`, `require_stable`;
`sk` amp: `beta`, `h`, and for finite-N AMP `N`, `seeds` (default 0-5), `iters` (default 100), `law`
(`gaussian`|`pm_one`; with `pm_one` every SE value is `not-computed`: SE is proved for Gaussian J only);
a finite-N parameter without `N` is refused `precondition:known-parameter` ("need N"); `pspin` and `ising_pspin`: `p` (default 3), `beta`; `perceptron` replica: `alpha`, `kappa` (default 0); `perceptron`
cavity: `alpha`, `kappa`, `variables` (`spherical`; `ising` is refused `not-implemented:cavity-ising`);
`teacher_student`: `alpha`, `learning` (`gibbs`|`bayes`); `teacher_student` amp: `alpha`, `learning`
(default `bayes`), and for finite-N GAMP `N`, `seeds` (default 0-5), `iters` (default 30; `seeds` and
`iters` without `N` are refused likewise); `hopfield` (both routes): `alpha`, `beta`,
`branch`; `viana_bray`: `graph` (`regular` with excess degree `k`, or `poisson` with mean degree `c`),
`J`, `beta`, `ansatz` (`rs`|`1rsb`), `m`, `ys`, and the sampling knobs `pop`, `inner`, `sweeps`,
`burn_in`, `runs`, `seed`, `require_equilibrated`, `field`, `init_field`, `init_spread`, `init`,
`bary_samples`; `xorsat` (cavity): `K` (integer >= 3, default 3), `alpha`, and for the finite-N
GF(2) check `N` (needs `alpha`; refused `precondition:state-point` without it) and `seeds` (default
0-19; without `N` refused `precondition:known-parameter`, "need N"); `ksat` (cavity): `K` (integer >= 2,
default 3), `alpha`, `N` (needs `alpha`; N <= 22 exhaustive counts, larger N BP and SP on instances),
`seeds` (default 0-19 for counts, 0-9 for BP / SP; needs `N`), `pop` (the SP population; needs
`alpha`), `sweeps` (default 30; needs `pop`), `crossing` (`true` or `false` only: the SP-population root, K = 3;
any other value, e.g. `yes`, `on`, `2`, is refused `precondition:known-parameter`); `ldpc_bec`
(cavity): `l`, `r` (integers 2 <= l < r, default 3, 6; the thresholds need l >= 3), `eps` (density
evolution), `N` (the peeling decoder; needs `eps`, else `precondition:state-point`; N l divisible by
r) and `seeds` (default 0-19; needs `N`); `spiked_wigner` (amp): `prior` (`gaussian`|`rademacher`,
default gaussian), `snr` (lambda >= 0: state evolution), and for finite-N Bayes AMP `N` (needs `snr`
> 1, else `precondition:state-point` / `precondition:snr-above-one`), `seeds` (default 0-7), `iters`
(default 30; both need `N`); `sbm` (cavity): `a`, `b` (both, the KS margin), `c` (the KS point at
mean degree c), and for BP `N` (needs `a`, `b`), `seeds` (default 0-9), `iters` (default 2000; both
need `N`); `ridge` (replica): `gamma` (required, > 0), `sigma2` (default 1), `signal` (|beta|^2,
default 1), `lam` (>= 0: the ridge closed form; > 0 also the RS saddle point), and for the finite-n
check `n` (an int >= 10; p = round(gamma n); lam defaults to 0 there) and `seeds` (default 0-19; needs
`n`); `series_ising2d` (series): `order` (even, >= 20, default 200: the v-order, i.e. u-terms 0 ..
100), `fit_terms` (default 62; fewer than the u-terms, the rest are held out), `max_order` (default
3), `max_degree` (default 9; fit_terms >= (max_degree + 1)(max_order + 1) + 5 + max_degree), `L`
(the torus, 3 .. 8, default 7). Without `beta` an entry returns its T = 0 / critical headline values. A composed spec
(not a catalog entry) does not refuse an unknown name: its notes list "unused parameters [...]".

OUTPUTS: the `Result` object (`values`, `errors`, `rigor`, `route`, `stable`, `final`, `finality`,
`statistical`, `notes`, `params`, `report`, `trace`, `entry`, `spec`; `reference_checks(slow=False)`
re-runs the entry's banked comparisons) and its JSON (`target`, `spec`, `route`, `rigor`, `values`,
`errors`, `statistical`, `stable`, `final`, `finality`, `notes`, `params`, `report`, `trace`,
`references`, `references_skipped_slow`). mpmath values are written as 20-digit decimal strings.
- PER-VALUE FINALITY (`finality[key]`: `state`, `ansatz`, `cite`, `note`): `stable` (stability computed
  by this run and positive), `unstable` (computed negative, or the entropy rules the ansatz out),
  `not-computed` (no stability computed here), `exact-by-literature` (the cited source establishes it;
  needs a cite). Only `stable` and `exact-by-literature` are FINAL; every other value gets a
  "NOT final" note and `final` is false. `run(require_final=True)` refuses if any value is not final.
- A value that is not `stable` or `exact-by-literature` is a CANDIDATE, not an answer. `stable: null`
  means UNKNOWN, not "fine" (a composed model whose energy or site has no replicon rule).
- CHECKS: `references` compares the entry's non-slow banked values within their stated tolerances;
  statistical keys use a 99% Student-t band on hypot(stderr, tol); slow and statistical comparisons
  (`references_skipped_slow`) run only through `Result.reference_checks(slow=True)` or `--full`.
- REFUSAL REASONS (`kind:detail`; all raised by the code; `Refusal.trace` carries the partial trace):

| reason | when |
|---|---|
| `missing-step:family:<f>`, `:site:<v>`, `:disorder:<d>`, `:constraint:<c>`, `:prior:<p>` | composer: the level-(a) hook the spec needs does not exist (the detail names what exists and the nearest catalog entry) |
| `missing-step:route:<r>` | unknown route, entry without that route, or cavity / amp for a spec outside the catalog (no generic cavity or AMP composer) |
| `missing-step:ansatz:<a>` | composer: an ansatz other than `rs`, `1rsb` |
| `precondition:counting-measure-for-frozen`, `:discrete-variables-for-frozen` | frozen 1RSB asked for an energy that is not an indicator pattern average, or continuous variables |
| `precondition:polynomial-energy-for-t0` | T = 0 1RSB asked for a Gaussian-integral energy (Hopfield, perceptron: needs q1 = 1 - C/beta, not implemented) |
| `no-solution:not-a-local-maximum` | `branch="max-f"`: the largest-f 1RSB root is not a local maximum of f (Schur complement over the conjugates) |
| `missing-step:derivation:<name>` | a requested document needs a derivation the entry does not provide |
| `precondition:<name>` | an input or state precondition fails (state point incomplete: `state-point`; out-of-range or non-numeric input; `known-parameter`, `rs-reducible`, `rs-site-trace-form`, `supported-disorder-law`, `nonnegative-alpha`, `finite-kappa`, `known-ansatz`, `known-graph`, `tex-engine`, `latex-path`, `known-target`, `complete-spec`, `known-variables`, `m-range`, `alpha-range`, `replicon-block`, `p-at-least-3`, ... and the step preconditions of the derivation) |
| `precondition:integer-k-at-least-3`, `:positive-alpha`, `:enough-variables`, `:named-seeds` | XORSAT (`csp/`): K not an integer >= 3 (K = 2 is not implemented), alpha <= 0 or non-numeric, N < max(12, 4K) (a finite-N check that small tests nothing), an empty or malformed seed list (seeds must be non-negative integers; the detail names the offending value) |
| `precondition:integer-k-at-least-2`, `:exhaustive-size`, `:damping`, `:iterations`, `:population-size`, `:runs`, `:sweeps`, `:burn-in`, `:crossing-grid`, `unsupported:sp-rule` | K-SAT (`csp/ksat.py`, `csp/sp.py`): K not an integer >= 2 (the zero-survey tangency `ksat_sp_support_threshold` needs K >= 3: `precondition:integer-k-at-least-3`), N > 22 for exhaustive counts, damping outside [0, 1), iters < 1, pop < 100, runs < 2, sweeps < 4, burn_in < max(2, sweeps // 2), fewer than three crossing densities (or `crossing` for K != 3), an SP rule name not in `sp.SP_UPDATES`; also `:positive-alpha`, `:enough-variables` (N < K) and `:named-seeds` as for XORSAT. `no-solution:bracket` / `no-convergence:equilibration` from `ksat_sp_crossing`: the mean Sigma does not change sign on the grid, a run's line is not decreasing, the population is trivial, or still drifting at a grid point |
| `precondition:degree-pair`, `:variable-degree-at-least-3`, `:erasure-probability`, `:enough-variables`, `:socket-count`, `:named-seeds`, `unsupported:degree-pair` | LDPC on the BEC (`inference/`): (l, r) not integers with 2 <= l < r (the design rate 1 - l/r must be positive); a threshold asked for l = 2 (eps(x) has no interior minimum: the BP threshold is the stability limit 1/(r - 1), not implemented); eps not a number in [0, 1]; N not an integer >= max(100, r); N l not divisible by r (the configuration model needs equal socket counts); malformed seeds as for XORSAT; a name not in `inference/ldpc.DEGREE_PAIRS`. `no-solution:bracket` / `:audit`: eps(x) not unimodal on its grid, or the trial entropy without exactly one sign change on (x_BP, 1) |
| `precondition:snr`, `:known-prior`, `:snr-above-one`, `:size`, `:square-matrix`, `:symmetric-matrix`, `:finite-matrix`, `:positive-iters`, `:known-init`, `:named-seeds`, `ambiguous:several-fixed-points` | spiked Wigner (`inference/spiked.py`): lambda not a finite number >= 0 (bools refused); a prior not in `spiked.PRIORS`; finite-N AMP at lambda <= 1 (the spectral start of MV Thm 2 needs lambda > 1); N not an int >= 10; Y not N x N, not symmetric or not finite; iters < 1; an init other than `spectral`; malformed seeds as for XORSAT; more than one nontrivial SE fixed point (none for these two priors) |
| `precondition:degree-parameters`, `unsupported:q`, `unsupported:disassortative`, `no-transition:ks-line`, `:size`, `:edge-probability`, `:sbm-graph`, `:positive-iters`, `:damping`, `:named-seeds` | block model (`inference/sbm.py`): a, b not finite numbers >= 0 or a + b = 0; q != 2; BP with a < b (the field equation is not monotone; BP did not converge at a = 0.5, b = 5.5); `sbm_ks_line(c)` with c <= 1 (tau <= c on the line); N not an int >= 10; a/N or b/N > 1; `sbm_bp` given something other than an `SBMGraph`; iters < 1; damping outside [0, 1); malformed seeds |
| `precondition:gamma`, `:lambda`, `:sigma2`, `:signal`, `:gamma-not-one`, `:size`, `:named-seeds`, `:two-samples`, `:two-sizes`, `no-solution:bracket` | ridge (`learning/`): gamma not a finite number > 0; lam, sigma2 or signal not a finite number >= 0 (bools refused; the RS saddle point needs lam > 0; `optimal_lambda` needs signal > 0); the ridgeless limit at gamma = 1 (it diverges for sigma2 > 0, and HMRT state it for gamma != 1 only); n or p not an int >= 10; malformed seeds as for XORSAT; a band from fewer than two seeds; a 1/n extrapolation from one size; `argmin_lambda` with its minimum at an end of the search interval |
| `precondition:order`, `:torus-size`, `:series-length`, `:fit-terms`, `:bounds`, `:exact-series`, `:two-primes`, `:prime`, `unsupported:variable`, `no-operator-within-bounds`, `no-solution:held-out`, `:prime-disagreement`, `:exact-reconstruction`, `:inconsistent-scan`, `:torus-disagreement`, `:critical-point`, `:analytic-point`, `unsupported:irregular-singularity`, `missing-tool:annihilator` | exact series (`series/`): order not an int in range (or odd / < 20 in the entry); torus L outside 3 .. 8; the fit window too short for the maximal ansatz, or no held-out terms; bounds not ints >= 1; a coefficient not int / Fraction; fewer than two primes, or a denominator divisible by one; no operator of theta-order <= max_order and degree <= max_degree on the fit window (the `Certificate`: full column rank mod both primes; a bound, never a proof of non-D-finiteness); the fitted recurrence fails on a held-out term; the two primes pin different (degree, order); the exact Q step fails; the scan and the maximal-ansatz rank disagree; the torus window is not the expected one; no positive singular point; every local solution analytic at the point; Fuchs fails; `tools/annihilator` not importable |
| `unstable:<ansatz>` | `require_final` or `require_stable`: a computed stability is negative (RS replicon or cavity lambda < 0); `unstable:1rsb` also from `pspin_gardner_temperature` when the spherical scan is negative everywhere |
| `unknown-stability:rs` | `require_final`: an RS value whose stability is not computed |
| `not-final:<ansatz>` | `require_final`: a non-RS value that is not final |
| `no-solution`, `no-solution:audit`, `:solver`, `:bracket`, `:retrieval`, `:retrieval-fold`, `:paramagnet`, `:spinglass`, `:no-maximum`, `:negative-entropy`, `:rs-t0`, `:graph`, `:estimator-mismatch` | no saddle/root in the searched range, or a found one fails its audit (residual, non-finite output, divergence) |
| `ambiguous:several-saddles` | more than one RS saddle found and no branch chosen: name one with `--param branch=...` |
| `no-convergence`, `no-convergence:equilibration`, `:tap` | BP or a survey did not converge; a population-dynamics run is still drifting and `require_equilibrated` is set; `amp.tap_fixed_point`: Newton from the AMP iterate left the TAP residual above tol |
| `no-transition` | the branching number rules out a spin-glass transition |
| `no-transition:gardner` | `pspin_gardner_temperature`: the pure spherical p-spin (no Gardner transition; the detail carries the computed replicon scan), or no sign change of the Ising static-line replicon on the scan |
| `unsupported:degree-law`, `unsupported:J-law` | a graph/coupling law the sparse code does not implement; for XORSAT a name not in `csp/xorsat.DEGREE_LAWS` |
| `not-implemented:annealed`, `:cavity-ising`, `:spherical-rsb`, `:mismatched-teacher-student`, `:frozen-1rsb-replicon` | the named case is outside v1 (`:mismatched-teacher-student`: GAMP state evolution off the Nishimori line; `:frozen-1rsb-replicon`: `ising_perceptron_1rsb_replicon`, the stability of the frozen Ising-perceptron 1RSB is not computed) |
| `unbanked:missing-entry`, `:malformed`, `:no-citation`, `:no-independent-route` | a banked reference value is absent or has no citation / no independent route (a value that fed a fit never certifies it) |
| `tex-engine:failed`, `tex-engine:unavailable` | `--pdf`: the engine ran and failed, or could not reach its offline resources |

CATALOG (v1; `list` shows the live table; values at the default / banked state points):

| entry | routes | headline values | banked references | finality |
|---|---|---|---|---|
| `rem` random energy model | replica | `beta_c` 1.66511 (`T_c` 0.600561, N_scale 1/2) | `T_c`, frozen ground energy, 1RSB `m beta/beta_c = 1` (Derrida) | exact-by-literature (1RSB) |
| `sk` Sherrington-Kirkpatrick | replica, cavity, amp | `tc` 1; RS `-sqrt(2/pi)` = -0.797885; 1RSB -0.765259; at `beta`: RS `q, f, e, replicon`; amp: SE `q_se`, with `N` finite-N `q_amp` | `tc`, RS, 1RSB; slow tier: 2RSB -0.76361, 3RSB -0.76331, full-RSB -0.763166 | `tc` stable; RS ground energy unstable; 1RSB not-computed; full RSB exact-by-literature (slow); amp `q_se` exact-by-literature (AMP limit, BM11 Thm 4), `q_amp` not-computed (statistical) |
| `pspin` spherical p-spin | replica | p = 3: `T_s` 0.586054, `T_d` 0.612372, `q_s` 0.645007; at `beta`: 1RSB `q1, m, f, e, replicon_1rsb` | `T_s`, `T_d`, `q_s`; slow: `e0` -1.172 | 1RSB values `stable` by the COMPUTED 1RSB replicon (= 1 - (p-1) y = 0.290015 at p = 3, the same for every T < T_s: no Gardner transition, `pspin_gardner_temperature(3)` refuses `no-transition:gardner`); `T_d` exact-by-literature (CC05 closed form) |
| `ising_pspin` Ising p-spin (Gardner 1985) | replica | p = 3: `T_s` 0.651385, `T_d` 0.681598, `T_G` 0.240260, `q_s` 0.813518; at `beta` (T < T_s): 1RSB `q1, m, f, e, replicon_1rsb` | `T_s`, `T_d`, `T_G` (Rizzo 2013), `q_s` and the replicon at `T_s` 0.3305 (Nishimori, PRE 2026) | `T_s`, `q_s`, `T_G` stable (computed replicon); state-point values `stable` for T_G < T < T_s, `unstable` below T_G (e.g. `--param beta=5`); `T_d` not-computed (marginal by construction) |
| `perceptron` spherical (Gardner) | replica, cavity | `alpha_c(kappa=0)` = 2; entropy `S` and its `replicon` at `alpha` | `alpha_c`, both routes | `alpha_c` stable for kappa >= 0 (RS exact), unstable and not final for kappa < 0; `S` carries its own computed replicon label. The entry is RS only: `--param ansatz=1rsb` is refused `precondition:known-parameter`; in Python `gardner_capacity("spherical", ansatz="1rsb")` is refused `not-implemented:spherical-rsb` |
| `teacher_student` | replica, amp | Gibbs 0.624658 / Bayes 0.441700 asymptotic constants (`eps ~ c/alpha`); amp: GAMP SE `q_se`, `mse_se`, `eps_se`, with `N` finite-N `q_gamp`, `eps_gamp` | both constants (bank tol is a 3-digit rounding) | exact-by-literature; amp values not-computed (GAMP SE is a claim) |
| `hopfield` (AGS) | replica, cavity | `alpha_c(T=0)` 0.137906, `m` 0.967417, `E` -0.501445, `alpha_global` 0.051854 | all four (Steffan-Kuehn; AGS 0.051 for the global one) | T = 0 values not-computed; state-point values carry their replicon |
| `xorsat` random K-XORSAT | cavity | K = 3: `alpha_d` 0.818469, `alpha_s` 0.917935, `lambda_d` 1.256431, `lambda_s` 2.149126; at `alpha`: 2-core `core_variables`, `core_checks`, `complexity`; with `N`: `sat_fraction` (exact GF(2)) and the sampled core | `alpha_d`, `alpha_s`, `lambda_d`, `lambda_s` (K = 3; MRZ 2003), `alpha_d` K = 4, 5, `alpha_s` K = 4, 5, 6 | thresholds and core exact-by-literature (Molloy 2005 core; Dubois-Mandler 2002 for K = 3, Pittel-Sorkin 2016 for K >= 3); `complexity` exact-by-literature below `alpha_s`, not-computed above; finite-N values not-computed (statistical) |
| `ksat` random K-SAT | cavity | K = 3: `alpha_1` 5.190893 (first-moment root), `alpha_t` 1.636938 (zero-survey tangency); at `alpha`: `annealed_entropy`; with `N`: exhaustive counts (N <= 22) or BP / SP on instances; with `pop`: SP-population `complexity`; `crossing=true`: `alpha_s_sp` (root of the SP complexity) | `alpha_t` K = 3 (MZ 2002 / MMZ 2006); slow, statistical: `alpha_s` K = 3 4.26675 (MMZ 2006 Eq. (43)); the other thresholds (MMZ, KMRSZ tables) banked but UNCLAIMED | `alpha_1`, `annealed_entropy` exact-by-literature (DSS Eq. (2.3)); everything else not-computed (the K = 3 threshold is a cavity conjecture) |
| `ldpc_bec` regular LDPC on the BEC | cavity | (3, 6): `eps_bp` 0.429440, `eps_map` 0.488151 (and `eps_map_area`), `x_bp`, `x_map`, `design_rate`; at `eps`: DE `de_edge_erasure`, `de_check_erasure`, `de_bit_erasure`; with `N`: peeling `peel_success_fraction`, `peel_residual_mean` | `eps_bp` (3, 6) (KRU 2010), (4, 8)-(7, 14) (Pfister slides, secondary); `eps_map` (3, 6) (MMU 2008 Ex. 4), (4, 8)-(7, 14) (KRU 2010), (3, 4), (3, 5), (4, 6), (3, 7) (KRU 2012 Table II, area route); `x_bp` and DE at eps = 0.46 (MMU Ex. 9) | `eps_bp`, `x_bp` exact-by-literature (LMSS 2001 Thm 2); `eps_map`, `x_map`, `eps_map_area` exact-by-literature (MMU Ex. 6 for (3, 6), KRU Lemma 4 otherwise); DE values exact-by-literature when converged; finite-N values not-computed (statistical) |
| `spiked_wigner` rank-one spiked Wigner | amp | `lambda_c` 1; at `snr`: SE `gamma_se`, `q_se`, `overlap_se`, `mmse_se`, `matrix_mmse_se` (snr = 2: 0.75, 0.866025, 0.25, 0.4375); with `N`: Bayes AMP `q_amp`, `overlap_amp`, `mmse_amp`, `matrix_mmse_amp` | `lambda_c` gaussian and rademacher (LM 2019 Sec. 2.3; DAM Thm 4.3 with Lemma 3.3); gaussian `q`, `matrix_mmse` at lambda = 0.8, 1.5, 2, 3 (LM Eq. (10)); the N = 2000 AMP bands against the bank are tests (slow), not catalog checks | `lambda_c` exact-by-literature; `gamma_se`, `q_se`, `matrix_mmse_se` exact-by-literature (LM Thm 1) when the run finds the SE root to be the RS potential's global minimum; `overlap_se`, `mmse_se` exact-by-literature (MV Thm 2) for lambda > 1; finite-N values not-computed (statistical) |
| `sbm` two-group block model | cavity | at `c`: the KS point `a_ks`, `b_ks`, `inverse_eps_ks` (c = 3: 3.732051); at `a`, `b`: `ks_margin` tau; with `N`: BP `bp_overlap` | `inverse_eps_ks` at q = 2, c = 3 (DKMZ 2011 Fig. 1: 3.73); the q = 4, c = 16 value 2.33 banked but UNCLAIMED (q = 2 only) | `ks_margin` exact-by-literature (MNS 2015 Thm 2.1 for tau <= 1 with a + b > 2; Massoulie 2014, MNS 2018 for tau > 1 with a > b), not-computed for a + b <= 2 below the line and for a < b above it; KS point exact-by-literature; `bp_overlap` not-computed (statistical) |
| `ridge` ridge / ridgeless least squares (isotropic Gaussian design) | replica | at `gamma`: `risk_ridgeless` (+ `bias_ridgeless`, `variance_ridgeless`, `test_mse_ridgeless`; not at gamma = 1), `lambda_opt`, `risk_opt`, `test_mse_opt`; with `lam`: `risk`, `bias`, `variance`, `test_mse`, RS `risk_rs`, `rs_V`, `rs_q`, `rs_m`; with `n`: `risk_sim` | ridgeless at gamma = 0.5, 0.9, 1.1, 2, 4 (HMRT Eq. (8)); ridge at three (gamma, lam) points (HMRT Eq. (43), also through `risk_rs`); optimal test MSE at four points (Dobriban-Wager Eq. (8)); lam* at two (HMRT Cor. 5): formula evaluations at chosen points (sympy exact), the papers print formulas, not numbers | closed forms exact-by-literature (HMRT Prop. 2 / Thm 1 / Cor. 5, Thm 6 for the split; DW Thm 2.1); RS values exact-by-literature (Loureiro et al. 2022 Thm 1, specialised here with the label noise); `risk_sim` not-computed (statistical) |
| `series_ising2d` 2D Ising (Onsager) free energy, exact series | series | `a4` 1 .. `a20` 8174.2 (v = tanh K); `operator_order` 3, `operator_degree` 8, `elliptic_factor_order` 2; `u_c` 3 - 2 sqrt 2, `v_c` 0.414214, `critical_exponent` 2, `alpha` 0 | `a4`, `a6`, `a8`, `a10` (Finch's table; independently the exact 7 x 7 torus), `v_c` (Kramers-Wannier / Onsager via Duminil-Copin), `alpha` (Onsager via Duminil-Copin, Baxter) | coefficients exact-by-literature (the cited integral, expanded exactly); operator values not-computed (L g = P proved exactly; minimal within bounds only); `u_c`, `v_c`, `alpha` exact-by-literature |
| `viana_bray` sparse +-J | cavity | `beta_c` (3-regular) 0.881374; T = 0 RS `p0` 1/3, `e0` -23/18; factorized 1RSB -1.27231, mu* 0.4174 | those, plus slow/statistical population-dynamics RS and 1RSB at k = 5, T = 0.8 | `beta_c` stable; the rest not-computed |

HELD-OUT EVIDENCE (the test of REUSE; `tests/test_heldout.py`). Two models that are not catalog entries
are assembled by `compose()` from level-(a) steps and checked by routes that did not feed them:
(1) SK with +-J couplings, against enumeration of `ln E Z` at n = 1 (N <= 12), `T_c` = 1, the RS ground
energy -sqrt(2/pi) and the Gaussian-catalog `q` (universality at leading order in 1/N), and the
de Almeida-Thouless eigenvalue recomputed by quadrature; (2) the ISING PERCEPTRON (J in {-1,+1}^N),
composed from the spherical entry, against S -> ln 2 as alpha -> 0, the zero of the composed RS entropy at
the banked Krauth-Mezard / Ding-Sun threshold (and its `q`, `qhat`), the annealed bound
`S <= ln 2 (1 - alpha)`, and exact enumeration at N <= 14. Positive controls (sign flips, a doubled
coupling variance, the spherical site instead of the Ising one) show the checks can fail. Catalog decision
(ruling R13): the Ising perceptron is DELIBERATELY NOT a catalog entry; it is the held-out model composed
from the spherical entry; its Gardner-Derrida RS endpoint 4/pi and Krauth-Mezard 0.8330786 numbers stay
banked in `reference/perceptron.json` as independent targets. Honest limits:
- The 4/pi RS endpoint of the composed Ising perceptron is NOT checked (the composed solver takes
  17-20 s per point at alpha >= 1.1).
- The small-N enumeration is coarse: its band (about +-0.09) cannot tell quenched from annealed; it
  catches gross errors only (e.g. the spherical site, -0.18 vs 0.515).
- The +-J SK agreement with the Gaussian `q` verifies the ADAPTED second-cumulant step, not universality
  independently; the only +-J-specific independent check is the n = 1 enumeration. The universality cite
  (Carmona and Hu) was read from its abstract only.
- The annealed bound does not detect the spherical-site swap (the other checks do).
- Enumeration slack c/N with c = 1 is an assumption of the test.
- (E3b) An independent finite-N route for (1): AMP run on actually sampled +-J matrices (`amp/`, which
  imports none of the replica, cavity or family code), N = 2000, seeds 0-7, t = 100, at beta = 1.5,
  h = 0.8 (RS-stable side): 0.60715 +- 0.01899 (99%), composed q 0.61309 inside; the composed q with
  the planted doubled +-J variance (0.66054) and naive mean-field AMP (0.82513) fall outside. AMP's SE
  is a theorem for Gaussian J only, so for +-J this is sampled numerical evidence, not a theorem, and
  the band (+-0.019) excludes errors of the size of the planted one (0.053), not small ones (slow).

ACCEPTANCE GATES: `python3 tools/replicant/selftest.py` (default tier: 334 tests, about 22 s on a
laptop; also `python3 run_selftests.py replicant`) covers the core and refusal paths, the symbolic
derivation against independent expressions, level agreement (a)=(b) for every family where both exist,
every catalog entry against its banked non-slow values, the cavity routes, the composer and the held-out
models, the CLI and the LaTeX writer (displayed equations re-checked numerically), the 1RSB replicon
the Gardner transition, random K-XORSAT (closed forms, small-N GF(2) elimination and leaf removal) and random K-SAT (exhaustive small-N counts, BP on trees, the SP-rule audit, small SP populations). `--full` adds 111 `@slow` tests (full-RSB Parisi up to K = 9, population dynamics at reference sizes, the XORSAT GF(2) crossing at N = 400-2000 and leaf removal at N = 100000, the K-SAT SP-population crossing (about 75 s with its planted control) and the BP window, the Hopfield replicon
scans, the p-spin T -> 0 energy, solver-based composed and LaTeX checks, a PDF compile when a TeX engine
is present) and takes about 12 minutes on a laptop (measured 12 min 7 s); run it before merging a
change to a family. The `@slow` marker is honoured by
`selftest.py` only: pytest ignores it and would run everything. Level agreement between the derivation
and the pipelines is a consistency check, not new evidence for the ansatz; the cavity route reproducing
the replica RS equations is an independent derivation but NOT independent evidence for the RS ansatz.

KNOWN HAZARDS:
- Rigor: PHYSICS-LEVEL, not proven. `rigor` says so on every result; the n -> 0 continuation and the
  RS/RSB ansaetze are assumptions. Do not quote a result as a theorem.
- An RS answer whose stability is NOT computed (`stable: null`, `not-computed`) is NOT final. `stable:
  null` means UNKNOWN. Level (a) derives the replicon for SK (pair-separable energy, Ising site), the
  spherical perceptron (Gaussian-integral energy, conjugate-free site), the Ising perceptron and Hopfield
  (Gaussian-integral energy, Ising site): `derive_replicon.py` and `derive_replicon_gauss.py` (Price
  covariance derivative, RS split, n -> 0 of the replica measure, replicon-eigenvector projection; the
  spherical site by an explicit-n projection of its registered trace at n = 4; that eps is a Hessian
  eigenvector and the value is the same at n = 5 is checked for the registered site in the slow tests,
  not on every call).
  A composed model whose energy is anything else (a LogDet energy, a covariance other than `q_ab`) or
  whose site has no rule is reported UNKNOWN (False only if the discrete-variable entropy is negative),
  and `final` is false. A composed RS value labelled `stable` means: derived replicon > 0, for Hopfield
  also the derived m-mode > 0 (`solve_derived_stability.py`: -d^2 phi_red/dm^2 with q, qhat held
  stationary, from the Hessian of the derived phi; it equals beta times the family's `mode_eig`), and
  (discrete variables) RS entropy >= 0 at that point; if a mode cannot be computed the value is UNKNOWN.
  It is LOCAL stability, not a proof that RS is exact. A lower (m-unstable) Hopfield retrieval root can
  have a positive replicon (e.g. beta = 20 at m = 0.98 m_fold: replicon +0.19, m-mode -2.24 beta): only
  the m-mode rules it out. The decoupling of the m_a from the replicon sector is argued, not
  mechanised. The Ising-site susceptibility chi_R = E sech^4 is a
  stated identity (the code differentiates the recorded site factor). The replicon value is accepted
  only when two quadrature rules agree (else fine-panel rules, else `no-convergence:quadrature`): at
  large beta the fixed splits alone can be off by 1e-3 (Hopfield beta ~ 43).
- 1RSB at level (a) (`solve_derived_1rsb`): every value has `stable: null` (`not-computed`: the
  1RSB replicon of E4 exists only for the two p-spin entries, below); physics-level. Float64 product quadrature for the search and
  polish (composite Gauss-Legendre; Gauss-Hermite only for the search, measured 1e-8 off on Hopfield
  integrands); the trapezoid audit of every candidate is a QUADRATURE check only (same compiled
  integrands); the accepted roots are audited through an independent evaluation path (`QuadNum`:
  scalar math-module code, adaptive QUADPACK; `audit="quadrature"` skips it and says so). The search is
  seeded (default: a table plus seeds from the derived RS solution) and includes the q0 = 0 edge
  (q0 pinned at 1e-9, its equation audited there); every note and refusal states the seed count and
  the edges searched; an absent root means "not found from these seeds". The x -> 0 and q1 -> 1 edges
  are NOT searched. An edge candidate whose pinned q0 equation fails while the others hold is "no edge
  root"; one dropped by the polish or an audit is listed as "a root may exist there" and, like any
  dropped nontrivial candidate, blocks the unique label (`ambiguous:several-saddles`). Main-rule window:
  the base rule (32 Gauss-Legendre panels) holds 1e-9 for SK up to about beta = 3; when the audit
  disagrees the candidate is re-polished with 64 and 128 panels (measured: SK beta = 4 needs 64, beta =
  6 needs 128); beyond that a candidate is dropped on audit (named, blocking), never silently. Roots with q1 - q0 < 1e-4 or
  x > 1 - 1e-4 are `rs-degenerate` (collapse to RS), not 1RSB roots. SK at h = 0 has two nontrivial
  1RSB saddles (q0 > 0 and the q0 = 0 edge, at T = 0 and at beta = 2): without `branch` the solver
  refuses `ambiguous:several-saddles`; `branch="max-f"` takes the root of largest f and requires a
  local maximum of f: the Schur complement of the derived phi's Hessian over the conjugates (qhat
  eliminated through its own equation) positive definite. Eliminating qhat through d phi/d q instead
  gives the same roots but a Hessian without max/min meaning (at the Hopfield spin-glass root it has an
  eigenvalue -557 while the Schur complement is positive definite, tested as a positive control); the
  T = 0 functional is built that way, so its local-maximum label rests on a sign agreement checked for
  SK at finite beta, not on a proof. Checked: SK T = 0 energy and beta = 2 / h = 0.3 values against
  `families/sk.py` (1e-8), the frozen Ising-perceptron capacity against the bank (Krauth-Mezard,
  Ding-Sun digits), each reduction rule at integer n against brute force, x = 1 against the derived RS.
  Full (q1 < 1) Ising perceptron at alpha = 0.8 (below the frozen capacity), default seeds and the q0 = 0
  edge: only rs-degenerate roots, so no full-1RSB capacity is claimed (no independent target). Hopfield
  1RSB: no independent target and nothing banked (its T = 0 limit is refused, so no code route could
  check a published T = 0 value); the finite-T spin-glass root at alpha = 0.05, beta = 3 is a local
  maximum of f with f_1RSB > f_RS.
- 1RSB stability, the Gardner transition (E4, `families/gardner.py`): the innermost-block 1RSB
  replicon Lambda_R = 1 - Sigma(q1) chi_R, Sigma = beta^2 p (p-1) q1^(p-2) / 2, chi_R = (1 - q1)^2
  (spherical; Crisanti-Leuzzi 2006 Eq. (31)) or E_m[sech^4] (Ising; Montanari-Ricci-Tersenghi 2003
  Eq. (2.6)). The formula is IMPLEMENTED AS CITED (neither p-spin is a level-(a) family, so the E1/E2
  rule machinery does not produce it); it is checked against an independent rule, the second
  derivative of the explicit integer-n replica functional along a replicon vector (finite
  differences, exact spin enumeration), at integer m (4, 5) only: its use at the solution's
  non-integer m is the analytic continuation in m. A planted sech^2 / (1 - q) is caught by that check
  and by the banked T_G. Computed: the innermost replicon and the outer one (exactly 1 at q0 = 0,
  Sigma(0) = 0); for the spherical model also the (q0, q, m) Hessian of f. NOT computed: the
  longitudinal / anomalous modes of the full replica Hessian and, for the Ising p-spin, the (q, m)
  Hessian; `stable` means "replicon positive", local in that sector. Spherical pure p-spin: on the
  static line Lambda_R = 1 - (p-1) y (0.290015 at p = 3) for every T < T_s, so there is no Gardner
  transition (MRT Sec. 1 states it for this model; Crisanti-Leuzzi App. C for the s+p model with
  s > 2, of which the pure model is the mu_s -> 0 edge); this replaces v1's blanket
  literature label by a computed one. Ising p-spin (float64 Gauss-Legendre, about 1e-12 on the
  equations; `audit_residuals` re-evaluates them by mpmath quadrature): the larger nonzero q root is
  the 1RSB branch (Rizzo 2013; the smaller one is an unstable fixed point, slope > 1, reported); the
  root finder assumes one hump of E_m tanh^2 - q (checked against a dense sign scan at four points,
  not proved); the static m is searched on a 21-point m grid (several roots refused
  `ambiguous:several-saddles`). T_G is the first zero of the static-line replicon below T_s (18-point scan in [0.15, 0.98] T_s,
  then Newton-continued brentq, the end point re-solved in full; a dense slow scan at p = 3 sees one
  sign change; p = 4 gives 0.127841, not banked: no fetched source tabulates it); T_d is the fold of the m = 1 q equation,
  where the m = 1 replicon vanishes like sqrt(T_d - T) (tested). Below T_G the 1RSB values are
  `unstable` and the FRSB solution is not computed. T_d is `not-computed` (the 1RSB threshold states
  are in the 1RSB-unstable region, MRT). Ising perceptron: the 1RSB that sets the capacity is
  frozen (q1 = 1), where Lambda_R is a 0 x infinity limit not evaluated here; its stability toward
  further RSB is NOT computed (`ising_perceptron_1rsb_replicon` refuses). Spherical
  `pspin_gardner_temperature` follows its own 5-point scan: all values > 1e-6 -> `no-transition:gardner`;
  a sign change -> a root-found T_G (audited by the signs either side); otherwise `unstable:1rsb`.
- AMP / state evolution / TAP (`amp/`): finite-N results are STATISTICAL (one run has no error bar;
  `sk_amp_ensemble` gives a 99% Student-t band over named seeds), never certified. SE is a theorem for
  each fixed t as N -> oo for Gaussian J (Bayati-Montanari 2011, Thm 4; Bolthausen 2014); Thm 4 is stated
  for a GOE-type matrix with diagonal variance 2/N, the code uses a zero diagonal (as Bolthausen's g):
  an O(N^-1/2) operator-norm perturbation not literally covered by Thm 4 as stated. Convergence of the
  iterates is proved only on the RS-stable side of the AT line (beta^2 E cosh^-4 <= 1) with h > 0
  (Bolthausen Thm 2.1). At h = 0, beta > 1 the iterates do not converge (measured: increment 0.73 +-
  0.12 at t = 10, N = 1500, two-time SE 0.706) and any fixed point need not describe the equilibrium
  measure; the TAP equations then have exponentially many solutions (and m = 0 at h = 0), so
  `tap_fixed_point` returns ONE of them. +-J: not covered by Bayati-Lelarge-Montanari 2015 (polynomial
  nonlinearities; abstract read); Chen & Lam, arXiv:2003.10431, claim universality for Lipschitz
  iterations with a symmetric random matrix (abstract only; entry assumptions not checked); the +-J
  test is numerical evidence. Measured 99% bands at N = 1500, seeds 0-5: q +- 0.031 (h = 0.8, t = 100)
  and +- 0.015 (h = 0, t = 10). At h = 0 the comparison is at t = 10 because SE is a fixed-t statement
  there and at t = 20-40 the finite-N spread roughly triples (+- 0.044). At h = 0.8 finite-N
  convergence is slower than SE predicts (one seed of six needs about 100 steps for an increment
  < 1e-10, SE about 45), hence t = 100 there. Resolving power (`test_onsager_coefficient_resolution`,
  slow): the Onsager coefficient scaled by 0.9 or 1.1 falls outside both bands, x0.95 only outside the
  h = 0 band, x0.97 and x1.05 inside both (measured with the same hook and seeds): errors below
  about 10% are not excluded. The TAP free-energy band (+- 0.0215 at beta = 1.5, h = 0.8) excludes only
  errors of about 25% or more in the (beta^2/4)(1 - Q)^2 term (that term is about 0.086 there); the
  stationarity check (directional derivatives of the functional at the AMP fixed point, < 1e-2 vs > 1
  without the term) is the sharper detector. damping changes the iterates (SE does not apply) but not
  the fixed points.
- GAMP / teacher-student (`amp/gamp.py`, `amp/gamp_se.py`, route `amp` of `teacher_student`): GAMP's
  state evolution for GLMs is stated as Claim 1 by Barbier, Krzakala, Macris, Miolane & Zdeborova (PNAS
  2019; from Rangan 2011, "some steps of the proof are missing"; Javanmard-Montanari 2013 state SE for
  general AMP covering GAMP, which BKMMZ do not count as a proof; abstracts read): every SE value is
  `not-computed`. The Bayes-optimal
  errors are their Thm 2 (the replica route). Bayes-optimal only on the Nishimori line (teacher drawn
  from the prior, matched channel). With the gaussian prior the SE fixed-point equation IS Gyorgyi's
  spherical Gibbs equation (r = q/(1 - q)): the 1e-6 agreement with `teacher_student_error` checks the
  implementation of an identity, not the physics. The catalog route uses the gaussian prior and an
  i.i.d. N(0, 1) teacher, not the sphere. Ising prior: no RS formula for the binary teacher-student
  overlap exists in the package; its checks are the banked thresholds (`reference/gamp.json`): alpha_IT
  1.244684 computed (BKMMZ state 1.249(1); a secondary source, Li et al. 2022, 1.245; the bank tol 0.004
  covers both, and a dps-25 recomputation agrees with ours: BKMMZ's value looks 0.004 high) and alpha_AMP
  1.492632 (BKMMZ 1.493(1), Li et al. 1.492). In (alpha_IT, alpha_AMP) GAMP's fixed point is NOT the
  Bayes-optimal one (`gamp_is_optimal` False). Finite N (N = 800, seeds 0-23, t = 30): the 99% bands
  (about +-0.011) contain the SE trajectory for the gaussian prior at alpha = 1, 2 and the ising prior at
  alpha = 1 and exclude Onsager-coefficient errors of 10% (closest margin 0.0035; each planted band
  also excludes the SE q_t); smaller errors are not claimed. Ising at alpha = 2 > alpha_AMP: SE reaches q = 1 but not every instance does (the rest
  freeze a few wrong spins; band 0.9928 +- 0.0074, which only just reaches 1). Observed, not explained:
  at N = 800 / 1600 / 3200 (seeds 0-23) the mean is 0.9928 / 0.9961 / 0.9975 (the deficit shrinks,
  asserted in the slow test) while 14 / 13 / 17 of 24 instances reach q = 1. When the
  ising variances collapse (<v> < 1e-6) the iterate is frozen (the scalar-variance update underflows
  there; the result note says so). The fixed-point list is a 240-point grid scan plus the trajectory
  limit and (ising) q = 1: roots closer than the grid spacing could be missed.
- Root searches are seeded: a saddle may be missed; the solver refuses with `ambiguous:several-saddles`
  when it sees more than one, but absence of that refusal is not a proof there is only one.
- Population dynamics near `T_c`: the spin-glass label is trustworthy only for `beta/beta_c <~ 0.95` or
  `>~ 1.1` at default settings (pop 2000, burn_in 50, sweeps 10, runs 8); closer, raise `burn_in` and
  `runs` and use `require_equilibrated`. Results are `value +- stderr`, statistical.
- Sparse 1RSB (population of populations) at finite T: a finite-inner-population dynamical bias of about
  +0.002 to +0.003 in the energy estimator (not removed); the reliable (m, T) window is not mapped; small
  inner populations (<~ 20) collapse towards `q1 = q0`. The banked 1RSB values are reproduction targets.
- The Ising perceptron RS solution is unphysical above alpha ~ 0.83 (negative entropy; the RS curve ends
  at 4/pi, the 1RSB frozen threshold 0.8330786 is the capacity); the replicon crosses zero near
  alpha ~ 1.015 (the level-(a) derived replicon and the family code locate the same root,
  `tests/test_replicon_derived.py`, slow), which agrees with the commonly quoted Gardner-Derrida value but
  is unsourced here and NOT banked; Gyorgyi's "AT-stable up to 4/pi" is believed to be a slip. The
  composed Ising perceptron is `stable` and final below the entropy zero (e.g. alpha = 0.5), `unstable`
  by the negative entropy between 0.833 and the AT point (its replicon is still positive there; the
  finality note says so) and `unstable` by the replicon above it.
- SK full-RSB ground energy (slow tier) is a K-level extrapolation with an empirical power law and a free
  exponent, still drifting at K = 8: -0.7631663 +- 6.8e-6 (the stated band is the whole extrapolated
  tail); `sk_ground_energy("fullrsb")` returns a bare value, use `parisi_ground_energy` for the band.
  The solver is capped at K <= 9 (T = 0) and K <= 5 (finite T); finite-T `e` is not extrapolated.
- The SK 1RSB ground energy lies ABOVE the RS one (-0.7653 vs -0.7979): the replica free energy is
  maximised, so a more broken ansatz raises the energy towards the true value.
- LaTeX writer: a heuristic converter turns code notation into math; some fragments stay in code font and
  a very wide expression can overrun the margin (`--pdf` warns `OverfullDisplay`). Writing a document
  checks nothing by itself; the checks are `tests/test_latex.py`.
- Spherical perceptron: above capacity the entropy is -inf (empty version space; no saddle, so `S` is
  `not-computed`). For kappa < 0 the capacity is labelled `unstable` and `final: false` (the entropy
  carries its own computed label: `S` is labelled by the sign of the RS replicon at that `alpha`, and
  `gardner_entropy` refuses `unstable:rs` where it is negative); e.g. `--param alpha=1.5 --param
  kappa=-1` gives `alpha_c` 13.27 with capacity replicon -1.106 (unstable) and `S` with its own
  replicon. The entry has no `ansatz` parameter: through the CLI or `compose(...).run`, `--param
  ansatz=1rsb` is refused `precondition:known-parameter`; at the Python level
  `gardner_capacity("spherical", ansatz="1rsb")` is refused `not-implemented:spherical-rsb`. Hopfield
  cavity gives the energy, not `f`.
- Level (a) hook names differ: the Theta rule refuses as `known-gaussian-integral`, the exp rule as
  `gaussian-integrable`. At K = 8 the top Hessian eigenvalue (about -2.6e-8) is near the finite-difference
  noise floor, so negative-definiteness is weakly established at the highest levels. The slow-tier catalog
  check of `fullrsb_ground_energy` uses the bank tolerance 2e-4; the solver band (6.8e-6) is checked only
  in `tests/test_parisi.py`. The `exact-by-literature` label of full RSB rests on the cited published
  Parisi value (Talagrand, Panchenko), not on this package's K-level extrapolation, whose band is stated
  separately.
- Random K-XORSAT (E5a, `csp/`, entry `xorsat`): the closed forms are computed from the leaf-removal
  recursion q = 1 - G1(1 - q^(K-1); K alpha) through the degree-law hook `DEGREE_LAWS` (Poisson
  only), 30 digits; the largest root is bracketed by a 64-point sign scan on [q_d, 1] (q_d the
  tangency point), alpha_d by the tangency Phi = dPhi/dq = 0 started from a 39-point scan of
  alpha*(q). They agree to 1e-20 with three published parametrisations written independently in
  the tests (Pittel-Sorkin g_k / psi, Dubois-Mandler (c, u), Cocco et al. s0); a planted law of
  mean (K + 1) alpha, or 1.01 K alpha, moves both thresholds out of the bank tolerance. The
  sources disagree in the sixth digit of alpha_d: K = 4, MRZ's table 0.772278 vs Mezard-Montanari
  0.772280 (computed 0.7722798; bank tol 2e-6 covers both); K = 6, MRZ's 0.637080 vs the computed
  0.6370811 with no second source, so alpha_d at K = 6 is NOT banked. What is proved: the 2-core
  threshold and size (Molloy 2005, read through Pittel-Sorkin Sec. 7; not fetched) and alpha_s
  (Dubois-Mandler 2002 for K = 3, from the C. R. note; Pittel-Sorkin 2016, Thm 16, for K >= 3);
  that alpha_d is also the clustering threshold is the cavity picture (MRZ; rigorous clustering
  results of Ibrahimi-Kanoria-Kraning-Montanari 2015 and Achlioptas-Molloy 2015 were read from the
  abstracts only; `complexity` is labelled as the count of core solutions). The `sat_fraction` error is
  the larger half-width of its 99% Clopper-Pearson interval (the binomial stderr is 0 when every seed
  agrees); the crossing widths were chosen after measuring seeds 0-199, which include the test seeds. Finite N: the GF(2)
  eliminator is checked against brute-force enumeration (N = 12) and an independent XOR-basis rank;
  measured P(sat) at alpha_s is 0.58 / 0.52 / 0.49 at N = 200 / 400 / 1000 (seeds 0-199), and
  the crossing band (99% Clopper-Pearson both sides of 1/2) is +-0.05 at N = 200 (default tier,
  seeds 0-39; it excludes a threshold shifted by +0.05) and +-0.04 / 0.03 / 0.02 at N = 400 /
  1000 / 2000 (slow, seeds 0-99; at N = 2000 shifts of +0.02 are excluded). The sampled core is
  compared in a 99% Student-t band plus a stated finite-size allowance (2e-3 at N = 20000,
  1e-3 at N = 100000); near alpha_d the finite-N core probability has a window of order
  N^(-1/2) (Mezard-Montanari Sec. 18.4.1, Dembo-Montanari 2008, not fetched), so the
  sampled checks sit 0.02-0.05 away from alpha_d. Not implemented: K = 2, the UNSAT
  (MAX-XORSAT) ground energy, non-Poisson degree laws.
- Random K-SAT (E5b, `csp/ksat.py`, `csp/sp.py`, entry `ksat`). The SP threshold is the cavity
  CONJECTURE alpha_s = root of the SP complexity (MMZ Sec. 2.3); Ding-Sly-Sun prove the 1RSB
  formula only for K >= k0, an unspecified constant, so no K of this package is covered and every
  threshold here is `not-computed`. The crossing (`ksat_sp_crossing`, settings `sp.CROSSING`: alpha =
  4.18, 4.22, 4.26, 4.30, 4.34, pop 30000, 30 measured sweeps, burn-in 100 then 33 per warm-started
  density, 6 runs, seed 0; about 40 s) fits a line to Sigma per run and averages the roots: measured
  4.2623 +- 0.0108 (99% Student-t half-width over runs), which contains MMZ's 4.26675 +- 0.00015.
  The catalog RefCheck (`run ksat --param crossing=true`, slow) uses t_{0.995,5} hypot(stderr, tol)
  with the bank tol 5e-4: 4.03 x hypot(0.00267, 0.0005) = +-0.011, the same band. It cannot
  separate MMZ from values a few 1e-3 away, e.g. the conditional upper bound alpha_3 <= 4.262 of
  Lundow-Markstrom 2019 (Monte Carlo plus a monotonicity assumption; abstract only; cited in the
  bank, not banked as a value). A planted 1.005 power on
  the SP factor moves the root to 4.3402 +- 0.0207 (excluded); the band excludes shifts of about
  0.03 and larger, not small ones. The population uses MMZ Eq. (18) rewritten with the fixed-point
  identity (17) (variance about 2.3x smaller; the unrewritten form `complexity_mmz18` is reported
  too and agrees within its band on an equilibrated population, slow test). Instance SP (BMZ Eqs.
  (39)-(41), N = 20000, seeds 0-7, all converged) gives 0.005996 +- 0.000232 at alpha = 4.2 against
  the population's 0.005983 +- 0.000460 (pop 50000, 8 runs): the 99% band 0.0018 is 0.3 Sigma and
  excludes |Delta Sigma| >~ 0.002; a planted 1.005 power in the population only (0.0139) is
  excluded (slow, about 60 s). The population keeps zero surveys
  (no Sec. 5.1 regularisation) and caps phi = -ln(1 - eta) at 50. The SP factor is a hook
  (`sp.SP_UPDATES`) audited against an enumeration of warning configurations (max deviation
  4e-16); a planted rule is caught there and changes the instance complexity. Near the SP onset
  (MMZ 3.927) the population relaxes slowly and is labelled `undetermined` (pop 2000, burn-in 40: 3.85
  and 3.9 undetermined, 4.0 non-trivial; with the catalog's burn-in 100 both collapse to trivial); the onset is bracketed (3.6 trivial, 4.2 non-trivial), not
  located. BP on instances: the measured convergence edge (synchronous BP, damping 0.5, 1000
  sweeps, seeds 0-19) is 3.5 -> 20/20, 3.8 -> 8, 3.9 -> 2, 4.0 -> 0 at N = 2000 and 3.7 -> 18, 3.8 ->
  3, 3.9 -> 0 at N = 6000 (damping 0.8 gives the same counts at N = 2000; undamped synchronous BP
  fails from 3.5 on, a schedule effect). The edge brackets alpha_d = 3.86 (KMRSZ 2007, where for
  K = 3 alpha_d is a local, Kesten-Stigum instability) but is NOT a measurement of it. SP on
  instances: synchronous sweeps, tol 1e-6; at N = 1000, 3 of seeds 0-3 converge at alpha = 4.2 (BMZ
  report non-convergence at small N too). MMZ find the 1RSB solution stable only for 4.15 < alpha <=
  alpha_s (K = 3); replicant computes no SP stability. Small N: exhaustive counts (N <= 22) agree
  with an independent backtracking counter; on named seeds mean ln Z <= ln mean Z holds (AM-GM:
  true for any data, a tautology kept because the brief asked for it); the content is that the
  sample mean of Z / E Z contains 1 (a (1 - 2^-(K-1))^M law is excluded); at N =
  12 the quenched-annealed entropy gap is NOT resolved (alpha = 2: 0.4177 +- 0.0042 against
  0.4261). `alpha_t` = 2 alpha_d^XOR(K) exactly: under q = u^(1/(K-1)), alpha -> alpha / 2 the
  zero-survey equation IS the XORSAT leaf-removal recursion, so this confirms the numerics
  through separate code, not by an independent derivation; the independent check is the
  population's measured zero-survey fraction against t(alpha). It is not a phase transition. Not implemented: SP stability,
  reconstruction / condensation thresholds (banked from KMRSZ, UNCLAIMED), SP decimation, the 3-
  coloring stretch.
- LDPC codes on the BEC (E6a, `inference/`, entry `ldpc_bec`). Thresholds come from the edge-
  perspective polynomials of the `DEGREE_PAIRS` hook at 30 digits: eps_BP is the minimum of eps(x) =
  x / lambda(1 - rho(1 - x)) (a 200-point scan, unimodality checked: MMU Fact 1, then the root of the
  derivative numerator); eps_MAP the root of the trial entropy P(x) on (x_BP, 1) (exactly one sign
  change required); the area route integrates the BP EXIT curve by Gauss-Legendre quadrature and
  agrees to 1e-12. The tests root the hook-free KRU polynomials p_BP, p_MAP (agreement 1e-20).
  Planting the node-perspective polynomials (x^l, x^r), or only rho(x) = x^r, moves both
  thresholds out of the bank tolerance. What is proved: below eps_BP the peeling decoder succeeds
  w.h.p. for l >= 3 (Luby-Mitzenmacher-Shokrollahi-Spielman 2001, Thm 2); DE gives the N -> oo BP
  erasure probabilities (Mezard-Montanari Sec. 15.3; the concentration theorem of Richardson-Urbanke
  2001 was not fetched); eps_MAP: MMU 2008 prove the Maxwell value an upper bound (Lemmas 2-3) and
  the MAP threshold where Theorem 10 / Corollary 2 apply, verified for (3, 6) in their Example 6;
  for other regular pairs the equality is cited from Kudekar-Richardson-Urbanke 2010 Lemma 4, which
  refers to Richardson-Urbanke's book (not fetched). Source slip: MMU Example 3 prints (102 - 7
  sqrt 21)/108 = 0.647426 as the MAP bound of (x^2, x^3); that is this ensemble's BP threshold (the
  MAP / area value is 0.746010, KRU 2012 Table II 0.7460), so it is not banked. The (4, 8) to (7, 14)
  BP digits come from slides (Pfister 2015), a secondary source. Finite N (independent code): the
  configuration model keeps multi-edges (mean 4.6 per graph for (3, 6) at N = 20000) and the
  decoder runs on the multigraph; the residual set is checked to be a stopping set and equal to a
  parallel-rounds reference peeler. Default tier (N = 2000, seeds 0-19): at eps = 0.36 every seed
  decodes; at eps = 0.55 the mean residual 0.5103 +- 0.0030 contains DE 0.5103 in its 99% t band
  (0.0086) and excludes DE at eps -+ 0.01. Slow (N = 20000): at eps = 0.50 (seeds 0-29) 0.4291 +-
  0.0015 vs DE 0.4293, band 0.0042, excludes DE at eps -+ 0.005; nearer eps_BP the finite-N mean is
  biased low (eps = 0.45: 0.3118 +- 0.0016 vs DE 0.3159). Crossing (seeds 0-59): success 54/60 at
  eps_BP - 0.005 (99% CP [0.759, 0.974]), 28/60 at eps_BP, 6/60 at eps_BP + 0.005 ([0.026, 0.241]):
  the crossing is within +-0.005 of eps_BP, thresholds shifted by +-0.01 are excluded, and so are
  the planted thresholds 0.3664 and 0.4366. Not implemented: l = 2 (stability-limited) thresholds,
  irregular ensembles in the catalog, the MAP conditional entropy above eps_MAP, finite-length
  scaling laws.
- Spiked Wigner (E6b, `inference/spiked.py`, entry `spiked_wigner`). Convention Y = (lambda/N) x x^T
  + W, W ~ GOE(N) (Montanari-Venkataramanan Eq. (2.1)); Lelarge-Miolane and Deshpande-Abbe-Montanari
  write sqrt(lambda_LM/n) X X^T + Z, so lambda_LM = lambda^2 and LM's Gaussian q* = 1 - 1/lambda_LM is
  q = 1 - 1/lambda^2 here. Four quantities, not one "MMSE": q (overlap of a posterior sample, LM
  Thm 2), the cosine sqrt(q), the sign-resolved vector MSE 1 - q (the posterior mean itself is 0 for
  a symmetric prior), and LM's matrix MMSE 1 - q^2 (the theorem quantity: (2 - 1/lambda^2)/lambda^2
  for the Gaussian prior, not 1/lambda^2). SE uses only the posterior-mean denoiser of the `PRIORS`
  hook, a trapezoid rule over G (Gauss-Hermite with 101 nodes was off by 3e-7 at gamma = 3.7) and
  1 - mmse = 2 E[X F] - E[F^2] (no cancellation against E X^2 = 1: computing mmse first gave a
  spurious root gamma ~ 1e-7 at lambda = 1 and a crash at 0.99999); roots below 1e-9 lambda^2 are
  not searched: for 1 < lambda < about 1 + 5e-10 the SE reports gamma = 0 with `se_below_floor`
  true and every SE value `not-computed` (a floor, not the fixed point; LM: q* > 0 for lambda > 1). The code reports the largest
  root, MV Thm 2 the smallest positive one (gamma_ALG): they coincide because several nontrivial
  roots are refused;
  planting the Gaussian denoiser y/(1 + 2 gamma) moves q out of the bank and makes the root
  non-stationary for the RS potential, which is evaluated from the cited mutual informations by
  QUADPACK. MV Prop. 2.2 says the Bayes gamma "maximizes" Psi; DAM Lemma 3.3 minimises the same
  function and Psi'' < 0 at gamma = 0 for lambda > 1: the code minimises. Rademacher <= Gaussian
  (MMSE): the linear estimator achieves 1/(1 + gamma) for every unit-variance prior and the SE map is
  monotone. AMP: spectral start (MV Thm 2 needs lambda > 1), gamma_t estimated from |x^t|^2/N (MV
  use the SE gamma_t: a plug-in, so AMP never calls the SE code). At N = 2000 the per-matrix spread
  of q_hat is about 0.03; with seeds 0-31 (slow test) the 99% band at lambda = 1.5 is 0.5524 +- 0.0174
  (bank 0.5556), at 2 0.7514 +- 0.0126 (0.75); without the Onsager term 0.6176 +- 0.0154 and 0.7763
  +- 0.0117, with it halved 0.5983 +- 0.0155: all three exclude the bank. Rademacher AMP at 1.5:
  0.6958 +- 0.0104 (SE 0.6923; the Gaussian 0.5556 is excluded). Not implemented: priors with a
  first-order transition (sparse PCA), several nontrivial SE roots (refused), the spectral threshold
  as a computed eigenvalue statement.
- Block model (E6b, `inference/sbm.py`, entry `sbm`). Sampler: labels i.i.d. uniform, per block an
  exact Binomial edge count and a uniform set of distinct pairs. BP is DKMZ Eq. (26) with the field
  (27), updated synchronously; the field is solved each sweep for consistency with the marginals (a
  lagging field oscillates with period 2 when (a - b)/2 > 1). The independent check `dkmz_residual`
  evaluates Eq. (26) in product form, edge by edge; with the field planted to zero the residual is
  above 1e-4 (0.63 measured), with it below 1e-6 (5e-8). Measured at N = 5000, c = 3, seeds 0-9 (slow test): tau = 1/3 and
  0.75 give bands that contain the random-guess value sqrt(2/(pi N)) = 0.0113 (factorized on 10/10
  and 9/10 graphs; 0.0112 +- 0.0072 and 0.0198 +- 0.0256); tau = 1.5 and 2.08 give 0.594 +- 0.018
  and 0.812 +- 0.011 (99% half-widths); shifted rules tau' = 2 tau and tau / 2 are
  contradicted. At b = 0 (tau = 3 at c = 3) the groups are disconnected and one seed converged to a
  fixed point with 88% of the nodes in one group: BP fixed points are initial-condition dependent and
  BP from random messages is not proved to reach the KS line (MNS 2018 Sec. 2.1). DAM's MMSE theorem
  for the SBM needs a diverging degree; the sparse-graph threshold theorems (MNS 2015, Massoulie
  2014, MNS 2018) are about detection, not about BP's overlap. Not implemented: q > 2, unequal group
  sizes, the Bethe free energy, the q = 4 hard phase, BP for a < b (refused; the KS margin is
  computed, but above the line with a < b it is not-computed: the possibility theorems fetched assume
  a > b, the impossibility one is symmetric).
- Ridge regression (E7, `learning/`, entry `ridge`). Convention HMRT (Hastie-Montanari-Rosset-
  Tibshirani 2022): rows x_i ~ N(0, I_p) (not N(0, I_p / p)), |beta|^2 = r^2 deterministic, gamma =
  p/n, ridge argmin (1/n)|y - X b|^2 + lam |b|^2; Dobriban-Wager use the same lam with unit noise and
  a random beta (Var = alpha^2 I/p), Advani-Saxe use inputs of variance 1/N, alpha_AS = n/p and a
  strength gamma_AS = lam / gamma; Loureiro et al. lam_L = lam / gamma. Reported: the EXCESS risk
  (HMRT's R_X, no test noise); `test_mse` = risk + sigma^2 is Dobriban-Wager's predictive risk. The
  closed form is evaluated from the Marchenko-Pastur quadratic in stable forms (m = 2/(b + sqrt D),
  m' = m (1 + gamma m)/sqrt D, m - lam m' = (1 - lam m)/sqrt D): the printed m(z) loses log10(1/lam)
  digits to cancellation for gamma < 1 as lam -> 0, and m - lam m' cancels for gamma > 1. The
  peak: the ridgeless risk diverges like sigma^2/|1 - gamma| from both sides, and at gamma = 1 the
  ridge risk grows like lam^(-1/2) as lam -> 0+ (measured: 500 at lam = 1e-6, 5000 at 1e-8). At lam*
  = sigma^2 gamma / r^2 the risk is finite at gamma = 1 and increasing in gamma on 0.1..10 (SNR 1 and
  5), below the ridgeless risk and the null risk r^2; a golden-section argmin of the closed form
  agrees with lam* to 1e-15 and not with sigma^2 / r^2. HMRT's minimum of the optimally tuned risk
  "around gamma = 1" is for the misspecified model (their Sec. 1.1, isotropic features, item 3: "For a
  misspecified model, optimally-tuned ridge regression attains its global minimum around gamma = 1");
  in the well-specified isotropic model implemented here R* is increasing in gamma; not banked. Replica route: replicant's formal layer does not derive
  the ridge free energy (no Family with a teacher replica and a quadratic loss); the RS saddle point
  is Loureiro et al.'s Eq. (A.38) (their Thm 1 Eq. (2.8) for the label-noise term), specialised to
  isotropic design and solved without the Stieltjes transform (V by a bracketed root, q and m from
  linear equations; but the V equation is the MP quadratic itself, V = gamma m(-lam), so the
  agreement checks the q, m and bias / variance algebra, not the MP law independently); it agrees
  with HMRT Eq. (43) to 1e-12 on a 12-point grid, and a bias or
  variance error planted in `ridge._SCALE` separates them by > 1e-3. Finite n: R_X from the SVD
  (Lemma 1 identities on the sampled spectrum, averaged exactly over the training noise), 99%
  Student-t bands over named seeds (disjoint seed sets per size, `ridge_sim.limit_ensemble`: with
  the same seed X_80 would be a block of X_160 and the sizes correlated), extrapolated in 1/n because the O(1/n) offset of the mean is as
  large as the band (gamma = 0.5, n = 200, seeds 0-19: 1.015 +- 0.011 misses the limit 1; the
  Wishart mean p/(n - p - 1) = 1.0101 is inside). Default tier n = 80, 160 (20 seeds; R_inf
  half-widths 0.043, 0.075, 0.061 at gamma = 0.5, 2 and gamma = 1, lam = 0.5): the closed form is
  inside, a missing bias (shifts 0.5, 1/3) and +10% variance at gamma = 0.5, 2 (+50% at the ridge
  point) are outside. Slow (n = 400, 1000, 24 seeds; half-widths 0.005, 0.020, 0.018, and 0.037 at
  lam* for gamma = 2, r^2 = 2): missing bias, +5% variance (gamma = 0.5, 2) and +25% (ridge points)
  excluded. The sampled-noise risk |beta_hat - beta|^2 (`RidgeFit.risk`) is much noisier and right-
  skewed for gamma < 1 (seeds 0-19 at n = 200 gave 0.899 +- 0.092 against an expectation 1.0101).
  Not implemented: anisotropic covariance (HMRT Thm 2/5), misspecified or latent-space models,
  random features, benign-overfitting spectra (Bartlett et al. 2020: downloaded, not used, context only), non-Gaussian
  design in the simulation.
- Exact series (E8, `series/`, entry `series_ising2d`). Variable v = tanh K (u = v^2 for the
  recognition); other variables are refused. The series comes from the Onsager integral (as printed
  by Duminil-Copin 2022, Eq. (3.6); the 1944 original not fetched) expanded in exact rational
  arithmetic: g = ln(1 + u) - sum_k [4^k M_2k / (4k)] W^k, W = u (1 - u)^2/(1 + u)^4, the moments
  M_m = <(cos t1 + cos t2)^m> integrated exactly; it matches Finch's table (Harris-Meir) through v^10.
  Independent exact route: the L x L torus by a parity transfer matrix (no Onsager integral). It
  equals the bulk for every v^k with k < L (a cluster of length k that wraps nothing has extent <=
  k/2 < L) and is larger by exactly 2/L at v^L (the 2L straight winding loops). For odd L the even
  coefficients agree through v^(2L - 2): an even subgraph of homology class (a, b) mod 2 has length
  = a + b mod 2 and contractible cycles are even, so an even-length wrapping cluster needs two odd
  windings or class (1, 1), length >= 2L; v^(2L) is the first even difference (verified L = 3, 5,
  7). The default L = 7 checks a4 .. a12, which covers the banked a4 .. a10; a planted moment error
  is caught. No finite lattice at this size constrains the recognised operator, which needs >= 62
  u-terms (v^122) at its (degree, order) = (8, 3): the operator rests on the exact identity below and
  on the held-out terms. The operator: the annihilator finds theta-order 3, degree 8 (recurrence
  order 8, degree 3 in n) on 62 u-terms (default bounds order <= 3, degree <= 9), 2-prime confirmed,
  and none of theta-order <= 2 with degree <= 9; it holds on 39 held-out terms (81 in the slow tier,
  fitted on 80 at bounds (4, 10)). It is INHOMOGENEOUS as an ODE: L g = P(u) = 2u^2(u^5 + 8u^4
  - 10u^3 + 28u^2 - 15u + 4) (the recurrence's boundary terms); a homogeneous operator needs one
  more order. The brief's "order 2 (elliptic)" is true for the right factor: L = M theta, and M
  (order 2) is the hypergeometric operator of F = 2F1(1/2, 1/2; 1; 16 W) = (2/pi) K(k), k = 4v(1 -
  u)/(1 + u)^2 (Onsager's modulus), pulled back and conjugated by s(u): `elliptic_factor_check`
  proves M(s F) = 0 and M(R) = P exactly, and `theta_g_identity` checks theta g = R + s F on the
  exact coefficients (u^40 in the run, u^59 in the tests), hence L g = P (given the hypergeometric
  equation). Finality: the operator values are `not-computed` (a proof is not a computed
  stability). Its
  singular points: u = 0 (exponents 0, 0, 0), 1 (0, 0, 0), -1, 3 + 2 sqrt 2 and u_c = 3 - 2 sqrt 2
  (0, 2, 2), none apparent. At u_c the logarithm multiplies the exponent-2 solution:
  (u - u_c)^2 ln|u - u_c| in g, 2 - alpha = 2, alpha = 0; a 10% change of the theta^2 coefficient
  moves the exponents off the integers. The ratio method on the exact coefficients gives 3.00 for
  n (1 - u_c b_n / b_(n-1)) (b_n ~ u_c^-n n^-3) and 2.00 for theta g. Local analysis: exact in
  Q(sqrt d) for factors of degree <= 2; logarithms are detected among integer exponents only
  (non-integer ones are reported numerically, not analysed). Stretch: the susceptibility (OEIS
  A002906, 400 terms) has no operator of theta-order <= 6 and degree <= 40 (360 equations, 287
  unknowns; slow tier, 6 s). Orrick, Nickel, Guttmann & Perk (arXiv:cond-mat/0103074, Secs. 1,
  3.2, 6.3) give strong evidence, not a proof, that the isotropic chi has the natural boundary |s| =
  1, s = sinh 2K, i.e. the circles |v +- 1| = sqrt 2 in v (|v| = 1 is the boundary of the anisotropic
  chi(v, v') at fixed v'); the check is a bound only. Partitions p(n) (natural boundary,
  Flajolet-Gerhold-Salvy 2005, Theorem 1 and the list after it) are refused at (order 3, degree 9)
  on 70 terms, and the Onsager series with one corrupted coefficient on its 62-term fit window.
- Banked values often come from secondary open sources where the original is paywalled; each bank
  entry names the source its digits were read from (`reference/*.json`, field `cite`), and where the
  cite says the original was not fetched the digits come from the secondary source named there. Check
  the originals by hand before quoting a digit.

RELATED: `tools/annihilator` (used by `series/` through its documented surface: `pf_from_series`,
`nullspace_mod_fast`, `rec_to_theta`; python-flint is not needed). Arbitrary precision is plain
mpmath (no ball enclosures; `baller` is not used).

CREDIT: the replica method, the Parisi ansatz and the cavity method are due to others; the package
reproduces their published calculations and credits them by key (bracketed keys resolve in
REFERENCES.md at the repository root). Models and methods: the SK model [SK75], its stability analysis
[AT78], the replica-symmetry-breaking scheme [Par79] [Par80] and its proof [Tal06] [Pan13]; the random
energy model [Der80] [Der81]; the spherical p-spin [CS92] with [CC05] [CLN17]; the Ising p-spin and its Gardner transition [Gar85]
[MRT03] [Riz13] [YM20] [Nis26], the spherical 1RSB replicon [CL06]; the Gardner capacity [Gar88]
[GD88], the Ising perceptron thresholds [KM89] [DS18] and the cavity treatment [Mez89]; teacher-student
learning [SST92] [OH91] [Gyo90] [BTG97] and its proof [BKMMZ19]; Hopfield model [AGS85] [AGS87] with the
RSB analysis of [SK94]; the Viana-Bray model [VB85], the sparse cavity and population-dynamics method
[MP01] [MP03] and the transition conditions [MK04] [GT03]; the dense cavity method [MPV87] with
[TAP77]; AMP, state evolution and the TAP free energy (`amp/`) follow [Bol14] [BM11] [FMM18] with
[Ple82], universality [BLM15] [CL20] and TAP complexity [BM80]; GAMP and its state evolution follow
[Ran11] [BKMMZ19] with [JM13], the binary teacher-student thresholds [Gyo90] [BKMMZ19] [LHZH22]. Random K-XORSAT (`csp/`): the cavity / leaf-removal solution [MRZ03] [CDMM03], the proofs of the threshold [DM02] [PS16] and of the 2-core [Mol05], the textbook treatment [MM09] and the solution geometry [IKKM15] [AM15]. Random K-SAT: BP and survey propagation [BMZ05] [MPZ02] [MZ02],
the SP population dynamics and thresholds [MMZ06] [LM19], the clustering / condensation thresholds
[KMRSZ07], the first-moment bound [FP83] and the proof for large K [DSS22]. LDPC codes on the BEC (`inference/`): density evolution and the peeling decoder [LMSS01] [MM09], the Maxwell construction and the MAP threshold [MMU08] [KRU10] [RU08], tabulated thresholds [KRU12] [Pfi15]. The universality of the SK free energy used in the held-out test is [CH04]. Reference digits are read
from [Sto25] [OSS07] [CRT03] [SY17] [DM15] [FP16] [Gyo01] [PRGE20] [Cov65] [Bar97] as stated in the bank
cites. The typed-step architecture, the symbolic level (a), the composer, the refusal and
finality discipline and the LaTeX writer are ours. Each bank entry names the source its digits were
read from; where a cite says the original was not fetched, the digits come from the secondary source
named there, and the REFERENCES.md note preserves that.
