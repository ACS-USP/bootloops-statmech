# replicant — guide

Tool page: none yet (this GUIDE and `README.md` are the documentation).

KIND: package (`tools/replicant/`; run with `tools/` on the path: `cd tools && python3 -m replicant ...`).
Requirements: Python 3 with `sympy`, `mpmath`, `numpy`, `scipy` and `python-flint` (Arb: `lattice/`
through `tools/baller`, and `tools/annihilator`); `tectonic` or `pdflatex` only for `--pdf`. Run the
battery in the project venv (INSTALL.md: `.venv`, which has python-flint); without python-flint the
lattice tests fail loudly (`unsupported:python-flint`), by design. Battery: `python3 tools/replicant/selftest.py [--full]`.

PURPOSE: published replica-method and cavity-method calculations of disordered systems (spin glasses,
neural-network capacity and learning) as TYPED, RECORDED STEPS. Each step names its preconditions and its
citation; a missing precondition is a named refusal, never a silent fallback; every run returns a trace a
reader can follow to the paper, a per-value finality label, and a comparison with banked, cited reference
values. Two levels: (a) a symbolic derivation (replicate, average the disorder, introduce order
parameters, RS ansatz, n -> 0, stationarity, solve) that works for models you describe, and (b) the
published pipelines of a catalog of 19 entries, each with a replica route and/or a cavity route (and an
`amp` route for `sk` and `teacher_student`; the constraint-satisfaction entries `xorsat` and `ksat`, the code entry `ldpc_bec` and the block model `sbm` are cavity-only; the spiked Wigner model `spiked_wigner` is amp-only; the ridge-regression entry `ridge` is replica-only, its RS saddle point a cited closed form, not a level-(a) derivation; the 2D Ising entry `series_ising2d` has the route `series` only: an exact series and its holonomic operator, no replica or cavity step; the solvable lattice entries `ising2d` and `ising1d` have the route `exact` and `bethe_ising` the route `cavity`, all three certified in ball arithmetic, `rigor` "certified"; the bound entry `sk_bounds` has the route `bound` only: Guerra's rigorous upper bounds on the SK pressure, certified, and a consistency check of the package's own RS / 1RSB / K-RSB optimisers against them; the online-learning entries `online_perceptron`, `online_linear` and `online_drift` have the route `online` only: the N -> oo order-parameter ODEs of online learning in alpha = examples / N, physics-level, checked against finite-N simulation). A composer
matches a model description against the catalog, reuses what applies, adapts what it can through
registered hooks, and says what it reused, adapted, or could not do. A LaTeX writer turns any run into a
derivation document. Monte Carlo (parallel tempering) and exact enumeration (`numerics/`) check the
held-out composed models by routes that share no code with them. For theoretical physicists and agents who need a published statistical-mechanics
result reproduced with its derivation, or the same calculation for a nearby model.

USE-WHEN:
- You need a published replica/cavity number (a critical temperature, a ground-state energy, a Gardner
  capacity, a learning-curve constant, a Hopfield capacity, a sparse-graph transition) WITH the recorded
  derivation and a check against a banked reference value from the paper.
- You want the calculation for a model near a catalogued one (a different coupling law, a different
  variable type) and want the composer to say what it reused, what it adapted, and where it refuses.
- You need a LaTeX appendix with the steps, equations, citations, check table and honest labels.

NOT-FOR:
- Rigorous proofs. Every result of the replica, cavity, amp and series routes is physics-level: the
  n -> 0 continuation and the RS/RSB ansaetze (or the cavity clustering assumptions) are not proved; the
  results say so (`rigor: "physics-level"`). The four entries `ising2d`, `ising1d`, `bethe_ising` and
  `sk_bounds` are `rigor: "certified"`: ball-arithmetic enclosures (python-flint) of a cited closed
  form, a cited theorem's formula or a rigorous bound, with the radius in `errors`; that certifies
  the evaluation, not the cited theorem. A value flagged `exact-by-literature` rests on the cited
  paper, not on this package.
- Population-dynamics results are statistical (value +- stderr over independent runs), never certified.
- Dynamics beyond AMP, DMFT, or quantum models. AMP, state evolution and the TAP free energy exist for
  SK, and GAMP with its state evolution for the Bayes-optimal teacher-student perceptron (gaussian or
  +-1 prior), in the module `amp/`, and Bayes AMP for the spiked Wigner model in `inference/`; the
  catalog route `amp` exists for `sk`, `teacher_student` and `spiked_wigner` only. Parallel tempering
  (`numerics/`) is an equilibrium sampler used as an independent check, not a dynamics.
- Automatic discovery of the RSB structure: the ansatz is an input; RS stability is computed or reported
  as unknown, and there is no search for the right ansatz.
- A model with no level-(a) hook (families: `sk`, `perceptron`, `hopfield`; variables `ising`,
  `spherical`; disorder `gaussian`, `pm_one`): the composer refuses (`missing-step:*`).

INVOKE:
CLI (from `tools/`):
`python3 -m replicant list` prints the catalog (entry, routes, citation).
`python3 -m replicant run <entry|spec.json> [--route replica|cavity|amp|exact|series|bound] [--param k=v ...] [--out result.json]
[--latex out.tex [--pdf]]`
- JSON result to `--out` (default stdout); the rendered step trace to stderr. Exit 0 ok, 2 refusal
  (`REFUSED: <reason>: <detail>` on stderr; with `--latex` the partial derivation and the reason are still
  written), 1 any other error (`ERROR: <type>: <message>`). `--pdf` without `--latex` is refused
  (`precondition:latex-path`).
- `--param k=v`: integers parse as int; decimals stay DECIMAL STRINGS (mpmath converts them, so `0.1` is
  not rounded through a binary float); `true`/`false` booleans (case-insensitive on the CLI: `TRUE`
  parses as true; at the Python API `ksat`'s `crossing` takes a bool or the exact strings "true" /
  "false", so `run(crossing="TRUE")` is refused `precondition:known-parameter`); a comma
  list is a tuple (a trailing comma
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
  --param ansatz=1rsb --param m=0.24` (1RSB population of populations; about 15 s measured), are
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
- Worked example, solvable lattice models, certified (`ising2d`, `ising1d` route `exact`; `bethe_ising`
  route `cavity`; python-flint balls through `tools/baller`; `rigor` "certified"; each value is a
  40-digit decimal string and `errors` an upper bound of |value - x| over the ball, so the printed
  value +- error (JSON, LaTeX "(certified ball)") is an enclosure): `python3 -m replicant run ising2d --route exact` (0.8 s) -> `beta_c`
  0.440687, `minus_beta_f` 0.929695 (the closed form (1/2) ln 2 + 2G/pi at beta_c, printed
  to 40 digits with the error bound 1.77e-41), `yang_exponent_loglog` 0.1249998 (the log-log slope of Yang's formula at beta - beta_c =
  1e-6, 1e-8; `not-computed`, a consistency check, so `final` is false), the 4 x 4 torus
  `torus_minus_beta_f` 0.970120 from Kaufman's formula and `torus_minus_beta_f_tm` from the transfer
  matrix, with `torus_coefficients` (2, 0, 0, 0, 32, ..., 20524 at x^16, ...) the integer polynomial
  both give (Kaufman's recovered from balls, `kaufman_max_radius` 9e-28 < 1/4). `--param beta=0.6`
  adds `magnetization` 0.973609 (Yang) and `minus_beta_f` 1.210132 (the certified single integral);
  at or below beta_c the Yang step is skipped. `python3 -m replicant run ising1d --route exact`
  -> `minus_beta_f` 1.126928 = ln(2 cosh 1) and `ring_Z` (N = 8). `python3 -m replicant run
  bethe_ising --route cavity` (k = 3, beta = 1, h = 0) -> three fixed points `lower`, `middle`
  (cavity field 0, `jacobian_middle` 1.523: unstable), `upper` (`magnetization_upper` 0.991757,
  `jacobian_upper` 0.313: stable), `phi` 1.503159 (Dembo-Montanari's phi(beta, 0), the common value
  of lower and upper); no `magnetization` at h = 0 (pass `--param branch=upper`); `--param h=0.02`
  selects `upper`.
- Worked example, Guerra's bounds on the SK pressure, certified (`sk_bounds`, route `bound`; python-flint
  balls through `tools/baller`; `rigor` "certified"): `python3 -m replicant run sk_bounds --route bound`
  (beta = 2, h = 0; 1.6-2.1 s measured) -> `rs_bound` "1.584785890269395881736457469249912060700"
  (`errors` 1.20e-40: the certified minimum over q of Guerra's RS bound, equal to the package's RS
  pressure -beta f_RS), `rs_q` "0.5303683920507946325937163727372733258213" (`errors` 2.97e-41: the
  certified minimiser, `stable`: `rs_second_variation` "1.3380657020581225064 +- 6.99e-21" > 0),
  `rs_stationary` ["max at q = 0", "min at q = 0.530368392051"] (every stationary point), `rs_certificate`
  "G' < 0 certified on 4 cell(s) covering [0, 1/8]; [1/8, 1]: Lipschitz exclusion, 1 root(s) on monotone
  runs", `rsb1_bound` "1.582417684758431402661933566110169094513" (`errors` 1.45e-26: Guerra's bound at
  the package's own 1RSB optimiser output `sk_1rsb(2)`), `chain_verdict` "consistent" (`chain_links`
  ["1rsb <= rs: ok"]), `rs_reported` "1.5847858902693959 +- 1.0e-12", `1rsb_reported` "1.5824176847584295
  +- 2.0e-11", `rs_functional` and `1rsb_functional` "ok" (the package's numbers equal the certified
  functionals within radius + the solver's stated tolerance). Both bounds are `exact-by-literature`
  (Guerra 2003, Theorem 3: an upper bound on lim (1/N) E ln Z_N for any admissible parameters); the chain
  verdict is a consistency check of the optimisers (the theorem orders the infima, not the package's
  outputs). `--param beta=1/2` -> `rs_bound` "0.7556471805599453094172321214581765680755" (= ln 2 + 1/16,
  `errors` 1.35e-43) at `rs_q` "0" exactly, `rs_stationary` ["min at q = 0"], `rs_certificate` "G' > 0
  certified on 1 cell(s) covering [0, 1]" (the high-temperature identity, Guerra 1992 Theorem 3),
  `chain_verdict` "rs-only" (sk_1rsb collapses to RS there). `--param digits=0` (or -5, x) is refused
  `precondition:digits`.
  `--param K=2` (h = 0, 1 < beta <= 3) adds `rsb2_bound` at the package's 2-step staircase: the package's
  own K = 1, 2 staircase solves (~5 s) and the triple nested integral (measured 37 s at `--param
  digits=12`, 116 s at the default 20 digits);
  `--param K=3` is refused `precondition:K` (bounds.check_bound_chain takes K <= 5, levels above 2
  numerical).
- Worked example, online learning (`online_perceptron`, route `online`; Biehl-Riegler convention: |B| =
  1, inputs i.i.d. N(0, 1), J <- J + (eta / N) F b xi, R = J . B, Q = J . J, eps = arccos(R / sqrt Q) / pi
  against the true teacher; 0.7-1.1 s each, measured): `python3 -m replicant run online_perceptron --route
  online --param rule=perceptron` -> `eps_exponent` 0.33349 (the log-slope of the integrated ODE between
  alpha = 5000 and 10000; derived 1/3, `eps_exponent_derived`), `eps_constant` 0.278147 (alpha^(1/3) eps at
  alpha = 1e4; Biehl-Riegler Eq. (8): (1/pi)(2/3)^(1/3) = 0.278069, the gap is the slow correction).
  `--param rule=optimal --param R0=0.01` -> `eps_exponent` 0.99965, `eps_constant` 0.883184 and
  `eps_constant_formula` 0.883400 (2 / I(0), the rho -> 1 limit; VKC 1998 print 0.88); without `R0` it is
  refused `precondition:initial-overlap` (W* carries 1/rho). `--param rule=hebb --param alpha=10 --param
  N=500` -> `eps` 0.125397 (ODE), `eps_hebb_closed_form` 0.125397 (`hebb_closed_form_gap` 2.3e-11),
  `eps_sim` 0.124991 (`errors` 9.0e-4, `eps_sim_halfwidth99` 0.00259: 20 named seeds at N = 500).
  `--param rule=optimal --param R0=0.01 --param noise=0.3` -> `no_learning_boundary` 0.100000 (Copelli et
  al.'s 2 lam - 1/2) and `robust_boundary` 0.149730 (their Fig. 1a solid line, read 0.150), both located
  on the ODE; the asymptote step is skipped with noise for rules other than Hebb. No value is final
  (`not-computed`: the ODE limit is physics-level); `run online_perceptron` without `--route online` is
  refused `missing-step:route:replica`, without `rule` `precondition:state-point`.
- Worked example, the linear perceptron online versus offline (`online_linear`, route `online`; Kinouchi &
  Caticha 1995 mapped into the same convention: |B|^2 = M, linear outputs b = B . xi, h = J . xi, J <- (1 -
  Omega/N) J + (eta/N) F xi, rho = R / sqrt(Q M), e_g = (M + Q - 2R)/2; 0.6-1.0 s each, measured):
  `python3 -m replicant run online_linear --route online --param alpha=0.5` (the defaults: the optimal rule
  F = (Q/R) b - h on the sphere Q = M = 1 from R0 = 1e-3) -> `rho` 0.627272 (KC Eq. (19): sqrt(1 - e^-0.5)
  = 0.627271; `closed_form_gap` 5.8e-11), `gamma_temperature` 1.541490 (KC's Gamma = (1 - rho^2)/rho^2),
  `rho_offline_standard` 0.5 and `rho_offline_optimal` 0.707107 (KC Eqs. (27), (49)),
  `rho_offline_optimal_ridge` 0.707107 (ridge's ridgeless risk at gamma = 1/alpha = 2, sigma2 = 0, mapped to the pseudoinverse
  direction).  Without `alpha` the entry reports the asymptotes and the crossing instead: `python3 -m
  replicant run online_linear --route online` -> `alpha_cross` 0.714556 (online optimal beats offline
  standard below it; KC print 0.71), `tail_constant` 0.4999995 (e^alpha (1 - rho) at alpha = 20; KC Eq.
  (31): 1/2), `rho_inf` 1.  `--param rule=standard --param Q0=0.25` -> `rho_inf` 0.800000 (KC Eq. (24) at M/Q = 4), `eg_inf` 0.225000.  `--param
  rule=standard --param constraint=none --param eta=0.5 --param alpha=2 --param N=200` -> `eg` 0.223130
  (ODE; `closed_form_gap` 1.0e-11), `eg_sim` 0.223871 (`eg_sim_halfwidth99` 0.0158: 20 named seeds at N =
  200).  `--param rule=standard --param eta=greedy --param alpha=1` -> `rho` 0.726230 (between the standard
  1 - e^-1 = 0.632 and the optimal 0.795), `eta_greedy` 1.188487; the ridge route is skipped at gamma = 1
  with a note (HMRT Eq. (8) is stated for gamma != 1 only; at sigma2 = 0 its bias tends to 0 there, but no
  value is claimed).  No value is final.
- Worked example, learning a drifting rule (`online_drift`, route `online`; the O1 engine with a teacher
  that drifts after each example, randomly (autocorrelation e^{-D alpha}, VKC 1998 Eqs. (4), (26); KC93
  Eqs. (3)-(6) with T = D) or deterministically away from the student (VKC Eqs. (28)-(31)); about 1 s each,
  measured): `python3 -m replicant run online_drift --route online` (the defaults: the optimal rule, random
  drift, no D) -> the small-drift law `eps_exponent` 0.333333, `eps_coefficient` 0.447321 (KC93 Eq. (25):
  0.447; VKC Table I: 0.45).  `--param rule=annealed_hebb --param D=0.1 --param alpha=5` ->
  `eps_inf` 0.226041 (`eps_inf_closed` 0.226041, the closed form), the KC93 bound `eps_inf_bound` 0.195856
  (the optimal rule), `eps` 0.229107 and `eps_bound` 0.199379 at alpha = 5.  `--param D=0.01 --param
  drift=deterministic` -> `eps_inf` 0.195820 (`eps_inf_kc93` 0.195820 from KC93's I(lambda)).  `--param
  rule=perceptron --param D=0.1 --param alpha=4 --param N=200` -> `eps_inf` 0.220719 at `Q_inf` 1.434972
  (|J| settles under drift), `eps` 0.247551, `eps_sim` 0.243858 (`eps_sim_halfwidth99` 0.0116: 20 named
  seeds at N = 200).  `--param rule=hebb --param D=0.1` is refused `no-solution:no-tracking` (plain Hebb:
  R -> eta a / D while Q grows, eps -> 1/2).  No value is final.
- Worked example, the tree committee machine (`online_tcm`, route `online`; K odd non-overlapping branches,
  majority vote, Copelli & Caticha 1995 (CC95) and Copelli, Kinouchi & Caticha 1996 (CKC96, abstract
  only); about 1 s each, measured): `python3 -m replicant run online_tcm --route online` (the defaults:
  CC95's nonlocal optimal rule at K = 3) -> `eps_constant` 0.883400 (CC95 Eq. (32): 0.88, the perceptron's
  constant), from the rate at theta -> 0.  `--param rule=local_optimal --param alpha=5` (the branch sees
  only its own field) -> `noise_equivalent` 0.250000 (the mapped output noise (1 - P0)/2, P0 = 1/2 the
  tie probability of the other two hidden units), `eps` 0.211221 (the TCM ODE),
  `rho_perceptron_mapped` 0.868537 (O1's noisy-perceptron ODE at alpha_p = 3 alpha; `equivalence_gap`
  <~ 1e-12), `eps_constant` 2.665178 against `eps_constant_perceptron` 2.665178 (P0 C(1/4)).  `--param
  rule=hebb --param alpha=3
  --param N=300` -> `eps_exponent` 0.497367 (alpha = 1e4), `eps_constant` 0.690985 (CC95: sqrt(3/2pi)),
  `eps` 0.321157, `eps_sim` 0.321281 (`eps_sim_halfwidth99` 0.0036, 20 named seeds at N = 3 x 100).
  `--param K=5` with the default rule is refused `unsupported:K` (the nonlocal rule is written for K = 3;
  `local_optimal` and `hebb` take any odd K).  No value is final.
- Worked example, the soft committee machine (`online_scm`, route `online`; online gradient descent, K
  student and M teacher erf units, Saad & Solla 1995 (SS; the PRL and PRE were NOT fetched, their NIPS 1996
  and MANNA 1997 restatements were read), Biehl & Schwarze 1995 (BS), Biehl, Riegler & Woehler 1996 (BRW);
  about 1.1-1.7 s each, measured): `python3 -m replicant run online_scm --route online` (K = M = 2, eta = 1.5)
  -> `eps_sym_identical` 0.041106 (the identical-student fixed point, BRW Fig. 2's plateau),
  `tau_esc_identical` 7.164603 (so BRW Eq. (17)'s shift per factor 100 in X is `plateau_shift_per_100x`
  16.497; Fig. 2: 16.4),
  `eps_sym_permutation` 0.034628 (Q > C: BRW Fig. 1's plateau), `eps_small_eta` 0.011611 (SS's eta^2-truncated
  point), `lambda1` -0.128365, `lambda2` -0.066277.  `--param alpha=120` (BRW Eq. (16) initial conditions,
  X = 1e-6, N = 1000) -> `plateau_status` escaped, `plateau_branch` identical, `plateau_height` 0.041106,
  `plateau_entry` 5.840280, `plateau_exit` 56.829106 (BRW's definition, B = 0.05; the escape moves by 6e-8
  at rtol x 1e-2), `eps` 5.1e-5; `--param alpha=40` -> `plateau_status` undetermined (no exit, no length:
  nothing is extrapolated).  `--param K=1 --param critical=1` (BS's single unit) -> `eta_max` 4.055779 (BS:
  pi sqrt(5/3)), `eta_crossing` 2.703852 = `eta_fastest` (BS's eta_opt = 2 eta_c/3); `--param K=3 --param
  critical=1` -> `eta_max` 1.620675 (SS Fig. 2: 1.62), `eta_crossing` 0.916824 (SS's condition lambda1 =
  2 lambda2), `eta_fastest` 1.280375 (the fastest asymptotic decay is NOT at SS's crossing at K = 3);
  `--param K=2 --param M=1 --param critical=1` -> `eta_c_specialised` 1.892595, `eta_s_symmetric` 2.169600
  (BS: 1.89, 2.17).  `--param alpha=16 --param N=200` -> `eps_sim` 0.041232 (`eps_sim_halfwidth99` 0.0026,
  12 named seeds) against the ODE's `eps` 0.041104.  No value is final.
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
- Level-(a) 1RSB: `--param ansatz=1rsb` for a spec that is NOT a catalog entry (composed / held-out
  models), or `--param level_a_ansatz=1rsb` for any spec with level-(a) hooks, catalog specs included
  (their own pipelines are then not used; `level_a_ansatz` takes only `1rsb`, else
  `missing-step:ansatz:<a>`). For a catalog entry `ansatz` keeps its v1 meaning, an entry parameter:
  `run sk --param ansatz=1rsb --param beta=2` runs the family 1RSB pipeline (`q0` 0.214631, `m`
  0.281843, `f` -0.791209, with its banked RefChecks), `viana_bray --route cavity --param ansatz=1rsb`
  the 1RSB population, and an entry that does not read it refuses `precondition:known-parameter`.
  `run sk --param level_a_ansatz=1rsb --param beta=2 --param h=0 --param branch=max-f` is the derived
  1RSB solve (`phi` 1.582418, about 5.5 s). The 1RSB functional derived by rules (`derive_rsb.py`) and solved
  (`solve_derived_rsb.py`); the report lists replicate / average-disorder / introduce-order-parameters
  as REUSED and the RS ansatz, n -> 0, stationarity and solve steps as ADAPTED, relative to the RS
  level-(a) derivation of the same spec. Knobs (also `--param`s): `frozen=true` (q1 = 1; discrete
  variables with a counting measure), `solve_for=<parameter>` (frozen threshold: x fixed to 1, the
  parameter solved for), `zero_temperature=true` (SK-type energies only), `branch`
  (`nontrivial-k` or `max-f`), `audit` (`independent` default, or `quadrature`). An unread parameter
  name is refused `precondition:known-parameter` on this route (reached through `ansatz` or
  `level_a_ansatz`). The frozen Ising-perceptron capacity (a composed spec, so `ansatz`):
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
`from replicant.online import fields, odes, optimal, simulate` (O1; floats; theta = arccos rho):
`odes.perceptron_flow(rule, eta=1.0, noise=0.0, noise_estimate=None, averages="closed")` -> `Flow`
(`rule` hebb | perceptron | adatron | optimal or a callable F(h, b, Q), numpy-vectorised; `eta` a number
> 0 or a schedule eta(alpha); `noise` the label flip probability in [0, 1/2); `noise_estimate` read by
the optimal rule only); `Flow.rates(alpha, R, Q)` -> (dR/dalpha, dQ/dalpha), `Flow.theta_rate(alpha,
theta, Q)`; `odes.integrate(flow, alpha_max, init=(R0, Q0), alphas=None, rtol=1e-9)` -> `Trajectory(alpha,
R, Q, eps)` (no init: `precondition:state-point`); `odes.hebb_closed_form(alpha, eta, noise, init)` -> (R,
Q, eps); `odes.asymptote(flow, init, alpha_big=1e4)` -> (log-slope, alpha_big^p eps); `odes.EXPONENT`;
`odes._PLANT` (private planted-error hook: the eta^2 term). `fields.closed(rule, theta, noise,
noise_est)` and `fields.quadrature(f, theta, noise, n_r=48, n_phi=32)` -> `Averages(ay, ax, a2, drive)`
(<f b y>, <f b x>, <f^2>, <f b (y - rho x)>; `fields._PLANT` hooks the sqrt(2 pi)). `optimal.w_star(delta,
theta, noise)`, `optimal.optimal_modulation(rho, noise)` -> a callable F(h, b, Q) with `.optimality_class`
(= `optimal.OPTIMALITY_CLASS`), `optimal.asymptotic_constant(noise, printed=False)`,
`optimal.noise_boundaries(noise)` -> `NoiseBoundaries(no_learning, robust, robust_error)`.
`simulate.simulate_perceptron(rule, N, alpha_max, seeds, eta, noise, noise_estimate, init, alphas)` ->
`SimResult(alpha, eps_mean, band, eps_seeds, N, seeds, alphas, curve)` (99% Student-t `band`). Example:
`odes.integrate(odes.perceptron_flow("adatron"), 1e4, (0, 1), alphas=(1e4,)).eps[-1] * 1e4` -> 1.49982.
`from replicant.online import linear, linear_offline, simulate_linear` (O2; floats):
`linear.linear_flow(rule="standard", eta=1.0, constraint="none", M=1.0, averages="closed")` -> `LinearFlow`
(`rule` standard | optimal or a callable F(b, h, Q, R), or F(b, h, Q, R, alpha) for a time-dependent rule
such as KC Eq. (20); `eta` > 0, a schedule eta(alpha) or "greedy";
`constraint` none | sphere); `LinearFlow.moments(R, Q, alpha)` -> (<b F>, <h F>, <F^2>),
`.rates(alpha, R, Q)`, `.rho_rate(alpha, R, Q)`; `linear.integrate_linear(flow, alpha_max, init=(R0, Q0),
alphas=None, rtol=1e-10, alpha0=0)` -> `LinTrajectory(alpha, R, Q, rho, eg)`; `linear.asymptote(flow, init,
alpha_big=60, alpha_tail=20)` -> `Asymptote(alpha, rho, eg, tail)`; the closed forms
`standard_closed_form(alpha, eta, init, M)`, `sphere_standard_closed_form(alpha, init, M)`,
`optimal_closed_form(alpha, init, M, constraint)`; `greedy_eta(R, Q, M)` (derived here, not in KC95), `hypothesis_temperature(rho)`,
`OPTIMALITY_CLASS`, `_PLANT` (private: the eta^2 term and the closed <F^2> cross term).
`linear_offline.offline(alpha, algorithm, constraint="sphere", M=1, Q=None)` -> `Offline(alpha, rho,
q_tilde, R, Q, eg)`; `linear_offline.ridge_route(alpha, M=1)` -> `RidgeRoute(gamma, risk, eg, rho)`;
`linear_offline.crossing_alpha(M=1)`. `simulate_linear.simulate_linear(rule, N, alpha_max, seeds, eta,
constraint, M, init)` and `simulate_linear.simulate_offline(algorithm, N, alpha, seeds, constraint, M, Q,
method="construct" | "descent")`
-> `LinSim(alpha, band, rho_band, eg_seeds, N, seeds)` (99% Student-t bands of e_g and rho). Example:
`linear.integrate_linear(linear.linear_flow("optimal", constraint="sphere"), 1, (1e-3, 1), alphas=(1,)).rho[0]`
-> 0.795060.
`from replicant.online import drift, drift_rules, simulate_drift` (O3; floats):
`drift.drift_flow(rule, D=0.0, kind="random", eta=1.0, noise=0.0, noise_estimate=None, averages="closed")`
-> `DriftFlow` (`rule` an O1 rule (hebb | perceptron | adatron | optimal or a callable F(h, b, Q)), a VKC rule
(annealed_hebb | step | symmetric, unit rate) or a `drift_rules.Modulation(W(Delta, theta))`; `D` >= 0 or a
schedule D(alpha); `kind` random | deterministic); `.theta_rate(alpha, theta, Q)`, `.rho_rate(alpha, theta,
Q)`; `drift.integrate_drift(flow, alpha_max, init=(R0, Q0), alphas=None, jumps=(), jump_overlap=0.0)` ->
`odes.Trajectory` (a jump replaces the teacher: R <- jump_overlap sqrt(Q)); `drift.stationary(flow)` ->
`Stationary(eps, theta, Q, unstable)`; `drift.small_drift(rule, kind, noise=0, eta=1, Ds=(1e-12, 1e-11))`
-> `SmallDrift(exponent, exponent_derived, coefficient, Ds, eps)`; `drift.drift_exponent(rule, kind)`;
`drift.hebb_drift_closed_form(alpha, D, eta, noise, init)`; `_PLANT` (private: "drift" scales the drift
term, "coupling" adds a W-dependent one). `drift_rules.vkc_modulation(rule, delta, theta, noise)`,
`vkc_averages(rule, theta, noise, route="closed" | "quadrature")`, `conditional_averages(g, theta, noise)`,
`VKC_CLASS`, `annealed_hebb_stationary(D, kind, noise)`, `kc93_I(lam, printed=False)`,
`kc93_stationary(D, kind)`. `simulate_drift.simulate_drift(rule, N, alpha_max, seeds, D, drift, eta, noise,
noise_estimate, init, alphas, jumps, jump_overlap)` -> `SimResult`. Example:
`drift.stationary(drift.drift_flow("optimal", D=0.1)).eps` -> 0.195856.
`from replicant.online import tcm, tcm_flow, simulate_tcm` (O4; floats): `tcm.tie_distribution(K)`,
`tcm.tie_probability(K)` (P0), `tcm.eps_committee(e, K)` (e a number or K branch errors; CC95 Eq. (5) at
K = 3), `tcm.averages(rule, theta, K)` -> `Averages(ay, ax, a2, drive)` (rule hebb | local_optimal |
optimal), `tcm.OPTIMALITY_CLASS`; the mapping, which the TCM route never calls: `tcm.effective_noise(K)`
(lambda_K = (1 - P0)/2), `tcm.perceptron_alpha(alpha, K)` (K alpha); `_PLANT` (private: "mapping",
"time"). `tcm_flow.tcm_flow(rule, K=3, eta=1.0)` -> `TcmFlow` (`.rates(alpha, R, Q)`,
`.theta_rate(theta, Q)`); `tcm_flow.integrate_tcm(flow, alpha_max, init=(R0, Q0), alphas=None)` ->
`TcmTrajectory(alpha, R, Q, rho, eps_branch, eps)`; `tcm_flow.hebb_closed_form(alpha, K, eta, init)`;
`tcm_flow.asymptote_tcm(flow, init, alpha_big=1e4)`; `tcm_flow.rate_limit_constant(flow)`; `_PLANT`
(private: "inv_k" multiplies the 1/K of the eta^2 term). `simulate_tcm.simulate_tcm(rule, K, N, alpha_max,
seeds, eta, init)` -> `TcmSimResult(alpha, eps_mean, band, eps_seeds, K, N, seeds)`. Example:
`tcm_flow.integrate_tcm(tcm_flow.tcm_flow("local_optimal", 3), 5, (0.01, 1), alphas=(5,)).eps[0]` -> 0.211221.
`from replicant.online import scm, scm_flow, plateau, simulate_scm` (O5; floats, numpy): `scm.i2(C)`,
`scm.i3(C)`, `scm.i4(C)` (SS's closed forms, C the 2x2 / 3x3 / 4x4 covariance), `scm.quadrature(C, n=24)`
(tensor Gauss-Hermite, independent), `scm.scm_flow(K, M, eta, teacher_kind="isotropic", truncated=False)` ->
`ScmFlow` (`.rates(Q, R)` -> (dQ, dR), `.vector_rate(y)`, `.pack`, `.unpack`, `.eps(Q, R)`), `scm.teacher(kind,
M)`; `_PLANT` (private: "eta2" scales the eta^2 term, "unit" the 1 of 1 + C_aa, "pref" the 2/pi and 4/pi^2).
`scm_flow.integrate_scm(flow, alpha_max, init=(Q, R), alphas=None, rtol=1e-9)` -> `ScmTrajectory(alpha, Q, R,
eps)`; `scm_flow.brw_initial(X, N=1000, Qt=0.5, u=(0.2, 1.0))`, `scm_flow.ss_initial(K, M, seed)`;
`scm_flow.symmetric_fixed_point(flow, branch="permutation")` -> `SymFP(branch, R, Q, C, eps, lam_esc, eigs)`
(branch identical | permutation); `scm_flow.optimum_modes(flow)` -> `Modes(lam1, lam2, rate)`;
`scm_flow.critical_rates(K)` (eta_max, eta_crossing, eta_fastest and their rates); `scm_flow.bs95_k2_rates()`;
`scm_flow.OPTIMALITY_CLASS`. `plateau.analyse(flow, init, alpha_max, B=0.05, rtol=1e-8, points=200)` ->
`PlateauReport(status, reason, branch, eps_fp, eps_mid, alpha_entry, alpha_exit, length, lam_esc, tau_esc, B,
rtol, exit_rtol_spread, Q_fp, R_fp, eps_end)`; `plateau.SPREAD_TOL` (0.01 tau_esc). `simulate_scm.simulate_scm(K,
M, N, eta, alphas, seeds, init, teacher)` -> `ScmSimResult(alphas, eps, bands, N, seeds, initial)`. Example:
`scm_flow.symmetric_fixed_point(scm.scm_flow(2, 2, 2.0), "identical").eps` -> 0.124368 (BRW Eq. (20): 0.124).
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
`from replicant.lattice import ...` (E9; needs python-flint and `tools/baller`; J = 1 unless given, K =
beta J; beta an int / Fraction / decimal string, or `BETA_C` = "beta_c" for K_c = (1/2) ln(1 + sqrt 2);
values -beta f): `onsager_free_energy(beta, J=1)` -> `Ball` (the closed form at `BETA_C`, else the
single integral by `acb.integral`); `critical_beta(J=1)`; `yang_magnetization(beta, J=1)` (beta >
beta_c only); `onsager.onsager_free_energy_mp(beta, dps=50, method="single" | "trapezoid")` (mpmath, the
independent route); `onsager.yang_loglog_slope(d1, d2)`; `kaufman_partition_function(Lx, Ly, beta)` (Z),
`kaufman_free_energy(Lx, Ly, beta)`, `kaufman_polynomial(Lx, Ly)` -> `Recovery(coefficients, max_radius,
prec)` (sides <= 12), `beta_of_x(x)` (beta through x = e^{-2K}); `torus_partition_polynomial(Lx, Ly)` ->
the exact integers c_0 .. c_2N (Ly <= 7, Lx <= 16), `ising_transfer_matrix_free_energy(L, beta, Lx=None)`;
`ising1d_free_energy(beta, h, J=1)`, `transfer.ising1d_ring_identity(N, a, b)` (exact, a = e^K, b =
e^{beta h} rational), `transfer.ising1d_ring_balls(N, beta, h)`; `bethe_critical_beta(k)`,
`bethe_ising_fixed_points(k, beta, h)` -> `BetheFixedPoint(branch, cavity_field, magnetization, phi,
phi_dm, jacobian, stable)` ("unique", or "lower" < "middle" < "upper"), `bethe_ising_free_energy(k, beta,
h, branch=None)` -> `BetheResult(..., fixed_points, selected, phi)` (`.magnetization` refuses
`ambiguous:branch` at h = 0 below T_c). `Ball`: `.mid()`, `.rad()`, `.value_and_error(dps)`,
`.contains(x)`, `.overlaps(other)`, `.render(digits)` (baller's fail-closed digits). Planted-error
hooks: `onsager._HOOKS["catalan_coefficient"]`, `kaufman._HOOKS["gamma0"]`, `transfer._HOOKS["d_sign"]`,
`bethe._HOOKS["link_weight"]`. Example: `onsager_free_energy(BETA_C).render(30)` ->
0.929695398341610214985384973666.
`from replicant.bounds import ...` (E10; needs python-flint and `tools/baller`; SK conventions of
families/sk.py, P = -beta f; parameters are taken as the exact rationals they denote, an mpf or float its
exact binary value): `guerra_rs_bound(beta, h, q, digits=40)` -> `Ball`; `guerra_rs_best_bound(beta, h=0,
digits=40)` -> `BestBound` (a `Ball`, the minimum over q in [0, 1], with `.q` the certified minimiser,
`.second_variation`, `.stable`, `.stationary` = every stationary point as `Stationary(q, value,
second_variation, kind)`, `.certificate` = how that list was shown complete; memoised); `guerra_1rsb_bound(beta, q0, q1, m, h=0, digits=20)`, `guerra_krsb_bound(beta,
qs, ms, h=0, digits=20)` (qs = q_0 .. q_K nondecreasing in [0, 1], ms = m_1 .. m_K nondecreasing in (0, 1];
else `precondition:order-parameter` / `precondition:levels`; `digits` an int 1 .. 200, else
`precondition:digits`); `rs_bound_mp`, `rs_stationary_q_mp`
(independent mpmath); `package_values(beta, h=0, K=1)` -> `PackageValue(level, qs, ms, pressure, tol,
source)` (sk_rs, sk_1rsb, families/parisi.py levels); `check_bound_chain(beta, h=0, K=1, *, digits=20,
certified_max=2, package=None)` -> `ChainReport(levels, links, violations, undecided, notes)` with `.ok`
(K = 1 .. 5, else `precondition:K`; K >= 2 with h != 0 `unsupported:field`). Planted-error hooks:
`bounds.krsb._HOOKS["inner_exponent_scale"]`, `bounds.guerra._HOOKS["dG_shift"]`. Example: `guerra_rs_best_bound("1/2", 0).render(30)` ->
0.755647180559945309417232121458.
`from replicant.numerics import ...` (E11; Monte Carlo and exact enumeration as independent checks of
the held-out models, not a catalog route; numpy, statistical results with named seeds, never certified;
SK conventions of families/sk.py, J_ij of variance 1/N, energies per spin unless stated): `random_pm_one_J(N,
seed)` (J_ij = +-1/sqrt N), `enumerate_pm_one_sk(J)` -> `PmOneSpectrum(N, M, counts)` (the exact integer
density of states by binary-reflected Gray-code blocks, N <= 26; `.log_Z(beta)`,
`.mean_energy_per_spin(beta)`, `.ground_energy_total` (TOTAL, not per spin)), `exact_ground_energy(J)`
(any real symmetric J, N <= 26), `ising_perceptron_counts(N, P, seed)` (exact solution counts after each
of P nested Gaussian patterns, meet-in-the-middle, N <= 30), `ising_perceptron_log_count(alpha, N, seed)`,
`ising_perceptron_capacity(counts)` (the largest p with a solution; `no-solution:capacity` if every
drawn pattern is stored); `pt_sk(J, betas, sweeps, seed)` (J: samples x N x N; parallel tempering,
Metropolis sweeps plus even/odd neighbour exchanges) -> `PTResult(betas, energy_per_spin_trace,
min_energy_per_spin, min_sweep, swap_acceptance, round_trips, final_energy_per_spin, ...)`,
`geometric_betas(lo, hi, R)`, `equilibration(result, beta_index, start=None)` -> `Equilibration(equilibrated,
drift, half_width, text)` (windows [T/4, T/2) vs [T/2, T), 99% Student-t over samples, >= 2 samples);
`sk_ground_energy_finite_size(Ns, samples, seed, *, sweeps=None, betas=None, conf=0.99)` -> `FSResult(e_inf,
band, stat_half_width, systematic, variants, sizes, means, stderrs, ...)` (Gaussian J; two PT runs per
sample must agree, else `undetermined:ground-states`; fit e_N = e_inf + a N^-2/3, band = 99% t + the
largest shift over the omega = 3/4 and drop-smallest-N fits); `numerics.finite_size.sk_energy_finite_size(beta,
Ns, samples, seed, *, law="pm_one", sweeps=None, nbetas=None, conf=0.99)` (thermal energy, beta > 0.5;
`undetermined:equilibration` when a size fails the diagnostic); `numerics.moments.log_moments(N, beta,
law)` (exact finite-N ln E Z^k, k <= 3, `pm_one` or `gaussian`, 2 <= N <= 200), `quenched_gap(N, beta,
order)` (the order-2 / order-3 cumulant closure of E ln Z - ln E Z; |order 2 - order 3| is the stated
truncation estimate, the closure error is not bounded), `pm_one_gap_limit(beta)` (its N -> oo limit
-1/4 [-ln(1 - beta^2) - beta^2 - beta^4/2], beta < 1). Planted-error hooks: `pt._HOOKS["acceptance_scale"]`,
`pt._HOOKS["exchange_sign"]`. Example: `J = random_pm_one_J(16, 0); enumerate_pm_one_sk(J).ground_energy_total
/ 16` -> -0.59375 (= `exact_ground_energy(J)`); `ising_perceptron_capacity(ising_perceptron_counts(16, 24,
0))` -> 12; `moments.quenched_gap(12, 0.7, 2)` -> -0.00935 (order 3: -0.00794).
`from replicant.latex import render_latex, write_latex` (`write_latex(result, path, pdf=False)`; a
`Refusal` is also accepted and writes its partial derivation).

INPUTS: a catalog entry name or a spec.json (above); run parameters by entry (exactly the names each
route reads, `CatalogEntry.parameters[route]`; any other name, e.g. a typo `kapa`, is refused
`precondition:known-parameter` with the accepted names, never ignored):
`rem`: `beta`, `N_scale` (default 1/2, Derrida's normalization); `sk` replica: `beta`, `h`, `ansatz`
(`rs`|`1rsb`, at a `beta`: the family pipelines; `level_a_ansatz=1rsb` instead sends the spec to the
level-(a) 1RSB solver), `fullrsb` (flag, slow tier); `sk` cavity: `beta`, `h`, `require_stable`;
`sk` amp: `beta`, `h`, and for finite-N AMP `N`, `seeds` (default 0-5; a non-empty list of
non-negative integers, else `precondition:named-seeds`, e.g. `--param seeds=1`), `iters` (default 100), `law`
(`gaussian`|`pm_one`; with `pm_one` every SE value is `not-computed`: SE is proved for Gaussian J only);
a finite-N parameter without `N` is refused `precondition:known-parameter` ("need N"); `pspin` and `ising_pspin`: `p` (default 3), `beta`; `perceptron` replica: `alpha`, `kappa` (default 0); `perceptron`
cavity: `alpha`, `kappa`, `variables` (`spherical`; `ising` is refused `not-implemented:cavity-ising`);
`teacher_student`: `alpha`, `learning` (`gibbs`|`bayes`); `teacher_student` amp: `alpha`, `learning`
(default `bayes`), and for finite-N GAMP `N`, `seeds` (default 0-5; malformed seeds refused
`precondition:named-seeds` as for `sk` amp), `iters` (default 30; `seeds` and
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
(cavity): `l`, `r` (integers 3 <= l < r, default 3, 6; the threshold step runs on every call and
needs l >= 3, so the catalog refuses l = 2 `precondition:variable-degree-at-least-3` even with `eps`;
density evolution for l = 2 is available as `ldpc.bec_density_evolution(2, r, eps)`), `eps` (density
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
(the torus, 3 .. 8, default 7); `ising2d` (exact): `beta` (default "beta_c"), `L` (torus side, 2 .. 6,
default 4), `Ly` (default L); `ising1d` (exact): `beta` (default 1), `h` (default 0), `N` (ring, default
8); `bethe_ising` (cavity): `k` (coordination, default 3), `beta` (default 1), `h` (default 0), `branch`;
`sk_bounds` (bound): `beta` (rational > 0, default 2), `h` (default 0), `K` (1 or 2, default 1), `digits`
(an int 1 .. 200: certified digits of the K-step balls, default 20, else `precondition:digits`; the RS values
always 40); `online_perceptron` (online): `rule` (required: hebb | perceptron | adatron | optimal), `eta`
(> 0, default 1; the optimal rule needs 1), `noise` (flip probability in [0, 1/2), default 0),
`noise_estimate` (optimal rule only; default the true noise), `R0`, `Q0` (default 0, 1; the optimal rule
needs R0 > 0), `alpha` (the learning curve), `N` (finite-N simulation; an int >= 10, needs `alpha`) and
`seeds` (default 0-19; needs `N`); `online_linear` (online): `rule` (optimal (default) | standard),
`constraint` (sphere (default) | none), `eta` (standard rule: > 0 or `greedy`, default 1; the optimal rule
needs 1), `M` (teacher norm |B|^2, default 1), `R0`, `Q0` (default R0 = 1e-3 M for the optimal rule or the
greedy eta, else 0; Q0 = M), `alpha` (the online curve and the offline solutions), `N` (finite-N online
simulation; needs `alpha`) and `seeds` (default 0-19; needs `N`); `online_drift` (online): `rule` (optimal
(default) | annealed_hebb | step | symmetric | hebb | perceptron | adatron), `drift` (random (default) |
deterministic), `D` (>= 0; without it the small-drift law, noiseless rules only), `eta` (O1 rules; the
optimal and VKC rules need 1), `noise`, `noise_estimate` (optimal rule only), `R0`, `Q0` (default R0 = 0.01
for the rules that read rho, else 0; Q0 = 1), `alpha` (the learning curve and its KC93 bound), `N`
(finite-N simulation with the teacher drifting; needs `alpha`) and `seeds` (default 0-19; needs `N`);
`online_tcm` (online): `rule` (optimal (default; CC95's nonlocal rule, K = 3) | local_optimal | hebb),
`K` (odd, default 3), `eta` (Hebb; the optimal rules need 1), `R0`, `Q0` (default R0 = 0.01 for the
optimal rules, else 0; Q0 = 1), `alpha` (the learning curve and, for the local rules, the mapped
perceptron), `N` (finite-N simulation; a multiple of K with N/K >= 10, needs `alpha`) and `seeds`
(default 0-19; needs `N`).
`online_scm` (online): `K` (student units, default 2), `M` (teacher units, default K), `eta` (default 1.5),
`teacher` (isotropic | graded), `sym` (all (default) | none | a comma list of identical, permutation,
small_eta: the symmetric points to compute), `critical` (eta_max: only that root; any other value: the
critical learning rates, K = M or (K, M) = (2, 1)),
`init` (brw16 (K = M = 2, the default there) with `X` (1e-6) and `Nx` (1000) | ss (default otherwise) with
`init_seed` (1)), `alpha` (the learning curve and the plateau report), `B` (BRW's plateau threshold, 0.05),
`N` (finite-N simulation, N >= 20, needs `alpha`) and `seeds` (default 0-11; needs `N`).
Without `beta` an entry returns its T = 0 / critical headline values. A composed spec
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
| `missing-step:ansatz:<a>` | composer: an ansatz other than `rs`, `1rsb` (CLI: `ansatz` on a spec that is not a catalog entry, or `level_a_ansatz` other than `1rsb`) |
| `precondition:named-seeds` | amp route of `sk` and `teacher_student` (finite N): `seeds` not a non-empty list of non-negative integers (`--param seeds=1`, an empty list, a negative or non-integer seed) |
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
| `precondition:beta`, `:J`, `:h`, `:size`, `:coordination`, `:branch`, `:ordered-phase`, `:certified-radius`, `ambiguous:branch`, `undetermined:spinodal`, `no-solution:certification`, `:radius-stall`, `:not-an-integer-polynomial`, `:torus-disagreement`, `:ring-identity`, `:audit`, `unsupported:python-flint`, `:baller-integrity` | lattice (`lattice/`): beta not a rational > 0 (bools, nan refused); J <= 0 (ferromagnet only; `beta_c` needs J = 1); h not rational; a torus side outside its range; k not an int >= 2; a branch not among the fixed points; Yang at or below beta_c (m* = 0 there); a certified route returning a value without its radius; the magnetization at h = 0 below T_c without a branch; a fixed-point count or stability sign undecided at 4096 bits; baller.solve could not certify the digits; a radius that does not halve when the precision doubles; Kaufman's coefficient balls (radius < 1/4) containing no integer; Kaufman's polynomial differing from the transfer matrix's; the 1D ring balls not overlapping; the two Bethe free-energy forms disagreeing; python-flint missing (use the project venv, INSTALL.md); `baller.verify()` failing |
| `unsupported:rule`, `precondition:eta`, `:noise`, `:optimal-eta`, `:initial-overlap`, `:alpha`, `:overlap`, `:averages`, `:asymptote`, `undetermined:asymptote`, `:rho-sign`, `no-solution:rho-zero`, `:step-size`, `:overflow` | online (`online/`, `online_perceptron`): a rule that is not hebb / perceptron / adatron / optimal or a callable; eta not a finite number > 0 (also a schedule turning non-positive); a flip probability outside [0, 1/2); the optimal rule with eta != 1 or a schedule; (R0, Q0) with Q0 <= 0 or rho0 >= 1, or R0 <= 0 for the optimal rule; alpha <= 0; rho <= 0 for the optimal modulation; a measured log-slope further than 0.01 from the derived exponent; a simulated seed of the optimal rule reaching rho <= 0; the optimal flow reaching rho = 0 at finite alpha (below the no-learning line); the step-size or ln Q > 700 guards |
| `unsupported:rule`, `unsupported:algorithm`, `precondition:eta`, `:optimal-eta`, `:constraint`, `:teacher-norm`, `:averages`, `:state-point`, `:initial-overlap`, `:alpha`, `:asymptote`, `:offline-window`, `:method`, `undetermined:asymptote`, `:crossing`, `:descent`, `no-solution:rho-zero` | online (`online/linear*.py`, `simulate_linear.py`, `online_linear`): a rule other than standard / optimal or a callable F(b, h, Q, R); an offline algorithm other than standard / optimal; eta not > 0 nor `greedy`; the optimal rule with eta != 1 or greedy; a constraint other than sphere / none; M <= 0; (R0, Q0) with Q0 < 0 (or = 0 on the sphere, for the optimal or greedy rule), R0^2 > Q0 M, J0 = B, or R0 <= 0 for the optimal rule or the greedy eta; alpha <= 0; an asymptote off the sphere or not converged to 1e-9 between 45 and 60; the standard offline algorithm on an unrealizable sphere (alpha > Q/M, or alpha >= 1 with Q != M); no single crossing on the grid; R <= 0 under the optimal modulation or the greedy eta; `simulate_offline` with method descent for the optimal algorithm (its finite-N zero-error set is degenerate) or descent not converged in 20000 steps |
| `unsupported:rule`, `unsupported:stationary`, `precondition:drift`, `:drift-kind`, `:optimal-eta`, `:known-parameter`, `:noise`, `:initial-overlap`, `:alpha`, `:averages`, `:overlap`, `undetermined:stationary`, `:small-drift`, `:rho-sign`, `no-solution:no-tracking` | online with drift (`online/drift*.py`, `simulate_drift.py`, `online_drift`): an unknown rule; a stationary state for a callable F(h, b, Q); D < 0 or a schedule where a constant D > 0 is needed (stationary, small drift); a drift other than random / deterministic; a VKC rule at eta != 1 or with noise_estimate; R0 <= 0 or jump_overlap <= 0 for a rule that reads rho (optimal, VKC); jumps outside (0, alpha_max); several stable stationary states; a small-drift log-slope more than 0.01 from the derived exponent; a simulated seed reaching rho <= 0 under a rule that reads rho; plain Hebb under drift (R -> eta a / D while Q grows: no stationary state) |
| `unsupported:rule`, `unsupported:K`, `precondition:K`, `:eta`, `:optimal-eta`, `:initial-overlap`, `:state-point`, `:alpha`, `:overlap`, `:asymptote`, `:size`, `:known-parameter`, `undetermined:asymptote`, `:rho-sign`, `no-solution:step-size` | tree committee machine (`online/tcm*.py`, `simulate_tcm.py`, `online_tcm`): a rule other than hebb / local_optimal / optimal; the nonlocal rule at K != 3; K not an odd integer >= 1 (or K branch errors missing); eta not > 0, or != 1 for an optimal rule; Q0 <= 0, rho0 >= 1, or R0 <= 0 for an optimal rule; integrating without a state or with alpha <= 0; theta outside (0, pi) (or rho <= 0 for an optimal rule); a rate-limit constant for Hebb; N not a multiple of K with N/K >= 10; seeds without N; a log-slope more than 0.01 from the derived exponent; a simulated branch reaching rho <= 0; the shared integrator's step underflow (e.g. ln Q overshooting from a tiny Q0) |
| `precondition:K`, `:M`, `:eta`, `:symmetric`, `:state-point`, `:initial-overlap`, `:alpha`, `:threshold`, `:grid`, `:size`, `:known-parameter`, `unsupported:teacher`, `:init`, `:sym`, `:branch`, `no-solution:fixed-point`, `:critical-rate`, `:step-size` | soft committee machine (`online/scm*.py`, `plateau.py`, `simulate_scm.py`, `online_scm`): K or M not an int in 1..16; eta not > 0; symmetric points with K < 2 or a graded teacher (or asymptotics with K != M); integrating without (Q, R) or with a covariance that is not positive semidefinite; alpha <= 0; B outside (0, 1); fewer than 50 plateau grid points; N < 20; seeds without N or N without alpha; a teacher other than isotropic / graded; init brw16 off K = M = 2; a sym entry other than identical / permutation / small_eta; a branch other than identical / permutation; Newton not converging (or the permutation branch merging with Q = C, as without the eta^2 term); no sign change of an eigenvalue in the scanned eta range; the shared integrator's step underflow |
| `precondition:order-parameter`, `:levels`, `:digits`, `:K`, `:temperature-range`, `unsupported:field`, `undetermined:stationary-points`, `ambiguous:minimiser`, `no-solution:certification`, `:radius-stall` | bounds (`bounds/`, `sk_bounds`): q or m outside Guerra's admissible set (q nondecreasing in [0, 1], m nondecreasing in (0, 1]); overlap / Parisi-parameter counts that do not match; digits not an int 1 .. 200; K outside 1 .. 5 (catalog: 1 or 2); K-step staircases outside 1 < beta <= 3 or with h != 0; the stationary points of P_RS not certified complete (a cover or monotone run undecided); two local minima with overlapping values; interval Newton or the precision escalation failing |
| `precondition:coupling-matrix`, `:pm-one-couplings`, `:enumeration-size`, `:integer-patterns`, `:seed`, `:betas`, `:sweeps`, `:nbetas`, `:windows`, `:sizes`, `:samples`, `:order`, `:supported-disorder-law`, `no-solution:capacity`, `undetermined:ground-states`, `undetermined:equilibration` | Monte Carlo and enumeration (`numerics/`): J not a finite symmetric matrix with zero diagonal (PT: a stack of them), or not +-1/sqrt N for the +-J spectrum; N above the enumeration limit (SK 26, perceptron 30); P not an int >= 0, or alpha N not an integer; a seed not an int >= 0 (or a list of them); betas not positive and strictly increasing (the finite-size thermal route also needs beta > 0.5, `moments.pm_one_gap_limit` 0 <= beta < 1); sweeps / nbetas not an int >= 1 or betas not a sequence (never a silent default); fewer than two samples or a window split that is empty; fewer than three distinct sizes, or a size without >= 2 samples; a closure order other than 2, 3; a law other than `pm_one`, `gaussian`; every drawn pattern stored (draw more); the two PT runs of a sample disagreeing on its ground state (raise sweeps); a size whose late and early energy windows differ beyond the 99% band (raise sweeps) |
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

CATALOG (24 entries: the 7 of v1, 12 added by the extensions E4-E10 and 5 by the online-learning tasks O1-O5; `list` shows the live table;
values at the default / banked state points):

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
| `ising2d` 2D Ising, exact solution, certified | exact | `beta_c` 0.440687, `minus_beta_f` 0.929695 at beta_c (at `beta`: the certified integral, and `magnetization` above beta_c), `yang_exponent_loglog` 0.1249998, `torus_minus_beta_f` (Kaufman) = `torus_minus_beta_f_tm` (transfer matrix) on the 4 x 4 torus | `onsager_critical_minus_beta_f` ((1/2) ln 2 + 2G/pi, OEIS A242743), `beta_c_square` (Kramers-Wannier via Duminil-Copin), `yang_exponent` 1/8 (Yang via Duminil-Copin; band 1e-5) | `beta_c`, `minus_beta_f`, `magnetization`, `torus_minus_beta_f` exact-by-literature (Kramers-Wannier, Onsager, Yang, Kaufman), `torus_minus_beta_f_tm` exact by construction (the integer transfer matrix, cited to Duminil-Copin Frame 2); `yang_exponent_loglog` not-computed (a consistency check) |
| `ising1d` 1D Ising in a field, certified | exact | `minus_beta_f` 1.126928 (beta = 1, h = 0), `ring_Z` | `ising1d_minus_beta_f_beta1_h0` (ln 2 cosh 1: the formula of Wang, Diaz-Mendez & Capdevila, arXiv:1907.11701, at a chosen point) | exact-by-literature |
| `bethe_ising` ferromagnet on the Bethe lattice / random k-regular graph, certified | cavity | `beta_c` 0.549306 (k = 3); per fixed point `cavity_field_*`, `magnetization_*`, `phi_*`, `jacobian_*`; `phi` 1.503159 (k = 3, beta = 1, h = 0) | `bethe_beta_c_k3` (atanh(1/2), Dembo-Montanari Eq. (2.5)) | `beta_c` not-computed (a closed-form threshold, DM's physics prediction); per fixed point stable / unstable by the computed BP Jacobian; `phi`, `magnetization` those of the selected fixed point (Dembo-Montanari Thm 2.4) |
| `series_ising2d` 2D Ising (Onsager) free energy, exact series | series | `a4` 1 .. `a20` 8174.2 (v = tanh K); `operator_order` 3, `operator_degree` 8, `elliptic_factor_order` 2; `u_c` 3 - 2 sqrt 2, `v_c` 0.414214, `critical_exponent` 2, `alpha` 0 | `a4`, `a6`, `a8`, `a10` (Finch's table; independently the exact 7 x 7 torus), `v_c` (Kramers-Wannier / Onsager via Duminil-Copin), `alpha` (Onsager via Duminil-Copin, Baxter) | coefficients exact-by-literature (the cited integral, expanded exactly); operator values not-computed (L g = P proved exactly; minimal within bounds only); `u_c`, `v_c`, `alpha` exact-by-literature |
| `sk_bounds` SK pressure: Guerra's RS and K-step RSB upper bounds, certified | bound | `rs_bound` 1.584786 (beta = 2: min over q of P_RS), `rs_q` 0.530368 and every stationary point; `rsb1_bound` 1.582418 at the package's 1RSB optimiser (`K=2`: `rsb2_bound`); `chain_verdict` | `annealed_pressure_beta_half` (ln 2 + 1/16: Guerra 1992 Thm 3, the annealed value for beta <= 1) | bounds exact-by-literature (Guerra 2003 Thm 3), certified radii; `rs_q` stable iff P_RS''(q*) > 0 certified; the chain verdict is a consistency check (meta), not a theorem |
| `online_perceptron` online learning of a perceptron teacher (Hebb, perceptron, AdaTron, optimal; output noise) | online | `eps_exponent`, `eps_constant` (alpha = 1e4; noiseless or Hebb), optimal `eps_constant_formula`; at `alpha`: `eps`, `R`, `Q`, `rho` (Hebb: `eps_hebb_closed_form`); optimal with noise: `no_learning_boundary`, `robust_boundary`; with `N`: `eps_sim` | exponents and constants for the four rules (Biehl-Riegler Eqs. (7), (8), (15); 0.88 from VKC 1998 Tables I-II, Kinouchi-Caticha 1992 not fetched); no-learning line 2 lam - 1/2 at lam = 0.3, 0.4 (Copelli et al. p. 5); robust line at lam = 0.2, 0.3, 0.4 READ OFF Copelli et al. Fig. 1a (tol 0.003) | all not-computed (physics-level ODE limit, self-averaging assumed); `eps_sim` statistical |
| `online_linear` online versus offline learning of a linear teacher (standard F = b - h and the optimal F = (Q/R) b - h; sphere or unconstrained; greedy eta) | online | at `alpha`: `rho`, `eg`, `R`, `Q`, `eps_sign`, optimal `gamma_temperature`, greedy `eta_greedy`, `rho_offline_standard` / `_optimal`, `eg_offline_*`, `rho_offline_optimal_ridge`, `eg_pinv_ridge`; on the sphere at eta = 1: `rho_inf`, `eg_inf`, optimal `tail_constant`, `alpha_cross`; with `N`: `eg_sim`, `rho_sim` | Kinouchi-Caticha 1995 (preprint IFUSP/P-1157 read): rho_opt = sqrt(1 - e^-alpha) at alpha = 1, 3 (Eq. (19)), rho_std = 1 - e^-1 (Eq. (26)), rho_max 0.8 at M/Q = 4 (Eq. (24)), e_g^oo 0.9 / 0.5 at Q = 4 (Eq. (25)), tail 1/2 (Eq. (31)), offline rho = alpha, sqrt(alpha) and e_g = 1 - alpha, 1 - sqrt(alpha) (Eqs. (27), (49), (53), (54)), the crossing 0.71 (p. 11, tol 0.005) | all not-computed (physics-level ODE limit; offline from the derived zero-energy geometry); `eg_sim` statistical |
| `online_drift` online learning of a drifting perceptron teacher (random or deterministic drift; the O1 rules and VKC's annealed Hebb, step, symmetric and optimal modulations) | online | without `D`: `eps_exponent`, `eps_coefficient` (small-drift law); with `D`: `eps_inf`, `theta_inf`, perceptron `Q_inf`, the KC93 bound `eps_inf_bound`, annealed Hebb `eps_inf_closed`, optimal `eps_inf_kc93`; at `alpha`: `eps`, `R`, `Q`, `rho`, `eps_bound`; with `N`: `eps_sim` | VKC 1998 (arXiv read): Tables I-II small-drift coefficients and exponents of the four rules (random and deterministic; the deterministic optimal formula as printed is inconsistent with its 0.63, the number is banked), Figs. 4 and 6 read off at two D each (tol 0.003); KC93 (preprint IFUSP/P-1044 read) C = 0.447 (Eq. (25)); VKC p. 15's perceptron delta = 5, 8 banked but UNCLAIMED (replicant derives 3, 4) | all not-computed (physics-level ODE limit); `eps_sim` statistical |
| `online_tcm` online learning in the tree committee machine (K odd non-overlapping branches, majority vote; CC95's nonlocal optimal rule, the local optimal rule, Hebb) | online | `eps_constant` (optimal rules: the rate limit; Hebb: closed form), `eps_exponent` (local rules, alpha = 1e4), local `eps_constant_perceptron`; local rules: `noise_equivalent`, `time_factor`, at `alpha` `rho_perceptron_mapped`, `equivalence_gap`; at `alpha`: `eps`, `eps_branch`, `R`, `Q`, `rho`; with `N`: `eps_sim` | CC95 (preprint IFUSP/P-1126 read): kappa 0.88 (Eq. (32)) and its printed integral, pseudo-Hebb rho(alpha) (Eq. (30)) at alpha = 1, 5, 20 and sqrt(3/2pi); Fig. 1 read off at four alpha (tol 0.003; UNCLAIMED, slow test); CKC96 (abstract only): exponents 1 (local learning) and 1/2 (Hebb) at finite K | all not-computed (physics-level ODE limit); `eps_sim` statistical |
| `online_scm` online gradient descent in the soft committee machine (K student, M teacher erf units; Saad-Solla ODEs) | online | K >= 2: `eps_sym_identical`, `eps_sym_permutation` (R, Q, C, `tau_esc`), `plateau_shift_per_100x`, `eps_small_eta`, `tau_small_eta`; K = M: `lambda1`, `lambda2`, `asymptotic_rate`; with `critical`: `eta_max`, `eta_crossing`, `eta_fastest` (K = M) or `eta_c_specialised`, `eta_s_symmetric` (K = 2, M = 1); at `alpha`: `eps`, `plateau_status`, `plateau_height`, `plateau_entry`, `plateau_exit`, `plateau_length`, `plateau_rtol_spread`; with `N`: `eps_sim` | SS (NIPS 1996 / MANNA 1997 read; PRL, PRE not fetched): the small-eta plateau 0.0203 and Eq. (7), tau at K = 2, 3, Fig. 1d heights (read off, tol 0.001), Fig. 2 eigenvalues and eta_max (read off), Eq. (9)'s coefficient (UNCLAIMED, slow), the 1/eta trapping law (UNCLAIMED, slow); BS (preprint read): eta_c = pi sqrt(5/3), eta_opt = 2 eta_c/3, the two-unit eta_c 1.89, eta_s 2.17; BRW (preprint read): Eqs. (20)-(21), Fig. 1 and Fig. 2 read off, Eq. (17)'s half-log slope (UNCLAIMED, slow) | all not-computed (physics-level ODE limit); `eps_sim` statistical |
| `viana_bray` sparse +-J | cavity | `beta_c` (3-regular) 0.881374; T = 0 RS `p0` 1/3, `e0` -23/18; factorized 1RSB -1.27231, mu* 0.4174 | those, plus slow/statistical population-dynamics RS and 1RSB at k = 5, T = 0.8 | `beta_c` stable; the rest not-computed |

HELD-OUT EVIDENCE (the test of REUSE; `tests/test_heldout_sk.py`, `tests/test_heldout_perceptron.py`,
`tests/test_heldout_mc.py` (E11), the positive controls, catalog guards and documents in
`tests/test_heldout.py`, shared helpers in `tests/heldout_common.py`). Two models that are not catalog entries
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
- The v1 small-N perceptron enumeration (N <= 14) is coarse: its band (about +-0.09) cannot tell
  quenched from annealed; it catches gross errors only (e.g. the spherical site, -0.18 vs 0.515). It is
  kept; the E11 route at N <= 24 below is the stronger one.
- The +-J SK agreement with the Gaussian `q` verifies the ADAPTED second-cumulant step, not universality
  independently. +-J-specific independent checks: the n = 1 enumeration (v1), AMP on sampled +-J
  matrices (E3b) and the E11 routes (f), (g) below. The universality cite (Carmona and Hu) was read from
  its abstract only.
- The annealed bound does not detect the spherical-site swap (the other checks do).
- Enumeration slack c/N with c = 1 is an assumption of the test.
- (E3b) An independent finite-N route for (1): AMP run on actually sampled +-J matrices (`amp/`, which
  imports none of the replica, cavity or family code), N = 2000, seeds 0-7, t = 100, at beta = 1.5,
  h = 0.8 (RS-stable side): 0.60715 +- 0.01899 (99%), composed q 0.61309 inside; the composed q with
  the planted doubled +-J variance (0.66054) and naive mean-field AMP (0.82513) fall outside. AMP's SE
  is a theorem for Gaussian J only, so for +-J this is sampled numerical evidence, not a theorem, and
  the band (+-0.019) excludes errors of the size of the planted one (0.053), not small ones (slow).
- (E11, `numerics/`, `tests/test_heldout_mc.py`, all slow, statistical: named seeds 2026102601..05, 99%
  Student-t bands plus a stated systematic; none uses the composer to measure, every comparison target is
  computed after the measurement; about 4.6 min together). What each check can and cannot exclude:
  (e) Gaussian SK ground energy by parallel tempering, N = 16 / 32 / 64 / 128 (1024 / 400 / 160 / 64
  samples; two independent PT runs must agree on every sample, and at N = 16 they equal exact
  enumeration), extrapolated in N^-2/3 (Boettcher 2005, Palassini 2003): e_inf = -0.7610 +- 0.0124
  (t-band 0.0059 + systematic 0.0066: omega = 3/4 and dropping N = 16). It CONTAINS the banked Parisi
  value (-0.76317 and its sources_disagree values) and EXCLUDES RS (-0.79788; the positive control); it
  does NOT resolve 1RSB (-0.76526, 2.1e-3 from Parisi: printed, not asserted; a band near 1e-3 would need
  N ~ 1000 and ~1e5 samples). (f) +-J SK at beta = 2 by PT, N = 64 / 128 / 256 (equilibration diagnostic
  passed at each size): at N = 256 the 99% band -0.68818 +- 0.00577 lies above both composed energies
  (RS -0.71871, 1RSB -0.71167): a finite-size statement only, it does not discriminate RS from 1RSB. The
  extrapolation -0.7064 +- 0.0186 (omega = 2/3 stressed by 3/4 and 1: no exponent is known deep in the
  ordered phase) contains the composed 1RSB energy and excludes the planted doubled +-J variance (about
  20 band widths); it does NOT exclude RS (0.66 band widths) and "1RSB closer than RS" is NOT resolved
  (printed, not asserted). (g) Exact E ln Z by Gray-code enumeration of +-J samples, N = 12 / 16 / 20 /
  22, beta = 0.7 and 0.9: the annealed value E ln Z = ln E Z is EXCLUDED at every (N, beta) (the first
  check here that tells quenched from annealed; at beta = 0.7 the gap is about -0.0095 against bands of
  0.0023-0.0068); at beta = 0.7 the +-J order-2 moment closure (exact finite-N moments) lies inside the
  band plus |order 2 - order 3|, asserted only because that tolerance excludes 0 at every N, and the
  Gaussian-law closure is excluded (the law matters); at beta = 0.9 the truncation estimate (0.03-0.065)
  is the size of the signal: printed "NOT resolved", not asserted. (Perceptron e) exact counts by
  meet-in-the-middle at N = 12 / 16 / 20 / 24 (8000 / 8000 / 3000 / 600 pattern sets; N stops at 24 for the
  budget), estimator E ln(1 + count)/N, 1/N extrapolation with a systematic over (drop N = 12, quadratic
  in 1/N, the estimator conditioned on count > 0): S = 0.5116 +- 0.0054, 0.3130 +- 0.0231, 0.0811 +- 0.0347
  at alpha = 0.25, 0.5, 0.75 contain the composed S (0.5128, 0.3144, 0.0865) and exclude the annealed
  ln 2 (1 - alpha) and the spherical-site entropy; the capacity M_N/N extrapolates to 0.8285 +- 0.0597,
  which contains the composed zero and the banked 0.8330786 (the same number, so they are not resolved
  from each other) and excludes the spherical capacity 2 and the annealed capacity 1. At alpha = 0.75 the
  empty-instance hazard (18-21% of sets) dominates the systematic. Detailed balance of the PT code
  (default tier): time averages at N = 8 equal the exact spectrum means within 99% bands at four betas,
  and the planted acceptance rule (x0.8) and the reversed exchange sign are detected at every beta.

ACCEPTANCE GATES: `python3 tools/replicant/selftest.py` (default tier: 425 tests, about 22 s on a laptop, measured 21.4 s wall; also `python3
run_selftests.py replicant`, which runs the `python3` on PATH; run both from the activated project venv,
which has python-flint) covers the core and
refusal paths, the symbolic derivation against independent expressions, level agreement (a)=(b) for
every family where both exist, every catalog entry against its banked non-slow values, the cavity routes,
the composer and the held-out models, the CLI and the LaTeX writer (displayed equations re-checked
numerically), the derived replicon (E1) and 1RSB (E2), AMP / GAMP and state evolution (E3), the 1RSB
replicon and the Gardner transition (E4), K-XORSAT and K-SAT (closed forms, small-N exact checks, BP on
trees, the SP-rule audit, small SP populations), LDPC density evolution and small peeling runs, the
spiked-Wigner and SBM state evolution / KS line, the ridge closed forms and small simulations, the exact
series and its operator, the certified lattice values and Guerra bounds, the PT detailed-balance gate
and exact enumeration (E11), and the online-learning ODEs, Gaussian averages and a small simulation
band (O1), the linear perceptron's ODEs, closed forms, offline geometry, ridge route and small online and
offline simulation bands (O2), the drift flow, the VKC modulations and their class maximisation, VKC's
drift-independence claim, the KC93 bound for every rule, stationary states and small-drift laws against
the bank and a small drifting-teacher simulation band (O3), the TCM's internal-representation averages,
CC95 Eqs. (5), (21), (24), (30), (32) against the TCM route, CKC96's equivalence with O1's noisy perceptron
(and a planted wrong mapping failing), the asymptotic constants against K and a K = 3 simulation band
(O4), the SCM's erf averages against Gauss-Hermite (with planted normalisation errors), the papers'
fixed points, eigenvalues and critical rates against the ODE, the plateau report (escaped, undetermined,
none, the rtol check) and a small simulation band with the eta^2 term planted out (O5). `--full` adds 156 `@slow` tests (full-RSB Parisi up to K = 9,
population dynamics at reference sizes, the XORSAT GF(2) crossing at N = 400-2000 and leaf removal at
N = 100000, the K-SAT SP-population crossing and BP window, the LDPC crossing at N = 20000, the spiked
AMP and SBM bands at reference sizes, the ridge 1/n bands at n = 400, 1000, the online ODE-vs-simulation
bands at N = 1000 (8 s; linear perceptron 2.3 s; drifting teacher 9 s; TCM at N = 3 x 300 4 s, CC95 Fig. 1
and the nonlocal asymptote 8 s; SCM through the plateau at N = 1000 13 s, BRW's Eq. (17) and SS's 1/eta law
6 s), the 2RSB certified bound and
the K = 5 chain, the E11 Monte Carlo / enumeration held-out routes (about 4.6 min), the Hopfield replicon
scans, the p-spin T -> 0 energy, solver-based composed and LaTeX checks, a PDF compile when a TeX engine
is present) and takes about 18 minutes on a laptop (measured 18 min 11 s, 557 tests); run it before merging a change to a family. The `@slow` marker is
honoured by `selftest.py` only: pytest ignores it and would run everything. Level agreement between the
derivation and the pipelines is a consistency check, not new evidence for the ansatz; the cavity route reproducing
the replica RS equations is an independent derivation but NOT independent evidence for the RS ansatz.

KNOWN HAZARDS:
- Rigor: PHYSICS-LEVEL, not proven, except the four certified entries (`ising2d`, `ising1d`,
  `bethe_ising`, `sk_bounds`). `rigor` says which on every result; the n -> 0 continuation and the
  RS/RSB ansaetze are assumptions. Do not quote a physics-level result as a theorem; a certified value
  is a rigorous enclosure of the cited formula or bound, not a proof of that formula.
- What the extensions (E1-E11) changed in these limits, stated precisely:
  (1) RS stability is now COMPUTED at level (a) for the spherical and Ising perceptron and Hopfield
  (E1: the derived replicon; for Hopfield also the derived m-mode), so composed Ising-perceptron values
  can be final. It is LOCAL stability in the replicon (and m) direction: the longitudinal / anomalous
  q-modes are not computed, and the Hopfield m-mode has no error estimate (the absolute quadrature floor
  1e-12 is amplified through the Schur complement of a near-singular (q, qhat) block: measured 7e-6
  relative at 0.9 m_fold; its sign was robust at every tested point). Level-(a) 1RSB values (E2) stay
  `not-computed`.
  (2) p-spin finality is conditional on the computed 1RSB replicon (E4): Ising p-spin state-point values
  are `stable` only for T_G < T < T_s (T_G = 0.240260 at p = 3, matching Rizzo 2013) and `unstable`
  below T_G; spherical pure p-spin 1RSB values are `stable` by the replicon computed at the requested
  point (1 - (p-1) y = 0.290015 at p = 3 on the static line; no Gardner transition on the 5-point scan). The formula is cited (Crisanti-Leuzzi 2006, Montanari-Ricci-
  Tersenghi 2003), checked against an explicit integer-n functional at integer m only; FRSB below T_G
  is not computed; the frozen Ising-perceptron 1RSB stability is refused.
  (3) Held-out evidence is stronger (E3b AMP on +-J matrices; E11 parallel tempering and exact
  enumeration; see HELD-OUT EVIDENCE for what each excludes): RS is now excluded for the SK ground
  energy and annealed is excluded for the +-J SK E ln Z and for the Ising-perceptron entropy and
  capacity. NOT resolved at the default budgets: 1RSB against Parisi at T = 0, RS against 1RSB for the +-J
  energy at beta = 2, the moment closure at beta = 0.9, perceptron sizes beyond N = 24.
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
  root" (that verdict rests on the trapezoid residuals only, a rule not validated at large beta); one dropped by the polish or an audit is listed as "a root may exist there" and, like any
  dropped nontrivial candidate, blocks the unique label (`ambiguous:several-saddles`). Main-rule window:
  the base rule (32 Gauss-Legendre panels) holds 1e-9 for SK up to about beta = 3; when the audit
  disagrees the candidate is re-polished with 64 and 128 panels (measured: SK beta = 4 needs 64, beta =
  6 needs 128); beyond that a candidate is dropped on audit (named, blocking), never silently. Cost: the
  SK level-(a) 1RSB solve takes about 5.5 s per state point (measured, beta = 2, h = 0, branch max-f,
  CLI). Roots with q1 - q0 < 1e-4 or
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
  the planted thresholds 0.3664 and 0.4366. The DE-against-peeling comparison has no planted error in
  the DE code itself (its controls shift eps; the planted degree polynomials act on the thresholds).
  Not implemented: l = 2 (stability-limited) thresholds,
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
  true and every SE value `not-computed` (a floor, not the fixed point; LM: q* > 0 for lambda > 1);
  the numbers in that window still print as gamma_se 0.0 etc. (only the labels say not-computed). The code reports the largest
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
  +- 0.0117, with it halved 0.5983 +- 0.0155: all three exclude the bank, the no-Onsager control at
  lambda = 2 only narrowly (0.7763 - 0.75 = 0.026 against the half-width 0.0117, about 2.2 half-widths).
  The spectral threshold and the eigenvector overlap (Feral-Peche 2007, Benaych-Georges-Nadakuditi 2011)
  are cited as stated by Lelarge-Miolane Sec. 2.3; those papers were not fetched. Rademacher AMP at 1.5:
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
  finite singular points and local exponents: u = 0 and u = 1, exponents (0, 0, 0) each; u = -1,
  u = 3 + 2 sqrt 2 and u_c = 3 - 2 sqrt 2, exponents (0, 2, 2) each, with a logarithm at 2; none
  apparent (`singularities(op)`; u = oo is not analysed). At u_c the logarithm multiplies the exponent-2 solution:
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
- Solvable lattice models (E9, `lattice/`, entries `ising2d`, `ising1d`, `bethe_ising`). Certified
  values are python-flint arb balls through `tools/baller`'s front door (`baller.solve` over
  `baller.run`, 40 digits by default; multi-output routines under `baller.hygiene.ctx_guard`), after
  one fail-closed `baller.verify()`. The entries report a 40-digit midpoint and in `errors` an upper
  bound of |value - x| over the ball; `CatalogEntry.rigor` marks the route "certified" and compose
  refuses a certified run whose value lacks a radius (`precondition:certified-radius`); every other
  entry stays "physics-level". Onsager (Duminil-Copin Eq. (3.6)): off beta_c the inner angle is
  integrated in closed form and the single integral is enclosed by Arb's rigorous integrator
  (`acb.integral`, analytic flags passed: a - S >= (1 - S)^2 > 0 off beta_c); at beta_c the closed form
  (1/2) ln 2 + 2G/pi. The integrand is evaluated cancellation-free (a - S = (1 - S)^2 + 2S sin^2(t/2),
  a^2 - S^2 = (a - S)(a + S)): the naive a*a - S*S stalled near beta_c (radius 4e-6 at beta = 0.4407,
  whatever the precision). Measured window: 40 digits in < 0.1 s from beta = 0.3 to beta_c +- 1e-22
  (0.4407, 0.44068, 0.44069, 0.4406867935097715126163). Any computation whose radius does not halve
  when the precision doubles is refused `no-solution:radius-stall` (lattice/ball.certify), instead of
  escalating to baller's cap. Independent: mpmath tanh-sinh on the single integral (default tier, 45-50 digits)
  and the periodic trapezoid on the double integral itself (160 nodes per axis: 1e-25 at beta = 0.3);
  Kaufman's L x L torus approaches the bulk (gaps 2e-4, 4e-7, 1.4e-9 at L = 8, 16, 24, beta = 0.3).
  Yang (Duminil-Copin Eq. (3.7)): refused at and below beta_c by a certified sinh(2K) > 1 test. The
  log-log slope at beta - beta_c = 1e-6, 1e-8 is 1/8 - 1.71e-7: M^8 = c d (1 - 6.36 d + ...) gives
  1/8 - (6.36/8)(d1 - d2)/ln(d1/d2) = 1/8 - 1.71e-7; the band 1e-5 excludes 1/4 and 1/16. It checks
  the implemented formula, it does not derive the exponent. Kaufman (Izmailian & Hu Eqs. (10)-(16),
  Kastner Eq. (8b) for the signed gamma_0 = 2K + ln tanh K): the exact comparison is route (b): balls
  at the 2N + 1 nodes x_i = i/(2N + 2), a rigorous Vandermonde solve, radius < 1/4, one integer per
  ball; equal to the transfer matrix's integers (Kronecker substitution x = 2^(N+1) in exact Python
  integers) for 2 x 2 .. 4 x 4 (default) and 5 x 5, 6 x 5, 6 x 6 (slow); held-out rationals (1/7,
  29/70 near x_c, 9/10) agree within the balls; the unsigned |gamma_0| (planted) is caught. Since
  Kaufman's expression is a polynomial of degree <= 2N in x (his theorem), agreement at 2N + 1 nodes is
  an identity of the two routes. 1D: Z_N = Tr T^N = lambda_+^N + lambda_-^N exactly in rationals
  (brute force, matrix trace, Q(sqrt D)) for N = 2 .. 10. Bethe (Dembo-Montanari): coordination k
  (viana_bray's k is the excess degree k - 1); all fixed points of x = B + (k - 1) atanh(t tanh x),
  located on the monotone intervals of the cavity equation and certified by interval Newton (Moore's
  test in arb; baller.certify's engines are model-specific or need a symmetric block-arrow system, so
  they do not apply), stability (k - 1) xi'(x) < 1 certified; two free-energy forms (the link/site
  Bethe form and DM Eq. (2.9)) must overlap. Comparison targets (deterministic implementation agreement, not statistical estimates): the
  viana_bray population dynamics in its paramagnet (beta = 0.4, k_vb = 2: every run collapses to h = 0,
  and above both transitions the +-J and the ferromagnetic Bethe free energies coincide) and BP
  (cavity/sparse.bp_fixed_point) on sampled ferromagnetic 3-regular graphs (n = 200, seeds 101-104,
  beta = 1, h = 0.1): BP converges to the homogeneous fixed point on every instance (spread at float
  rounding), so both comparisons check the sparse code's float implementation against the ball, with
  tolerances 1e-12 (the collapsed population) and 1e-9 (BP), not sampling bands.
- Guerra's bounds on the SK pressure (E10, `bounds/`, entry `sk_bounds`). Conventions are families/sk.py's
  (J_ij ~ N(0, 1/N), P = -beta f); Guerra's Hamiltonian (1) with unit Gaussian couplings over sqrt N is
  the same, and his Eq. (14) with identity (26) gives the package's bracket (1 - q_K)^2 - sum m_k (q_k^2 -
  q_{k-1}^2) (a sympy test: equal to families/sk.py's quadratic_term for RS and 1RSB); the brief's
  P_RS(q) = ln 2 + beta^2 (1 - q)^2/4 + E ln cosh(beta sqrt(q) z + beta h) matches. What is proved:
  Guerra 2003 Theorem 3, (1/N) E ln Z_N <= P_K(q, m) for every N and every admissible (q, m), so each
  certified P_K at any parameters is a certified upper bound (labelled exact-by-literature, with its
  radius); Talagrand 2006 (not fetched) proves the infimum is the limit. What is checked: the chain
  P_K(params_K) <= ... <= P_RS(q_RS) at the package's own optimiser outputs (not a theorem: the theorem
  orders the infima; a violation is a solver defect), and that each package-reported pressure equals its
  certified functional within radius + the solver's stated tolerance. Certified Gaussian expectations:
  `acb.integral` on |z| <= Z (Z grows with the precision) plus an analytic tail bound as radius (ln cosh x
  <= |x|, tanh^2 in [0, 1], (tanh^2)'' in [-2/3, 2]). K-step levels are nested `acb.integral`s: real
  nodes get the accurate inner integral, complex boxes a cheap rigorous enclosure (cosh(x + iv) = cosh x
  (cos v + i tanh x sin v); a positive average of points in the cone |arg| <= m v < pi/2 stays in it, so
  Z_k(x + iv) = Z_k(x) zeta with |arg zeta| <= v, c_k <= |zeta| <= 1), which also proves analyticity on
  the strip |Im| < pi/2. Measured at beta = 2: RS 40 digits in < 0.01 s; 1RSB 8 / 12 / 20 / 40 digits
  0.10 / 0.26 / 0.54 / 3.3 s; 2RSB 12 / 20 digits 37 / 116 s; K >= 3 is not certified (the cost grows
  as nodes^(K+1); nested mpmath quadrature at 40 digits was not attempted, for the same reason), so
  higher levels enter the chain as the package's value +-
  its own error terms, labelled numerical). Minimum of the RS bound: P_RS'(q) = (beta^2/2) G(q), G(q) =
  q - E tanh^2. At h = 0, G(0) = 0. For beta < 1, G' > 0 is certified on an adaptive cover of [0, 1] (G'
  enclosed on a cell [a, c] through the argument beta * hull(sqrt a, sqrt c); 1, 4, 8 cells at beta =
  1/2, 9/10, 99/100), so G > 0 on (0, 1] and q = 0 is the unique minimiser; the minimum ln 2 + beta^2/4
  is P_RS(0) as a ball, compared with ln 2 + beta^2/4 built separately. For beta > 1, G' < 0 is certified
  on [0, a] (a = 1/2, 1/4, ...), and [a, 1] is treated as at h != 0. Only beta = 1 exactly (G'(0) = 0) rests
  on a stated lemma, not a computation: tanh^2 x < x^2, so E tanh^2(sqrt(q) z) < q (the Latala-Guerra
  lemma of Talagrand's Vol. I, Sec. 1.3 was not fetched and is not used). At h != 0 (and on [a, 1]) the
  roots are located by Lipschitz exclusion (|G'| <= 1 + beta^2), each surviving cluster split into runs of
  certified sign of G' (a tiny cell around a zero of G' is dropped only when Lipschitz exclusion shows G
  has no root on it) and a run holds one root when G changes sign across it; each root by interval Newton,
  inside its bracket; several local minima are compared by certified values (`ambiguous:minimiser`).
  `BestBound.certificate` (catalog `rs_certificate`) says which argument was used; a planted shift of G' by
  -1 (`guerra._HOOKS["dG_shift"]`) is refused `undetermined:stationary-points` at beta = 1/2 and 9/10, and
  a shift of +3 or more at beta = 2 (G'(0) = -3 there). A shift that keeps the sign pattern (-1 or +1 at
  beta = 2) is NOT detected: the hook perturbs the completeness argument, not G or P_RS, and the
  minimiser and value come out unchanged. The functional check
  catches only functional errors above the solver's stated accuracy (1e-12 for sk_rs, 2e-11 for sk_1rsb at
  beta = 2); smaller discrepancies pass by design. Normalisation: Guerra 1992 sums over ordered pairs i !=
  j with sqrt(N - 1) (his (2.3)); his log 2 + beta^2/4 with the critical point at beta = 1 is the package's
  unordered-pair convention (J_ij ~ N(0, 1/N) over i < j), as the annealed computation (2.17) shows
  (the E10 implementer's reading of Guerra 1992; the E10 review did not re-check it).
  Measured chain at beta = 2, K = 5
  (slow test, about 100 s with the package's own staircase solves): rs 1.5847858902693959 (functional
  gap 1.2e-17), 1rsb 1.5824176847584295 (gap 1.9e-15, tol 2e-11), 2rsb 1.5823642818021404 (certified
  1.58236428180213960 +- 2.7e-18: gap 8e-16, tol 5e-15, i.e. parisi.py's trapezoid h-grid agrees with
  the nested balls), 3rsb 1.5823576213517763, 4rsb 1.582356070022068, 5rsb 1.5823555659659725
  (numerical, +- 1.0e-10, 9.9e-11, 4.2e-14); every link "ok". Controls: a planted optimiser output (q0 =
  0, m = 1: its functional is ln 2 + beta^2/4 > P_RS) is reported `ordering:1rsb>rs`; a reported value
  off by 1e-6 `functional:1rsb`; the evaluator's innermost exponent planted at 1.01 m `functional:1rsb`
  (against an independent mpmath value).
- Online learning (O1, `online/`, entry `online_perceptron`). Convention Biehl & Riegler 1994 (preprint
  read): |B| = 1, inputs i.i.d. N(0, 1), J <- J + (eta / N) F(h, b, Q) b xi, R = J . B, Q = J . J (BR's
  Q is |J|), alpha = examples / N, eps = arccos(R / sqrt Q) / pi against the TRUE teacher (the
  prediction error on noisy labels is not reported); output noise flips the label with probability
  lam (Copelli et al. Eq. (7)); VKC 1998 use normalised fields and a modulation W = F / |J|, and their
  noise chi is the flip probability in their Eq. (1) but chi/2 in their Eqs. (20), (25): replicant follows
  (20)/(25), noise = chi/2. A mis-estimated optimal rule (noise_estimate != noise) is outside
  OPTIMALITY_CLASS: the entry then reports no `eps_constant_formula` and says so in a note. The N -> oo
  ODEs assume self-averaging (no concentration theorem fetched): every ODE value is `not-computed`.
  The flow is integrated in (theta = arccos rho, ln Q) with the averages' `drive` <F b (y - rho x)>
  computed without cancellation; the perceptron rule is stiff at large alpha (its theta relaxes on a
  scale alpha^(1/3) much shorter than alpha), which is why the asymptote is taken at alpha = 1e4 (0.03 s), where the measured
  constants sit 9e-7 (Hebb), 7.8e-5 (perceptron), 1.8e-4 (AdaTron) and 2.2e-4 (optimal) from their
  limits. "Optimal" is optimal only in `optimal.OPTIMALITY_CLASS` (single-example modulated Hebbian rules
  that know rho and the noise level, eta = 1, greedy per example); it is about twice the batch Bayes
  constant of `teacher_student` (0.8834 vs 0.4417, a comparison, never an input). Kinouchi & Caticha
  1992 itself was NOT fetched (IOP behind a bot check): its 0.88 is banked from VKC 1998 Tables I-II
  and BR p. 4. Copelli et al. Eq. (8) print I(lam) with Dt e^{-t^2}, which gives C(0) = 1.325, not 0.88:
  replicant derives C = 2 / I with Dt e^{-t^2/2} (0.8834) and records the printed form as a
  discrepancy (`asymptotic_constant(printed=True)`). Below the no-learning line the optimal flow
  reaches rho = 0 at finite alpha (W* ~ 1/rho) and is refused `no-solution:rho-zero`; at finite N the
  optimal rule from a small R0 can cross rho = 0 (refused `undetermined:rho-sign`), so start it at R0
  well above 1/sqrt(N). With noise > 0 only the Hebb asymptote is reported (no power law is claimed
  for the other rules).
- Online versus offline linear perceptron (O2, `online/linear.py`, `linear_offline.py`, `simulate_linear.py`,
  entry `online_linear`). Kinouchi & Caticha 1995 (preprint IFUSP/P-1157, May 1995, read; journal version
  not checked) use h = sqrt(N) J . S with X . Y = (1/N) sum X_j Y_j and I = S . S = 1; replicant maps
  their M = B . B, Q = J . J onto |B|^2 = M, Q = |J|^2 with unit-variance inputs (their Eqs. (7)-(8) at
  I = 1 are replicant's rates at eta = 1). e_g is the quadratic error (M + Q - 2R)/2 (their Eq. (4)), not
  a probability; `eps_sign` = arccos(rho)/pi is their e_g^B. Their 'constrained' learning is the sphere
  (Q fixed by the decay Omega; the simulator rescales |J|^2 = Q0 after each example, the same to O(1/N)).
  "Optimal" is optimal only in `linear.OPTIMALITY_CLASS`: it reads the true rho and M, maximises rho (not
  e_g: unconstrained, from Q0 off sqrt(Q0) = rho0 sqrt(M), its e_g grows with |J|), eta = 1; `greedy` (eta* derived
  here, not in KC95) is greedy within F = eta (b - h). The offline values are the zero-temperature solutions from the derived
  zero-energy-set geometry (J = c P B + w, |P B|^2 = alpha M), not the replica calculation; KC's Eq. (27)
  quotes Seung, Sompolinsky & Tishby 1992 (not fetched). The ridge route (E7) maps n = P, p = N, gamma =
  p/n = 1/alpha, sigma2 = 0, signal r^2 = M, excess risk = 2 e_g, and uses R = Q of the pseudoinverse (a
  projection of B) to read rho = sqrt(1 - risk/M); ridge's numbers never enter the online computation.
  The crossing is 0.71456 on the ODE (the root of sqrt(1 - e^-alpha) = alpha), printed 0.71 by KC.
  KC Eq. (20), the optimal rule with the theoretical rho(alpha) substituted, F = b / sqrt(1 - e^-alpha) - h,
  is checked as a time-dependent callable rule from alpha0 = 0.01 (banked, unclaimed by the catalog). The
  offline simulation has two methods: `construct` builds the solutions with a uniform w (the derivation's
  own assumption), `descent` finds the standard ones by gradient descent from a random start.
- Drifting rules (O3, `online/drift.py`, `drift_rules.py`, `simulate_drift.py`, entry `online_drift`).
  Papers read: Vicente, Kinouchi & Caticha 1998 (arXiv:cond-mat/9801297) and Kinouchi & Caticha 1993 (J.
  Phys. A 26, 6161; the preprint IFUSP/P-1044, March 1993, scanned, read page by page). Biehl & Schwarze
  (Europhys. Lett. 20, 733 (1992); J. Phys. A 26, 2651 (1993), doi:10.1088/0305-4470/26/11/014): the
  full texts were NOT read (IOP bot check on the PDFs, no preprint found); the existence and scope of the
  EPL paper were confirmed from its IOP landing page (abstract only) by the O3 reviewer (the
  implementer's own fetch met the bot check), the J. Phys. A paper from the Groningen portal; their drift models are used as VKC and KC93 restate them and nothing is banked from
  them. Conventions: random drift B <- (B + sqrt(2D)/N u)/|.| (VKC's <eta_i eta_j> =
  2D, KC93's T = D: B . B' = 1 - D/N per example); deterministic drift turns B away from J by
  arccos(1 - D/N^2); noise = VKC's chi / 2 (O1). VKC's "Hebb" is the annealed Hebb W_AH (their Eq. (22)),
  not plain Hebb, which under drift does not track (eps -> 1/2). Each VKC rule is optimal only in its
  information class (`drift_rules.VKC_CLASS`), at unit rate, reading the true rho. VKC's claim (Sec. III,
  p. 9) that the modulations "do not depend on the drift parameters" under uncorrelated drift is tested as
  stated: the class maximiser of drho/dalpha is the same at D = 0, 0.3 (random) and 0.01
  (deterministic), and changes with the noise. KC93 Eq. (22) prints e^{-(1 - lambda^2) u^2/2}; replicant derives (1 + lambda^2) (it agrees
  with the engine's <W*^2>/2; the printed form diverges for lambda >= 1) and keeps their Eq. (25) constant
  0.447, unaffected. VKC Table II's deterministic optimal formula (8D/A^2)^(1/4) does not give its printed
  0.63 (A is not defined in the arXiv text; (8D/(pi^2 A^4))^(1/4), derived, does). VKC p. 15's perceptron
  delta = 5 (random), 8 (deterministic) apply Eq. (34) outside its derivation: Eq. (34) rests on the
  decoupled form (35), and KC93 p. 6 state that the rho and |J| equations decouple only when W does not
  depend on J. The fixed-rate free-norm perceptron's |J| self-tunes under drift, and replicant's ODE
  gives delta = 3, 4 with eps_inf = (4D)^(1/3)/pi and 2^(5/4)/pi D^(1/4) (the step rule's coefficients;
  derived in `drift.drift_exponent`, checked against the stationary ODE); delta = 3 also violates VKC's
  inequality (42), delta > 1/beta + 2 = 5. VKC's values are banked, UNCLAIMED, recorded as a
  discrepancy. VKC p. 15 also prints "beta = 1 for the Hebb algorithm", which should read beta = 1/2
  (their Table I, annealed Hebb 0.40 alpha^-1/2).
- Tree committee machine (O4, `online/tcm.py`, `tcm_flow.py`, `simulate_tcm.py`, entry `online_tcm`).
  Papers: Copelli & Caticha 1995 (J. Phys. A 28, 1615; the preprint IFUSP/P-1126, October 1994, scanned,
  read page by page); Copelli, Kinouchi & Caticha 1996 (Phys. Rev. E 53, 6341): ONLY THE ABSTRACT was read
  (APS 403, closed access at repositorio.usp.br, no arXiv copy, the neuroprose file gone, not among the
  IFUSP preprints P-1127-P-1300), so its epsilon(K), its figures and its K -> oo analysis are not used.
  Conventions: K odd branches of N/K inputs, |B_k| = 1 (CC95's B_k^2 = 1), Gaussian inputs (CC95 use
  +-1), J_k <- J_k + (1/N) F_k xi_k, alpha = examples / N (= CKC96's alpha_CM, examples per adjustable
  weight), eps = P(TCM outputs differ), branch-symmetric overlaps. CC95's "optimal" (expected stability)
  rule is NONLOCAL (branch k reads every student field) and is implemented at K = 3 only; it is optimal
  only in `tcm.OPTIMALITY_CLASS["optimal"]` (greedy, true rho, eta = 1). `local_optimal` reads only the
  branch's own field; it is the CKC96 "local learning", identical (derived here) to O1's optimal noisy
  perceptron at lambda_K = (1 - P0)/2 (P0 the tie probability of the other K - 1 hidden units: P0 = 1/2, lambda_K =
  1/4 at K = 3; P0 = 3/8, lambda_K = 5/16 at K = 5) on N/K inputs at alpha_p = K alpha. The TCM route computes its own field averages
  (the posterior over the other units' internal representations) and never calls the mapping; the
  identity holds to <~ 1e-12 measured (1e-8 asserted) and fails under a planted wrong noise or time map. CC95's Eq. (32) integral
  gives 0.883400, the perceptron's C(0), which also supports O1's reading of Copelli et al. 1997 Eq. (8)
  as a typo. CC95's Eq. (21) as printed agrees with the TCM route to 2e-7 (quadrature), and their Eqs.
  (19)-(20), typed into the simulator, reproduce the nonlocal ODE at N = 3 x 300. CC95's pseudo-Hebb
  (constant step) from J = 0 is plain Hebb: rho does not depend on the step. The local rule's constant
  P0 C(lambda_K) (0.883, 2.665, 3.652 at K = 1, 3, 5) grows with K, while the nonlocal rule keeps 0.88
  (K = 3; CEKBSRC97 p. 4 state it for every K, not checked beyond K = 3).
- Soft committee machine (O5, `online/scm.py`, `scm_flow.py`, `plateau.py`, `simulate_scm.py`, entry
  `online_scm`). Papers: Saad & Solla, PRL 74, 4337 and PRE 52, 4225 (1995) were NOT fetched (APS 403; not on
  Aston's eprints or arXiv; the neuroprose file saad.online.ps.Z is not reachable); what is banked from SS is
  read in their NIPS 8 (1996) paper and MANNA 1995 proceedings (Aston eprints 640, 657), which restate the
  PRE. Biehl & Schwarze 1995 (preprint LU TP 94-10) and Biehl, Riegler & Woehler 1996 (preprint WUE-ITP-96-003;
  J. Phys. A 29, 4769-4780 confirmed) were read. Convention: SS's (K student, M teacher units; the O5 brief
  swapped the letters), g = erf(x/sqrt 2), eps = (sigma - tau)^2/2, eta/N, alpha = examples/N; BS's Q is |J|
  and their beta = 1 two-unit model is K = 2, M = 1; BRW use SS's convention. The symmetric manifold has two
  branches: identical students (Q = C; BRW Eq. (20), Fig. 2) and permutation-symmetric (Q > C; BRW Eq. (21),
  Fig. 1, and SS Fig. 1's plateau, Q = Q* + Delta); they merge as eta -> 0 (SS's truncated point). The tau of
  SS NIPS p. 307 has (2K - 1)^(+1/2) (read at 300 dpi) and is in units of 1/eta. SS NIPS p. 306 writes "the
  increase in the height of the plateau with decreasing eta"; their Fig. 1d and the ODE show the height
  increasing with eta (recorded, not banked). SS's eta_opt condition lambda1 = 2 lambda2 (no number printed
  in the sources read) is computed, not banked: it only balances the two decay exponents (SS p. 307's
  inequality "2 lambda_2 < lambda_1" is reversed for the negative eigenvalues of their Fig. 2), and it is the
  fastest asymptotic decay max(lambda1, 2 lambda2) only where lambda1 is non-decreasing at the crossing: true
  for K = 1 (BS's 2 eta_c/3), false for K = 2, 3, 4, 5, 8 (crossing -> fastest: 1.275 -> 1.881, 0.917 ->
  1.280, 0.700 -> 0.970, 0.564 -> 0.780, 0.355 -> 0.492); the full Jacobian (15-D at K = 3: the leading
  linear-term mode is lambda1, the leading mode overall lambda2) confirms the restricted analysis. SS's Eq. (9)
  simplifies: (75 - 42 sqrt3)/(25 sqrt3 - 42) = sqrt3, so eta_max ~ pi sqrt3 / K. "Optimal" rates
  are optimal only in `scm_flow.OPTIMALITY_CLASS` (fixed eta, asymptotic decay, K = M, isotropic). Plateaus:
  BRW's definition with B = 0.05 (the entry alpha is replicant's choice; BRW's alpha_0 is arbitrary, so only
  differences of escape alphas are meaningful); `undetermined` when not left by alpha_max, when the escape is
  seeded below round-off (initial symmetry breaking < 1e-13, e.g. an exactly symmetric start, which still
  leaves through floating-point error: K = 3, eta = 0.5 from (0.3 I, 0) "escapes" at alpha = 1683), or when the
  norms run away (max Q_ii > 3 max T_nn and growing, e.g. K = 3 at eta = 1.5 from SS's start, near eta_max);
  symmetric roots no better than the zero student or with Q > 3 max T_nn are noted as odd (K = 3, eta = 1.5:
  the identical point has eps 0.512 > 0.5); the escape alpha is recomputed at
  rtol x 1e-2 (spreads measured: 6e-8 to 5e-7 on BRW's runs, 0.02-0.22, i.e. <= 1e-3 tau_esc, on SS's
  K = 3 runs at eta = 0.05-0.2; even rtol 1e-2 moves the BRW escape by only 0.016: the explicit R asymmetry,
  not the integrator, drives it). BRW's Eq. (17) half-log law is reproduced (slope -0.4995 tau_esc
  per ln X).
- families/sk.py's float64 1RSB functional (`_rsb_phi_e`, 80 x 80 Gauss-Hermite; v1 code, unchanged) is
  inaccurate in the q0 = 0 / large q1 - q0 sector, where the branch points of cosh^m (zeros of cosh at
  Im = pi/2) are not smoothed by an outer Gaussian average. Against the certified balls of bounds/krsb.py:
  at the solver's own q0 = 0 candidate (the h = 0 seed of `_rsb`, sk.py:224) the error is 2.8e-9 at beta
  = 2, 2.8e-7 at beta = 2.5 and 3.8e-6 at beta = 3, and 6.6e-10 at (q0 = 0, q1 = 3/5, m = 1/2, beta = 2)
  (1.5962745995322627 vs 1.596274598870465316814...). No v1 result is affected: that candidate loses the
  largest-f selection by 1.9e-4, 6.2e-4, 1.3e-3 in -beta f at beta = 2, 2.5, 3, far above those errors.
  sk.py's docstring claim "validated against 160 x 160 nodes" (to ~1e-11 at beta = 2) holds at the
  selected optimum (where the certified ball agrees to 1.9e-15), not in that sector. (A v1 follow-up.)
- Monte Carlo and enumeration (E11, `numerics/`). Statistical, never certified, and not a catalog
  route: they serve the held-out tests. The +-J spectrum is exact (integer counts; the cross term is a
  float64 product of integers below 2^53); the perceptron counts use float64 Gaussian fields (a tie
  J.xi = 0, probability zero, counts as not stored). PT ground states: two independent runs per sample
  must agree (evidence, not proof; checked against exact enumeration at N = 12 in the default tier and
  for all 1024 samples at N = 16); the default 12N sweeps were raised from 6N after the agreement rule
  fired at N = 128. Equilibration is a 99% window-drift t-test, so about 1% of passing cases are
  refused by chance (it fails closed); round trips are reported, not gated. The PT exchange rule
  (Hukushima-Nemoto 1996, not fetched) is validated numerically against exact enumeration, not by the
  paper. Finite-size exponents: omega = 2/3 is supported for ground states by Boettcher 2005 (+-J,
  0.672(5)) and Palassini 2003 (Gaussian, 0.673 +- 0.002 with subleading corrections visible); at finite
  T deep in the ordered phase no exponent is known (Palassini), so the thermal band adds the omega = 3/4
  and omega = 1 shifts; omega = 3/4 is a stress test, not a literature exponent. The dropped-N and
  quadratic variants with four sizes are volatile and dominate several systematics (deliberately not
  narrowed). The moment closure's error is not bounded; |order 2 - order 3| is only a truncation
  estimate. Limits: SK enumeration N <= 26 (N = 26 in 0.07-0.1 s), perceptron N <= 30 (N = 24 about
  0.1 s per pattern set, N = 28 about 1.5 s); the held-out perceptron route stops at N = 24 for the
  test budget.
- Banked values often come from secondary open sources where the original is paywalled; each bank
  entry names the source its digits were read from (`reference/*.json`, field `cite`), and where the
  cite says the original was not fetched the digits come from the secondary source named there. Check
  the originals by hand before quoting a digit.

RELATED: `tools/annihilator` (used by `series/` through its documented surface: `pf_from_series`,
`nullspace_mod_fast`, `rec_to_theta`); `tools/baller` (used by `lattice/` through its documented front
door: `verify`, `solve`, `run`, `render`, `certified_digits`, `hygiene.ctx_guard`; needs python-flint), and by `bounds/` the same way (`solve` over `run`, with
`start_dps`, and `hygiene.ctx_guard`).
Elsewhere arbitrary precision is plain mpmath (no ball enclosures).

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
[KMRSZ07], the first-moment bound [FP83] and the proof for large K [DSS22]. LDPC codes on the BEC (`inference/`): density evolution and the peeling decoder [LMSS01] [MM09], the Maxwell construction and the MAP threshold [MMU08] [KRU10] [RU08], tabulated thresholds [KRU12] [Pfi15]. Spiked Wigner and the block model (`inference/`): the Bayes-optimal limits and state evolution [LeMi19] [MV21] [DAM15] [BDMKLZ16], the spectral threshold [FP07] [BGN11], the block-model BP and thresholds [DKMZ11] [MNS15] [Mas14] [MNS18]. Ridge regression (`learning/`): [HMRT22] [DW18] [AS20] [LGCGKMZ21] with [SS07] [TV04], context [BLLT20]. Exact series (`series/`): Onsager's integral [Ons44] [KW41] as printed by [DC22] [Bax10], the coefficient table [Fin99] [HM87], the hypergeometric identities [DLMF], the susceptibility series and its natural boundary [A002906] [Nic99] [ONGP01], the non-holonomic control [FGS05]. The universality of the SK free energy used in the held-out test is [CH04]. Solvable lattice models (`lattice/`): Onsager's free energy [Ons44] and Yang's magnetization [Yan52] as printed by [DC22], the critical constant [A242743], Kaufman's torus [Kau49] as printed by [IH02] [Kas06], the Bethe-lattice ferromagnet [DM10], the 1D transfer matrix [WDC19] [Bax82] [Wiki-Ising]; ball arithmetic is Arb [Joh17] through tools/baller. Bounds on the SK
pressure (`bounds/`): the broken-replica bound [Gue03], the high-temperature identity [Gue92] [ALR87]. Online learning (`online/`): the perceptron learning curves [BR94], the variational optimal modulation [KC92] as restated by [VKC98] and [BR94], output-noise robustness [CEKBSRC97], the linear perceptron online versus offline [KC95], drifting rules [VKC98] [KC93], the tree committee machine [CC95] [CKC96], the soft committee machine [SS95a] [SS95b] (read through [SS96] [SS97]) [BS95] [BRW96]. Monte Carlo and enumeration (`numerics/`): the finite-size exponent and the banked finite-N ground energies [Boe05] [Pal03], the exchange rule of parallel tempering [HN96]. Reference digits are read
from [Sto25] [OSS07] [CRT03] [SY17] [DM15] [FP16] [Gyo01] [PRGE20] [Cov65] [Bar97] as stated in the bank
cites. The typed-step architecture, the symbolic level (a), the composer, the refusal and
finality discipline and the LaTeX writer are ours. Each bank entry names the source its digits were
read from; where a cite says the original was not fetched, the digits come from the secondary source
named there, and the REFERENCES.md note preserves that.
