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
published pipelines of a catalog of 8 entries, each with a replica route and/or a cavity route (and an
`amp` route for `sk` and `teacher_student`). A composer
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
`bary_samples`. Without `beta` an entry returns its T = 0 / critical headline values. A composed spec
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
| `unstable:<ansatz>` | `require_final` or `require_stable`: a computed stability is negative (RS replicon or cavity lambda < 0); `unstable:1rsb` also from `pspin_gardner_temperature` when the spherical scan is negative everywhere |
| `unknown-stability:rs` | `require_final`: an RS value whose stability is not computed |
| `not-final:<ansatz>` | `require_final`: a non-RS value that is not final |
| `no-solution`, `no-solution:audit`, `:solver`, `:bracket`, `:retrieval`, `:retrieval-fold`, `:paramagnet`, `:spinglass`, `:no-maximum`, `:negative-entropy`, `:rs-t0`, `:graph`, `:estimator-mismatch` | no saddle/root in the searched range, or a found one fails its audit (residual, non-finite output, divergence) |
| `ambiguous:several-saddles` | more than one RS saddle found and no branch chosen: name one with `--param branch=...` |
| `no-convergence`, `no-convergence:equilibration`, `:tap` | BP or a survey did not converge; a population-dynamics run is still drifting and `require_equilibrated` is set; `amp.tap_fixed_point`: Newton from the AMP iterate left the TAP residual above tol |
| `no-transition` | the branching number rules out a spin-glass transition |
| `no-transition:gardner` | `pspin_gardner_temperature`: the pure spherical p-spin (no Gardner transition; the detail carries the computed replicon scan), or no sign change of the Ising static-line replicon on the scan |
| `unsupported:degree-law`, `unsupported:J-law` | a graph/coupling law the sparse code does not implement |
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

ACCEPTANCE GATES: `python3 tools/replicant/selftest.py` (default tier: 306 tests, about 21 s on a
laptop; also `python3 run_selftests.py replicant`) covers the core and refusal paths, the symbolic
derivation against independent expressions, level agreement (a)=(b) for every family where both exist,
every catalog entry against its banked non-slow values, the cavity routes, the composer and the held-out
models, the CLI and the LaTeX writer (displayed equations re-checked numerically), the 1RSB replicon
and the Gardner transition. `--full` adds 103 `@slow` tests (full-RSB Parisi up to K = 9, population dynamics at reference sizes, the Hopfield replicon
scans, the p-spin T -> 0 energy, solver-based composed and LaTeX checks, a PDF compile when a TeX engine
is present) and takes about 9-10 minutes on a laptop (measured 9 min 28 s); run it before merging a
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
- Banked values often come from secondary open sources where the original is paywalled; each bank
  entry names the source its digits were read from (`reference/*.json`, field `cite`), and where the
  cite says the original was not fetched the digits come from the secondary source named there. Check
  the originals by hand before quoting a digit.

RELATED: none upstream or downstream in this repository. Arbitrary precision is plain mpmath (no ball
enclosures; `baller` is not used).

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
[Ran11] [BKMMZ19] with [JM13], the binary teacher-student thresholds [Gyo90] [BKMMZ19] [LHZH22]. The universality of the SK free energy used in the held-out test is [CH04]. Reference digits are read
from [Sto25] [OSS07] [CRT03] [SY17] [DM15] [FP16] [Gyo01] [PRGE20] [Cov65] [Bar97] as stated in the bank
cites. The typed-step architecture, the symbolic level (a), the composer, the refusal and
finality discipline and the LaTeX writer are ours. Each bank entry names the source its digits were
read from; where a cite says the original was not fetched, the digits come from the secondary source
named there, and the REFERENCES.md note preserves that.
