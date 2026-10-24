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
published pipelines of a catalog of 7 entries, each with a replica route and/or a cavity route. A composer
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
- Dynamics, TAP, DMFT, or quantum models (excluded from v1; TAP is explicitly out).
- Automatic discovery of the RSB structure: the ansatz is an input; RS stability is computed or reported
  as unknown, and there is no search for the right ansatz.
- A model with no level-(a) hook (families: `sk`, `perceptron`, `hopfield`; variables `ising`,
  `spherical`; disorder `gaussian`, `pm_one`): the composer refuses (`missing-step:*`).

INVOKE:
CLI (from `tools/`):
`python3 -m replicant list` prints the catalog (entry, routes, citation).
`python3 -m replicant run <entry|spec.json> [--route replica|cavity] [--param k=v ...] [--out result.json]
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
Python API (put `tools/` on `sys.path`):
`from replicant.catalog import catalog` (dict of `CatalogEntry`);
`from replicant.compose import compose, nearest_entry`; `compose(spec, route).run(require_final=False,
**params) -> Result`; `from replicant.spec import ModelSpec`;
`from replicant.solve_derived import solve_derived_rs` (solve the DERIVED RS equations of a spec at a
state point: all roots found from seeded searches, an audit residual, a computed replicon where derived);
`from replicant.latex import render_latex, write_latex` (`write_latex(result, path, pdf=False)`; a
`Refusal` is also accepted and writes its partial derivation).

INPUTS: a catalog entry name or a spec.json (above); run parameters by entry (exactly the names each
route reads, `CatalogEntry.parameters[route]`; any other name, e.g. a typo `kapa`, is refused
`precondition:known-parameter` with the accepted names, never ignored):
`rem`: `beta`, `N_scale` (default 1/2, Derrida's normalization); `sk` replica: `beta`, `h`, `ansatz`
(`rs`|`1rsb`, at a `beta`), `fullrsb` (flag, slow tier); `sk` cavity: `beta`, `h`, `require_stable`;
`pspin`: `p` (default 3), `beta`; `perceptron` replica: `alpha`, `kappa` (default 0); `perceptron`
cavity: `alpha`, `kappa`, `variables` (`spherical`; `ising` is refused `not-implemented:cavity-ising`);
`teacher_student`: `alpha`, `learning` (`gibbs`|`bayes`); `hopfield` (both routes): `alpha`, `beta`,
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
  means UNKNOWN, not "fine" (a composed Ising perceptron or Hopfield RS value has no derived replicon).
- CHECKS: `references` compares the entry's non-slow banked values within their stated tolerances;
  statistical keys use a 99% Student-t band on hypot(stderr, tol); slow and statistical comparisons
  (`references_skipped_slow`) run only through `Result.reference_checks(slow=True)` or `--full`.
- REFUSAL REASONS (`kind:detail`; all raised by the code; `Refusal.trace` carries the partial trace):

| reason | when |
|---|---|
| `missing-step:family:<f>`, `:site:<v>`, `:disorder:<d>`, `:constraint:<c>`, `:prior:<p>` | composer: the level-(a) hook the spec needs does not exist (the detail names what exists and the nearest catalog entry) |
| `missing-step:route:<r>` | unknown route, entry without that route, or cavity for a spec outside the catalog (no generic cavity composer) |
| `missing-step:derivation:<name>` | a requested document needs a derivation the entry does not provide |
| `precondition:<name>` | an input or state precondition fails (state point incomplete: `state-point`; out-of-range or non-numeric input; `known-parameter`, `rs-reducible`, `rs-site-trace-form`, `supported-disorder-law`, `nonnegative-alpha`, `finite-kappa`, `known-ansatz`, `known-graph`, `tex-engine`, `latex-path`, `known-target`, `complete-spec`, ... and the step preconditions of the derivation) |
| `unstable:<ansatz>` | `require_final` or `require_stable`: a computed stability is negative (RS replicon or cavity lambda < 0) |
| `unknown-stability:rs` | `require_final`: an RS value whose stability is not computed |
| `not-final:<ansatz>` | `require_final`: a non-RS value that is not final |
| `no-solution`, `no-solution:audit`, `:solver`, `:bracket`, `:retrieval`, `:retrieval-fold`, `:paramagnet`, `:spinglass`, `:no-maximum`, `:negative-entropy`, `:rs-t0`, `:graph`, `:estimator-mismatch` | no saddle/root in the searched range, or a found one fails its audit (residual, non-finite output, divergence) |
| `ambiguous:several-saddles` | more than one RS saddle found and no branch chosen: name one with `--param branch=...` |
| `no-convergence`, `no-convergence:equilibration` | BP or a survey did not converge; a population-dynamics run is still drifting and `require_equilibrated` is set |
| `no-transition` | the branching number rules out a spin-glass transition |
| `unsupported:degree-law`, `unsupported:J-law` | a graph/coupling law the sparse code does not implement |
| `not-implemented:annealed`, `:cavity-ising`, `:spherical-rsb` | the named case is outside v1 |
| `unbanked:missing-entry`, `:malformed`, `:no-citation`, `:no-independent-route` | a banked reference value is absent or has no citation / no independent route (a value that fed a fit never certifies it) |
| `tex-engine:failed`, `tex-engine:unavailable` | `--pdf`: the engine ran and failed, or could not reach its offline resources |

CATALOG (v1; `list` shows the live table; values at the default / banked state points):

| entry | routes | headline values | banked references | finality |
|---|---|---|---|---|
| `rem` random energy model | replica | `beta_c` 1.66511 (`T_c` 0.600561, N_scale 1/2) | `T_c`, frozen ground energy, 1RSB `m beta/beta_c = 1` (Derrida) | exact-by-literature (1RSB) |
| `sk` Sherrington-Kirkpatrick | replica, cavity | `tc` 1; RS `-sqrt(2/pi)` = -0.797885; 1RSB -0.765259; at `beta`: RS `q, f, e, replicon` | `tc`, RS, 1RSB; slow tier: 2RSB -0.76361, 3RSB -0.76331, full-RSB -0.763166 | `tc` stable; RS ground energy unstable; 1RSB not-computed; full RSB exact-by-literature (slow) |
| `pspin` spherical p-spin | replica | p = 3: `T_s` 0.586054, `T_d` 0.612372, `q_s` 0.645007 | `T_s`, `T_d`, `q_s`; slow: `e0` -1.172 | exact-by-literature (1RSB) |
| `perceptron` spherical (Gardner) | replica, cavity | `alpha_c(kappa=0)` = 2; entropy `S` and its `replicon` at `alpha` | `alpha_c`, both routes | `alpha_c` stable for kappa >= 0 (RS exact), unstable and not final for kappa < 0; `S` carries its own computed replicon label. The entry is RS only: `--param ansatz=1rsb` is refused `precondition:known-parameter`; in Python `gardner_capacity("spherical", ansatz="1rsb")` is refused `not-implemented:spherical-rsb` |
| `teacher_student` | replica | Gibbs 0.624658 / Bayes 0.441700 asymptotic constants (`eps ~ c/alpha`) | both constants (bank tol is a 3-digit rounding) | exact-by-literature |
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

ACCEPTANCE GATES: `python3 tools/replicant/selftest.py` (default tier: 247 tests, about 17 s on a
laptop; also `python3 run_selftests.py replicant`) covers the core and refusal paths, the symbolic
derivation against independent expressions, level agreement (a)=(b) for every family where both exist,
every catalog entry against its banked non-slow values, the cavity routes, the composer and the held-out
models, the CLI and the LaTeX writer (displayed equations re-checked numerically). `--full` adds 68
`@slow` tests (full-RSB Parisi up to K = 9, population dynamics at reference sizes, the Hopfield replicon
scans, the p-spin T -> 0 energy, solver-based composed and LaTeX checks, a PDF compile when a TeX engine
is present) and takes about 6-7 minutes on a laptop (measured 6 min 42 s); run it before merging a
change to a family. The `@slow` marker is honoured by
`selftest.py` only: pytest ignores it and would run everything. Level agreement between the derivation
and the pipelines is a consistency check, not new evidence for the ansatz; the cavity route reproducing
the replica RS equations is an independent derivation but NOT independent evidence for the RS ansatz.

KNOWN HAZARDS:
- Rigor: PHYSICS-LEVEL, not proven. `rigor` says so on every result; the n -> 0 continuation and the
  RS/RSB ansaetze are assumptions. Do not quote a result as a theorem.
- An RS answer whose stability is NOT computed (`stable: null`, `not-computed`) is NOT final. `stable:
  null` means UNKNOWN. Level (a) derives the replicon only for the SK site; for composed perceptrons it
  is reported unknown (False only if the discrete-variable entropy is negative), and `final` is false.
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
  alpha ~ 1.015, which agrees with the commonly quoted Gardner-Derrida value but is unsourced here and
  NOT banked; Gyorgyi's "AT-stable up to 4/pi" is believed to be a slip.
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
energy model [Der80] [Der81]; the spherical p-spin [CS92] with [CC05] [CLN17]; the Gardner capacity [Gar88]
[GD88], the Ising perceptron thresholds [KM89] [DS18] and the cavity treatment [Mez89]; teacher-student
learning [SST92] [OH91] [Gyo90] [BTG97] and its proof [BKMMZ19]; Hopfield model [AGS85] [AGS87] with the
RSB analysis of [SK94]; the Viana-Bray model [VB85], the sparse cavity and population-dynamics method
[MP01] [MP03] and the transition conditions [MK04] [GT03]; the dense cavity method [MPV87] with
[TAP77]. The universality of the SK free energy used in the held-out test is [CH04]. Reference digits are read
from [Sto25] [OSS07] [CRT03] [SY17] [DM15] [FP16] [Gyo01] [PRGE20] [Cov65] [Bar97] as stated in the bank
cites. The typed-step architecture, the symbolic level (a), the composer, the refusal and
finality discipline and the LaTeX writer are ours. Each bank entry names the source its digits were
read from; where a cite says the original was not fetched, the digits come from the secondary source
named there, and the REFERENCES.md note preserves that.
