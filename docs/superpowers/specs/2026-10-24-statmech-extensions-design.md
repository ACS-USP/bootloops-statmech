# Statistical-mechanics extensions to `replicant` — design

Status: approved by instruction ("Implemente as 11 sugestões na ordem em que estão enumeradas", 2026-10-24); the user waived the question phase. Defaults below are the controller's calls and are recorded as rulings in the execution ledger.
Parent design: `2026-10-23-replica-machinery-design.md` (v1, implemented).

## Intent

Extend the `replicant` package (`tools/replicant/`) with eleven techniques, in this order, each following the house discipline of v1: banked cited references (never from memory), a verification route independent of the code that produced the answer, refusal over invention, per-value finality (stable / unstable / not-computed / exact-by-literature), physics-level labelling (statistical or certified where it applies), catalog entry + LaTeX derivation where the item produces a catalog model, GUIDE/README/REFERENCES/CHANGELOG updates.

Success for the whole extension: (1) the two largest honesty gaps of v1 are closed (stability unknown for perceptrons/Hopfield; held-out evidence coarse); (2) the package covers the neighbouring problem classes (constraint satisfaction, inference, codes, high-dimensional learning); (3) the BootLoops-specific strength is used: exact series with holonomic recognition and certified ball-arithmetic evaluation.

## The eleven items

| # | Item | Deliverable | Independent checks (must be able to fail) |
|---|---|---|---|
| E1 | Replicon (AT) at level (a) for Gaussian-integral energies | `solve_derived_rs` returns computed `stable` for spherical and Ising perceptron and Hopfield (second variation of the RS functional in the replicon direction, produced by rules, not typed) | derived replicon = the families' replicon at ≥ 3 points; planted hook error changes it; composed Ising perceptron finality becomes computed |
| E2 | 1RSB at level (a) | derived 1RSB functional (block overlap matrices of `ansatz.py`) and `solve_derived_1rsb` for SK, Ising perceptron (frozen and full) and Hopfield; composer `ansatz=1rsb` route | SK T=0 −0.76526 and finite-T vs `sk_1rsb`; Ising perceptron frozen-1RSB zero of S at 0.8330786 (bank, target); Hopfield 1RSB vs a banked literature value if one exists |
| E3 | TAP, AMP and state evolution | `amp/` module: SK AMP (Bolthausen) with state evolution, SK TAP free energy, GAMP for the perceptron / teacher–student with state evolution; catalog route `amp`; independent route for held-out models | state-evolution fixed point = RS equations (derived); finite-N AMP over named seeds within a Student-t band; ±J SK composed q vs AMP on a ±J matrix |
| E4 | Gardner transition | 1RSB replicon (stability of 1RSB toward full RSB); `pspin_gardner_temperature(p)`; Ising perceptron 1RSB stability; p-spin finality becomes "final only above T_G" | banked T_G from the literature; scan of the computed replicon sign; planted error |
| E5 | Constraint satisfaction | XORSAT (BP, 1RSB thresholds) and K-SAT (BP + survey propagation, complexity); coloring is a stretch | XORSAT: finite-N GF(2) elimination (exact) and leaf-removal core; 3-SAT: banked SP thresholds, finite-N small-N exact counts vs annealed bound |
| E6 | Inference and codes | LDPC on the BEC (density evolution), spiked Wigner (Bayes-optimal state evolution, threshold), SBM (Kesten–Stigum, BP) | BEC: closed-form fixed point + finite-N peeling; spiked Wigner: AMP simulation + banked formula; SBM: BP on simulated graphs |
| E7 | High-dimensional learning | ridge / ridgeless regression, double descent (Marchenko–Pastur), teacher–student linear model; replica route and closed form | direct simulation with SVD over named seeds; closed-form ridgeless limits |
| E8 | Exact series + holonomic recognition | exact high-T series of a solvable model → `tools/annihilator` → ODE, singularities and exponents with certificate; impossibility certificate where applicable | recovered ODE verified on held-out coefficients beyond the fitting window; banked critical point and exponent |
| E9 | Solvable lattice models, certified | Onsager free energy, Yang magnetization, 1D Ising/Potts transfer matrices, Bethe lattice; evaluation with `tools/baller` ball arithmetic and stated error radii | exact finite-lattice partition functions (transfer matrix / Kac–Ward determinant, exact rationals) vs the finite-size formula |
| E10 | Guerra and Aizenman–Sims–Starr bounds | certified evaluation (ball arithmetic) of the Guerra RS/1RSB bounds on the SK free energy; chain-of-inequalities check on the package's own numbers | certified enclosures; the K-RSB staircase and the Parisi extrapolation must respect the bound chain; planted error violates it |
| E11 | Monte Carlo and larger exact enumeration | parallel tempering and exact enumeration (Gray code up to N≈26) with finite-size extrapolation; used to strengthen the held-out checks (±J SK energy; Ising perceptron entropy) | finite-size scaling with stated bands; comparison with composed RS/1RSB numbers; the old coarse enumeration test is superseded |

## Goals

- Every new catalog model has: banked references (`reference/*.json` with `cite`, `independent_route`), a declared parameter set, a per-value finality declaration, a LaTeX derivation block, and a GUIDE row.
- New routes appear in `python3 -m replicant list` and are exercised by `python3 -m replicant run <entry> --route <route>`.
- Default battery growth is bounded: the whole extension adds ≲ 8 s to the default tier (17 s → ≲ 25 s); everything heavy is `@slow`.

## Non-goals

- Rigorous proofs beyond what an item states (E10 is certified numerics of a proven inequality, not a proof of the inequality).
- Quantum models; DMFT/real-time dynamics; automatic discovery of the RSB structure.
- Full coloring phase diagram (E5 stretch only).
- Nothing in the existing v1 behaviour may change except: finality labels that become computed (E1, E4) and held-out test strengthening (E3, E11).

## Anti-goals

- A banked number used as an input of the computation it checks.
- A check that cannot fail (every item ships a positive control).
- Silent downgrade of rigor labels; statistical results labelled certified; certified results without error radii.
- Growth of the default battery beyond the stated bound.

## Left to the builder

Internal representation; module names below the package; how each derivation is staged inside the existing step framework; algorithmic choices (as long as the independent checks hold).

## Calls made (controller defaults, user waived questions)

1. Work directly on `main` (the user's standing choice for this repository); no push until the user asks.
2. Hopfield 1RSB (E2) is included only if a banked literature value is found; otherwise it is derived and reported with "no independent target" in its finality note.
3. Coloring (E5) and the impossibility certificate for the 2D Ising susceptibility (E8) are stretch goals; if they do not fit at quality they are reported as not done.
4. E8 and E9 reuse `tools/annihilator`, `tools/ratfit` and `tools/baller` through their documented CLIs/APIs; no change to those packages.
5. Reference values come from papers fetched during the build; secondary sources are recorded as such.

## As built

Deviations of the implementation from the text above, taken from the build ledger
(`.superpowers/sdd/2026-10-24-statmech-extensions/progress.md`, not committed) and the task
reports. Commits 3f149c4 (this design) .. 288f133 (E11), then the documentation and cleanup task.

- **E1 (replicon at level (a)).** Built as specified for the spherical and Ising perceptron and
  Hopfield. For Hopfield the stability label also needs the m-mode, computed as a Schur complement
  of the Hessian of the derived functional; it has no error estimate (the quadrature floor is
  amplified through a near-singular block: 7e-6 relative at 0.9 m_fold, sign robust). The decoupling
  of the m_a from the replicon sector is argued, not mechanised; only replicon and m-modes are
  computed (no longitudinal / anomalous q-modes), so `stable` is local. The spherical site is
  projected at explicit n = 4 (n = 5 checked in the slow tier). One v1 test that asserted "NOT final"
  for the Ising perceptron at alpha = 0.5 changed, as the spec allows. Default tier +0.75 s.
- **E2 (1RSB at level (a)).** SK (T = 0 and finite T), the frozen Ising perceptron (Krauth-Mezard
  capacity reproduced) and Hopfield are derived and solved. The full (q1 < 1) Ising perceptron gives
  only roots that collapse to RS below the frozen capacity, so no full-1RSB capacity is claimed.
  Hopfield 1RSB has no independent target (call made 2): the T = 0 limit is implemented for
  SK-type energies only, so the Steffan-Kuehn T = 0 value is not banked. The root search is seeded;
  the q0 = 0 edge is searched, the x -> 0 and q1 -> 1 edges are not; SK at h = 0 has two nontrivial
  saddles and refuses without a branch. The T = 0 local-maximum label rests on a sign agreement
  checked at finite beta. Every 1RSB value is `not-computed`, never final. The SK solve takes about
  5 s per point.
- **E3 (TAP, AMP, state evolution).** Split in two tasks (E3a SK, E3b GAMP and the catalog route).
  The TAP free energy and its fixed points are an API, not a catalog route. GAMP's state evolution
  for GLMs is a claim (Barbier et al. 2019, Claim 1), so the `teacher_student` amp values are
  `not-computed`; only SK's SE on Gaussian J is `exact-by-literature`. The binary-prior alpha_IT
  sources disagree by 0.004 and both are covered by the tolerance. Universality for +-J is
  numerical evidence (the cited universality results were read from abstracts).
- **E4 (Gardner transition).** The 1RSB replicon is implemented as cited (Crisanti-Leuzzi 2006,
  Montanari-Ricci-Tersenghi 2003), not produced by the E1/E2 rules (neither p-spin is a level-(a)
  family); it is checked against an explicit integer-n functional at integer m only. A new Ising
  p-spin family and the catalog entry `ising_pspin` were added (T_G = 0.240260 against Rizzo 2013).
  The spherical pure p-spin has no Gardner transition (refused with the computed scan), so its 1RSB
  values became computed `stable`. The frozen Ising-perceptron 1RSB stability is refused
  (`not-implemented:frozen-1rsb-replicon`). Full RSB below T_G is not computed.
- **E5 (constraint satisfaction).** XORSAT and K-SAT built as specified. The 3-coloring stretch was
  not done (call made 3). No K-SAT threshold is labelled final: Ding-Sly-Sun prove the 1RSB formula
  only for an unspecified large k. The SP-population root 4.2623 +- 0.0108 contains Mertens-Mezard-
  Zecchina's 4.26675 but cannot separate it from values a few 1e-3 away. The K = 6 XORSAT alpha_d is
  not banked (one source, a 1.1e-6 discrepancy).
- **E6 (inference and codes).** LDPC on the BEC, spiked Wigner and the SBM built as specified. The
  MAP-threshold equality is proved for (3, 6) in the cited paper and cited from Kudekar-Richardson-
  Urbanke Lemma 4 for the other pairs (their proof is in a book not fetched). A misprint in
  Measson-Montanari-Urbanke Example 3 was found and that value not banked. The spiked-Wigner
  conventions differ from Lelarge-Miolane (lambda_LM = lambda^2) and four quantities are reported,
  not one "MMSE". BP for a disassortative block model is refused. The controller's briefs gave a wrong arXiv
  identifier twice (Montanari-Ricci-Tersenghi in E4, the Maxwell-construction paper in E6a); both
  were corrected from the sources.
- **E7 (high-dimensional learning).** The replica route is a cited closed form (the RS saddle point
  of Loureiro et al. 2022, Eq. (A.38), with replicant's specialisation of the label-noise term), not
  a level-(a) derivation: the formal layer has no family with a teacher replica and a quadratic loss.
  Bank values are formula evaluations at chosen points (the papers print formulas). The simulation
  extrapolates in 1/n from two sizes on disjoint seed sets.
- **E8 (exact series, holonomic recognition).** The operator found is of theta-order 3 and degree 8
  and is inhomogeneous (L g = P); the "order 2, elliptic" expectation holds for its right factor
  (L = M theta, M the pulled-back hypergeometric operator of K(k), proved exactly). The operator
  values are `not-computed` (proved identity and minimality within bounds, not a computed
  stability). The impossibility stretch was done as a bound only: the susceptibility has no operator
  of theta-order <= 6 and degree <= 40, which is not a proof of non-D-finiteness.
- **E9 (solvable lattice models, certified).** Ball arithmetic through `tools/baller` (python-flint)
  as planned. A new `CatalogEntry.rigor` value "certified" marks `ising2d`, `ising1d` and
  `bethe_ising`; every other entry is unchanged. The finite-lattice check uses the exact integer
  transfer matrix (no Kac-Ward determinant). The Bethe fixed points are certified by a self-written
  interval Newton (baller's certify engines do not apply). The Bethe beta_c is `not-computed` (a
  physics prediction in the cited paper). The lattice tests fail loudly without python-flint (ruling:
  the batteries run in the project venv).
- **E10 (bounds).** Guerra's RS, 1RSB and 2RSB bounds are certified; levels K >= 3 enter the chain
  as numerical values (the nested certified integral grows as nodes^(K+1)). The Aizenman-Sims-Starr
  variational principle was not implemented (Guerra's bound only). The chain is a consistency check
  of the package's optimisers against the theorem's ordering of the infima, not a theorem. The h = 0
  RS minimiser is certified by sign covers of G' except at beta = 1 exactly, which rests on a stated
  lemma.
- **E11 (Monte Carlo and enumeration).** Built as specified, with the perceptron counts stopping at
  N = 24 (not 26-30) for the test budget. Two contrasts are not resolved at the budget and are
  printed, not asserted: 1RSB against Parisi at T = 0 and "1RSB closer than RS" at beta = 2; the
  beta = 0.9 moment closure is not asserted. The v1 coarse enumeration test is kept, not superseded:
  the new routes are added beside it.
- **Known issue found (v1 code, not fixed).** `families/sk.py` `_rsb_phi_e` (float64 Gauss-Hermite)
  is inaccurate in the q0 = 0 / large q1 - q0 sector (up to 3.8e-6 at beta = 3 against the E10
  certified balls). No v1 result is affected: that candidate loses the largest-f selection by
  1.9e-4 .. 1.3e-3. Documented in the GUIDE hazards; a follow-up.
- **Cleanup (documentation task).** tests/test_heldout.py was split by model (test_heldout_sk.py,
  test_heldout_perceptron.py, test_heldout_mc.py, helpers in heldout_common.py) without changing an
  assertion. Several deferred review minors were fixed (a malformed `seeds` on the amp route is now a
  refusal, not a TypeError; the LDPC MAP cite and DE note; the spherical no-transition detail tied to
  its scan; an exact ball no longer counts as a radius stall; stronger enclosure, cache and planted
  tests); the rest are documented in the GUIDE hazards.
- **Time standard.** The default battery takes about 22 s (425 tests, measured 21.4 s; v1 was about 16 s, so the extension added about 5.4 s against the 8 s bound); the slow tier (`--full`, 557 tests) about 18 minutes (measured 18 min 11 s).
