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
