# replicant

Published replica-method and cavity-method calculations of disordered systems as typed,
recorded steps. Each step names its preconditions and its citation; a missing precondition
is a named refusal (`precondition:<name>`, `missing-step:<hook>`, ...), never a silent
fallback; every run returns a trace, a per-value finality label (`stable`, `unstable`,
`not-computed`, `exact-by-literature`) and a comparison with banked, cited reference values.
Results are physics-level (the n -> 0 continuation and the RS/RSB ansaetze are not proved), except
the four entries certified in ball arithmetic (`ising2d`, `ising1d`, `bethe_ising`, `sk_bounds`);
population-dynamics, AMP and Monte Carlo results are statistical. Full documentation: [GUIDE.md](GUIDE.md).

Requirements: Python 3 with `sympy`, `mpmath`, `numpy`, `scipy` and `python-flint` (the certified
entries go through `tools/baller`). Run the battery from the project venv (INSTALL.md: `.venv`, which
has python-flint): without python-flint the lattice and bounds tests fail loudly
(`unsupported:python-flint`), by design.

## Quick start

From the repository root:

```
python3 tools/replicant/selftest.py                        # default battery, about 22 s
cd tools && python3 -m replicant list                      # the 19 catalog entries
python3 -m replicant run perceptron --param alpha=1.5 --out /tmp/result.json --latex /tmp/p.tex
```

`run` also takes a spec file for a model near a catalogued one
(`python3 -m replicant run spec.json --param beta=1.5 --param h=0`; schema in the GUIDE),
`--route amp` for `sk`, `teacher_student` and `spiked_wigner` (state evolution, and finite-N AMP / GAMP with
`--param N=...`), `--route cavity` for the entries that have a cavity route (required for `viana_bray`, `xorsat`,
`ksat`, `ldpc_bec`, `sbm` and `bethe_ising`, which are cavity-only: `python3 -m replicant run viana_bray --route
cavity`, `python3 -m replicant run xorsat --route cavity`), `--route exact` for `ising2d` and `ising1d`, `--route
series` for `series_ising2d`, `--route bound` for `sk_bounds`, and `--pdf` (with `--latex`) to compile the derivation.

## Layout

- `core.py`, `spec.py`, `testing.py`: steps, state, trace, `Refusal`, `ModelSpec`, the `@slow` marker.
- `refdata.py`, `reference/*.json`: the reference store, banked cited values (schema in
  `reference/_schema.md`).
- `derive*.py`, `gauss.py`, `ansatz.py`, `agree.py`, `solve.py`, `solve_derived.py`: level (a),
  the symbolic derivation (replicate, average, order parameters, RS ansatz, n -> 0,
  stationarity), the derived replicon (`derive_replicon*.py`: SK, perceptrons, Hopfield), the
  solver for the derived equations, and the 1RSB functional (`derive_rsb*.py`, `gauss_rsb.py`,
  `solve_derived_rsb.py`, `_rsb_num.py`; composer route `compose_rsb.py`, `--param ansatz=1rsb`).
- `families/`: level (b), the published pipelines (SK, REM, spherical and Ising p-spin,
  perceptron, teacher-student, Hopfield, Parisi full RSB); `families/gardner.py`: the 1RSB replicon
  (stability toward further RSB, the Gardner transition) for the two p-spin entries.
- `cavity/`: dense cavity, sparse BP / population dynamics, sparse 1RSB.
- `amp/`: AMP, state evolution and the TAP free energy for SK, GAMP and its state evolution for the
  teacher-student perceptron (finite N, statistical; independent of the replica and cavity code);
  catalog route `amp` (`catalog_amp.py`, `latex_amp.py`) for `sk` and `teacher_student`.
- `csp/`: random K-XORSAT (E5a): leaf-removal / zero-temperature cavity closed forms (`alpha_d`,
  `alpha_s`, 2-core, complexity) through a degree-law hook, and independent finite-N checks (exact GF(2)
  elimination, leaf removal on sampled hypergraphs); catalog entry `xorsat` (`catalog_csp.py`,
  `latex_csp.py`). Random K-SAT (E5b): `csp/ksat.py` (instances, exhaustive counts, the first-moment
  bound, BP) and `csp/sp.py` (survey propagation on instances, the SP population dynamics and the
  root of its complexity); catalog entry `ksat`.
- `inference/`: regular (l, r) LDPC codes on the binary erasure channel (E6a): density evolution,
  the BP threshold, the MAP threshold (Maxwell construction: trial-entropy root and area theorem)
  through a degree-polynomial hook, and independent finite-N checks (configuration-model graphs, the
  peeling decoder); catalog entry `ldpc_bec` (`catalog_inference.py`, `latex_inference.py`).
  E6b: the rank-one spiked Wigner model (`inference/spiked.py`: Bayes-optimal state evolution, the
  threshold, Bayes AMP with spectral start) and the two-group stochastic block model
  (`inference/sbm.py`: Kesten-Stigum line, sampler, BP with the external field); catalog entries
  `spiked_wigner` (amp) and `sbm` (cavity) (`catalog_spiked_sbm.py`, `latex_spiked_sbm.py`).
- `learning/` (E7): ridge and ridgeless least squares on isotropic Gaussian design (`ridge.py`: the
  Marchenko-Pastur closed forms of Hastie-Montanari-Rosset-Tibshirani, the optimal penalty;
  `ridge_rs.py`: the RS saddle point of the linear student, a cited closed form solved here;
  `ridge_sim.py`: SVD fits over named seeds, extrapolated in 1/n); catalog entry `ridge` (replica)
  (`catalog_learning.py`, `latex_learning.py`).
- `series/` (E8): the exact high-temperature series of the 2D Ising free energy from the Onsager
  integral (`ising2d.py`, with the independent L x L torus route), holonomic recognition through
  `tools/annihilator` (`recognize.py`), exact local exponents (`local.py`), the elliptic operator
  identity (`elliptic.py`), the banked susceptibility series (`susceptibility.py`); catalog entry
  `series_ising2d` (route `series`) (`catalog_series.py`, `latex_series.py`).
- `lattice/` (E9): solvable lattice models certified in ball arithmetic (python-flint arb through
  `tools/baller`: `ball.py`): Onsager's free energy and Yang's magnetization (`onsager.py`), Kaufman's
  finite torus and its integer polynomial recovered from balls (`kaufman.py`), the exact integer
  transfer matrix and the 1D ring (`transfer.py`), every BP fixed point of the Bethe-lattice
  ferromagnet (`bethe.py`); catalog entries `ising2d`, `ising1d` (route `exact`) and `bethe_ising`
  (route `cavity`), rigor "certified" (`catalog_lattice.py`, `latex_lattice.py`).
- `bounds/` (E10): Guerra's rigorous upper bounds on the SK pressure, certified in ball arithmetic
  (python-flint arb through `tools/baller`): certified Gaussian expectations (`gauss.py`), the K-step
  broken-replica bound by nested `acb.integral` (`krsb.py`), the RS bound with all its stationary points
  and its certified minimum (`guerra.py`), the chain check of the package's own RS / 1RSB / K-RSB
  optimisers (`chain.py`); catalog entry `sk_bounds` (route `bound`), rigor "certified"
  (`catalog_bounds.py`, `latex_bounds.py`).
- `numerics/` (E11): Monte Carlo and exact enumeration as independent checks of the held-out models
  (not a catalog route): Gray-code enumeration of +-J SK spectra and Ising-perceptron solution counts
  (`enumeration.py`), parallel tempering with an equilibration diagnostic (`pt.py`), finite-size
  extrapolation (`finite_size.py`), exact finite-N moments and the quenched-annealed closure (`moments.py`);
  used by `tests/test_heldout_mc.py` through `tests/heldout_mc.py`.
- `catalog*.py`, `compose.py` (+ `compose_result.py`: `Result` and the finality rule), `cli.py`,
  `__main__.py`: the 19-entry catalog (`catalog_ts.py` holds the teacher-student entry,
  `catalog_pspin.py` the two p-spin entries; LaTeX in `latex_pspin.py`), the composer,
  the command line.
- `latex*.py`, `tex*.py`, `derive_tex.py`: the LaTeX derivation writer.
- `tests/`, `selftest.py`: the battery.

## Test tiers

- Default: `python3 tools/replicant/selftest.py` (425 tests, about 22 s; measured 21.4 s). Also
  `python3 run_selftests.py replicant` from the repository root, with the venv activated (it runs
  the `python3` on PATH; without python-flint the lattice and bounds tests fail, 31 of them).
- Full: `python3 tools/replicant/selftest.py --full` adds the 132 `@slow` tests (full RSB,
  population dynamics at reference sizes, the K-SAT and LDPC crossings, Monte Carlo and enumeration
  held-out routes, solver-based checks); about 18 minutes on a laptop (measured 18 min 11 s, 557 tests).
- The held-out tests are `tests/test_heldout*.py` (helpers `tests/heldout_common.py`, `tests/heldout_mc.py`).
- `@slow` is a plain function attribute honoured by `selftest.py` only. pytest ignores the
  marker and would run every test, slow ones included.
