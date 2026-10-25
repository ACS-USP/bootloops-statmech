# replicant

Published replica-method and cavity-method calculations of disordered systems as typed,
recorded steps. Each step names its preconditions and its citation; a missing precondition
is a named refusal (`precondition:<name>`, `missing-step:<hook>`, ...), never a silent
fallback; every run returns a trace, a per-value finality label (`stable`, `unstable`,
`not-computed`, `exact-by-literature`) and a comparison with banked, cited reference values.
Results are physics-level (the n -> 0 continuation and the RS/RSB ansaetze are not proved);
population-dynamics results are statistical. Full documentation: [GUIDE.md](GUIDE.md).

## Quick start

From the repository root:

```
python3 tools/replicant/selftest.py                        # default battery, about 16 s
cd tools && python3 -m replicant list                      # the 7 catalog entries
python3 -m replicant run perceptron --param alpha=1.5 --out /tmp/result.json --latex /tmp/p.tex
```

`run` also takes a spec file for a model near a catalogued one
(`python3 -m replicant run spec.json --param beta=1.5 --param h=0`; schema in the GUIDE),
`--route amp` for `sk` and `teacher_student` (state evolution, and finite-N AMP / GAMP with
`--param N=...`), `--route cavity` for the entries that have a cavity route (required for `viana_bray`, which is
cavity-only: `python3 -m replicant run viana_bray --route cavity`), and `--pdf` (with `--latex`) to
compile the derivation.

## Layout

- `core.py`, `spec.py`, `testing.py`: steps, state, trace, `Refusal`, `ModelSpec`, the `@slow` marker.
- `refdata.py`, `reference/*.json`: the reference store, banked cited values (schema in
  `reference/_schema.md`).
- `derive*.py`, `gauss.py`, `ansatz.py`, `agree.py`, `solve.py`, `solve_derived.py`: level (a),
  the symbolic derivation (replicate, average, order parameters, RS ansatz, n -> 0,
  stationarity), the derived replicon (`derive_replicon*.py`: SK, perceptrons, Hopfield), the
  solver for the derived equations, and the 1RSB functional (`derive_rsb*.py`, `gauss_rsb.py`,
  `solve_derived_rsb.py`, `_rsb_num.py`; composer route `compose_rsb.py`, `--param ansatz=1rsb`).
- `families/`: level (b), the published pipelines (SK, REM, p-spin, perceptron, teacher-student,
  Hopfield, Parisi full RSB).
- `cavity/`: dense cavity, sparse BP / population dynamics, sparse 1RSB.
- `amp/`: AMP, state evolution and the TAP free energy for SK, GAMP and its state evolution for the
  teacher-student perceptron (finite N, statistical; independent of the replica and cavity code);
  catalog route `amp` (`catalog_amp.py`, `latex_amp.py`) for `sk` and `teacher_student`.
- `catalog*.py`, `compose.py`, `cli.py`, `__main__.py`: the 7-entry catalog, the composer,
  the command line.
- `latex*.py`, `tex*.py`, `derive_tex.py`: the LaTeX derivation writer.
- `tests/`, `selftest.py`: the battery.

## Test tiers

- Default: `python3 tools/replicant/selftest.py` (289 tests, about 20 s). Also
  `python3 run_selftests.py replicant` from the repository root.
- Full: `python3 tools/replicant/selftest.py --full` adds the 101 `@slow` tests (full RSB,
  population dynamics at reference sizes, solver-based checks); about 9-10 minutes on a laptop (measured 9 min 28 s).
- `@slow` is a plain function attribute honoured by `selftest.py` only. pytest ignores the
  marker and would run every test, slow ones included.
