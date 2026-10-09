# cutbasis — GUIDE

Tool page: https://bootloops.ai/tools/cutbasis.html

## KIND

Package: `cutbasis.py` (CLI with `census`, `select`, `evaluate`), `geo_master_select.py` and
`imode.py` (the two selection drivers), `evaluate.py` and `tablecheck.py` (reducer comparison and
certification), the `geomorder/` library (Baikov representations, the integral-to-form map, the
ordering integers, the maximal-cut Laporta, Algorithms 1–3 for up to two cut variables,
linear-combination pre-images with symbolic coefficients), `selftest.py`, `fixtures/`,
`examples/moeller_tower/`. Python 3.9 or later with sympy, python-flint and pyyaml; Kira with a
Fermat executable for `evaluate` only.

## PURPOSE

Choose the master integrals of a Feynman integral family the way the geometric construction of
arXiv:2511.15381 (Algorithms 1–3) does: on the maximal cut of every sector, write the integrand as
a twisted differential form in a Baikov representation, assign to every form the integers a
(localization preference), w (Hodge weight), o (pole order after blow-ups) and |μ| (total pole
multiplicity), and take as masters the simplest forms under that order. The tool then realizes
those masters as Feynman integrals and writes Kira `preferred_masters` files, so that an unchanged
Laporta reduction runs on the geometric basis. Its `evaluate` subcommand runs the reducer with each
candidate list, certifies every table against the stock table and reports the smallest certified
table, with the stock table in the running.

## SCOPE

Sectors with a good loop-by-loop Baikov representation in at most two cut variables (and the
n = 0 sectors, which have one master, the corner integral) are analyzed: ordering integers by
local analysis modulo word-size primes at numeric kinematics (one sample for n ≤ 1; for n = 2 a
two-sample consensus, with any disagreement flagged in `summary.json`), point and curve
localizations, merge, global cut
Laporta. Sectors with three or more cut variables, or only a democratic representation, are
reported as "other" and keep the reducer's own masters. The `census` subcommand reports the
representation of every sector first; a family whose heavy sectors have a smaller-n representation
under a different loop-momentum routing or ISP basis should be relabeled before `select`.

Three lists are written and must be measured, not assumed: `<fam>.echelon.preferred`
(single-integral pre-images of the leading forms of the integral span, one per class that has one;
the reducer completes the rest), `<fam>.lc.preferred` (the same plus BASISLC linear combinations
with symbolic coefficients for the other classes: the class-pure basis), `<fam>.integrals.preferred`
(the maximal-cut Laporta with the Feynman integrals as columns: single integrals, dotted where the
corner integrand has no residue). Which one makes the reducer's table smallest depends on the
family; `evaluate` decides.

## USE-WHEN

- A Kira (or any preferred-masters Laporta) reduction carries mixed d–kinematics denominator
  factors, spurious polynomials, or high coefficient degrees that the stock basis choice causes.
- A master basis is wanted whose maximal-cut differential equation is a Laurent polynomial in ε
  inside the F-filtration window (the paper's refined statement, TeX label `refined_statement`; `geomorder/cutde.py` tests it).
- Before a long reduction of a new family: `census` is seconds per sector and tells whether the
  family is in scope and whether its representation should be relabeled.

## NOT-FOR

- Sectors with three or more cut variables (n ≥ 3) or democratic-only representations: they keep
  the reducer's masters.
- Replacing the reducer: the tool writes preferred-masters input; it does not reduce.
- Taking the linear-combination basis into a Laporta run by default: it is correct (certified) but
  the coefficients propagate into every table entry; use it for the differential equation, and let
  `evaluate` pick the Laporta basis.
- The ε-form rotation itself (the paper's step 2): `geomorder/step2.py` handles one variable and
  two blocks only and is not part of the command line.

## INVOKE

From the package directory (`tools/cutbasis/`); the lines below are the worked example and run as
written (see `examples/moeller_tower/README.md` for what they produce and the measured cost):

- `python3 cutbasis.py census --config examples/moeller_tower/input --family moeller --top 107`
- `python3 cutbasis.py select --config examples/moeller_tower/input --family moeller --top 107 --out OUT --workers 8`
- `python3 cutbasis.py evaluate --config examples/moeller_tower/input --family moeller --jobs examples/moeller_tower/input/jobs_template.yaml --targets examples/moeller_tower/input/targets --out OUT/eval --candidates echelon=OUT/moeller.echelon.preferred integrals=OUT/moeller.integrals.preferred --symbols d,t,m2`

Options: `--sectors a,b,c` (analyze these sectors only) or `--sectormappings DIR` (Kira's
`nonTrivialSector` list; by default every subsector of `--top` is analyzed); `select`: `--M 5`
(truncation of the cut system), `--dots 2 --sps 2` (candidate integrals), `--workers 8`,
`--seed 5`, `--no-lc`, `--no-integrals`; `evaluate`: `--alphabet 't,m2,...'` (expected denominator
factors; others are counted as `kinematic_not_in_alphabet`), `--kira PATH` (default: `KIRA`, then
`kira` on PATH), `--fermat PATH` (default: `FERMATPATH`, then `fer64` on PATH), `--parallel 4`.
Library: `geomorder.lcbasis.SectorLC(spec, sector).base()/.structure()`, `reconstruct_sector`,
`geomorder.masters.sector_masters` (integrals mode), `geomorder.cutde.run_sector` (F-test).

## INPUTS

A Kira configuration directory (`integralfamilies.yaml`, `kinematics.yaml`); optionally
`sectormappings/<fam>/nonTrivialSector`; for `evaluate` a jobs template containing the line
`preferred_masters: PREFERRED` and a workload target list. Configuration files and reduction
tables are trusted input (their expressions are parsed by SymPy); do not point the tool at files
from an untrusted source.

## OUTPUTS

`select`: the three preferred-masters files, `summary.json` (per sector: representation, H-basis
with a/w/o/|μ|, leading forms, supports, reconstructed coefficients, flags, `failed_sectors`),
per-sector checkpoints under `ckpt/` (Python pickles; a rerun resumes from them, so they are
trusted local state: do not copy them in from elsewhere). `evaluate`: `evaluate.json`
(per run: wall, bytes, rules, max degrees, coefficient bits, denominator-factor classes,
certification), `evaluate_summary.json` (the same without stamps, walls and paths: the part an
example's `expected/` directory carries) and the Kira tables under `run_<name>/`.

## GATES

Exit codes. `cutbasis.py select`: 0 every analyzed sector completed; 1 a sector analysis failed
(line `[select] FAILED: ...` naming the sectors; the lists are written but incomplete). A sector
whose cut system has no non-pivot column inside the truncation (dim H = 0 with columns left at the
boundary |μ| = M, e.g. `--sectors 42 --M 1` on the worked example) is such a failure, named
`[select] sector 42 FAILED: dim H = 0 at M=1 ... raise --M`; a scaleless sector (dim H = 0, no
boundary column) is not. `cutbasis.py evaluate` and `evaluate.py`: 0 every run finished and every
candidate certified; 1 a Kira run failed or a candidate did not certify (named line, recorded in
`evaluate.json`); 4 no Kira or no Fermat executable found: one line `[evaluate] REFUSED: <reason>`
and nothing written. Every driver (`cutbasis.py`, `geo_master_select.py`, `evaluate.py`): 2 an
option the tool cannot act on (argparse's usage-error code): an unknown `--family` (the families
of the config are listed), a `--top` or `--sectors` entry that is not a sector of the family or
not a subsector of `--top`, a missing `nonTrivialSector` file, a `--candidates` entry without `=`
or named `stock`, a missing candidate, jobs or targets file. One line
`[cutbasis] usage error: <option>: <reason>` (`[select] ...`, `[evaluate] ...` standalone) on
stderr, checked before the reducer lookup, nothing written. `tablecheck.py`: 0 consistent, 1 not
consistent. `selftest.py`: 0 every check passed (the reducer check prints a named SKIP line when
Kira or Fermat is not found and still exits 0; the package's verification class is `partial`);
1 a check failed; 4 only with `CUTBASIS_REQUIRE_REDUCER=1` set and no reducer: the last line is
`REFUSED: <reason>`.

Known-answer checks (`selftest.py`): 23 statements of arXiv:2511.15381 reproduced by script
(double-box twist and images; Møller sector 93 twist, pre-images and Hodge table; three-loop
electron self-energy twist and blow-up pole orders); the double-box top sector's H-basis and
masters (corner plus one ISP integral) against `fixtures/expected_doublebox.json`; sector 42 of the
worked example against `examples/moeller_tower/expected/quick/` byte for byte.

Planted defects (committed fixtures, built by `fixtures/build_fixtures.py`, which the self-test
never runs): a reduction table with one wrong coefficient (`fixtures/tablecheck/Bbad.txt`) must be
refused by `tablecheck.py` with exit 1 and the rule named; a mod-p sample set with one corrupted
value (`fixtures/ratrec/samples.json`) must be rejected by the dense fit; `evaluate` with no Kira on
PATH must refuse with exit 4 before writing anything; five bad options (`--sectors 1000` on the
worked example, `--family nosuch`, `--candidates X`, a candidate name with a path separator, a
missing candidate file) must each exit 2 with one named line and write nothing; with Kira and Fermat, two candidate lists for
the double box (`fixtures/doublebox_kira/echelon.preferred`, `alt.preferred`) must both certify
against the stock table, the three tables must differ, and the larger candidate must not be called
best. Coefficient strings must contain no spaces (see FOOTGUNS).

Worked example (`examples/moeller_tower/`, the two-loop Møller double box with a massive loop:
13 masters, 105 workload targets). The committed `expected/` files are the output of the
INVOKE lines above; the self-test compares the quick part, the full run is compared by hand as
`examples/moeller_tower/README.md` says. From `expected/evaluate_summary.json`: stock table
1550369 B (max degrees 24/24, 1 mixed d–kinematics denominator factor), echelon
list 507364 B (degrees 21/18, 0 mixed factor), integrals list 2877633 B (degrees
20/19, 0 mixed factor); both candidates certified equal to stock as a reduction
modulo two primes at two points; best certified candidate echelon. Which list wins is a property
of the family, not a rule.

Certification: `tablecheck.py` rotates one basis into the other and compares every common rule
modulo two primes at two random points (linear-combination masters through their definitions).
Coefficient reconstruction: dense multivariate rational fit at two 61-bit primes, Chinese
remaindering, rational reconstruction, verification at a third prime on fresh points.

Conditions under which the output is wrong or meaningless: a sector flagged `NOT trusted` in
`summary.json` (a master at the truncation boundary |μ| ≥ M − 1 in every loop order, or a point on
two even divisors without a residue candidate); a preferred list with more objects than the sector
has masters (symmetries): the tool emits only the echelon-in-V objects, but a hand-edited list
displaces masters in lower sectors inside Kira; an accidental degeneracy of the numeric kinematics
(rerun with another `--seed`); a Kira run that fails (the run is marked `failed` and the candidate
is never best).

## FOOTGUNS

- With Kira 3.1 as used here, a run whose preferred-masters coefficient strings contained spaces
  did not progress past "Generate equations"; the writer strips spaces, and `selftest.py` checks
  that. A Kira run idle in "Generate equations" right after a sector with a BASISLC block is
  this, not a long computation.
- Supplying more preferred objects than the sector has masters (dim H > dim V, symmetries)
  displaces masters in lower sectors inside Kira; the tool emits only the echelon-in-V objects.
- The truncation |μ| ≤ M of the cut system is checked (a master at the truncation boundary or a
  two-even-divisor point without a residue candidate triggers a retry with another loop order and,
  failing that, a `NOT trusted` flag in `summary.json`); read the flags.
- Numeric kinematics modulo primes can hit accidental degeneracies; the base analysis uses one
  seed and the reconstruction two primes plus a verification prime. Rerun with another `--seed`
  if a sector is flagged.
- The integers for n = 2 are computed by local analysis at every column of the cut system; the
  example's top sector (n = 2, six masters) is the slow part of its `select` run (its measured
  wall is in `examples/moeller_tower/README.md`). `--M 4` is a cheaper first pass; use `--workers`.
- `evaluate` runs the reducer with no resource limits of its own; on a shared machine wrap it in
  yours.
- A `select` run with a failed sector exits 1 and still writes the lists; they are incomplete for
  that sector, and `summary.json` names it under `failed_sectors`.

CREDIT: the construction is the one of [GeoBasis] (geometric master integrals and the ε-factorized
differential equation) and [GeoInt] (intersection matrices); the implementation here is ours. The
loop-by-loop Baikov representation follows Frellesvig and Papadopoulos [FP]. Kira is the Laporta
program of the Kira developers [Kira1, Kira2, Kira3]; FireFly [FF1] and Fermat are
its back ends. Bracketed keys resolve in REFERENCES.md at the repository root.

## License

MIT License. Copyright (c) 2026 Anthropic, PBC. Created by Matthew D. Schwartz; code written by Claude (Anthropic) under his supervision. Documentation is licensed CC BY 4.0. See LICENSE, LICENSE-CONTENT and NOTICE at the repository root; third-party components keep their own licenses (THIRD_PARTY.md).
