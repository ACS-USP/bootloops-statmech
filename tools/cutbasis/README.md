# cutbasis — geometric master-basis selector on the maximal cut

Chooses master integrals the way the geometric construction of arXiv:2511.15381 (Algorithms 1–3)
does: on the maximal cut of each sector the integrand is a twisted differential form in a Baikov
representation; every form gets the integers a (localization preference), w (Hodge weight), o
(pole order after blow-ups) and |μ|, and the masters are the simplest forms under that order. The
tool realizes them as Feynman integrals (single integrals, or linear combinations with symbolic
coefficients) and writes Kira `preferred_masters` files; its `evaluate` subcommand runs the
reducer with each candidate list, certifies every table against the stock table and reports the
smallest certified table, the stock table included (a candidate larger than stock is never called
best). The local analysis runs modulo word-size primes at numeric kinematics: one sample for sectors with
one cut variable, two independent (prime, kinematics) samples for two cut variables, with any
disagreement flagged in `summary.json`.

Run lines (from this directory; these are the worked example and run as written):

    python3 cutbasis.py census   --config examples/moeller_tower/input --family moeller --top 107
    python3 cutbasis.py select   --config examples/moeller_tower/input --family moeller --top 107 --out OUT --workers 8
    python3 cutbasis.py evaluate --config examples/moeller_tower/input --family moeller \
        --jobs examples/moeller_tower/input/jobs_template.yaml --targets examples/moeller_tower/input/targets \
        --out OUT/eval --candidates echelon=OUT/moeller.echelon.preferred integrals=OUT/moeller.integrals.preferred \
        --symbols d,t,m2

`evaluate` needs Kira and a Fermat executable (`--kira`/`KIRA`, `--fermat`/`FERMATPATH`, or both
on PATH) and refuses with exit 4 otherwise. Exit codes and the checks are in `GUIDE.md` (GATES).

Example (`examples/moeller_tower/`): the two-loop Møller double box with a massive loop, the
paper's elliptic example. Input: Kira family files, the 105-integral workload, the jobs
template. Committed expected output: the three preferred-masters lists (12 echelon
integrals, 17 integrals-mode integrals, the lc list with 5 linear-combination
blocks) and `evaluate_summary.json`: stock table 1550369 B with 1 mixed
d–kinematics denominator factor; echelon basis 507364 B, 0 mixed factor, degrees
24/24 → 21/18, certified equal to stock as a reduction; integrals basis 2877633 B,
certified. Measured cost of the two run lines on one machine is in the example's README. The
self-test (`python3 selftest.py`) reproduces 23 statements of the paper, the double-box selection
and sector 42 of the example against committed expected output, runs the committed planted-defect
fixtures, and runs the reducer comparison when Kira and Fermat are found (a named SKIP line
otherwise).

Files: `cutbasis.py` (CLI), `geo_master_select.py` and `imode.py` (the two selection drivers),
`evaluate.py` + `tablecheck.py` (reducer comparison and certification), `geomorder/` (library),
`fixtures/` (paper families, expected outputs, planted-defect fixtures and their builder, a Kira
configuration for the self-test), `examples/`, `GUIDE.md` (manual: scope, gates, footguns).

## License

MIT License. Copyright (c) 2026 Anthropic, PBC. Created by Matthew D. Schwartz; code written by Claude (Anthropic) under his supervision. Documentation is licensed CC BY 4.0. See LICENSE, LICENSE-CONTENT and NOTICE at the repository root; third-party components keep their own licenses (THIRD_PARTY.md).
