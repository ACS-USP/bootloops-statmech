# Worked example: the two-loop Møller double box with a massive loop

Family `moeller` (nine propagators, top sector 107, invariants t and m2 with s = 1), the elliptic
example of arXiv:2511.15381; `input/` holds the Kira configuration (`integralfamilies.yaml`,
`kinematics.yaml`), the jobs template with the line `preferred_masters: PREFERRED` and the
105-integral workload `targets`. Both commands run from the package directory
(`tools/cutbasis/`).

## 1. Select the geometric masters

    python3 cutbasis.py select --config examples/moeller_tower/input --family moeller --top 107 --out OUT --workers 8

Writes `OUT/moeller.echelon.preferred`, `OUT/moeller.lc.preferred`, `OUT/moeller.integrals.preferred`
and `OUT/summary.json`. Compare the three lists with `expected/`: they are byte for byte equal
(`cmp OUT/moeller.echelon.preferred expected/moeller.echelon.preferred`, and so on). Measured on
the run that produced the committed files (an x86-64 server, eight workers, `/usr/bin/time -v`):
about 32 min wall, 1.5 GB peak memory.

## 2. Run Kira with each list and certify

    python3 cutbasis.py evaluate --config examples/moeller_tower/input --family moeller \
        --jobs examples/moeller_tower/input/jobs_template.yaml --targets examples/moeller_tower/input/targets \
        --out OUT/eval --candidates echelon=OUT/moeller.echelon.preferred integrals=OUT/moeller.integrals.preferred \
        --symbols d,t,m2

Needs Kira with a Fermat executable (`--kira`/`KIRA`, `--fermat`/`FERMATPATH`, or both on PATH;
exit 4 with a `REFUSED:` line otherwise). Writes `OUT/eval/evaluate.json` and
`OUT/eval/evaluate_summary.json`; compare the latter with `expected/evaluate_summary.json` (it
carries no stamps, walls or paths). Measured on the same machine: about 14 min wall (three Kira runs of about ten
seconds each; the rest is the certification and the metrics in `tablecheck.py`).

From the committed `expected/evaluate_summary.json`:

| run | bytes | rules | terms | max degree num/den | coefficient bits | mixed d–kinematics factors | certified |
|---|---|---|---|---|---|---|---|
| stock | 1550369 | 102 | 1150 | 24/24 | 32 | 1 | (reference) |
| echelon | 507364 | 102 | 1131 | 21/18 | 26 | 0 | yes |
| integrals | 2877633 | 102 | 1220 | 20/19 | 32 | 0 | yes |

Best certified candidate: echelon. The byte counts are those of Kira 3.1 tables; another Kira
version can format its tables differently, in which case the certified flags, the master lists and
the degree and factor columns are the comparison.

## 3. Quick check (part of the self-test)

    python3 cutbasis.py select --config examples/moeller_tower/input --family moeller --top 107 --sectors 42 --out OUT_QUICK --workers 4

Sector 42 only (n = 1, two masters, one of them a linear combination). The three lists in
`OUT_QUICK/` equal `expected/quick/` byte for byte; `selftest.py` runs this command and makes that
comparison (measured wall 63 s with four workers).
