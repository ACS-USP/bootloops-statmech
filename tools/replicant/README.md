# replicant

Published replica/cavity calculations as typed, recorded steps: each step
names its preconditions and its citation, a missing precondition is a named
refusal (`precondition:<name>`), and every run returns a trace.

Battery: `python3 tools/replicant/selftest.py [--full]`.

CLI (from `tools/`): `python3 -m replicant list` prints the catalog (entries, routes,
citations); `python3 -m replicant run <entry|spec.json> [--route replica|cavity]
[--param k=v ...] [--out result.json]` composes and runs a calculation (catalog.py,
compose.py, cli.py), writes the JSON result and prints the step trace to stderr; a
refusal prints `REFUSED: <reason>: <detail>` and exits 2.

LaTeX (latex.py): `python3 -m replicant run <entry|spec.json> ... --latex out.tex [--pdf]`
writes the calculation as a paper appendix: model, route, every step with its equations
(level (a) rendered from the sympy objects in the State; the family code's equations in
sympy form, each checked against the code at three points, or cited), the solution with
per-value finality, stability, the banked-versus-computed check table, the honest labels
and a bibliography from the citations.  `--pdf` compiles it with tectonic or pdflatex.
A refused run writes its partial derivation and the reason.
