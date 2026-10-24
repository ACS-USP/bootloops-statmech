"""Command line (run with tools/ on the path, e.g. `cd tools && python3 -m replicant ...`):

  python3 -m replicant list
  python3 -m replicant run <entry|spec.json> [--route replica|cavity]
                           [--param k=v ...] [--out result.json]
                           [--latex derivation.tex [--pdf]]

`run` composes the calculation (compose.compose), runs it, writes the JSON result
(values, errors, rigor, route, stable, final, notes, the composition report, the trace
and the entry's non-slow banked comparisons) to --out (default stdout) and the rendered
trace to stderr.  A refusal prints `REFUSED: <reason>: <detail>` to stderr and exits 2;
any other exception prints `ERROR: <type>: <message>` and exits 1; success exits 0.
--param values: integers parse as int; decimals stay DECIMAL STRINGS (the families convert
them with mpmath, so 0.1 is not rounded through a binary float); true/false -> bool; a
comma list (a trailing comma makes a 1-list: ys=0.3,) -> tuple of those; anything else
stays a string (e.g. learning=bayes).  spec.json: {"family", "variables",
"disorder", "constraint", "prior", "params"}; a param given as null is left symbolic
(beta, h, alpha, kappa: the derivation's own symbols); numbers are run parameters.
--param ansatz=1rsb selects the level-(a) 1RSB route (compose.compose(..., ansatz="1rsb");
its knobs frozen=true, zero_temperature=true, solve_for=<parameter>, branch=... are also
--params).  `build_result(...)` returns the Result object, so writers (e.g. LaTeX) can be added.
--latex PATH writes the derivation document (latex.write_latex: the steps with their
equations, citations, the check table and the labels); --pdf also compiles it (tectonic
or pdflatex; refused precondition:tex-engine when neither is on PATH).  A refused run
with --latex still writes the document of its partial derivation and the reason, then
exits 2.  --pdf without --latex is refused (precondition:latex-path).  The JSON result is
written before the document, so a refused compile (REFUSED: latex: ..., exit 2) or an
unwritable path (ERROR: latex: OSError ..., exit 1) never loses it; a display wider than
the page prints WARNING: (latex_pdf.OverfullDisplay).
"""
import argparse
import json
import sys
import warnings

import mpmath as mp
import sympy as sp

from . import derive as D
from .catalog import catalog
from .compose import compose
from .core import Refusal
from .derive_families import ALPHA, KAPPA
from .latex import OverfullDisplay, write_latex
from .spec import ModelSpec

_SYMBOLS = {"beta": D.BETA, "h": D.H_FIELD, "alpha": ALPHA, "kappa": KAPPA}


def _number(text):
    """int for an integer literal, the decimal string itself for a finite decimal, else
    None."""
    t = text.strip()
    try:
        return int(t)
    except ValueError:
        pass
    try:
        x = mp.mpf(t)
    except (ValueError, TypeError):
        return None
    return t if mp.isfinite(x) else None


def parse_param(item):
    if "=" not in item:
        raise Refusal("precondition:param-syntax", f"--param {item!r}: use key=value")
    k, v = item.split("=", 1)
    low = v.strip().lower()
    if low in ("true", "false"):
        return k, low == "true"
    if "," in v:
        parts = [_number(x) for x in v.split(",") if x.strip()]
        if all(x is not None for x in parts):
            return k, tuple(parts)
    n = _number(v)
    return k, (v if n is None else n)


def load_spec(target):
    """(name, ModelSpec) for a catalog entry name or a spec.json path."""
    cat = catalog()
    if target in cat:
        return target, cat[target].spec
    if not target.endswith(".json"):
        raise Refusal("precondition:known-target",
                      f"{target!r} is neither a catalog entry ({', '.join(cat)}) nor a .json spec")
    try:
        with open(target) as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise Refusal("precondition:readable-spec", f"{target}: {exc}")
    missing = [k for k in ("family", "variables", "disorder") if k not in raw]
    if missing:
        raise Refusal("precondition:complete-spec", f"{target}: missing {', '.join(missing)}")
    params = {}
    for k, v in (raw.get("params") or {}).items():
        if v is None:
            if k not in _SYMBOLS:
                raise Refusal("precondition:known-symbol",
                              f"param {k!r} = null: symbolic only for {', '.join(_SYMBOLS)}")
            params[k] = _SYMBOLS[k]
        else:
            params[k] = v
    return target, ModelSpec(raw["family"], raw["variables"], raw["disorder"],
                             raw.get("constraint"), raw.get("prior"), params)


def jsonable(x):
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else repr(x)
    if isinstance(x, mp.mpf):
        return mp.nstr(x, 20)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, sp.Basic):
        return str(x)
    try:                                  # numpy scalars
        return jsonable(x.item())
    except AttributeError:
        return str(x)


def build_result(target, route, params):
    name, spec = load_spec(target)
    params = dict(params)
    ansatz = params.pop("ansatz", "rs")
    return spec, compose(spec, route, ansatz).run(**params)


def result_json(target, spec, res):
    refs, skipped = [], []
    if res.entry is not None:
        try:
            refs = [vars(c) for c in res.reference_checks()]
        except Refusal as r:
            refs = [{"refused": f"{r.reason}: {r.detail}"}]
        skipped = [f"{r.file}/{r.key}" for r in res.entry.refs
                   if r.route == res.route and (r.slow or r.statistical)]
    return jsonable({"target": target, "spec": vars(spec), "route": res.route,
                     "rigor": res.rigor, "values": res.values, "errors": res.errors,
                     "statistical": res.statistical, "stable": res.stable, "final": res.final,
                     "finality": {k: {**vars(st), "final": st.final}
                                  for k, st in res.finality.items()},
                     "notes": res.notes, "params": res.params, "report": res.report.to_json(),
                     "trace": res.trace.to_json(), "references": refs,
                     "references_skipped_slow": skipped})


def main(argv=None, stdout=None, stderr=None):
    stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
    ap = argparse.ArgumentParser(prog="replicant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("target")
    run.add_argument("--route", default="replica")
    run.add_argument("--param", action="append", default=[])
    run.add_argument("--out")
    run.add_argument("--latex")
    run.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for name, e in catalog().items():
            print(f"{name:16s} routes={','.join(sorted(e.routes)):15s} {e.cite}", file=stdout)
        return 0
    try:
        if args.pdf and not args.latex:
            raise Refusal("precondition:latex-path", "--pdf needs --latex PATH")
        params = dict(parse_param(p) for p in args.param)
        spec, res = build_result(args.target, args.route, params)
        doc = result_json(args.target, spec, res)
    except Refusal as r:
        if args.latex:
            try:
                write_latex(r, args.latex)
                print(f"latex: wrote the refused calculation to {args.latex}", file=stderr)
            except OSError as exc:
                print(f"ERROR: latex: {type(exc).__name__}: {exc}", file=stderr)
        print(f"REFUSED: {r.reason}: {r.detail}", file=stderr)
        return 2
    except Exception as exc:          # not a named refusal: report it, never a traceback
        print(f"ERROR: {type(exc).__name__}: {exc}", file=stderr)
        return 1
    text = json.dumps(doc, indent=2)
    if args.out:
        try:
            with open(args.out, "w") as f:
                f.write(text + "\n")
        except OSError as exc:
            print(f"ERROR: --out: {type(exc).__name__}: {exc}", file=stderr)
            return 1
    else:
        print(text, file=stdout)
    print(f"trace ({res.route}, {res.rigor}):\n{res.trace.render()}", file=stderr)
    if args.latex:                    # after the JSON: a writer failure never loses the result
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OverfullDisplay)
                written = write_latex(res, args.latex, pdf=args.pdf)
            print(f"latex: wrote {written}", file=stderr)
            for w in caught:
                print(f"WARNING: {w.message}", file=stderr)
        except Refusal as r:
            print(f"REFUSED: latex: {r.reason}: {r.detail}", file=stderr)
            return 2
        except OSError as exc:
            print(f"ERROR: latex: {type(exc).__name__}: {exc}", file=stderr)
            return 1
    return 0
