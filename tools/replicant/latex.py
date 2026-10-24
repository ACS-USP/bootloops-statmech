"""LaTeX writer: a calculation as the appendix of a paper.

  render_latex(result, *, title=None, standalone=True, extra=(), slow=False) -> str
  write_latex(result, path, *, pdf=False, **kw) -> Path    (.tex, or the .pdf with pdf=True)
  document_blocks(result) -> [TexBlock]                     every block the document shows
  report_blocks(report)   -> [TexBlock]                     the composition report

`result` is a compose.Result (catalog or composed) or a core.Refusal (its partial trace
is rendered with the reason).  Sections, in order: Model (spec, conventions, the
coupling law); Method and route (replica | cavity, the honest caveat, for a composed
model what was reused / adapted and relative to what); Derivation (catalog replica
entries: the level-(a) derivation of the same spec when a derive.FAMILIES hook exists,
rendered from the sympy objects in its State, and its derived replicon (derive_replicon:
the rules, then the slot weight, E''_R, chi_R and lambda_R as the code holds them, or the
reason it is not derived), then the equations the family code
implements, latex_entries / latex_cavity; composed models: the level-(a) trace itself);
Calculation (the run's trace, in order); Solution (values with per-key finality);
Stability; Reference check (key, computed, banked, tolerance, verdict; statistical rows
with stderr; slow / statistical rows "not run" unless slow=True); Labels; bibliography
(thebibliography from every cite string: no BibTeX, compiles offline).

Text and cites are escaped (tex.escape); math bodies are not.  A math block that is
neither rendered from a sympy object (tex.sym) nor cited is refused
(precondition:equation-source): the writer never displays an equation the code does
not hold.  write_latex(pdf=True) compiles with tectonic (preferred) or pdflatex when on
PATH, else refuses precondition:tex-engine; a failed compile is tex-engine:failed, or
tex-engine:unavailable when the engine reports it could not fetch its packages.
"""
from functools import lru_cache
from pathlib import Path

import mpmath as mp

from .core import Refusal, State, run_pipeline
from .latex_bib import _Bib
from .latex_pdf import OverfullDisplay, compile_pdf, overfull  # noqa: F401 (re-exported)
from .tex import Check, TexBlock, escape, invented, prose  # noqa: F401 (re-exported)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{iftex}
\ifPDFTeX
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\fi
\usepackage{amsmath,amssymb,graphicx}
\allowdisplaybreaks
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\sloppy
\setlength{\emergencystretch}{3em}
"""
ROUTES = {"replica": "replica method: replicated partition function Z^n, disorder average, "
                     "order parameters q_ab (and conjugates), saddle point, RS / 1RSB "
                     "ansatz, continuation n -> 0",
          "cavity": "cavity method: remove a site, the law of its cavity field, add the site "
                    "back (marginal and reaction), self-consistency; on sparse graphs "
                    "population dynamics for the law of the cavity fields"}
_STAB = ("replicon", "stability", "capacity_stable", "stable", "mode_eig")


def _num(v):
    if isinstance(v, (mp.mpf, mp.mpc)):
        return mp.nstr(v, 15)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _block(b: TexBlock, bib):
    if invented(b):
        raise Refusal("precondition:equation-source",
                      f"block {b.title!r}: an equation with neither a sympy expression the "
                      "code holds nor a citation")
    head = f"\\paragraph{{{prose(b.title)}}}"
    if b.source:
        head += " " + bib.cite(b.source)
    if b.check is not None:
        head += " " + prose("(sympy form; checked against the code by tests/test_latex.py)"
                             if b.expr is not None else "(evaluated by hand-coded quadrature; "
                             "checked against the code by tests/test_latex.py)")
    if b.kind == "text":
        return f"{head}\n{prose(b.body)}\n"
    env = "multline" if b.kind == "equation" and "\\\\" in b.body else b.kind
    return f"{head}\n\\begin{{{env}}}\n{b.body}\n\\end{{{env}}}\n"


def _blocks(blocks, bib):
    return "".join(_block(b, bib) for b in blocks)


def _entry(k, e, bib, blocks, fallback_cite=None):
    head = f"\\subsection{{Step {k}: {escape(e.step)}}}\n"
    cite = e.cite or fallback_cite
    src = f"Source: {bib.cite(cite)}. " if cite else ""
    # a step that displays blocks shows its note there; the header keeps the State keys set
    summary = e.summary.split("; ")[0] if blocks else e.summary
    return head + f"{src}Trace: {prose(summary)}\n\n" + _blocks(blocks, bib)


@lru_cache(maxsize=None)
def _level_a(key):
    """(trace, replicon blocks, None) of the symbolic level-(a) derivation of a spec, or
    (None, [], reason) when level (a) has no hook for it."""
    from . import compose as C
    from . import derive as D
    from .derive_replicon import replicon
    from .derive_tex import replicon_blocks
    from .spec import ModelSpec
    *fields, params = key
    spec = ModelSpec(*fields, params=dict(params))
    if spec.family not in D.FAMILIES:
        return None, [], f"derive.FAMILIES has no level-(a) hook for family {spec.family!r}"
    try:
        C._check_hooks(spec, None)
        state, trace = run_pipeline(C.level_a_steps(spec)[:6], State())
        return trace, replicon_blocks(replicon(state)), None
    except Refusal as r:
        return None, [], f"{r.reason}: {r.detail}"


def _spec_key(spec):
    from .compose import FIELDS
    return (*(getattr(spec, f) for f in FIELDS), tuple(sorted(spec.params.items(), key=str)))


def _model_blocks(res):
    from .latex_entries import model_blocks
    return model_blocks(res.spec or res.entry.spec)


def report_blocks(report):
    from .tex import text
    out = [text("Composition", f"kind: {report.kind}; nearest catalog entry: "
                f"{report.nearest}; compared against: {report.relative_to}")]
    out.append(text("Reused steps (the same registry hooks as the nearest entry)",
                    ", ".join(report.reused) or "none"))
    out.append(text("Adapted steps (a different hook)",
                    "; ".join(str(a) for a in report.adapted) or "none"))
    out.append(text("Missing steps", ", ".join(report.missing) or
                    "none (a missing step is a refusal, never a partial calculation)"))
    return out


def _derivation(res):
    """[(trace entry or None, title, blocks)] of the Derivation section."""
    from .compose import LEVEL_A_CITE
    from .solve_derived import lift_numeric_beta
    rep_title = "Replicon of the RS saddle (level (a))"
    if res.report is not None and res.report.kind == "composed":
        rows = [(e, None, e.tex) for e in res.trace.entries]
        if res.values.get("ansatz") == "1rsb":        # the RS replicon is not this saddle's
            return rows, LEVEL_A_CITE, None
        _, rep, _ = _level_a(_spec_key(lift_numeric_beta(res.spec, {})[0]))
        return rows + ([(None, rep_title, rep)] if rep else []), LEVEL_A_CITE, None
    trace, rep, why = (None, [], "cavity route") if res.route != "replica" else \
        _level_a(_spec_key(res.entry.spec))
    rows = [(e, None, e.tex) for e in trace.entries] if trace else []
    if rep:
        rows.append((None, rep_title, rep))
    rows.append((None, "Equations implemented by the family code"
                 if res.route == "replica" else "The cavity argument",
                 res.entry.derivation(res.route)))
    return rows, LEVEL_A_CITE, why


def document_blocks(result):
    if isinstance(result, Refusal):
        return [b for e in (result.trace.entries if result.trace else []) for b in e.tex]
    rows, _, _ = _derivation(result)
    out = _model_blocks(result) + [b for _, _, bl in rows for b in bl]
    if result.report is not None and result.report.kind == "composed":
        out += report_blocks(result.report)
    return out


def _solution(res, bib):
    stat = set(res.statistical)
    rows = []
    for k, v in res.values.items():
        st = res.finality.get(k)
        if st is None:
            continue
        val = escape(_num(v))
        if k in stat:
            val += f" $\\pm$ {escape(_num(res.errors.get(k)))} (statistical)"
        rows.append(f"{escape(k)} & {val} & {escape(st.state)} ({escape(st.ansatz)}) "
                    f"{bib.cite(st.cite)} & {'yes' if st.final else 'no'} \\\\")
    meta = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.values.items()
            if k not in res.finality]
    errs = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.errors.items() if k not in stat]
    notes = [n for n in res.notes if "NOT final" in n]
    out = ("\\section{Solution}\nValues at the requested parameters (" +
           prose(", ".join(f"{k} = {_num(v)}" for k, v in res.params.items()) or "none") +
           "), each with its finality: only stable and exact-by-literature values are final.\n"
           "\\begin{center}\n\\begin{tabular}{llll}\nkey & value & finality (ansatz) & final \\\\"
           "\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n")
    if notes:
        out += "\\begin{itemize}\n" + "".join(f"\\item {prose(n)}\n" for n in notes) + \
            "\\end{itemize}\n"
    if meta:
        out += "Other outputs (parameters, labels, diagnostics): " + "; ".join(meta) + ".\n\n"
    if errs:
        out += "Numerical error estimates: " + "; ".join(errs) + ".\n"
    return out


def _stability(res):
    comp = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.values.items()
            if any(s in k for s in _STAB)]
    by = {}
    for k, st in res.finality.items():
        by.setdefault(st.state, []).append(k)
    out = "\\section{Stability}\n"
    out += ("Computed stability quantities (replicon / AT / cavity susceptibility): "
            + "; ".join(comp) + ".\n\n") if comp else \
        "No stability quantity (replicon / AT) is computed by this run.\n\n"
    out += "".join(f"{prose(s)}: {prose(', '.join(ks))}.\n\n" for s, ks in sorted(by.items()))
    return out


def _references(res, bib, slow):
    from .refdata import load
    out = "\\section{Reference check}\n"
    if res.entry is None:
        near = res.report.nearest if res.report else None
        return out + prose(f"No banked keys: a composed model is not a catalog entry, so "
                            f"nothing is compared here (nearest entry: {near}).  This document "
                            "does not certify its numbers: a composed model is validated only "
                            "by an independent route (enumeration at small N, or a limit with "
                            "a known answer) in the test suite, where such a test exists.") + "\n"
    try:
        done = {(o.file, o.key): o for o in res.reference_checks(slow=slow)}
    except Refusal as r:
        return out + prose(f"refused: {r.reason}: {r.detail}") + "\n"
    rows, stat = [], []
    for ref in res.entry.refs:
        if ref.route != res.route:
            continue
        want = load(ref.file)[ref.key]
        o = done.get((ref.file, ref.key))
        if o is None:
            got, verdict = "not run (" + ("statistical" if ref.statistical else "slow") + \
                ")", "--"
        else:
            got, verdict = escape(_short(o.got)), ("agrees" if o.ok else "DISAGREES")
            if o.statistical:
                got += " (statistical)"
                stat.append(f"\\item {prose(ref.key)}: {prose(o.detail)}\n")
        rows.append(f"{escape(ref.key)} & {got} & {escape(_short(want.value))} & "
                    f"{escape(want.tol)} & {verdict} & {bib.cite(want.cite)} \\\\")
    return (out + "Banked values (reference/*.json) against this pipeline's output; numbers "
            "are shown to 16 significant digits, the comparison uses them in full; slow and "
            "statistical rows run only with --full (\\texttt{render\\_latex(slow=True)}).\n"
            "\\begin{center}\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n" + _FIT +
            "{\\begin{tabular}{llllll}\n"
            "key & computed & banked & tolerance & verdict & source \\\\\n\\hline\n" +
            "\n".join(rows) + "\n\\end{tabular}}\n\\end{center}\n" +
            ("Statistical comparisons:\n\\begin{itemize}\n" + "".join(stat) +
             "\\end{itemize}\n" if stat else ""))


_FIT = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}"   # shrink if wide


def _short(s, digits=16):
    """A decimal string to `digits` significant digits (display only); other text as is."""
    with mp.workdps(40):
        try:
            x = mp.mpf(str(s))
        except (ValueError, TypeError):
            return str(s)
        return mp.nstr(x, digits) if len(str(s)) > digits + 3 else str(s)


def _caveat(res):
    if res.entry is not None:
        return res.entry.caveats[res.route]
    return res.notes[0] if res.notes else ""


def _labels(res):
    out = (f"\\section{{Labels}}\nRigor: \\textbf{{{escape(res.rigor)}}}. "
           f"Route: \\textbf{{{prose(res.route)}}}.\n\n{prose(_caveat(res))}\n\n")
    if res.statistical:
        out += prose(f"statistical: {', '.join(res.statistical)} are sampling estimates "
                      "(value +- stderr over independent runs), not certified.") + "\n\n"
    rest = [n for n in res.notes if n and n != _caveat(res) and "NOT final" not in n]
    if rest:
        out += "\\begin{itemize}\n" + "".join(f"\\item {prose(n)}\n" for n in rest) + \
            "\\end{itemize}\n"
    return out


def _result_body(res, bib, extra, slow):
    spec = res.spec or (res.entry.spec if res.entry is not None else None)
    if spec is None:
        raise Refusal("precondition:result-spec", "the Result carries no ModelSpec "
                      "(compose.ComposedCalculation.run sets it)")
    sp_txt = ", ".join(f"{f} = {getattr(spec, f)}" for f in
                       ("family", "variables", "disorder", "constraint", "prior"))
    out = ["\\section{Model}\n" + prose(sp_txt) + ".\n\n" +
           (prose(res.entry.description) + " " + bib.cite(res.entry.cite) + "\n\n"
            if res.entry else "") + _blocks(_model_blocks(res), bib)]
    out.append(f"\\section{{Method and route}}\nRoute: \\textbf{{{escape(res.route)}}} "
               f"({prose(ROUTES.get(res.route, res.route))}). Rigor: "
               f"\\textbf{{{prose(res.rigor)}}}. {prose(_caveat(res))}\n")
    if res.report is not None and res.report.kind == "composed":
        out.append("\\subsection{Reused and adapted steps}\n" +
                   _blocks(report_blocks(res.report), bib))
    rows, a_cite, why = _derivation(res)
    composed = res.entry is None
    if composed or rows[0][0] is not None:
        intro = ("Level (a): each step is rendered from the sympy objects that step put in "
                 "the State (sympy.latex); nothing is retyped.")
    elif res.route == "cavity":
        intro = ("The cavity equations as the cavity code implements them (cavity/*.py): for "
                 "the dense models in the order of its four steps (remove a site, the cavity-"
                 "field law, add the site, self-consistency); for sparse graphs the recursion, "
                 "observables and population dynamics.  Each equation is in sympy form "
                 "(checked against the code by tests/test_latex.py) or cited.")
    else:
        intro = ("No step-by-step symbolic derivation is available in this package for this "
                 "model; the governing equations implemented by the code are shown below, "
                 "each in sympy form (checked against the code by tests/test_latex.py) or "
                 "cited.  No intermediate step the code does not perform is shown.")
    out.append("\\section{Derivation}\n" + prose(intro) + "\n")
    if not composed and rows[0][0] is not None:
        out.append(prose("The level-(a) steps below derive the same model symbolically "
                          "(tests/test_agree.py checks them against the family equations); "
                          f"method as in {res.entry.cite}.") + "\n")
    for k, (e, title, blocks) in enumerate(rows, 1):
        out.append(_entry(k, e, bib, blocks, a_cite) if e is not None else
                   f"\\subsection{{{escape(title)}}}\n" + _blocks(blocks, bib))
    if extra:
        out.append(_blocks(extra, bib))
    if not composed:
        out.append("\\section{Calculation at the requested parameters}\nThe run's trace, in "
                   "order (skipped steps say why).\n" +
                   "".join(_entry(k, e, bib, e.tex) for k, e in
                           enumerate(res.trace.entries, 1)))
    out += [_solution(res, bib), _stability(res), _references(res, bib, slow), _labels(res)]
    return "".join(out)


def _refusal_body(r, bib):
    from .compose import LEVEL_A_CITE
    from .families.common import CAVEAT, CAVITY_CAVEAT
    entries = r.trace.entries if r.trace is not None else []
    route = getattr(r, "route", None)
    out = ["\\section{Refused calculation}\n" + prose(
        f"Refused: {r.reason}" + (f" at step {r.step}" if r.step else "") +
        (f": {r.detail}" if r.detail else "") + ". No value is reported as a result; "
        "numbers in the trace are diagnostics of the refused computation, not results. "
        "The steps applied before the refusal are shown below.") + "\n"]
    out.append("\\section{Partial derivation}\n" + (
        "".join(_entry(k, e, bib, e.tex, LEVEL_A_CITE) for k, e in enumerate(entries, 1))
        or escape("The refusal came before any step was applied.") + "\n"))
    out.append("\\section{Labels}\nRigor: \\textbf{physics-level}. " +
               (f"Route: \\textbf{{{escape(route)}}}. " if route else "") +
               prose(CAVITY_CAVEAT if route == "cavity" else CAVEAT) + "\n")
    return "".join(out)


def render_latex(result, *, title=None, standalone=True, extra=(), slow=False) -> str:
    bib = _Bib()
    if isinstance(result, Refusal):
        body = _refusal_body(result, bib)
        title = title or f"Refused calculation ({result.reason})"
    else:
        body = _result_body(result, bib, list(extra), slow)
        if title is None:
            s = result.spec
            title = (f"{result.entry.name}: {result.entry.description}" if result.entry
                     else f"Composed model: {s.family}, {s.variables}, {s.disorder}")
    body += bib.render()
    if not standalone:
        return body
    return (PREAMBLE + f"\\title{{{prose(title)}}}\n\\date{{}}\n\\begin{{document}}\n"
            "\\maketitle\n" + body + "\\end{document}\n")


def write_latex(result, path, *, pdf=False, **kw) -> Path:
    """Write the document to `path`; with pdf=True also compile it and return the PDF."""
    path = Path(path)
    path.write_text(render_latex(result, **kw))
    return compile_pdf(path) if pdf else path


__all__ = ["TexBlock", "compile_pdf", "document_blocks", "render_latex", "report_blocks",
           "write_latex"]
