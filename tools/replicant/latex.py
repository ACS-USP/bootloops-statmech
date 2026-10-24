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
rendered from the sympy objects in its State, then the equations the family code
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
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import mpmath as mp

from .core import Refusal, State, run_pipeline
from .tex import Check, TexBlock, escape, invented  # noqa: F401 (re-exported)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb}
\allowdisplaybreaks
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
"""
ROUTES = {"replica": "replica method: replicated partition function Z^n, disorder average, "
                     "order parameters q_ab (and conjugates), saddle point, RS / 1RSB "
                     "ansatz, continuation n -> 0",
          "cavity": "cavity method: remove a site, the law of its cavity field, add the site "
                    "back (marginal and reaction), self-consistency; on sparse graphs "
                    "population dynamics for the law of the cavity fields"}
_STAB = ("replicon", "stability", "capacity_stable", "stable", "mode_eig")


class _Bib:
    """Cite strings -> \\cite keys; items are split at top-level '; '."""
    def __init__(self):
        self.keys = {}

    @staticmethod
    def items(cite):
        out, depth, cur = [], 0, ""
        for i, ch in enumerate(cite):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if ch == ";" and depth == 0 and cite[i + 1:i + 2] == " ":
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        return [x for x in out + [cur.strip()] if x]

    def cite(self, cite):
        if not cite:
            return ""
        keys = [self.keys.setdefault(it, f"b{len(self.keys) + 1}") for it in self.items(cite)]
        return f"\\cite{{{','.join(keys)}}}"

    def render(self):
        if not self.keys:
            return ""
        rows = "\n".join(f"\\bibitem{{{k}}} {escape(it)}" for it, k in self.keys.items())
        return f"\n\\begin{{thebibliography}}{{99}}\n{rows}\n\\end{{thebibliography}}\n"


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
    head = f"\\paragraph{{{escape(b.title)}}}"
    if b.source:
        head += " " + bib.cite(b.source)
    if b.check is not None:
        head += " (sympy form, checked against the code)"
    if b.kind == "text":
        return f"{head}\n{escape(b.body)}\n"
    return f"{head}\n\\begin{{{b.kind}}}\n{b.body}\n\\end{{{b.kind}}}\n"


def _blocks(blocks, bib):
    return "".join(_block(b, bib) for b in blocks)


def _entry(k, e, bib, blocks, fallback_cite=None):
    head = f"\\subsection{{Step {k}: {escape(e.step)}}}\n"
    cite = e.cite or fallback_cite
    src = f"Source: {bib.cite(cite)}. " if cite else ""
    return head + f"{src}Trace: {escape(e.summary)}\n\n" + _blocks(blocks, bib)


@lru_cache(maxsize=None)
def _level_a(key):
    """(trace, None) of the symbolic level-(a) derivation of a catalog spec, or
    (None, reason) when level (a) has no hook for it."""
    from . import compose as C
    from .spec import ModelSpec
    *fields, params = key
    spec = ModelSpec(*fields, params=dict(params))
    try:
        C._check_hooks(spec, None)
        return run_pipeline(C.level_a_steps(spec)[:6], State())[1], None
    except Refusal as r:
        return None, f"{r.reason}: {r.detail}"


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
    if res.report is not None and res.report.kind == "composed":
        return [(e, None, e.tex) for e in res.trace.entries], LEVEL_A_CITE, None
    trace, why = (None, "cavity route") if res.route != "replica" else \
        _level_a(_spec_key(res.entry.spec))
    rows = [(e, None, e.tex) for e in trace.entries] if trace else []
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
    meta = [f"{escape(k)} = {escape(_num(v))}" for k, v in res.values.items()
            if k not in res.finality]
    errs = [f"{escape(k)} = {escape(_num(v))}" for k, v in res.errors.items() if k not in stat]
    notes = [n for n in res.notes if "NOT final" in n]
    out = ("\\section{Solution}\nValues at the requested parameters (" +
           escape(", ".join(f"{k} = {_num(v)}" for k, v in res.params.items()) or "none") +
           "), each with its finality: only stable and exact-by-literature values are final.\n"
           "\\begin{center}\n\\begin{tabular}{llll}\nkey & value & finality (ansatz) & final \\\\"
           "\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n")
    if notes:
        out += "\\begin{itemize}\n" + "".join(f"\\item {escape(n)}\n" for n in notes) + \
            "\\end{itemize}\n"
    if meta:
        out += "Other outputs (parameters, labels, diagnostics): " + "; ".join(meta) + ".\n\n"
    if errs:
        out += "Numerical error estimates: " + "; ".join(errs) + ".\n"
    return out


def _stability(res):
    comp = [f"{escape(k)} = {escape(_num(v))}" for k, v in res.values.items()
            if any(s in k for s in _STAB)]
    by = {}
    for k, st in res.finality.items():
        by.setdefault(st.state, []).append(k)
    out = "\\section{Stability}\n"
    out += ("Computed stability quantities (replicon / AT / cavity susceptibility): "
            + "; ".join(comp) + ".\n\n") if comp else \
        "No stability quantity (replicon / AT) is computed by this run.\n\n"
    out += "".join(f"{escape(s)}: {escape(', '.join(ks))}.\n\n" for s, ks in sorted(by.items()))
    return out


def _references(res, bib, slow):
    from .refdata import load
    out = "\\section{Reference check}\n"
    if res.entry is None:
        near = res.report.nearest if res.report else None
        return out + escape(f"No banked keys: a composed model is not a catalog entry, so "
                            f"nothing is compared here (nearest entry: {near}).  This document "
                            "does not certify its numbers: a composed model is validated only "
                            "by an independent route (enumeration at small N, or a limit with "
                            "a known answer) in the test suite, where such a test exists.") + "\n"
    try:
        done = {(o.file, o.key): o for o in res.reference_checks(slow=slow)}
    except Refusal as r:
        return out + escape(f"refused: {r.reason}: {r.detail}") + "\n"
    rows = []
    for ref in res.entry.refs:
        if ref.route != res.route:
            continue
        want = load(ref.file)[ref.key]
        o = done.get((ref.file, ref.key))
        if o is None:
            got, verdict = "not run (slow" + (", statistical" if ref.statistical else "") + \
                "; --full)", "--"
        else:
            got, verdict = escape(o.got), ("agrees" if o.ok else "DISAGREES")
            if o.statistical:
                got += f" ({escape(o.detail)})"
        rows.append(f"{escape(ref.key)} & {got} & {escape(want.value)} & {escape(want.tol)} & "
                    f"{verdict} & {bib.cite(want.cite)} \\\\")
    return (out + "Banked values (reference/*.json) against this pipeline's output.\n"
            "\\begin{center}\n\\begin{tabular}{llllll}\nkey & computed & banked & tolerance & "
            "verdict & source \\\\\n\\hline\n" + "\n".join(rows) +
            "\n\\end{tabular}\n\\end{center}\n")


def _caveat(res):
    if res.entry is not None:
        return res.entry.caveats[res.route]
    return res.notes[0] if res.notes else ""


def _labels(res):
    out = (f"\\section{{Labels}}\nRigor: \\textbf{{{escape(res.rigor)}}}. "
           f"Route: \\textbf{{{escape(res.route)}}}.\n\n{escape(_caveat(res))}\n\n")
    if res.statistical:
        out += escape(f"statistical: {', '.join(res.statistical)} are sampling estimates "
                      "(value +- stderr over independent runs), not certified.") + "\n\n"
    rest = [n for n in res.notes if n and n != _caveat(res) and "NOT final" not in n]
    if rest:
        out += "\\begin{itemize}\n" + "".join(f"\\item {escape(n)}\n" for n in rest) + \
            "\\end{itemize}\n"
    return out


def _result_body(res, bib, extra, slow):
    spec = res.spec or (res.entry.spec if res.entry is not None else None)
    if spec is None:
        raise Refusal("precondition:result-spec", "the Result carries no ModelSpec "
                      "(compose.ComposedCalculation.run sets it)")
    sp_txt = ", ".join(f"{f} = {getattr(spec, f)}" for f in
                       ("family", "variables", "disorder", "constraint", "prior"))
    out = ["\\section{Model}\n" + escape(sp_txt) + ".\n\n" +
           (escape(res.entry.description) + " " + bib.cite(res.entry.cite) + "\n\n"
            if res.entry else "") + _blocks(_model_blocks(res), bib)]
    out.append(f"\\section{{Method and route}}\nRoute: \\textbf{{{escape(res.route)}}} "
               f"({escape(ROUTES.get(res.route, res.route))}). Rigor: "
               f"\\textbf{{{escape(res.rigor)}}}. {escape(_caveat(res))}\n")
    if res.report is not None and res.report.kind == "composed":
        out.append("\\subsection{Reused and adapted steps}\n" +
                   _blocks(report_blocks(res.report), bib))
    rows, a_cite, why = _derivation(res)
    composed = res.entry is None
    if composed or rows[0][0] is not None:
        intro = ("Level (a): each step is rendered from the sympy objects that step put in "
                 "the State (sympy.latex); nothing is retyped.")
    elif res.route == "cavity":
        intro = ("The cavity argument as the cavity code implements it (cavity/*.py), step "
                 "by step; each equation in sympy form checked against the code or cited.")
    else:
        intro = (f"No level-(a) derivation for this model ({why}): the governing equations "
                 "the family code implements are shown, each in sympy form checked against "
                 "the code or cited; no intermediate step the code does not perform is shown.")
    out.append("\\section{Derivation}\n" + escape(intro) + "\n")
    if not composed and rows[0][0] is not None:
        out.append(escape("The level-(a) steps below derive the same model symbolically "
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
    from .families.common import CAVEAT
    entries = r.trace.entries if r.trace is not None else []
    out = ["\\section{Refused calculation}\n" + escape(
        f"Refused: {r.reason}" + (f" at step {r.step}" if r.step else "") +
        (f": {r.detail}" if r.detail else "") + ". No value is reported; the steps "
        "applied before the refusal are shown below.") + "\n"]
    out.append("\\section{Partial derivation}\n" + (
        "".join(_entry(k, e, bib, e.tex, LEVEL_A_CITE) for k, e in enumerate(entries, 1))
        or escape("The refusal came before any step was applied.") + "\n"))
    out.append("\\section{Labels}\nRigor: \\textbf{physics-level}. " + escape(CAVEAT) + "\n")
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
    return (PREAMBLE + f"\\title{{{escape(title)}}}\n\\date{{}}\n\\begin{{document}}\n"
            "\\maketitle\n" + body + "\\end{document}\n")


_OFFLINE = re.compile(r"network|download|resolve host|could not connect|offline|"
                      r"failed to fetch", re.I)


def compile_pdf(tex: Path) -> Path:
    engine = shutil.which("tectonic") or shutil.which("pdflatex")
    if engine is None:
        raise Refusal("precondition:tex-engine", "neither tectonic nor pdflatex is on PATH")
    d = str(tex.parent)
    if Path(engine).name == "tectonic":
        runs = [[engine, "--chatter", "minimal", "--outdir", d, str(tex)]]
    else:
        runs = [[engine, "-interaction=nonstopmode", "-halt-on-error",
                 f"-output-directory={d}", str(tex)]] * 2      # twice: citations
    for cmd in runs:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            log = (p.stdout + p.stderr)[-1500:]
            reason = "tex-engine:unavailable" if _OFFLINE.search(log) else "tex-engine:failed"
            raise Refusal(reason, f"{Path(engine).name} exit {p.returncode}: {log}")
    pdf = tex.with_suffix(".pdf")
    if not pdf.exists():
        raise Refusal("tex-engine:failed", f"{engine} wrote no {pdf.name}")
    return pdf


def write_latex(result, path, *, pdf=False, **kw) -> Path:
    """Write the document to `path`; with pdf=True also compile it and return the PDF."""
    path = Path(path)
    path.write_text(render_latex(result, **kw))
    return compile_pdf(path) if pdf else path


__all__ = ["TexBlock", "compile_pdf", "document_blocks", "render_latex", "report_blocks",
           "write_latex"]
