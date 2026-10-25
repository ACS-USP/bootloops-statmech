"""LaTeX derivation writer (latex.py, tex.py, latex_entries.py, latex_cavity.py).

The default tier renders documents from cheap catalog runs (headline quantities, the
dense SK cavity at one state point, the T = 0 sparse cavity) and checks the displayed
equations against the family code at their parameter points (the checks marked slow,
which need a saddle solve per point, run with --full).  Composed (level-(a) + solver)
documents and the PDF compiles are slow."""
import io
import os
import re
import shutil
import sys
import tempfile
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import sympy as sp

from replicant import derive as D
from replicant.catalog import catalog
from replicant.cli import main
from replicant.compose import compose
from replicant.core import Refusal, State, run_pipeline
from replicant.families.common import CAVEAT, CAVITY_CAVEAT
from replicant.latex import document_blocks, render_latex, write_latex
from replicant.spec import ModelSpec
from replicant.testing import slow
from replicant.tex import Check, TexBlock, escape, invented, prose, render, verify_block
from replicant.tex_print import WIDTH, lines, width

# cheap parameters per (entry, route): every route of every entry gets a document
CHEAP = {("sk", "cavity"): {"beta": "1.5", "h": "0.3"},
         ("hopfield", "cavity"): {"alpha": "0.05", "beta": 3},
         ("sk", "amp"): {"beta": "1.5", "h": "0.3"}, ("teacher_student", "amp"): {"alpha": "2"},
         ("spiked_wigner", "amp"): {"snr": "2"}, ("sbm", "cavity"): {"c": 3},
         ("ridge", "replica"): {"gamma": "2", "lam": "0.1"}}
PM_ONE_SK = ModelSpec("sk", "ising", "pm_one", params={"beta": "0.5"})


@lru_cache(maxsize=None)
def _result(name, route):
    e = catalog()[name]
    return compose(e.spec, route).run(**CHEAP.get((name, route), {}))


@lru_cache(maxsize=None)
def _doc(name, route):
    return render_latex(_result(name, route))


def _routes():
    return [(n, r) for n, e in catalog().items() for r in sorted(e.routes)]


def _subsections(doc):
    """[(title, body)] for every \\subsection of the document."""
    parts = re.split(r"\\subsection\*?\{", doc)[1:]
    out = []
    for p in parts:
        depth, i = 1, 0
        while depth:
            depth += {"{": 1, "}": -1}.get(p[i], 0)
            i += 1
        out.append((p[:i - 1], p[i:]))
    return out


def _has_math(body):
    return any(f"\\begin{{{e}" in body for e in ("equation", "align", "multline"))


# ---- structure ------------------------------------------------------------------------
def test_sk_latex_contains_steps_in_order():
    doc = _doc("sk", "replica")
    subs = _subsections(doc)
    titles = [t for t, _ in subs]
    order = []
    for step in ("replicate", "average-disorder", "introduce-order-parameters", "rs-ansatz",
                 "replica-limit", "rs-stationarity"):
        i = next(k for k, t in enumerate(titles) if step in t)
        assert _has_math(subs[i][1]), (step, subs[i][1][:400])
        order.append(i)
    assert order == sorted(order), order
    # then the calculation at the requested parameters, in trace order
    run = [titles.index(next(t for t in titles if s in t)) for s in
           ("sk-critical-temperature", "sk-ground-energy", "sk-rs-saddle", "sk-1rsb-saddle")]
    assert run == sorted(run) and run[0] > order[-1], run
    for sec in ("Model", "Method and route", "Solution", "Stability", "Reference check",
                "Labels"):
        assert f"\\section{{{sec}" in doc, sec
    assert "\\begin{thebibliography}" in doc and doc.rstrip().endswith("\\end{document}")


def test_every_catalog_entry_has_derivation():
    for name, route in _routes():
        e = catalog()[name]
        blocks = e.derivation(route)
        assert any(b.kind in ("equation", "align") for b in blocks), (name, route)
        assert any(b.source for b in blocks), (name, route)       # a paper citation
        doc = _doc(name, route)
        assert doc.count("\\bibitem") >= 1 and "\\cite{" in doc, (name, route)
        for ref in e.refs:
            if ref.route == route:
                row = next((l for l in doc.splitlines() if l.startswith(escape(ref.key) + " &")),
                           None)
                assert row is not None, (name, route, ref.key)
                assert ("not run" in row) == (ref.slow or ref.statistical), row


def test_latex_environments_balanced_and_specials_escaped():
    assert escape("q_EA 50% & #1 $x^2~{a}\\") == \
        "q\\_EA 50\\% \\& \\#1 \\$x\\^{}2\\~{}\\{a\\}\\textbackslash{}"
    assert escape("n -> 0, a < b, |z|") == \
        "n -\\textgreater{} 0, a \\textless{} b, \\textbar{}z\\textbar{}"
    assert "\\usepackage[T1]{fontenc}" in _doc("rem", "replica")      # guarded for pdflatex
    for name, route in _routes():
        doc = _doc(name, route)
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([a-z*]+)\}", doc):
            if kind == "begin":
                stack.append(env)
            else:
                assert stack and stack.pop() == env, (name, route, env)
        assert not stack, (name, route, stack)
        for line in doc.splitlines():   # bibliography items and text are escaped
            if line.startswith("\\bibitem"):
                assert not re.search(r"(?<!\\)[_%&#]", line.split("}", 1)[1]), line
        text_mode = re.sub(r"\\begin\{(equation|align|multline)\}.*?\\end\{\1\}", "", doc,
                           flags=re.S)
        text_mode = re.sub(r"\$[^$]*\$|\\(cite|bibitem|begin|end)\{[^}]*\}", "", text_mode)
        for line in text_mode.splitlines():      # every text-mode line is escaped
            assert not re.search(r"(?<!\\)[_%#]", line), (name, route, line)
    # a trace summary with specials is escaped where it is shown
    sk = _doc("sk", "replica")
    assert "T = 0 energies; RS replicon at T = 0.5" in sk
    assert escape("rs_ground_energy") in sk and "rs_ground_energy" not in sk


def test_no_equation_is_invented():
    n = 0
    for name, route in _routes():
        for b in document_blocks(_result(name, route)):
            assert b.kind in ("equation", "align", "text"), b
            assert not invented(b), b
            if b.expr is not None:
                assert b.body == render(b.expr), b.title
            n += 1
    assert n > 40, n
    bare = TexBlock("equation", "made up", "E = m c^2")
    assert invented(bare)
    try:
        render_latex(_result("rem", "replica"), extra=[bare])
    except Refusal as r:
        assert r.reason == "precondition:equation-source" and "made up" in r.detail
    else:
        raise AssertionError("an equation with neither a sympy expr nor a source was rendered")
    assert not invented(TexBlock("equation", "cited", "E = m c^2", source="Einstein (1905)"))


# ---- the displayed equations are the implemented ones ---------------------------------
def _checked_blocks():
    out = []
    for name, route in _routes():
        for b in catalog()[name].derivation(route):
            if b.expr is not None:
                assert b.check is not None, (name, route, b.title)
            if b.check is not None:          # incl. cited blocks with a hand-coded display
                out.append((name, route, b))
    return out


def test_displayed_equations_match_implementation():
    dps = mp.mp.dps
    blocks = _checked_blocks()
    fast = [x for x in blocks if not x[2].check.slow]
    assert len(fast) >= 15, len(fast)
    for name, route, b in fast:     # default tier: the first point of each (--full: all)
        bad = verify_block(b, max_points=1)
        assert not bad, (name, route, b.title, bad)
        pts = b.check.points() if callable(b.check.points) else b.check.points
        assert len(pts) == 3 or not b.check.args, (name, b.title)
    assert {n for n, _, _ in blocks} >= {"sk", "rem", "pspin", "perceptron", "hopfield",
                                         "teacher_student", "viana_bray"}
    assert mp.mp.dps == dps


@slow
def test_displayed_equations_match_implementation_all_points():
    """Every checked block at all its points, including the ones that need a saddle
    solve per point (Check.slow)."""
    dps = mp.mp.dps
    for name, route, b in _checked_blocks():
        bad = verify_block(b)
        assert not bad, (name, route, b.title, bad)
    assert mp.mp.dps == dps


def test_consistency_check_catches_a_flipped_sign():
    """Negative control: the same check on a deliberately wrong expression fails."""
    b = next(b for b in catalog()["sk"].derivation("replica") if "replicon" in b.title.lower())
    assert not verify_block(b, max_points=2)
    wrong_rhs = sp.Eq(b.expr.lhs, 2 - b.expr.rhs)      # 1 - x  ->  1 + x
    wrong = TexBlock(b.kind, b.title, render(wrong_rhs), wrong_rhs, check=b.check)
    bad = verify_block(wrong, max_points=2)          # (all 3 points: the --full test)
    assert len(bad) == 2, bad


# ---- refusals, labels, finality -------------------------------------------------------
def test_refusal_renders_partial_derivation():
    assert Refusal("x").trace is None                      # backward compatible
    spec = ModelSpec("sk", "ising", "gaussian")
    n = sp.Symbol("n", positive=True)
    steps = [D.replicate(spec, n), D.average_disorder(), D.introduce_order_parameters(),
             D.saddle_equations()]
    try:
        run_pipeline(steps, State())
    except Refusal as r:
        ref = r
    else:
        raise AssertionError("saddle-equations at symbolic n must refuse")
    assert ref.reason == "precondition:explicit-replica-number", ref.reason
    assert [e.step for e in ref.trace.entries] == ["replicate", "average-disorder",
                                                   "introduce-order-parameters"]
    assert ref.step == "saddle-equations"
    doc = render_latex(ref, title="SK at symbolic n")
    subs = _subsections(doc)
    for step in ("replicate", "average-disorder", "introduce-order-parameters"):
        assert any(step in t and _has_math(b) for t, b in subs), step
    assert "Refused" in doc and escape("precondition:explicit-replica-number") in doc
    assert escape("saddle-equations") in doc
    # a refusal inside a catalog pipeline carries its partial trace too
    try:
        compose(catalog()["sk"].spec, "cavity").run()
    except Refusal as r:
        assert r.reason == "precondition:state-point" and r.trace is not None
        assert r.trace.entries == [] and r.step == "cavity-model"
        assert escape("the sk cavity route needs beta") in render_latex(r)


def test_refusal_document_says_trace_numbers_are_diagnostics():
    """Final review, Minor 5: the partial trace below a require_final refusal prints q,
    phi, replicon digits, so the document must not claim that no number appears."""
    try:
        compose(catalog()["sk"].spec).run(beta=2, require_final=True)
    except Refusal as r:
        ref = r
    else:
        raise AssertionError("SK RS at beta = 2 is unstable: require_final must refuse")
    doc = render_latex(ref)
    assert "No value is reported; " not in doc
    flat = " ".join(doc.split())
    assert ("No value is reported as a result; numbers in the trace are diagnostics of the "
            "refused computation, not results.") in flat, flat[:800]


def test_pdflatex_argv_disables_shell_escape_and_guards_the_path():
    """Final review, Minor 3: argv only (no engine is run)."""
    from pathlib import Path
    from replicant.latex_pdf import engine_commands
    runs = engine_commands("/usr/bin/pdflatex", Path("-evil.tex"))
    assert len(runs) == 2 and runs[0] == runs[1]
    assert "-no-shell-escape" in runs[0] and runs[0][-1] == "./-evil.tex", runs[0]
    assert not any(a.startswith("-") and a.endswith(".tex") for a in runs[0]), runs[0]
    runs = engine_commands("/usr/bin/pdflatex", Path("/tmp/d/doc.tex"))
    assert runs[0][-1] == "/tmp/d/doc.tex" and "-output-directory=/tmp/d" in runs[0]
    (t,) = engine_commands("/opt/bin/tectonic", Path("-x/-y.tex"))
    assert t[-1] == "./-x/-y.tex" and t[t.index("--outdir") + 1] == "./-x", t


def test_statistical_and_physics_labels_present():
    for name, route in _routes():
        doc = _doc(name, route)
        assert "physics-level" in doc and f"Route: \\textbf{{{route}}}" in doc, (name, route)
    vb = compose(catalog()["viana_bray"].spec, "cavity").run(
        beta="0.8", pop=200, sweeps=4, burn_in=4, runs=2, seed=3)
    doc = render_latex(vb)
    assert set(vb.statistical) == {"q", "e", "f"}
    assert "statistical" in doc and "stderr" in doc and prose(CAVITY_CAVEAT) in doc
    for k in vb.statistical:
        row = next(l for l in doc.splitlines() if l.startswith(f"{escape(k)} &"))
        assert "statistical" in row and "\\pm" in row, row


def test_latex_shows_per_key_finality():
    doc = _doc("sk", "replica")
    row = next(l for l in doc.splitlines() if l.startswith(escape("rs_ground_energy") + " &"))
    assert "unstable" in row and "no" in row.split("&")[-1], row
    assert "NOT final" in doc and "computed RS replicon" in doc
    tc = next(l for l in doc.splitlines() if l.startswith("tc &"))
    assert "stable" in tc and "yes" in tc.split("&")[-1], tc
    assert "not-computed" in doc          # the 1RSB ground energy


def test_cavity_document_uses_cavity_caveat():
    from replicant.cavity.dense import cavity_hopfield_rs, cavity_sk_rs
    assert cavity_sk_rs("1.5", "0.3").caveat == CAVITY_CAVEAT
    assert cavity_hopfield_rs("0.05", 3).caveat == CAVITY_CAVEAT
    for name in ("sk", "hopfield", "perceptron", "viana_bray"):
        doc = _doc(name, "cavity")
        assert prose(CAVITY_CAVEAT) in doc and prose(CAVEAT) not in doc, name
        assert "Route: \\textbf{cavity}" in doc
    for step in ("remove_site", "cavity_field_law", "add_site", "self_consistency"):
        assert escape(step) in _doc("sk", "cavity"), step


def test_composed_model_document_names_adapted_steps():
    """Cheap tier: the composition report section, from the composer alone (no solve)."""
    from replicant.latex import report_blocks
    calc = compose(PM_ONE_SK)
    text = "\n".join(escape(b.body) for b in report_blocks(calc.report))
    assert escape(calc.report.relative_to) in text and "sk" in text
    assert escape("average-disorder") in text and escape("disorder:pm_one") in text
    for s in calc.report.reused:
        assert escape(s) in text, s


@slow
def test_composed_model_document_full():
    res = compose(PM_ONE_SK).run()
    doc = render_latex(res)
    assert "Composed model" in doc and escape(res.report.relative_to) in doc
    assert escape("disorder:pm_one") in doc and "\\cosh" in doc     # the pm_one cumulant
    subs = _subsections(doc)
    for step in ("replicate", "average-disorder", "introduce-order-parameters", "rs-ansatz",
                 "replica-limit", "rs-stationarity", "solve-derived"):
        assert any(step in t for t, _ in subs), step
    assert "No banked keys" in doc
    assert not _raw_code_in_prose(doc), _raw_code_in_prose(doc)[:5]


# ---- writer, CLI, PDF -----------------------------------------------------------------
def test_cli_latex_flag_writes_tex():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rem.tex")
        out, err = io.StringIO(), io.StringIO()
        code = main(["run", "rem", "--out", os.path.join(d, "r.json"), "--latex", path],
                    stdout=out, stderr=err)
        assert code == 0, err.getvalue()
        text = open(path).read()
        assert text.startswith("\\documentclass") and escape("rem-critical-temperature") in text
        # a refused run still writes the partial document and exits 2
        bad = os.path.join(d, "bad.tex")
        code = main(["run", "sk", "--route", "cavity", "--latex", bad], stdout=out, stderr=err)
        assert code == 2 and "Refused" in open(bad).read()
        assert "REFUSED: precondition:state-point" in err.getvalue()
        # write_latex returns the .tex path without pdf
        p = write_latex(_result("rem", "replica"), os.path.join(d, "w.tex"))
        assert p.suffix == ".tex" and p.read_text().startswith("\\documentclass")
        code = main(["run", "rem", "--pdf"], stdout=out, stderr=err)
        assert code == 2 and "precondition:latex-path" in err.getvalue()


def _engine_or_skip():
    if not (shutil.which("tectonic") or shutil.which("pdflatex")):
        print("SKIP test_pdf_compiles: neither tectonic nor pdflatex is on PATH")
        return False
    return True


@slow
def test_pdf_compiles():
    if not _engine_or_skip():
        return
    docs = {"sk": _result("sk", "replica"), "hopfield": _result("hopfield", "replica"),
            "sk_cavity": _result("sk", "cavity"), "sk_pm_one": compose(PM_ONE_SK).run()}
    with tempfile.TemporaryDirectory() as d:
        for name, res in docs.items():
            try:
                pdf = write_latex(res, os.path.join(d, f"{name}.tex"), pdf=True)
            except Refusal as r:
                if r.reason == "tex-engine:unavailable":
                    print(f"SKIP test_pdf_compiles: {r.detail}")
                    return
                raise
            assert pdf.suffix == ".pdf" and pdf.stat().st_size > 1000, (name, pdf)


# ---- typesetting (fix round 1) --------------------------------------------------------
def test_long_equation_breaks_at_terms_and_stays_the_same_object():
    x = sp.symbols("x0:30")
    q, n = sp.IndexedBase("q"), sp.Symbol("n", positive=True)
    a, b = sp.symbols("a b", integer=True, positive=True)
    pair = sp.Sum(q[a, b]**2, (b, a + 1, n), (a, 1, n))
    rhs = sum(sp.log(1 + xi**2) * xi for xi in x) - pair + sp.Symbol("alpha") * (
        sum(sp.log(1 + xi) for xi in x[:12]))
    eq = sp.Eq(sp.Symbol("Phi"), rhs)
    rows = lines(eq.lhs, eq.rhs)
    assert len(rows) > 3 and all(width(r) <= WIDTH for r in rows), [width(r) for r in rows]
    b_ = TexBlock("equation", "long", render(eq), eq)
    assert not invented(b_) and "\\\\" in b_.body
    assert "\\sum_{a < b \\leq n}" in b_.body and "substack" not in b_.body
    assert "\\Bigl(" in b_.body and "\\Bigr)" in b_.body      # alpha (...) split inside
    # the broken lines are the same terms: re-joined they equal the unbroken printout
    flat = lambda s: s.replace("\\quad", "").replace("\\Bigl(", "\\left(").replace(
        "\\Bigr)", "\\right)").replace("\\\\", "").replace(" ", "").replace("\n", "")
    from replicant.tex_print import latex
    assert flat(render(eq)) == flat(f"{latex(eq.lhs)} = {latex(eq.rhs)}")
    short = sp.Eq(sp.Symbol("q"), x[0] + 1)
    assert render(short) == "q = x_{0} + 1"
    doc = render_latex(_result("rem", "replica"), extra=[b_])
    assert "\\begin{multline}" in doc


def test_refused_cavity_document_carries_cavity_caveat():
    try:
        compose(catalog()["sk"].spec, "cavity").run()
    except Refusal as r:
        assert r.route == "cavity"
        doc = render_latex(r)
        assert prose(CAVITY_CAVEAT) in doc and prose(CAVEAT) not in doc
        assert "Route: \\textbf{cavity}" in doc
    try:
        run_pipeline([D.saddle_equations()], State())
    except Refusal as r:
        assert r.route is None and prose(CAVEAT) in render_latex(r)


def test_cli_keeps_the_json_when_the_document_fails():
    from replicant import latex_pdf
    with tempfile.TemporaryDirectory() as d:
        out, err = io.StringIO(), io.StringIO()
        j, t = os.path.join(d, "r.json"), os.path.join(d, "r.tex")
        which, latex_pdf.shutil.which = latex_pdf.shutil.which, lambda name: None
        try:
            code = main(["run", "rem", "--out", j, "--latex", t, "--pdf"], stdout=out, stderr=err)
        finally:
            latex_pdf.shutil.which = which
        assert code == 2 and os.path.exists(j) and os.path.exists(t), err.getvalue()
        assert "REFUSED: latex: precondition:tex-engine" in err.getvalue()
        err = io.StringIO()
        code = main(["run", "rem", "--out", j, "--latex", os.path.join(d, "no", "x.tex")],
                    stdout=out, stderr=err)
        assert code == 1 and "ERROR: latex: FileNotFoundError" in err.getvalue()
        assert "Traceback" not in err.getvalue() and os.path.exists(j)


def test_bibliography_lists_each_reference_once_and_no_code_notation():
    doc = _doc("sk", "replica")
    bib = [l for l in doc.splitlines() if l.startswith("\\bibitem")]
    assert sum("Sherrington \\& Kirkpatrick" in l for l in bib) == 1, bib
    assert sum(bool(re.match(r"\\bibitem\{b\d+\} Parisi, J\. Phys\. A 13, L115", l))
               for l in bib) == 1, bib
    for name, route in _routes():
        body = _doc(name, route).split("\\begin{document}")[1]
        text_only = re.sub(r"\\begin\{(equation|align|multline)\}.*?\\end\{\1\}", "", body,
                           flags=re.S)
        assert "**" not in text_only, (name, route)
    assert "\\sloppy" in doc


@slow
def test_compiled_documents_have_no_overfull_display():
    """Hopfield (the widest level-(a) functionals), SK, SK cavity, the p-spin functional, the
    sparse cavity and the composed +-J SK: no display wider than the text block by 1 pt."""
    import warnings
    from replicant.latex import OverfullDisplay, overfull
    if not _engine_or_skip():
        return
    docs = {"hopfield": _result("hopfield", "replica"), "sk": _result("sk", "replica"),
            "sk_cavity": _result("sk", "cavity"), "pspin": _result("pspin", "replica"),
            "viana_bray": _result("viana_bray", "cavity"),
            "sk_pm_one": compose(PM_ONE_SK).run()}
    x = sp.symbols("x0:40")
    probe = sp.Eq(sp.Symbol("F"), sp.exp(sum(xi**2 for xi in x)) * sp.Symbol("a"))
    extra = {"sk": [TexBlock("equation", "indivisible probe", render(probe), probe)]}
    with tempfile.TemporaryDirectory() as d:
        for name, res in docs.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OverfullDisplay)
                try:
                    write_latex(res, os.path.join(d, f"{name}.tex"), pdf=True,
                                extra=extra.get(name, ()))
                except Refusal as r:
                    if r.reason == "tex-engine:unavailable":
                        print(f"SKIP test_compiled_documents_have_no_overfull_display: {r.detail}")
                        return
                    raise
            log = os.path.join(d, f"{name}.log")
            assert os.path.exists(log), name
            bad = [o for o in overfull(log, 1.0) if o[2] == "display"]
            assert not bad and not caught, (name, bad, [str(w.message) for w in caught])
    # the detector itself sees a planted overfull display
    planted = "Overfull \\hbox (74.6pt too wide) detected at line 67\n" \
              "Overfull \\hbox (3.0pt too wide) in paragraph at lines 3--4\n"
    assert overfull(planted) == [(74.6, 67, "display"), (3.0, 3, "paragraph")]


# ---- prose and robustness (fix round 2) -------------------------------------------------
PROSE_TABLE = {   # every pattern class found in the notes of the 11 catalog + composed docs
    "m_a": "$m_{a}$",
    "q_ab": "$q_{ab}$",
    "qhat_ab": "$\\hat{q}_{ab}$",
    "s_i^a": "$s_{i}^{a}$",
    "Phi_n^RS": "$\\Phi _{n}^{RS}$",
    "s_i^a/sqrt(N),": "$s_{i}^{a}/\\sqrt{N}$,",
    "sqrt(1 - q)": "$\\sqrt{1 - q}$",
    "n->0": "$n \\to 0$",
    "a -> b": "a $\\to$ b",
    "lim_{n->0}": "$\\lim _{n \\to 0}$",
    "-t**4/(12*N**2)": "$-t^{4}/(12\\,N^{2})$",
    "2.25*q;": "$2.25\\,q$;",
    "sum_i": "$\\sum _{i}$",
    "prod_a": "$\\prod _{a}$",
    "E_z": "$E_{z}$",
    "[E_z g]^n,": "$[E_{z} g]^{n}$,",
    "(sum_a s_a)^2,": "$(\\sum _{a} s_{a})^{2}$,",
    "tanh^2(beta_c J)": "$\\tanh ^{2}(\\beta _{c} J)$",
    "int_x^inf": "$\\int _{x}^{\\infty }$",
    "g(y*)^2/2": "$g(y^*)^{2}/2$",
    "(a != b);": "$(a  \\neq  b)$;",
    "alpha_c(kappa),": "$\\alpha _{c}(\\kappa )$,",
    "ansatz.n_to_zero,": "\\texttt{ansatz.n\\_to\\_zero},",
    "ansatz.sum_offdiag_power('rs',": "\\texttt{ansatz.sum\\_offdiag\\_power('rs'},",
    "(families/sk.py": "(\\texttt{families/sk.py}",
    "rs_ground_energy:": "\\texttt{rs\\_ground\\_energy}:",
    "q_rs_T0,": "\\texttt{q\\_rs\\_T0},",
    "e0_rs": "\\texttt{e0\\_rs}",
    "N_scale": "\\texttt{N\\_scale}",
    "pm_one;": "\\texttt{pm\\_one};",
    "factor^n;": "\\texttt{factor\\^{}n};",          # an English word in it: code font
    "s_a)^2,": "\\texttt{s\\_a)\\^{}2},",             # unbalanced fragment: code font
    "beta = 2": "beta = 2",                          # plain prose stays text
    "50% & #1": "50\\% \\& \\#1",
}


def test_prose_converts_the_code_notation_of_the_notes():
    for src, want in PROSE_TABLE.items():
        assert prose(src) == want, (src, prose(src), want)
    for src in PROSE_TABLE:
        out = prose(src)
        assert out.count("$") % 2 == 0 and _brace_balanced(out), (src, out)


def _brace_balanced(s):
    depth, i = 0, 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth < 0:
            return False
        i += 1
    return depth == 0


def _strip_texttt(s):
    out, i = "", 0
    while True:
        j = s.find("\\texttt{", i)
        if j < 0:
            return out + s[i:]
        out, k, depth = out + s[i:j], j + 8, 1
        while depth:
            if s[k] == "\\":
                k += 2
                continue
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        i = k


def _raw_code_in_prose(doc):
    body = doc.split("\\begin{document}")[1].split("\\begin{thebibliography}")[0]
    body = re.sub(r"\\begin\{(equation|align|multline|tabular)\}.*?\\end\{\1\}", "", body,
                  flags=re.S)
    body = re.sub(r"\$[^$]*\$", "", _strip_texttt(body))
    return [l for l in body.splitlines() if not l.startswith("\\subsection")
            and re.search(r"\\_|\\\^\{\}|\\textgreater", l)]


def test_documents_have_no_raw_code_notation_in_prose():
    for name in ("sk", "hopfield"):
        bad = _raw_code_in_prose(_doc(name, "replica"))
        assert not bad, (name, bad[:5])
    for name, route in _routes():                 # math mode and braces stay balanced
        doc = _doc(name, route)
        assert _brace_balanced(doc), (name, route)
        for line in doc.splitlines():
            if not line.startswith(("\\begin", "\\end")):
                assert line.replace("\\$", "").count("$") % 2 == 0, (name, route, line)
        for b in re.findall(r"\\begin\{(?:equation|align|multline)\}(.*?)\\end", doc, re.S):
            assert "+ - " not in b and "- - " not in b, (name, route, b[:200])


def test_step_note_is_printed_once():
    doc = _doc("hopfield", "replica")
    note = "condensed pattern gauged to +1"
    assert doc.count(note) == 1, doc.count(note)
    assert "Trace: set avg\n" in doc


def test_indivisible_wide_term_is_scaled_to_the_line():
    x = sp.symbols("x0:40")
    probe = sp.Eq(sp.Symbol("F"), sp.exp(sum(xi**2 for xi in x)) * sp.Symbol("a"))
    body = render(probe)
    assert "\\resizebox" in body and body.startswith("F = \\resizebox"), body[:80]
    assert render([probe]).startswith("F &= \\resizebox")          # the & stays outside
    assert not invented(TexBlock("equation", "probe", body, probe))


def test_missing_log_is_an_audit_failure():
    import warnings
    from replicant import latex_pdf
    from replicant.latex import OverfullDisplay

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):            # a TeX engine that writes a PDF and no log
        open(cmd[-1][:-4] + ".pdf", "wb").write(b"%PDF-1.5 stub" * 200)
        return Done()
    with tempfile.TemporaryDirectory() as d:
        tex = os.path.join(d, "x.tex")
        open(tex, "w").write("x")
        run, which = latex_pdf.subprocess.run, latex_pdf.shutil.which
        latex_pdf.subprocess.run, latex_pdf.shutil.which = fake_run, lambda n: "/bin/tectonic"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OverfullDisplay)
                latex_pdf.compile_pdf(tex)
        finally:
            latex_pdf.subprocess.run, latex_pdf.shutil.which = run, which
        assert any("overflow audit unavailable: no log" in str(w.message) for w in caught)


def test_cli_unwritable_out_is_an_error_not_a_traceback():
    out, err = io.StringIO(), io.StringIO()
    code = main(["run", "rem", "--out", "/nonexistent-dir/r.json"], stdout=out, stderr=err)
    assert code == 1 and "ERROR: --out: FileNotFoundError" in err.getvalue()


def test_replicon_derivation_in_gauss_entry_documents():
    # E1: the documents of the perceptron and Hopfield entries (and SK) show the level-(a)
    # replicon, rendered from the sympy objects derive_replicon holds; the Gaussian-energy
    # entries show the covariance-derivative pieces
    for name in ("perceptron", "hopfield", "sk"):
        subs = dict(_subsections(_doc(name, "replica")))
        body = subs.get("Replicon of the RS saddle (level (a))")
        assert body is not None and _has_math(body), (name, list(subs)[:12])
        assert r"\lambda_{R}" in body and r"\chi_{R}" in body, name
        if name != "sk":
            assert "Price covariance derivative" in body and r"G_{0}" in body, name
    blocks = document_blocks(_result("hopfield", "replica"))
    rep = [b for b in blocks if b.title.startswith("Energy replicon eigenvalue with")]
    assert rep and rep[0].expr is not None and not invented(rep[0])
