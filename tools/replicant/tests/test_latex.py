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
from replicant.tex import Check, TexBlock, escape, invented, render, verify_block

# cheap parameters per (entry, route): every route of every entry gets a document
CHEAP = {("sk", "cavity"): {"beta": "1.5", "h": "0.3"},
         ("hopfield", "cavity"): {"alpha": "0.05", "beta": 3}}
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
    return "\\begin{equation" in body or "\\begin{align" in body


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
        assert len(b.check.resolved_points()) == 3 or not b.check.args, (name, b.title)
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
    assert not verify_block(b)
    wrong_rhs = sp.Eq(b.expr.lhs, 2 - b.expr.rhs)      # 1 - x  ->  1 + x
    wrong = TexBlock(b.kind, b.title, render(wrong_rhs), wrong_rhs, check=b.check)
    bad = verify_block(wrong)
    assert len(bad) == 3, bad
    assert len(verify_block(wrong, max_points=1)) == 1


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


def test_statistical_and_physics_labels_present():
    for name, route in _routes():
        doc = _doc(name, route)
        assert "physics-level" in doc and f"Route: \\textbf{{{route}}}" in doc, (name, route)
    vb = compose(catalog()["viana_bray"].spec, "cavity").run(
        beta="0.8", pop=200, sweeps=4, burn_in=4, runs=2, seed=3)
    doc = render_latex(vb)
    assert set(vb.statistical) == {"q", "e", "f"}
    assert "statistical" in doc and "stderr" in doc and escape(CAVITY_CAVEAT) in doc
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
        assert escape(CAVITY_CAVEAT) in doc and escape(CAVEAT) not in doc, name
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
