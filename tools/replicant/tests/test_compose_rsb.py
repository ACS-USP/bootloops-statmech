"""Composer route ansatz="1rsb" (compose_rsb.py): the report relative to the RS level-(a)
derivation of the same spec, refusals, and a composed Ising perceptron whose 1RSB values
are never presented as final (1RSB stability is not computed: E4).

Conventions as in test_derive_rsb.py; the Ising perceptron spec is not a catalog entry.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.agree import default_spec
from replicant.compose import compose
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _ising_perceptron():
    s = default_spec("perceptron")
    return ModelSpec("perceptron", "ising", s.disorder, s.constraint, s.prior, dict(s.params))


def test_compose_1rsb_reports_steps_relative_to_rs():
    c = compose(_ising_perceptron(), ansatz="1rsb")
    rep = c.report
    assert rep.kind == "composed" and "RS level-(a) derivation of the same spec" in rep.relative_to
    assert rep.reused == ["replicate", "average-disorder", "introduce-order-parameters"]
    assert [(a.step, a.hook) for a in rep.adapted] == [
        ("rs-ansatz", "rsb-ansatz"), ("replica-limit", "rsb-replica-limit"),
        ("rs-stationarity", "rsb-stationarity"), ("solve-derived", "solve-derived-1rsb")]
    assert all(a.was == "rs" and a.now == "1rsb" for a in rep.adapted)
    assert [s.name for s in c.steps][3:] == ["rsb-ansatz", "rsb-replica-limit",
                                             "rsb-stationarity", "solve-derived-1rsb"]


def test_compose_1rsb_report_is_computed_from_the_step_lists():
    from replicant.compose_rsb import rsb_report
    rep = rsb_report(_ising_perceptron(), {"zero_temperature": True})
    assert rep.reused == ["replicate", "average-disorder", "introduce-order-parameters"]
    assert [a.hook for a in rep.adapted] == ["rsb-ansatz", "rsb-replica-limit",
                                             "zero-temperature-limit", "rsb-stationarity",
                                             "solve-derived-1rsb"], rep.adapted
    assert rep.adapted[2].step == "(new step)", rep.adapted[2]


def test_compose_unknown_ansatz_and_route_refuse():
    assert _refusal(lambda: compose(_ising_perceptron(), ansatz="2rsb")).reason == \
        "missing-step:ansatz:2rsb"
    assert _refusal(lambda: compose(_ising_perceptron(), "cavity", ansatz="1rsb")).reason == \
        "missing-step:route:cavity"


@slow
def test_composed_ising_perceptron_1rsb_capacity_is_not_final():
    c = compose(_ising_perceptron(), ansatz="1rsb")
    res = c.run(frozen=True, solve_for="alpha", kappa=0)
    assert res.values["ansatz"] == "1rsb" and res.stable is None and res.final is False
    assert all(st.state == "not-computed" and st.ansatz == "1rsb" for st in res.finality.values())
    assert all("no independent target" in st.note for st in res.finality.values())
    assert check(load("perceptron")["ising_1rsb_capacity"], mp.nstr(res.values["alpha"], 15),
                 label="composed frozen alpha_c").ok
    assert any("NOT final" in n for n in res.notes)
    r = _refusal(lambda: c.run(frozen=True, solve_for="alpha", kappa=0, require_final=True))
    assert r.reason == "not-final:1rsb", r
    from replicant.latex import render_latex
    tex = render_latex(res)
    for title in ("One-step replica symmetry breaking", "Continuation n", "1RSB saddle-point",
                  "Solution of the derived 1RSB equations", "Reused and adapted steps"):
        assert title in tex, title
