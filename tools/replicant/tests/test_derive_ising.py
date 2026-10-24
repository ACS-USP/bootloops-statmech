"""Level (a) for a model not in the derivation catalog: the Ising perceptron.

Model (families/perceptron.py conventions): P = alpha N Gaussian patterns, binary
couplings J in {-1, +1}^N, J stores pattern mu if J.xi^mu / sqrt(N) >= kappa.
Derived by the same steps as the spherical perceptron; only the SITE changes
(derive.SITES["ising"], the trace SK uses).  Expected RS equations (Gardner-
Derrida 1988, Krauth-Mezard 1989; families/perceptron.py):
  q = E_z tanh^2(sqrt(qhat) z),  qhat = alpha E_t E(x)^2 / (1-q),
  x = (kappa - sqrt(q) t)/sqrt(1-q),  E = phi/H.
"""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant.agree import (compare_equations, default_spec, derive_rs,
                             derived_rs_equations, pipeline_rs_equations)
from replicant.core import Refusal
from replicant.gauss import GaussE, numeric
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected Refusal")


def _names(eqs):
    return {s.name for e in eqs for s in e.free_symbols}


def test_ising_perceptron_derived_equals_pipeline():
    spec = default_spec("perceptron_ising")
    assert spec.family == "perceptron" and spec.variables == "ising"
    a, state, trace = derive_rs(spec)
    b = pipeline_rs_equations("perceptron_ising")
    assert len(a) == len(b) == 2
    assert _names(a) == _names(b) == {"alpha", "kappa", "q", "qhat"}
    res = compare_equations(a, b)
    assert res.agree and res.route.startswith("symbolic"), res
    # the site is the Ising trace (Hubbard-Stratonovich), not the spherical volume
    text = trace.render()
    assert "Hubbard-Stratonovich" in text and "LogDet" not in text, text
    F = state["functional"]
    assert not F.entropic.has(D.LogDet) and F.conjugate
    assert "physics-level" in state["rs_free"].caveat


def test_ising_derived_differs_from_spherical():
    # positive control: the same pattern average with the other site is a
    # different saddle, and the comparison says so
    ising = derive_rs(default_spec("perceptron_ising"))
    sph = derive_rs(default_spec("perceptron"))
    assert len(ising[0]) == 2 and len(sph[0]) == 1
    assert sph[1]["rs_free"].expr.has(sp.log(1 - D.Q_RS))         # (1/2) ln det Q, RS
    assert not ising[1]["rs_free"].expr.has(sp.log(1 - D.Q_RS))
    assert ising[0] != sph[0]
    # the comparison cannot be satisfied by the other model's level (b)
    res = compare_equations(sph[0], pipeline_rs_equations("perceptron_ising"))
    assert not res.agree, res
    # (derived Ising vs derived spherical through compare_equations: slow tier)


@slow
def test_ising_and_spherical_derived_sets_do_not_agree():
    a = derived_rs_equations(default_spec("perceptron_ising"))
    b = derived_rs_equations(default_spec("perceptron"))
    assert not compare_equations(a, b).agree
    assert not compare_equations(b, a).agree


def test_planted_error_in_ising_site_hook_is_caught():
    # a wrong factor in the Ising SITE hook (the entropic term doubled) must make
    # the derived Ising perceptron disagree with level (b)
    from dataclasses import replace
    good = D.SITES["ising"]
    D.SITES["ising"] = replace(good, trace=lambda n, q, qh, field=0:
                               2 * good.trace(n, q, qh, field))
    try:
        a = derived_rs_equations(default_spec("perceptron_ising"))
    finally:
        D.SITES["ising"] = good
    # a site-only plant changes the d/dqhat equation; listing it first lets the
    # one-point numeric screen reject it before the (slow) symbolic proof of the
    # untouched d/dq pair (the verdict does not depend on the order)
    res = compare_equations(a[::-1], pipeline_rs_equations("perceptron_ising"))
    assert res.agree is False and res.route == "numeric", res
    assert D.SITES["ising"] is good         # the registry is restored


def test_perceptron_refuses_unsupported_variables_with_reason():
    r = _refusal(lambda: derived_rs_equations(ModelSpec(
        "perceptron", "potts", "gaussian", params={"alpha": 1, "kappa": 0})))
    assert r.reason == "precondition:supported-variables", r
    assert "perceptron" in r.detail and "ising" in r.detail and "spherical" in r.detail, r.detail


@slow
def test_ising_pipeline_equations_are_the_solved_point():
    # the symbolic level-(b) form is tied to the running numeric code: zero
    # residual at a gardner_rs_point, and its expectations are the solver's
    from replicant.families import perceptron as P
    eqs = pipeline_rs_equations("perceptron_ising")
    s = {x.name: x for e in eqs for x in e.free_symbols}
    with mp.workdps(20):
        pnt = P.gardner_rs_point("ising", "0.7", "0.3")
        pt = {s["q"]: pnt.q, s["qhat"]: pnt.qhat, s["kappa"]: pnt.kappa, s["alpha"]: pnt.alpha}
        for e in eqs:
            assert abs(numeric(e.lhs - e.rhs, pt, dps=16)) < 1e-12, e
        by = {e.lhs.name: e for e in eqs}
        G = next(iter(by["qhat"].rhs.atoms(GaussE)))
        q_, k_ = mp.mpf("0.3"), mp.mpf("-0.2")
        assert abs(G.doit_numeric({s["q"]: q_, s["kappa"]: k_}, dps=16)
                   - P._T(q_, 1 - q_, k_).eh2()) < 1e-12
        (Gz,) = by["q"].rhs.atoms(GaussE)
        assert abs(Gz.doit_numeric({s["qhat"]: mp.mpf("0.9")}, dps=16)
                   - P._ising_q(mp.mpf("0.9"))[0]) < 1e-12


@slow
def test_spherical_derivation_unchanged_by_accepting_ising():
    a = derived_rs_equations(default_spec("perceptron"))
    res = compare_equations(a, pipeline_rs_equations("perceptron"))
    assert res.agree and res.route == "symbolic+stein", res
