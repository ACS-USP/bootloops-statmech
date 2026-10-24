"""Level agreement: (a) derived RS equations == (b) the families' equations.

Conventions.  Level (a) runs replicate -> average-disorder ->
introduce-order-parameters -> rs-ansatz -> replica-limit -> rs-stationarity at
symbolic n (derive.py, derive_families.py, gauss.py).  Level (b) is each
family's `rs_equations_symbolic()`, the equations its numeric solver solves,
written with the GaussE placeholder (tied to the running code by the tests at
the bottom).  Models:
  SK          J_ij ~ N(0, 1/N), field h:      q = E tanh^2 beta(h + sqrt(q) z)
  perceptron  spherical, Gaussian patterns, margin kappa:
                                              q/(1-q) = alpha E_t E(x)^2
  Hopfield    +-1 patterns, p = alpha N, one condensed:
              m = E tanh Xi, q = E tanh^2 Xi, r = q/(1 - beta(1-q))^2,
              Xi = beta(m + sqrt(alpha r) z)
Canonical form of the comparison: agree.py module docstring.
"""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import agree as A
from replicant.agree import (compare_equations, default_spec, derive_rs,
                             derived_rs_equations, equations_agree,
                             pipeline_rs_equations)
from replicant.core import Refusal
from replicant.gauss import GaussE, numeric
from replicant.spec import ModelSpec
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def _names(eqs):
    return {s.name for e in eqs for s in e.free_symbols}


def _sym(eqs, name):
    return next(s for e in eqs for s in e.free_symbols if s.name == name)


def _plant(eqs, old, new):
    return [sp.Eq(e.lhs.xreplace({old: new}), e.rhs.xreplace({old: new})) for e in eqs]


# ---- the three families ----------------------------------------------------------

def test_sk_derived_equals_pipeline():
    a = derived_rs_equations(default_spec("sk"))
    b = pipeline_rs_equations("sk")
    # (a): d/dq and d/dqhat of the n -> 0 functional; (b): one fixed-point equation
    assert len(a) == 2 and len(b) == 1
    assert _names(a) == {"beta", "h", "q", "qhat"} and _names(b) == {"beta", "h", "q"}
    res = compare_equations(a, b)
    assert res.agree and res.route == "symbolic", res
    assert "qhat" in res.detail          # the conjugate was eliminated, and it says so


@slow
def test_spherical_perceptron_derived_equals_pipeline():
    a = derived_rs_equations(default_spec("perceptron"))
    b = pipeline_rs_equations("perceptron")
    assert len(a) == 1 and _names(a) == _names(b) == {"alpha", "kappa", "q"}
    res = compare_equations(a, b)
    assert res.agree and res.route.startswith("symbolic"), res
    # the identity needs Gaussian integration by parts beyond one Stein round
    assert res.route == "symbolic+stein", res


@slow
def test_hopfield_derived_equals_pipeline():
    a = derived_rs_equations(default_spec("hopfield"))
    b = pipeline_rs_equations("hopfield")
    assert len(a) == len(b) == 3
    assert _names(a) == {"alpha", "beta", "m", "q", "qhat"}
    assert _names(b) == {"alpha", "beta", "m", "q", "r"}
    res = compare_equations(a, b)
    assert res.agree and res.route == "symbolic", res
    assert "qhat" in res.detail and "r" in res.detail


def test_agreement_detects_a_planted_sign_error():
    a = derived_rs_equations(default_spec("sk"))
    b = pipeline_rs_equations("sk")
    q = _sym(b, "q")
    # symbolic plant: a factor 2 on the overlap inside the expectation
    bad = [sp.Eq(b[0].lhs, b[0].rhs.xreplace({q: 2 * q}))]
    res = compare_equations(a, bad)
    assert not res.agree and res.route == "numeric", res
    # sign flip of the whole right-hand side
    assert not equations_agree(a, [sp.Eq(b[0].lhs, -b[0].rhs)])
    # numeric-only plant: a 1e-6 relative error, invisible to rewriting, caught by
    # the residual at the random points; the same forced route accepts the truth
    eq = b[0]
    tiny = [sp.Eq(eq.lhs, eq.rhs * (1 + sp.Rational(1, 10**6)))]
    assert not compare_equations(a, tiny, route="numeric").agree
    # (that the same forced route accepts the true system: slow tier, below)


@slow
def test_sk_symmetric_flip_is_not_an_error_and_symbolic_route_certifies_no_plant():
    a = derived_rs_equations(default_spec("sk"))
    b = pipeline_rs_equations("sk")
    # sign flip of the field: h -> -h is NOT an error (tanh^2 is even, z -> -z),
    # and the check must not call it one (decided numerically: not pointwise)
    assert equations_agree(a, _plant(b, _sym(b, "h"), -_sym(b, "h")))
    # a forced symbolic route never certifies a planted system
    tiny = [sp.Eq(b[0].lhs, b[0].rhs * (1 + sp.Rational(1, 10**6)))]
    assert not compare_equations(a, tiny, route="symbolic").agree
    # the forced numeric route accepts the true system (the default-tier plant
    # test shows it rejects the 1e-6 plant)
    assert compare_equations(a, b, route="numeric").agree


@slow
def test_agreement_detects_planted_errors_in_hopfield_and_perceptron():
    a = derived_rs_equations(default_spec("hopfield"))
    b = pipeline_rs_equations("hopfield")
    beta = _sym(b, "beta")
    q = _sym(b, "q")
    flip = [e if e.lhs.free_symbols != {_sym(b, "r")} else
            sp.Eq(e.lhs, e.rhs.xreplace({1 - beta * (1 - q): 1 + beta * (1 - q)})) for e in b]
    assert flip != b
    assert not equations_agree(a, flip)
    a = derived_rs_equations(default_spec("perceptron"))
    b = pipeline_rs_equations("perceptron")
    alpha = _sym(b, "alpha")
    assert not equations_agree(a, _plant(b, alpha, 2 * alpha))
    # the Stein-closure prover itself must not certify a wrong pair: a factor 2
    # inside the expectation, symbolic route only (no numeric screen)
    G = next(iter(b[0].rhs.atoms(GaussE)))
    inside = [sp.Eq(b[0].lhs, alpha * GaussE(2 * G.f, G.z))]
    assert not compare_equations(a, inside, route="symbolic").agree


def test_planted_error_in_a_level_a_hook_is_caught():
    # A wrong MODEL piece on the derivation side (SK pair energy doubled in the
    # family hook) must make level (a) disagree with level (b).
    from dataclasses import replace
    from replicant import derive as D
    good = D.FAMILIES["sk"]

    def doubled(avg, n, sum_pairs):
        r = good.order_parameters(avg, n, sum_pairs)
        return D.Reduction(2 * r.energy, r.correction, r.missing, r.field, r.extra)
    D.FAMILIES["sk"] = replace(good, order_parameters=doubled)
    try:
        a = derived_rs_equations(default_spec("sk"))
    finally:
        D.FAMILIES["sk"] = good
    assert compare_equations(a, pipeline_rs_equations("sk")).agree is False


def test_hopfield_trace_states_the_gaussian_integrability_assumption():
    _, _, trace = derive_rs(default_spec("hopfield"))
    text = trace.render()
    assert "assumed 1 - 2A = beta*q - beta + 1 > 0" in text, text


# ---- the canonicalisation itself -------------------------------------------------

def test_zero_test_handles_radicals_and_rejects_nonzero():
    q = sp.Symbol("q", positive=True)
    x = sp.Symbol("x", real=True)
    e = sp.sqrt(1 - q) ** 3 / (1 - q) - sp.sqrt(1 - q)
    assert A.is_identically_zero(e)
    assert A.is_identically_zero(sp.tanh(x)**2 - (1 - 1 / sp.cosh(x)**2))
    assert not A.is_identically_zero(sp.sqrt(1 - q) - sp.sqrt(q))
    assert not A.is_identically_zero(sp.exp(x) - sp.exp(2 * x))


def test_mismatched_systems_do_not_agree():
    a = derived_rs_equations(default_spec("sk"))
    q = _sym(a, "q")
    assert not equations_agree(a, [sp.Eq(q, sp.Rational(1, 2))])   # no common target
    res = compare_equations(a, [])
    assert not res.agree and "count" in res.detail


# ---- level (a) is a recorded pipeline --------------------------------------------

def test_derivation_is_a_recorded_pipeline():
    eqs, state, trace = derive_rs(default_spec("sk"))
    assert trace.to_json()["steps"] == [
        "replicate", "average-disorder", "introduce-order-parameters",
        "rs-ansatz", "replica-limit", "rs-stationarity"]
    text = trace.render()
    for rule in ("Hubbard-Stratonovich", "sum_offdiag_power", "n_to_zero", "E ln f"):
        assert rule in text, rule
    assert "physics-level" in state["rs_free"].caveat


def test_rs_steps_refuse_what_they_cannot_do():
    spec = default_spec("sk")
    assert _refusal(lambda: A.derive_rs(spec, n=2)) == "precondition:symbolic-replica-number"
    assert _refusal(lambda: derived_rs_equations(ModelSpec("potts", "ising", "gaussian"))) \
        == "precondition:supported-family"
    assert _refusal(lambda: derived_rs_equations(
        ModelSpec("perceptron", "spherical", "pm_one"))) == "precondition:supported-disorder-law"
    assert _refusal(lambda: derived_rs_equations(
        ModelSpec("perceptron", "ising", "gaussian"))) == "precondition:supported-variables"
    assert _refusal(lambda: pipeline_rs_equations("rem")) == "precondition:supported-family"


@slow
def test_derived_functionals_equal_the_families_free_energies():
    # The n -> 0 functional itself (not only its gradient) matches level (b),
    # constants included, at a solved point of each family.
    from replicant.families.hopfield import hopfield_free_energy, hopfield_rs
    from replicant.families.perceptron import gardner_rs_point
    from replicant.families.sk import sk_rs
    with mp.workdps(20):
        _, st, _ = derive_rs(default_spec("sk"))
        phi = st["rs_free"].expr
        s = {x.name: x for x in phi.free_symbols}
        sol = sk_rs("1.5", "0.3")
        pt = {s["beta"]: 1.5, s["h"]: 0.3, s["q"]: sol.q, s["qhat"]: 1.5**2 * sol.q}
        assert abs(numeric(phi, pt) - (-1.5 * sol.f)) < 1e-12

        _, st, _ = derive_rs(default_spec("perceptron"))
        phi = st["rs_free"].expr
        s = {x.name: x for x in phi.free_symbols}
        p = gardner_rs_point("spherical", "0.4", "0.5")
        pt = {s["q"]: p.q, s["kappa"]: p.kappa, s["alpha"]: p.alpha}
        assert abs(numeric(phi, pt) - p.S) < 1e-12

        _, st, _ = derive_rs(default_spec("hopfield"))
        phi = st["rs_free"].expr
        s = {x.name: x for x in phi.free_symbols}
        al, be = mp.mpf("0.05"), mp.mpf(2)
        h = hopfield_rs(al, be)
        pt = {s["alpha"]: al, s["beta"]: be, s["m"]: h.m, s["q"]: h.q,
              s["qhat"]: al * be**2 * h.r}
        # derived H keeps the i = j terms: -beta f_AGS = phi - alpha beta / 2
        f = hopfield_free_energy(h.m, h.q, h.r, al, be)
        assert abs(numeric(phi, pt) - al * be / 2 - (-be * f)) < 1e-12


# ---- level (b): the symbolic equations are the ones the solvers solve -----------

def test_sk_pipeline_equation_is_the_solved_map():
    from replicant.families import sk
    (eq,) = pipeline_rs_equations("sk")
    s = {x.name: x for x in eq.free_symbols}
    with mp.workdps(20):
        for b_, h_, q_ in (("1.3", "0.2", "0.4"), ("0.7", "0.6", "0.25")):
            pt = {s["beta"]: b_, s["h"]: h_, s["q"]: q_}
            got = numeric(eq.rhs - eq.lhs, pt, dps=16)
            want = sk._rs_map(mp.mpf(b_), mp.mpf(h_), mp.mpf(q_)) - mp.mpf(q_)
            assert abs(got - want) < 1e-14
        sol = sk.sk_rs("2.0", "0.3")         # the point test_sk solves (cached)
        res = numeric(eq.rhs - eq.lhs, {s["beta"]: "2.0", s["h"]: "0.3", s["q"]: sol.q}, dps=16)
        assert abs(res) < 1e-11


@slow
def test_perceptron_pipeline_equation_is_the_solved_map():
    from replicant.families import perceptron as P
    (eq,) = pipeline_rs_equations("perceptron")
    s = {x.name: x for x in eq.free_symbols}
    with mp.workdps(20):
        pnt = P.gardner_rs_point("spherical", "0.4", "0.5")
        pt = {s["q"]: pnt.q, s["kappa"]: pnt.kappa, s["alpha"]: pnt.alpha}
        assert abs(numeric(eq.lhs - eq.rhs, pt, dps=16)) < 1e-12
        # the expectation is the one the solver computes (_T.eh2), off the solution too
        G = next(iter(eq.rhs.atoms(GaussE)))
        q_, k_ = mp.mpf("0.3"), mp.mpf("-0.2")
        assert abs(G.doit_numeric({s["q"]: q_, s["kappa"]: k_}, dps=16)
                   - P._T(q_, 1 - q_, k_).eh2()) \
            < 1e-12


def test_hopfield_pipeline_equations_are_the_solved_gradient():
    from replicant.families import hopfield as H
    eqs = pipeline_rs_equations("hopfield")
    s = {x.name: x for e in eqs for x in e.free_symbols}
    by_lhs = {e.lhs.name: e for e in eqs}
    with mp.workdps(20):
        m, q, r, al, be = (mp.mpf(v) for v in ("0.6", "0.5", "0.9", "0.08", "1.4"))
        pt = {s["m"]: m, s["q"]: q, s["r"]: r, s["alpha"]: al, s["beta"]: be}
        gm, gq, gr = H.hopfield_gradient(m, q, r, al, be)
        res = {k: numeric(e.lhs - e.rhs, pt, dps=16) for k, e in by_lhs.items()}
        assert abs(res["m"] - gm) < 1e-12                      # df/dm = m - E tanh
        assert abs(res["q"] + 2 * gr / (al * be)) < 1e-12      # df/dr = (ab/2)(E tanh^2 - q)
        assert abs(res["r"] + 2 * gq / (al * be)) < 1e-12      # df/dq = (ab/2)(q/D^2 - r)
        al, be = mp.mpf("0.05"), mp.mpf(2)
        sol = H.hopfield_rs(al, be)
        pt = {s["m"]: sol.m, s["q"]: sol.q, s["r"]: sol.r, s["alpha"]: al, s["beta"]: be}
        assert all(abs(numeric(e.lhs - e.rhs, pt, dps=16)) < 1e-12 for e in eqs)
