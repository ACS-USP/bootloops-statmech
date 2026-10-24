import itertools
import os
import sys
from fractions import Fraction
from types import SimpleNamespace

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive
from replicant.core import Refusal, State, run_pipeline
from replicant.derive import (Averaged, annealed_log_Z, average_disorder,
                              introduce_order_parameters, replicate,
                              saddle_equations)
from replicant.spec import ModelSpec

beta = sp.Symbol("beta", positive=True)
N = sp.Symbol("N", positive=True, integer=True)


def _sk(disorder="gaussian"):
    return ModelSpec("sk", "ising", disorder, params={"beta": beta})


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def _run(spec, n, upto=4):
    steps = [replicate(spec, n), average_disorder(),
             introduce_order_parameters(), saddle_equations()][:upto]
    return run_pipeline(steps, State())


def _configs(N_):
    return list(itertools.product((1, -1), repeat=N_))


def test_sk_gaussian_annealed_exact():
    got = annealed_log_Z(_sk(), N)
    assert sp.simplify(got - (N*sp.log(2) + beta**2*(N - 1)/4)) == 0
    # Brute force: E exp(beta J s_i s_j) = exp(beta^2 (s_i s_j)^2 / (2N)),
    # the exact Gaussian moment, summed over all 2^N states.
    with mp.workdps(40):
        b = Fraction(7, 10)
        for N_ in range(2, 11):
            total = mp.mpf(0)
            for s in _configs(N_):
                expo = sum(b**2 * (s[i]*s[j])**2 / (2*N_)
                           for i in range(N_) for j in range(i + 1, N_))
                total += mp.exp(mp.mpf(expo.numerator) / expo.denominator)
            want = got.subs({N: N_, beta: sp.Rational(7, 10)})
            assert abs(mp.log(total) - mp.mpf(sp.N(want, 40))) < mp.mpf("1e-30")


def test_sk_pm_one_annealed_exact():
    got = annealed_log_Z(_sk("pm_one"), N)
    want = N*sp.log(2) + (N*(N - 1)/2)*sp.log(sp.cosh(beta/sp.sqrt(N)))
    assert sp.simplify(got - want) == 0
    with mp.workdps(40):
        b = mp.mpf(7) / 10
        # N <= 4: average over every coupling sign pattern, no cosh identity used.
        for N_ in range(2, 5):
            pairs = [(i, j) for i in range(N_) for j in range(i + 1, N_)]
            J = 1 / mp.sqrt(N_)
            total = mp.mpf(0)
            for signs in itertools.product((1, -1), repeat=len(pairs)):
                for s in _configs(N_):
                    total += mp.exp(b * J * sum(g * s[i] * s[j]
                                                for g, (i, j) in zip(signs, pairs)))
            avg = total / 2**len(pairs)
            val = got.subs({N: N_, beta: sp.Rational(7, 10)})
            assert abs(mp.log(avg) - mp.mpf(sp.N(val, 40))) < mp.mpf("1e-30")
        # N <= 10: per-pair exact average over the two signs, all 2^N states.
        for N_ in range(5, 11):
            J = 1 / mp.sqrt(N_)
            total = mp.mpf(0)
            for s in _configs(N_):
                term = mp.mpf(1)
                for i in range(N_):
                    for j in range(i + 1, N_):
                        t = b * J * s[i] * s[j]
                        term *= (mp.exp(t) + mp.exp(-t)) / 2
                total += term
            val = got.subs({N: N_, beta: sp.Rational(7, 10)})
            assert abs(mp.log(total) - mp.mpf(sp.N(val, 40))) < mp.mpf("1e-30")


def test_unsupported_disorder_refuses():
    spec = _sk("lognormal")
    assert _refusal(lambda: annealed_log_Z(spec, N)) == \
        "precondition:supported-disorder-law"
    assert _refusal(lambda: _run(spec, 2)) == \
        "precondition:supported-disorder-law"


def test_unsupported_family_and_variables_refuse():
    assert _refusal(lambda: _run(ModelSpec("hopfield", "ising", "gaussian"), 2)) \
        == "precondition:supported-family"
    assert _refusal(lambda: _run(ModelSpec("sk", "spherical", "gaussian"), 2)) \
        == "precondition:supported-variables"
    assert _refusal(lambda: _run(_sk(), 0)) == "precondition:replica-number"


def _permuted(expr, F, perm):
    sub = {}
    for (a, b), sym in F.q.items():
        pa, pb = sorted((perm[a], perm[b]))
        sub[sym] = F.q[(pa, pb)]
        sub[F.qhat[(a, b)]] = F.qhat[(pa, pb)]
    return expr.xreplace(sub)


def test_functional_is_symmetric_in_replica_indices():
    for n in (3, 4):
        F = _run(_sk(), n, upto=3)[0]["functional"]
        labels = list(range(1, n + 1))
        for perm in ({1: 2, 2: 1}, {1: 2, 2: 3, 3: 1}):
            perm = {a: perm.get(a, a) for a in labels}
            diff = F.expr - _permuted(F.expr, F, perm)
            assert sp.simplify(diff) == 0, (n, perm)
        # Positive control: an asymmetric perturbation is detected.
        bad = F.expr + F.q[(1, 2)]
        assert sp.simplify(bad - _permuted(bad, F, {**{a: a for a in labels},
                                                    1: 3, 3: 1})) != 0


def test_replicated_exponent_matches_enumeration_n2():
    # For Gaussian couplings the disorder average is exact:
    #   ln E exp(beta sum_{i<j} J_ij X_ij) = sum_{i<j} beta^2 X_ij^2 / (2N),
    # X_ij = sum_a s_i^a s_j^a.  The order-parameter form must equal it
    # configuration by configuration (leading part plus O(1) correction).
    F = _run(_sk(), 2, upto=3)[0]["functional"]
    b = sp.Rational(7, 10)
    N_ = 5
    form = (F.N * F.energy + F.correction).subs({F.N: N_, beta: b})
    for s1 in _configs(N_):
        for s2 in _configs(N_):
            exact = sum(b**2 * (s1[i]*s1[j] + s2[i]*s2[j])**2 / (2*N_)
                        for i in range(N_) for j in range(i + 1, N_))
            q = sp.Rational(sum(x*y for x, y in zip(s1, s2)), N_)
            assert form.subs(F.q[(1, 2)], q) == exact


def test_order_parameters_reproduce_annealed_at_n1():
    F = _run(_sk(), 1, upto=3)[0]["functional"]
    assert sp.simplify(F.N * F.energy + F.correction
                       - (annealed_log_Z(_sk(), F.N) - F.N*sp.log(2))) == 0
    assert sp.simplify(F.energy - beta**2/4) == 0


def test_saddle_equations_sk_n2():
    st, _ = _run(_sk(), 2)
    F, eqs = st["functional"], st["equations"]
    q, qh = F.q[(1, 2)], F.qhat[(1, 2)]
    sol = {qh: beta**2 * q}
    lhs = {sp.simplify(e.lhs - e.rhs) for e in eqs}
    assert len(eqs) == 2
    same = lambda x, y: sp.simplify((x - y).rewrite(sp.exp)) == 0
    assert any(same(x, beta**2*q - qh) for x in lhs)
    assert any(same(x, sp.tanh(qh) - q) for x in lhs)
    # Paramagnet q = 0 solves both.
    assert all(sp.simplify((e.lhs - e.rhs).subs(sol).subs(q, 0)) == 0 for e in eqs)


def test_pm_one_trace_states_dropped_order():
    _, trace = _run(_sk("pm_one"), 2)
    text = trace.render()
    assert "fourth cumulant" in text and "O(1/N)" in text
    _, trace = _run(_sk(), 2)
    assert "exact" in trace.render()
    avg = _run(_sk("pm_one"), 2)[0]["avg"]
    # log cosh(t/sqrt N) = t^2/(2N) - t^4/(12 N^2) + ...: the kappa_4 term.
    assert sp.simplify(avg.dropped + avg.t**4 / (12 * avg.N**2)) == 0


def test_symbolic_n_functional_then_saddle_refuses():
    n = sp.Symbol("n", positive=True, integer=True)
    st, _ = _run(_sk(), n, upto=3)
    F = st["functional"]
    assert n in F.expr.free_symbols
    # Formal energy: beta^2/4 (n + 2 sum_{a<b} q_ab^2); at q = 0 it is n beta^2/4.
    assert sp.simplify(F.energy.xreplace(F.q_zero).doit() - n*beta**2/4) == 0
    assert sp.simplify(F.correction + beta**2*n**2/4) == 0
    assert _refusal(lambda: _run(_sk(), n)) == \
        "precondition:explicit-replica-number"


def test_refusal_inside_step_carries_reason():
    st, _ = _run(_sk(), 2, upto=2)
    avg = st["avg"]
    X = avg.X
    bad = Averaged(avg.spec, avg.n, avg.law, X, avg.t, avg.N,
                   leading=X**4 / avg.N**2, dropped=sp.S(0), note="planted")
    step = introduce_order_parameters()
    assert _refusal(lambda: step.apply(st.with_(avg=bad))) == \
        "precondition:quadratic-in-coupling-field"


def test_functional_records_missing_order():
    assert _run(_sk(), 2, upto=3)[0]["functional"].missing is None
    F = _run(_sk("pm_one"), 1, upto=3)[0]["functional"]
    assert F.missing == "O(1)"
    gap = annealed_log_Z(_sk("pm_one"), F.N) - F.N*sp.log(2) \
        - (F.N * F.energy + F.correction)
    assert sp.simplify(sp.limit(gap, F.N, sp.oo) + beta**4/24) == 0


def test_law_with_vanishing_third_and_fourth_cumulant_refuses():
    derive.LAWS["k6"] = derive.DisorderLaw(
        "k6", lambda t, N_: t**2/(2*N_) + t**6/N_**3)
    try:
        assert _refusal(lambda: _run(_sk("k6"), 2, upto=2)) == \
            "precondition:dropped-cumulant-identified"
    finally:
        del derive.LAWS["k6"]


def test_toy_family_flows_through_the_hooks():
    # A disorder-free family (H = 0, Z = 2^N) registered from outside:
    # the steps only route to its hooks.
    calls = []

    def average(zn):
        calls.append("average")
        return SimpleNamespace(note="toy: nothing to average")

    def order_parameters(avg, n, sum_pairs):
        calls.append("order_parameters")
        return sp.S(0), sp.S(0), None

    derive.FAMILIES["toy"] = derive.Family(
        formal=lambda n, b: derive.Tr(sp.S(1)),
        annealed=lambda spec, N_: N_ * sp.log(2),
        supports_disorder=lambda d: d == "none",
        average=average, order_parameters=order_parameters)
    try:
        spec = ModelSpec("toy", "ising", "none")
        st, trace = _run(spec, 2, upto=3)
        assert calls == ["average", "order_parameters"]
        assert "toy: nothing to average" in trace.render()
        F = st["functional"]
        at0 = F.expr.subs({F.qhat[(1, 2)]: 0, F.q[(1, 2)]: 0})
        assert sp.simplify(at0 - 2*sp.log(2)) == 0   # ln 2^n per spin
        assert _refusal(lambda: _run(ModelSpec("toy", "ising", "gaussian"), 2)) \
            == "precondition:supported-disorder-law"
    finally:
        del derive.FAMILIES["toy"]
