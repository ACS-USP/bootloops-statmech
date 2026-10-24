"""The formal Gaussian-integral layer (gauss.py): placeholder and rules.

Conventions: E_z[f] = int f(z) exp(-z^2/2)/sqrt(2 pi) dz, z ~ N(0, 1).  Each rule
used by the RS derivation (derive.rs_ansatz / replica_limit) is tested here on
its own, against independent quadrature or an explicit finite-n computation.
"""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import gauss as G
from replicant.core import Refusal
from replicant.gauss import GaussE
from replicant.solve import gauss_expect

z, t = sp.symbols("z t", real=True)
a, b, c = sp.symbols("a b c", positive=True)


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def test_placeholder_binds_its_variable_and_evaluates_moments():
    e = GaussE(sp.tanh(a * z + b), z)
    assert e.free_symbols == {a, b}
    assert GaussE(z**2, z) == 1 and GaussE(3 + z**4 + z**3, z) == 6
    assert GaussE(a, z) == a                       # E of a constant
    # derivative passes under the expectation; the bound variable is untouched
    d = sp.diff(e, a)
    assert isinstance(d, GaussE) and sp.simplify(d.f - z * (1 - sp.tanh(a*z + b)**2)) == 0
    assert e.subs(z, 5) == e
    # bound-variable renaming is alpha-equivalence
    assert G.rename_bound(GaussE(sp.cos(a * t), t), z) == GaussE(sp.cos(a * z), z)


def test_numeric_evaluation_goes_through_gauss_expect():
    with mp.workdps(30):
        v = GaussE(sp.cos(a * z), z).doit_numeric({a: sp.Rational(7, 10)})
        assert abs(v - mp.exp(-mp.mpf("0.49") / 2)) < mp.mpf("1e-18")
        expr = 2 * GaussE(sp.cos(a * z), z) + a
        w = G.numeric(expr, {a: sp.Rational(7, 10)})
        assert abs(w - (2 * mp.exp(-mp.mpf("0.49") / 2) + mp.mpf("0.7"))) < mp.mpf("1e-18")
    before = mp.mp.dps
    G.numeric(expr, {a: 0.5}, dps=40)
    assert mp.mp.dps == before


def test_linearity_and_one_round_of_stein():
    e = GaussE(2 * a * z * sp.tanh(a * z + b) + 3 * sp.tanh(a * z + b)**2, z)
    lin = G.linear(e)
    assert lin == 2 * a * GaussE(z * sp.tanh(a*z + b), z) + 3 * GaussE(sp.tanh(a*z + b)**2, z)
    st = G.stein(e)          # E[z g(z)] = E[g'(z)]
    want = 2 * a**2 * (1 - GaussE(sp.tanh(a*z + b)**2, z)) + 3 * GaussE(sp.tanh(a*z + b)**2, z)
    assert sp.expand(st - want) == 0
    pt = {a: sp.Rational(3, 5), b: sp.Rational(1, 4)}
    assert abs(G.numeric(e, pt, dps=16) - G.numeric(want, pt, dps=16)) < 1e-13
    # higher powers: E[z^3 g] = E[z^2 g'] + 2 E[z g]
    assert abs(G.numeric(G.stein(GaussE(z**3 * sp.cos(a*z), z)), pt, dps=16)
               - G.numeric(GaussE(z**3 * sp.cos(a*z), z), pt, dps=16)) < 1e-13


def test_step_expectation_rule():
    # E_z Theta(a z + b) = H(-b/a), H(x) = erfc(x/sqrt 2)/2  (a > 0)
    r = G.expect_closed(sp.Heaviside(a * z + b - c), z)
    assert not r.has(z)
    with mp.workdps(30):
        for av, bv, cv in (("0.6", "0.3", "0.5"), ("1.3", "0.1", "0.9")):
            got = mp.mpf(sp.N(r.subs({a: sp.Rational(av), b: sp.Rational(bv),
                                      c: sp.Rational(cv)}), 40))
            x0 = (mp.mpf(cv) - mp.mpf(bv)) / mp.mpf(av)
            want = mp.quad(lambda u: mp.exp(-u * u / 2) / mp.sqrt(2 * mp.pi), [x0, mp.inf])
            assert abs(got - want) < mp.mpf("1e-25")
    assert _refusal(lambda: G.expect_closed(sp.Heaviside(z**2 - 1), z)) == \
        "precondition:known-gaussian-integral"
    assert _refusal(lambda: G.expect_closed(sp.Heaviside(-a * z + b), z)) == \
        "precondition:known-gaussian-integral"


def test_exp_quadratic_rule():
    # E_z exp(A z^2 + B z + C) = exp(C + B^2 / (2 (1 - 2A))) / sqrt(1 - 2A),  A < 1/2
    f = sp.exp(a * (b + c * z)**2 / 2)
    r = G.expect_closed(f, z)
    pt = {a: sp.Rational(1, 3), b: sp.Rational(1, 2), c: sp.Rational(4, 5)}
    with mp.workdps(30):
        want = gauss_expect(lambda u: mp.exp((mp.mpf("0.5") + mp.mpf("0.8")*u)**2 / 6))
        assert abs(mp.mpf(sp.N(r.subs(pt), 40)) - want) < mp.mpf("1e-20")


def test_hubbard_stratonovich_rule():
    S = sp.Symbol("S", real=True)
    hs = G.hubbard_stratonovich(a, S, z)          # exp(a S^2 / 2) = E_z exp(sqrt(a) z S)
    assert isinstance(hs, GaussE)
    pt = {a: sp.Rational(3, 4), S: sp.Rational(-6, 5)}
    assert abs(G.numeric(hs, pt) - mp.exp(mp.mpf("0.75") * mp.mpf("1.44") / 2)) < 1e-14


def test_rs_split_has_the_rs_covariance():
    q = sp.Symbol("q", positive=True)
    z1, z2 = sp.symbols("z1 z2", real=True)
    l1, l2 = G.rs_split(q, t, z1), G.rs_split(q, t, z2)
    cov = lambda e: GaussE(GaussE(GaussE(sp.expand(e), t), z1), z2)
    assert sp.simplify(cov(l1 * l2) - q) == 0 and sp.simplify(cov(l1**2) - 1) == 0


def test_rs_logdet_matches_the_determinant():
    d, off = sp.symbols("d c", real=True)
    nn = sp.Symbol("n")
    closed = G.rs_logdet(d, off, nn)
    for n in range(1, 5):
        M = sp.Matrix(n, n, lambda i, j: d if i == j else off)
        assert sp.expand(sp.expand_log(sp.exp(closed.subs(nn, n)), force=True)
                         - M.det()) == 0, n


def test_replica_limit_rule():
    n = sp.Symbol("n", positive=True)
    f = sp.exp(a * z + b) + sp.exp(-a * z - b)          # > 0
    T = n * (n - 1) * c / 2 + sp.log(GaussE(f**n, z))
    lim = G.replica_limit(T, n)
    assert sp.simplify(lim - (-c / 2 + GaussE(sp.log(f), z))) == 0
    # numerically: (1/n) T(n) at small n approaches the limit
    pt = {a: sp.Rational(3, 5), b: sp.Rational(1, 5), c: sp.Rational(1, 3)}
    with mp.workdps(30):
        nv = mp.mpf("1e-8")
        Ef = gauss_expect(lambda u: (2 * mp.cosh(mp.mpf("0.6") * u + mp.mpf("0.2")))**nv)
        small = (nv * (nv - 1) / 6 + mp.log(Ef)) / nv
        assert abs(small - G.numeric(lim, pt)) < mp.mpf("1e-7")
    assert _refusal(lambda: G.replica_limit(T + 1, n)) == "precondition:vanishes-at-n-zero"
