"""The 1RSB rules of the formal Gaussian layer (gauss_rsb.py, gauss.expect_closed), each
tested on its own against an independent computation: explicit moments, an explicit
determinant, brute-force sums over spins or quadrature.

Conventions: E_z[f] = int f(z) exp(-z^2/2)/sqrt(2 pi) dz.  Parisi 1RSB: blocks of m
replicas, q1 within a block, q0 between blocks (ansatz.py).
"""
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import gauss as G
from replicant import gauss_rsb as GR
from replicant.ansatz import rsb_matrix
from replicant.core import Refusal
from replicant.gauss import GaussE

z, t, u, x = sp.symbols("z t u x", real=True)
a, b, mu = sp.symbols("a b mu", positive=True)


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def test_rsb_split_has_the_block_covariance():
    q0, q1 = sp.symbols("q0 q1", positive=True)
    u1, u2, z1, z2, z3 = sp.symbols("u1 u2 z1 z2 z3", real=True)
    la, lb, lc = (GR.rsb_split(q0, q1, t, uu, zz) for uu, zz in ((u1, z1), (u1, z2), (u2, z3)))
    E = lambda e: sp.expand(GaussE(GaussE(GaussE(GaussE(GaussE(GaussE(
        sp.expand(e), t), u1), u2), z1), z2), z3))
    assert sp.simplify(E(la * lb) - q1) == 0          # same block
    assert sp.simplify(E(la * lc) - q0) == 0          # different blocks
    assert sp.simplify(E(la**2) - 1) == 0
    # the RS split is the q1 = q0 case (positive control: q0 <-> q1 swapped fails)
    assert sp.simplify(E(la * lc) - q1) != 0


def test_rsb_logdet_matches_the_determinant():
    d, c0, c1 = sp.Rational(7, 5), sp.Rational(1, 5), sp.Rational(2, 3)
    nn, mm = sp.symbols("n m")
    closed = GR.rsb_logdet(d, c0, c1, mm, nn)
    for n, m in ((4, 2), (6, 3), (6, 2), (8, 4)):
        M = rsb_matrix([m], [c0, c1], n) + (d - 1) * sp.eye(n)
        assert abs(sp.N(closed.subs({nn: n, mm: m}) - sp.log(M.det()), 30)) < 1e-25, (n, m)
    # m = 1 and m = n are RS with off-diagonal c0 and c1 respectively
    assert abs(sp.N((closed - G.rs_logdet(d, c0, nn)).subs({mm: 1, nn: 5}), 30)) < 1e-25
    assert abs(sp.N((closed - G.rs_logdet(d, c1, nn)).subs({mm: 5, nn: 5}), 30)) < 1e-25


def test_indicator_power_rule():
    m = sp.Symbol("m", positive=True)
    e = sp.Heaviside(x - a)**m * sp.exp(x)**m
    out = GR.indicator_power(e)
    assert out == sp.Heaviside(x - a) * sp.exp(x)**m
    assert _refusal(lambda: GR.indicator_power(sp.Heaviside(x)**sp.Symbol("k", real=True))) \
        == "precondition:positive-exponent"


def test_exp_linear_times_step_rule():
    # E_z exp(B z + C) Theta(a z + b) for both signs of the slope, against quadrature
    for slope in (sp.Rational(3, 4), sp.Rational(-3, 4)):
        f = sp.exp(sp.Rational(1, 2) * z + sp.Rational(1, 3)) * \
            sp.Heaviside(slope * z + sp.Rational(1, 5))
        closed = G.expect_closed(f, z)
        fn = sp.lambdify(z, f.rewrite(sp.Piecewise), "mpmath")
        with mp.workdps(30):
            cut = -mp.mpf(1) / 5 / mp.mpf(slope)
            c = 1 / mp.sqrt(2 * mp.pi)
            direct = mp.quad(lambda w: fn(w) * c * mp.exp(-w * w / 2), [-mp.inf, cut, mp.inf])
            assert abs(sp.N(closed, 30) - direct) < 1e-25, slope
    assert _refusal(lambda: G.expect_closed(sp.exp(z) * sp.Heaviside(z**2 - 1), z)) \
        == "precondition:known-gaussian-integral"


def test_abs_split_rule():
    e = sp.exp(mu * sp.Abs(a * x + b))
    s = GR.abs_split(e)
    for xv in (-3, sp.Rational(-1, 5), sp.Rational(2, 5), 2):
        pt = {mu: sp.Rational(13, 10), a: sp.Rational(1, 2), b: sp.Rational(1, 4), x: xv}
        assert abs(sp.N((s - e).subs(pt).rewrite(sp.Piecewise), 30)) < 1e-25
    # with the exp-step rule: E_u exp(mu |a + s u|) in closed form, against quadrature
    sv = sp.Rational(3, 5)
    closed = G.linear(GaussE(sp.expand(GR.abs_split(sp.exp(mu * sp.Abs(a + sv * u)))), u))
    closed = closed.replace(lambda g: isinstance(g, GaussE), lambda g: G.expect_closed(g.f, g.z))
    pt = {mu: sp.Rational(13, 10), a: sp.Rational(1, 3)}
    with mp.workdps(30):
        direct = mp.quad(lambda w: mp.exp(mp.mpf(13) / 10 * abs(mp.mpf(1) / 3 + mp.mpf(3) / 5 * w))
                         * mp.exp(-w * w / 2) / mp.sqrt(2 * mp.pi),
                         [-mp.inf, -mp.mpf(5) / 9, mp.inf])
        assert abs(sp.N(closed.subs(pt), 30) - direct) < 1e-25


def test_abs_split_refuses_a_non_linear_exponent():
    assert _refusal(lambda: GR.abs_split(sp.exp(sp.sqrt(sp.Abs(x))))) == \
        "precondition:abs-linear-exponent"
    assert _refusal(lambda: GR.abs_split(sp.exp(sp.Abs(x) + sp.Abs(x - 1)))) == \
        "precondition:abs-linear-exponent"


def test_zero_temperature_power_rule():
    beta = sp.Symbol("beta", positive=True)
    e = sp.log(GaussE((sp.exp(beta * (a + b * u)) + sp.exp(-beta * (a + b * u)))**(mu / beta), u))
    out = GR.zero_temperature_power(e, beta)
    assert out == sp.log(GaussE(sp.exp(mu * sp.Abs(a + b * u)), u))
    # the limit is approached: at beta = 400 the power is within 2^(mu/beta) of the limit
    pt = {a: sp.Rational(-1, 3), b: 1, mu: sp.Rational(3, 2), u: sp.Rational(1, 10)}
    base = ((sp.exp(beta * (a + b * u)) + sp.exp(-beta * (a + b * u)))**(mu / beta)).subs(pt)
    lim = sp.exp(mu * sp.Abs(a + b * u)).subs(pt)
    assert abs(sp.N(sp.log(base.subs(beta, 400)) - sp.log(lim), 30)) < mp.log(2) * 1.5 / 400 + 1e-20
    # a power whose exponent is not mu/beta, or an argument depending on beta, is refused
    assert _refusal(lambda: GR.zero_temperature_power(
        (sp.exp(beta * x) + sp.exp(-beta * x))**mu, beta)) == "precondition:tempered-power"
    assert _refusal(lambda: GR.zero_temperature_power(
        (sp.exp(beta**2 * x) + sp.exp(-beta**2 * x))**(mu / beta), beta)) \
        == "precondition:tempered-power"


def test_block_quadratic_identity_matches_brute_force():
    # sum_{a<b} A_ab s_a s_b for a 1RSB A equals A0/2 S^2 + (A1-A0)/2 sum_k S_k^2 - n A1/2
    import itertools
    A0, A1 = sp.Rational(1, 3), sp.Rational(4, 5)
    for n, m in ((4, 2), (6, 3), (6, 2)):
        A = rsb_matrix([m], [A0, A1], n)
        for s in itertools.product((1, -1), repeat=n):
            lhs = sum(A[i, j] * s[i] * s[j] for i in range(n) for j in range(i + 1, n))
            blocks = [sum(s[k * m:(k + 1) * m]) for k in range(n // m)]
            assert lhs == GR.block_quadratic(A0, A1, sum(s), blocks, n), (n, m, s)
