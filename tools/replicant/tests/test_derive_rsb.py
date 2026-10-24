"""Level (a), 1RSB (derive_rsb.py, derive_rsb_t0.py, solve_derived_rsb.py).

Conventions: Parisi 1RSB, blocks of x replicas (x = Parisi's m; the Hopfield
magnetization keeps the name m), q1 in a block, q0 between blocks.  phi = lim (1/(N n))
ln E Z^n: -beta f for SK and Hopfield, the entropy per coupling for the perceptron
(Ising J, Gaussian patterns, kappa = 0).  The families' 1RSB numbers (families/sk.py
sk_1rsb, sk_ground_energy("1rsb")) and the banked Krauth-Mezard capacity are TARGETS only:
nothing of them enters the derivation or the solver.

Independent checks of the reduction rules: each 1RSB reduction (Ising site, frozen site,
LnGaussProd with an exp-quadratic or an indicator weight) is evaluated at an integer n
(n = 4, x = 2) and compared with a brute-force sum over spins, an exact determinant built
from ansatz.rsb_matrix, or a direct bivariate quadrature; the x = 1 and q1 = q0 reductions
are compared with the DERIVED RS functional (derive.rs_ansatz, a separate rule set).
"""
import itertools
import os
import random
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive as D
from replicant import derive_rsb as R
from replicant._rsb_num import NestedNum
from replicant.agree import default_spec, derive_rs
from replicant.ansatz import rsb_matrix
from replicant.core import Refusal
from replicant.spec import ModelSpec
from replicant.testing import slow

n = sp.Symbol("n", positive=True)
_l = sp.Symbol("l", real=True)


def _refusal(fn):
    try:
        out = fn()
    except Refusal as r:
        return r
    raise AssertionError(f"expected Refusal, got {out!r}")


def _sk():
    return ModelSpec("sk", "ising", "gaussian", params={"beta": D.BETA, "h": D.H_FIELD})


def _ising_perceptron():
    s = default_spec("perceptron")
    return ModelSpec("perceptron", "ising", s.disorder, s.constraint, s.prior, dict(s.params))


def _value(expr, point, rule="audit"):
    syms = sorted(expr.free_symbols, key=str)
    return NestedNum(expr, syms).value([float(point[s.name]) for s in syms], rule)


def _functional(expr, energy=0, variables="ising"):
    """A formal Functional holding only `expr` (the reduction rules read expr and, for the
    frozen precondition, the energy)."""
    q, qh, _ = D._overlap_symbols(n)
    return D.Functional(ModelSpec("t", variables, "gaussian"), n, D.N_SYM, q, qh, energy, 0,
                        expr, expr, {}, None, {}, True), q, qh


def _at(e, nv, xv):
    return e.subs({n: nv, R.X: xv})


# ---- the reduction rules at integer n -------------------------------------------------
def test_ising_site_reduction_matches_brute_force_at_integer_n():
    B = sp.Symbol("B", real=True)
    F, q, qh = _functional(0)
    F = F.__class__(**{**vars(F), "expr": D._ising_site(n, q, qh, B)})
    e = _at(R._reduce(F, False)[0], 4, 2)
    A0, A1, Bv = sp.Rational(1, 3), sp.Rational(9, 10), sp.Rational(1, 5)
    A = rsb_matrix([2], [A0, A1], 4)
    brute = sp.log(sum(sp.exp(sum(A[i, j] * s[i] * s[j] for i in range(4) for j in range(i + 1, 4))
                              + Bv * sum(s)) for s in itertools.product((1, -1), repeat=4)))
    got = _value(e, {"qhat0": A0, "qhat1": A1, "B": Bv})
    assert abs(got - float(brute)) < 1e-12, (got, float(brute))
    # positive control: the blocks swapped (q0 <-> q1) is a different number
    assert abs(_value(e, {"qhat0": A1, "qhat1": A0 + 1, "B": Bv}) - float(brute)) > 1e-3


def test_frozen_site_reduction_matches_brute_force_at_integer_n():
    B = sp.Symbol("B", real=True)
    g = sp.Lambda(_l, sp.Heaviside(_l))
    F, q, qh = _functional(0)
    F = F.__class__(**{**vars(F), "expr": D._ising_site(n, q, qh, B),
                       "energy": D.LnGaussProd(g, q[D._a, D._b], n)})
    e = _at(R._reduce(F, True)[0], 4, 2)
    A0, Bv = sp.Rational(2, 5), sp.Rational(-1, 7)
    # s1 = s2 = sigma1, s3 = s4 = sigma2: 4 between-block pairs, no within-block coupling
    brute = sp.log(sum(sp.exp(4 * A0 * s1 * s2 + 2 * Bv * (s1 + s2))
                       for s1 in (1, -1) for s2 in (1, -1)))
    got = _value(e, {"qhat0": A0, "B": Bv})
    assert abs(got - float(brute)) < 1e-12, (got, float(brute))


@slow
def test_gaussian_energy_reduction_matches_the_determinant_at_integer_n():
    # E_y prod_a exp(b y_a^2 / 2), y ~ N(0, C), C 1RSB: = det(I - b C)^(-1/2) exactly
    bv = sp.Rational(1, 5)
    q = D._overlap_symbols(n)[0]
    w = D.LnGaussProd(sp.Lambda(_l, sp.exp(bv * _l**2 / 2)), q[D._a, D._b], n)
    F, q, qh = _functional(w, energy=w)
    e = _at(R._reduce(F, False)[0], 4, 2)
    R_ = sp.Rational
    for q0, q1 in ((R_(1, 4), R_(3, 5)), (R_(1, 10), R_(9, 10))):
        C = rsb_matrix([2], [q0, q1], 4)
        exact = -sp.log((sp.eye(4) - bv * C).det()) / 2
        assert abs(_value(e, {"q0": q0, "q1": q1}) - float(exact)) < 1e-12


@slow
def test_frozen_indicator_energy_matches_a_direct_quadrature():
    # frozen, n = 4, x = 2: E prod_a Theta(lambda_a - kappa) = P(l1 > k, l3 > k), corr q0
    kap = sp.Rational(1, 3)
    q = D._overlap_symbols(n)[0]
    w = D.LnGaussProd(sp.Lambda(_l, sp.Heaviside(_l - kap)), q[D._a, D._b], n)
    F, q, qh = _functional(w, energy=w)
    e = _at(R._reduce(F, True)[0], 4, 2)
    q0 = mp.mpf(3) / 10
    with mp.workdps(20):
        dens = lambda a, b: mp.exp(-(a * a - 2 * q0 * a * b + b * b) / (2 * (1 - q0**2))) / (
            2 * mp.pi * mp.sqrt(1 - q0**2))
        direct = mp.quad(dens, [kap, 4, mp.inf], [kap, 4, mp.inf])
    assert abs(_value(e, {"q0": "0.3"}) - float(mp.log(direct))) < 1e-10


def test_frozen_rule_refuses_a_non_counting_measure():
    r = _refusal(lambda: R.derive_1rsb(_sk(), frozen=True))
    assert r.reason == "precondition:counting-measure-for-frozen", r
    r = _refusal(lambda: R.derive_1rsb(default_spec("perceptron"), frozen=True))
    assert r.reason == "precondition:discrete-variables-for-frozen", r


# ---- reductions to the derived RS functional ---------------------------------------------
def _rs_phi(spec):
    return derive_rs(spec)[1]["rs_free"].expr


@slow
def test_m_equal_one_reduces_to_rs():
    """x = 1 (and, separately, q1 = q0 at any x) turns the derived 1RSB phi into the
    derived RS phi at q = q0, qhat = qhat0, for any q1, qhat1: SK, Ising perceptron,
    Hopfield, at three random points each (seeded)."""
    rng = random.Random(7)
    for spec, extra in ((_sk(), {"beta": 1.3, "h": 0.4}),
                        (_ising_perceptron(), {"alpha": 0.6, "kappa": 0.2}),
                        (default_spec("hopfield"), {"alpha": 0.08, "beta": 1.2, "m": 0.6})):
        P = R.derived_1rsb_functional(spec)
        rs = _rs_phi(spec)
        for _ in range(3):
            q0, q1 = sorted(rng.uniform(0.25, 0.85) for _ in range(2))   # Hopfield: D > 0
            h0 = rng.uniform(0.2, 1.5)
            pt = {**extra, "q0": q0, "q1": q1, "qhat0": h0, "qhat1": h0 + rng.uniform(0.1, 1)}
            one = _value(P.phi.subs(R.X, 1), pt)
            ref = _value(rs, {**pt, "q": q0, "qhat": h0})
            assert abs(one - ref) < 1e-11, (spec.family, one, ref)
            same = {**pt, "q1": q0, "qhat1": h0, "x": rng.uniform(0.2, 0.8)}
            assert abs(_value(P.phi, same) - ref) < 1e-11, (spec.family, "q1 = q0")
        # positive control: at x = 1/2 with q1 > q0 the 1RSB phi is NOT the RS one
        assert abs(_value(P.phi, {**pt, "x": 0.5}) - ref) > 1e-6, spec.family
