import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.solve import fixed_point, gauss_expect


def _sk_F(beta):
    return lambda q: gauss_expect(lambda z: mp.tanh(beta * mp.sqrt(q) * z)**2)


def test_gauss_expect_known():
    with mp.workdps(40):
        assert abs(gauss_expect(lambda z: z**2) - 1) < mp.mpf("1e-25")
        assert abs(gauss_expect(lambda z: mp.cos(z)) - mp.exp(-mp.mpf(1) / 2)) < mp.mpf("1e-25")
        assert abs(gauss_expect(lambda z: z**4) - 3) < mp.mpf("1e-25")


def test_gauss_expect_restores_precision():
    before = mp.mp.dps
    gauss_expect(lambda z: z, dps=50)
    assert mp.mp.dps == before


def test_gauss_expect_logcosh():
    # E[log cosh(a z)] has no closed form; check against the exp-series-free
    # identity E[exp(a z)] = exp(a^2/2) and an independent high-precision quad.
    with mp.workdps(40):
        a = mp.mpf("1.3")
        assert abs(gauss_expect(lambda z: mp.exp(a * z)) - mp.exp(a * a / 2)) < mp.mpf("1e-24")
        f = lambda z: mp.log(mp.cosh(a * z)) * mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)
        ref = mp.quad(f, [-mp.inf, -5, 0, 5, mp.inf])
        assert abs(gauss_expect(lambda z: mp.log(mp.cosh(a * z))) - ref) < mp.mpf("1e-24")


def test_fixed_point_sk_q_below_tc_is_trivial():
    r = fixed_point(_sk_F(mp.mpf("0.8")), mp.mpf("0.5"))
    assert r.branch == "trivial" and r.value == 0 and r.converged
    assert r.branches_seen == ("trivial",)


def test_fixed_point_sk_q_above_tc_nontrivial():
    r = fixed_point(_sk_F(mp.mpf("1.5")), mp.mpf("0.5"))
    assert r.branch == "nontrivial" and r.converged
    assert r.value > 0 and r.residual < mp.mpf("1e-20")
    assert "trivial" in r.branches_seen and "nontrivial" in r.branches_seen


def test_fixed_point_select_trivial_branch():
    r = fixed_point(_sk_F(mp.mpf("1.5")), mp.mpf("0.5"), branch="trivial")
    assert r.branch == "trivial" and r.value == 0


def test_fixed_point_sk_agrees_with_independent_root():
    beta = mp.mpf("1.5")
    r = fixed_point(_sk_F(beta), mp.mpf("0.5"))
    with mp.workdps(40):
        G = lambda q: mp.quad(lambda z: mp.tanh(beta * mp.sqrt(q) * z)**2
                              * mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi),
                              [-mp.inf, -6, 0, 6, mp.inf]) - q
        ref = mp.findroot(G, r.value * (1 + mp.mpf("1e-6")), tol=mp.mpf("1e-60"))
        assert abs(r.value - ref) < mp.mpf("1e-18")


def test_two_nontrivial_attractors_both_found():
    F = lambda x: mp.mpf("1.5") * x - mp.mpf("0.5") * x**3  # fixed points 0 (unstable), +-1
    r = fixed_point(F, mp.mpf("0.5"))
    assert r.branch == "nontrivial" and abs(r.value - 1) < mp.mpf("1e-20")
    assert set(r.branches_seen) == {"trivial", "nontrivial", "nontrivial-2"}
    vals = sorted(v for _, v in r.branch_values if v != 0)
    assert abs(vals[0] + 1) < mp.mpf("1e-20") and abs(vals[1] - 1) < mp.mpf("1e-20")
    r2 = fixed_point(F, mp.mpf("0.5"), branch="nontrivial-2")
    assert abs(r2.value + 1) < mp.mpf("1e-20")


def test_unknown_branch_refuses():
    try:
        fixed_point(_sk_F(mp.mpf("0.8")), mp.mpf("0.5"), branch="nontrivial")
    except Refusal as e:
        assert e.reason == "no-solution"
    else:
        raise AssertionError("expected Refusal")


def test_no_convergence_refuses():
    # x -> x + 1 has no fixed point; x -> 2x + 1 has only a repelling one
    # (x = -1) that damped iteration from the seeds never reaches.
    for F in (lambda x: x + 1, lambda x: 2 * x + 1):
        try:
            fixed_point(F, mp.mpf("0.5"))
        except Refusal as e:
            assert e.reason == "no-solution"
        else:
            raise AssertionError("expected Refusal")
