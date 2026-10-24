import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.testing import slow
from replicant.solve import fixed_point, gauss_expect


def test_gauss_expect_known():
    with mp.workdps(40):
        assert abs(gauss_expect(lambda z: z**2) - 1) < mp.mpf("1e-25")
        assert abs(gauss_expect(lambda z: mp.cos(z)) - mp.exp(-mp.mpf(1) / 2)) < mp.mpf("1e-25")
        assert abs(gauss_expect(lambda z: z**4) - 3) < mp.mpf("1e-25")


def test_gauss_expect_restores_precision():
    before = mp.mp.dps
    gauss_expect(lambda z: z, dps=50)
    assert mp.mp.dps == before


def _trapz_ref(g):
    """Independent reference: trapezoid rule (spectrally accurate for smooth
    integrands) on [-12, 12] with the Gaussian weight, in float64."""
    z = np.linspace(-12, 12, 240001)
    w = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    return np.trapezoid(g(z) * w, z)


def test_gauss_expect_large_beta_independent():
    b = 10
    got = gauss_expect(lambda z: mp.tanh(b * z)**2, dps=20)
    assert abs(float(got) - _trapz_ref(lambda z: np.tanh(b * z)**2)) < 1e-10
    got = gauss_expect(lambda z: mp.log(mp.cosh(b * z)), dps=20)
    assert abs(float(got) - _trapz_ref(lambda z: np.logaddexp(b * z, -b * z) - np.log(2))) < 1e-9


_DPS, _TOL = 15, "1e-10"
_cache = {}


def _sk(beta, dps=_DPS, tol=_TOL, **kw):
    F = lambda q: gauss_expect(lambda z: mp.tanh(mp.mpf(beta) * mp.sqrt(q) * z)**2, dps=dps)
    return fixed_point(F, mp.mpf("0.5"), dps=dps, tol=tol, **kw)


def _sk15():
    if "sk15" not in _cache:
        _cache["sk15"] = _sk("1.5")
    return _cache["sk15"]


def test_fixed_point_sk_q_below_tc_is_trivial():
    r = _sk("0.8")
    assert r.branch == "trivial" and r.value == 0 and r.converged
    assert r.branches_seen == ("trivial",)


def test_fixed_point_sk_q_above_tc_nontrivial():
    r = _sk15()
    assert r.branch == "nontrivial" and r.converged
    assert r.value > 0 and r.residual < mp.mpf(_TOL)
    assert r.branches_seen == ("trivial", "nontrivial")
    assert abs(r.value - mp.mpf("0.3526017967141234")) < mp.mpf("1e-9")


def test_fixed_point_select_trivial_branch():
    r = _sk("1.5", branch="trivial")
    assert r.branch == "trivial" and r.value == 0


def test_fixed_point_near_tc_finds_nontrivial():
    r = _sk("1.05", dps=10, tol="1e-7")
    assert r.branches_seen == ("trivial", "nontrivial") and r.value > 0


@slow
def test_fixed_point_sk_agrees_with_independent_root():
    beta = mp.mpf("1.5")
    F = lambda q: gauss_expect(lambda z: mp.tanh(beta * mp.sqrt(q) * z)**2)
    r = fixed_point(F, mp.mpf("0.5"))
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
        fixed_point(lambda x: x / 2, mp.mpf("0.5"), dps=15, tol="1e-10", branch="nontrivial")
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
