"""Sparse cavity: belief propagation, RS population dynamics, Viana-Bray transition.

Conventions (stated once; the same as cavity/sparse.py):
  E = -sum_<ij> J_ij s_i s_j - sum_i H s_i, beta = 1/T, s_i = +-1.
  ("regular", k): every spin has k + 1 neighbours (cavity / excess degree k).
  ("poisson", c): degree ~ Poisson(c) (Erdos-Renyi, mean degree c; excess degree Poisson(c)).
  ("pm1", J): J_ij = +-J with probability 1/2;  ("gauss", sigma): J_ij ~ N(0, sigma^2).
  Transition: B E_J tanh^2(beta_c J) = 1 with branching B = k (regular) or c (Poisson).
Statistical outputs (population dynamics) are compared within stated stderr bands; the
brute-force enumeration and the exact T = 0 recursion are this file's independent routes.
"""
import itertools
import os
import sys
import time

import mpmath as mp
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.cavity import graphs as G
from replicant.cavity import sparse as S
from replicant.cavity.sparse import (bp_fixed_point, pd_growth_factor, population_dynamics,
                                     rs_zero_temperature_pm1, viana_bray_beta_c, viana_bray_tc)
from replicant.core import Refusal
from replicant.families.common import CAVITY_CAVEAT
from replicant.refdata import check, load
from replicant.testing import slow

REF = None


def _ref():
    global REF
    if REF is None:
        REF = load("viana_bray")
    return REF


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        assert r.detail, "refusal must say why"
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _band(got, err, want, tol, runs, conf=0.99):
    """Statistical comparison: |got - want| <= t_{runs-1} sqrt(err^2 + tol^2), with the
    two-sided Student-t quantile at `conf` (our stderr comes from only `runs` runs, so
    normal quantiles would be too narrow)."""
    t = S.t_quantile(1 - (1 - conf) / 2, runs - 1)
    return abs(got - want) <= t * np.hypot(err, tol), (got, err, want, tol, t)


# ---- BP on trees vs exhaustive enumeration (independent route) -----------------------
def _enumerate(graph, beta, field):
    """All 2^n states: ln Z and <s_i>; nothing from the BP code is used."""
    n = graph.n
    states = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    i, j = graph.edges[:, 0], graph.edges[:, 1]
    energy = -(states[:, i] * states[:, j] * graph.J).sum(1) - field * states.sum(1)
    a = -beta * energy
    amax = a.max()
    w = np.exp(a - amax)
    z = w.sum()
    return amax + np.log(z), (w[:, None] * states).sum(0) / z


def test_bp_on_tree_matches_enumeration():
    cases = [(("tree", 9), ("pm1", 1.0), 0.9, 0.0), (("tree", 10), ("gauss", 1.0), 1.7, 0.3),
             (("tree", 7), ("pm1", 0.5), 3.0, -0.2), (("tree", 2), ("gauss", 2.0), 0.4, 0.0)]
    for seed, (law, jlaw, beta, field) in enumerate(cases):
        r = bp_fixed_point(law, beta, jlaw, seed=seed, field=field)
        assert r.converged, r
        lnz, mag = _enumerate(r.graph, beta, field)
        assert abs(r.log_z - lnz) < 1e-12, (law, r.log_z, lnz)
        assert np.max(np.abs(r.marginals - mag)) < 1e-12, (law, r.marginals, mag)
        assert r.messages.shape == (2 * len(r.graph.edges),)
        assert r.caveat == CAVITY_CAVEAT


def test_bp_accepts_an_explicit_graph_and_its_message_layout():
    """Path 0-1-2: messages[2e] = u_{i->j}, messages[2e+1] = u_{j->i} for edges[e] = (i, j)."""
    g = G.Graph(3, np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
    beta, H = 0.7, 0.4
    r = bp_fixed_point(g, beta, None, seed=0, field=H)
    u = lambda J, h: np.arctanh(np.tanh(beta * J) * np.tanh(beta * h)) / beta
    u01 = u(1.0, H)                      # leaf 0 sends with cavity field H
    u21 = u(-1.0, H)
    u12 = u(-1.0, H + u01)
    u10 = u(1.0, H + u21)
    assert np.allclose(r.messages, [u01, u10, u12, u21], atol=1e-14), r.messages
    _refuses(lambda: bp_fixed_point(g, beta, ("pm1", 1.0), seed=0), "precondition:j-law")


def test_bp_random_regular_high_temperature_converges_to_paramagnet():
    r = bp_fixed_point(("regular", 2), 0.5, ("pm1", 1.0), seed=3, n=300)
    assert r.converged and np.max(np.abs(r.messages)) < 1e-10, r.residual
    assert np.all((r.graph.degrees() == 3))


def test_bp_reports_non_convergence_and_refuses_on_request():
    # loopy 3-regular graph deep in the spin-glass phase, no damping, few iterations
    args = (("regular", 2), 3.0, ("pm1", 1.0))
    r = bp_fixed_point(*args, seed=1, n=200, max_iter=30)
    assert not r.converged and r.residual > 1e-6, r.residual
    assert r.marginals is None and r.log_z is None     # no Bethe quantities off a fixed point
    e = _refuses(lambda: bp_fixed_point(*args, seed=1, n=200, max_iter=30,
                                        require_converged=True), "no-convergence")
    assert "damping" in e.detail and "residual" in e.detail


# ---- Viana-Bray transition (closed form, banked) --------------------------------------
def test_viana_bray_tc_closed_form_regular_graph():
    ref = _ref()
    bc = viana_bray_beta_c(("regular", 2), ("pm1", 1))
    assert check(ref["beta_c_regular_pm1_k2"], mp.nstr(bc, 25), label="k=2").ok, bc
    tc = viana_bray_tc(("regular", 2), ("pm1", 1))
    with mp.workdps(30):
        assert abs(tc * bc - 1) < mp.mpf("1e-25")
        assert abs(2 * mp.tanh(bc)**2 - 1) < mp.mpf("1e-25")      # k tanh^2(beta_c J) = 1
    bp = viana_bray_beta_c(("poisson", 3), ("pm1", 1))
    assert check(ref["beta_c_poisson_pm1_c3"], mp.nstr(bp, 25), label="c=3").ok, bp
    # J scales temperature: beta_c(J) = beta_c(1) / J
    with mp.workdps(30):
        assert abs(viana_bray_beta_c(("regular", 2), ("pm1", 2)) - bc / 2) < mp.mpf("1e-25")


def test_viana_bray_gaussian_condition_independent_quadrature():
    """Gaussian J: c E tanh^2(beta_c sigma z) = 1, checked by numpy Gauss-Hermite (not mpmath)."""
    before = mp.mp.dps
    bc = float(viana_bray_beta_c(("poisson", 4), ("gauss", 1.0)))
    assert mp.mp.dps == before, "mpmath global precision must be left unchanged"
    z, w = np.polynomial.hermite_e.hermegauss(120)
    val = 4 * float(w @ np.tanh(bc * z)**2) / np.sqrt(2 * np.pi)
    assert abs(val - 1) < 1e-10, (bc, val)
    assert abs(float(viana_bray_beta_c(("poisson", 4), ("gauss", 2.0))) - bc / 2) < 1e-12


def test_viana_bray_refusals():
    _refuses(lambda: viana_bray_tc(("regular", 1), ("pm1", 1)), "no-transition")
    _refuses(lambda: viana_bray_tc(("poisson", 0.8), ("gauss", 1)), "no-transition")
    _refuses(lambda: viana_bray_tc(("scale-free", 3), ("pm1", 1)), "unsupported:degree-law")
    _refuses(lambda: viana_bray_tc(("regular", 2), ("cauchy", 1)), "unsupported:J-law")
    _refuses(lambda: viana_bray_tc(("regular", 2.5), ("pm1", 1)), "unsupported:degree-law")
    _refuses(lambda: viana_bray_tc(("regular", "two"), ("pm1", 1)), "unsupported:degree-law")
    _refuses(lambda: viana_bray_tc(("poisson", None), ("pm1", 1)), "unsupported:degree-law")
    _refuses(lambda: viana_bray_tc(("regular", 2), ("pm1", "one")), "unsupported:J-law")


# ---- population dynamics -------------------------------------------------------------
_SMALL = dict(pop=300, sweeps=10, burn_in=30, runs=4)


def test_pd_same_seed_same_result():
    a = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, seed=7, **_SMALL)
    b = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, seed=7, **_SMALL)
    c = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, seed=8, **_SMALL)
    assert a.value == b.value and a.stderr == b.stderr
    assert a.value != c.value


def test_pd_result_is_labelled_statistical():
    r = population_dynamics(("poisson", 3), ("gauss", 1.0), 1.2, seed=1, **_SMALL)
    assert r.label == "statistical" and r.caveat == CAVITY_CAVEAT
    assert set(r.value) == set(r.stderr) == {"q", "e", "f"}
    assert all(np.isfinite(r.stderr[k]) and r.stderr[k] >= 0 for k in r.stderr)
    assert [e["step"] for e in r.trace.to_json()["entries"]] == list(S.STEP_NAMES)


def test_pd_stderr_shrinks_with_population():
    kw = dict(sweeps=5, burn_in=20, runs=8)
    small = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, pop=1000, seed=2, **kw)
    big = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, pop=10000, seed=2, **kw)
    for key in ("q", "e"):
        assert big.stderr[key] * 2 <= small.stderr[key], (key, small.stderr, big.stderr)


def test_pd_paramagnet_above_tc_and_spin_glass_below():
    """k = 2, J = +-1, beta_c = 0.8814.  From the field-biased start (all h = 1):
    beta = 0.7 collapses to h = 0 (q = 0, e = -(3/2) tanh beta); beta = 1.5 keeps q > 0."""
    long = {**_SMALL, "burn_in": 200}           # decay ~ (k tanh^2 beta)^(t/2) per sweep
    hi_t = population_dynamics(("regular", 2), ("pm1", 1.0), 0.7, seed=0, **long)
    assert hi_t.value["q"] < 1e-8 and hi_t.phase == "paramagnet", hi_t.value
    assert abs(hi_t.value["e"] + 1.5 * np.tanh(0.7)) < 1e-6, hi_t.value
    lo_t = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5, seed=0, **_SMALL)
    assert lo_t.value["q"] > 0.2 + 5 * lo_t.stderr["q"] and lo_t.phase == "spin-glass", lo_t


def test_pd_transition_location_from_linear_growth():
    """PD's own transition (no closed form): a tiny-amplitude population grows iff
    beta > beta_c.  Checked on both graph laws against the banked beta_c.  Note: at tiny
    amplitude this measures the linearisation u ~ tanh(beta J) h, the same one the closed
    form comes from, so it is a WEAK independent route for beta_c (it checks the code, not
    the physics); the stronger evidence is the slow bisection with Gaussian J and the
    equilibrated PD phases below."""
    for law, key in ((("regular", 2), "beta_c_regular_pm1_k2"),
                     (("poisson", 3), "beta_c_poisson_pm1_c3")):
        bc = float(_ref()[key].value)
        below = pd_growth_factor(law, ("pm1", 1.0), 0.95 * bc, pop=2000, sweeps=8, seed=4)
        above = pd_growth_factor(law, ("pm1", 1.0), 1.05 * bc, pop=2000, sweeps=8, seed=4)
        assert below.value["growth"] + 3 * below.stderr["growth"] < 1, (law, below)
        assert above.value["growth"] - 3 * above.stderr["growth"] > 1, (law, above)


def _planted_transfer(beta, J, h):
    """PLANTED ERROR: beta dropped from the coupling, atanh(tanh(J) tanh(beta h)) / beta."""
    return np.arctanh(np.tanh(J) * np.tanh(beta * h)) / beta


def test_planted_transfer_error_moves_the_transition_and_fails_the_check():
    bc = float(_ref()["beta_c_regular_pm1_k2"].value)
    below = pd_growth_factor(("regular", 2), ("pm1", 1.0), 0.95 * bc, pop=2000, sweeps=8,
                             seed=4, _transfer=_planted_transfer)
    assert below.value["growth"] - 3 * below.stderr["growth"] > 1, below   # wrong side
    hi_t = population_dynamics(("regular", 2), ("pm1", 1.0), 0.7, seed=0,
                               _transfer=_planted_transfer, **_SMALL)
    assert hi_t.value["q"] > 1e-3, hi_t.value       # no paramagnet above the true T_c


def test_rs_zero_temperature_exact_matches_bank():
    ref = _ref()
    sol = rs_zero_temperature_pm1(2)
    assert check(ref["p0_rs_T0_regular_pm1_k2"], mp.nstr(sol["p0"], 25), label="p0").ok
    assert check(ref["e0_rs_regular_pm1_k2"], mp.nstr(sol["e"], 25), label="e0").ok
    R = sp.Rational
    assert sol["p_exact"] == {0: R(1, 3), 1: R(2, 9), 2: R(1, 9)}, sol["p_exact"]
    assert sol["e_exact"] == R(-23, 18), sol["e_exact"]


def test_pd_zero_temperature_matches_exact_rs():
    """T = 0 PD (small pop) vs the exact integer-field RS solution: q = P(H != 0) = 20/27
    and e = -23/18, within 99% Student-t bands (4 runs)."""
    sol = rs_zero_temperature_pm1(2)
    assert sol["q_exact"] == sp.Rational(20, 27)
    r = population_dynamics(("regular", 2), ("pm1", 1.0), mp.inf, pop=500, sweeps=10,
                            burn_in=40, runs=4, seed=5)
    for obs in ("q", "e"):
        ok, info = _band(r.value[obs], r.stderr[obs], float(sol[obs]), 0.0, 4)
        assert ok, (obs, info)
    assert r.stderr["q"] < 0.005 and r.phase == "spin-glass", r


def test_pd_near_tc_is_never_a_confident_spin_glass():
    """0.9 beta_c (k = 2, +-1), short burn-in: the biased start is still decaying to h = 0.
    The drift diagnostic (dynamics only) must not report a spin glass with q > 0."""
    bc = float(_ref()["beta_c_regular_pm1_k2"].value)
    for seed in range(4):
        r = population_dynamics(("regular", 2), ("pm1", 1.0), 0.9 * bc, seed=seed, **_SMALL)
        assert r.phase in ("paramagnet", "undetermined"), (seed, r.phase, r.value)
        assert r.phase == "paramagnet" or "not equilibrated" in r.note, r.note
    _refuses(lambda: population_dynamics(("regular", 2), ("pm1", 1.0), 0.9 * bc, seed=0,
                                         require_equilibrated=True, **_SMALL),
             "no-convergence:equilibration")
    # well below beta_c with enough burn-in: the paramagnet
    r = population_dynamics(("regular", 2), ("pm1", 1.0), 0.6 * bc, seed=0,
                            **{**_SMALL, "burn_in": 150})
    assert r.phase == "paramagnet" and r.value["q"] < 1e-12, r


def test_pd_above_tc_is_a_spin_glass_with_the_right_q():
    """1.5 beta_c (k = 2): spin glass, stationary, for several seeds; and k = 5 at T = 0.8
    (beta = 2.6 beta_c) small pop vs Mezard-Parisi RS q (Student-t band, 4 runs)."""
    bc = float(_ref()["beta_c_regular_pm1_k2"].value)
    for seed in range(3):
        r = population_dynamics(("regular", 2), ("pm1", 1.0), 1.5 * bc, seed=seed, **_SMALL)
        assert r.phase == "spin-glass" and r.value["q"] > 0.4, (seed, r)
    ref = _ref()["q_rs_regular_pm1_k5_T0p8"]
    r = population_dynamics(("regular", 5), ("pm1", 1.0), 1.25, seed=0, **_SMALL)
    ok, info = _band(r.value["q"], r.stderr["q"], float(ref.value), float(ref.tol), 4)
    assert ok and r.phase == "spin-glass", info


def test_pd_refusals():
    ok = dict(pop=300, sweeps=2, seed=0)
    _refuses(lambda: population_dynamics(("star", 2), ("pm1", 1), 1.0, **ok),
             "unsupported:degree-law")
    _refuses(lambda: population_dynamics(("regular", 2), ("uniform", 1), 1.0, **ok),
             "unsupported:J-law")
    _refuses(lambda: population_dynamics(("regular", 2), ("pm1", 1), -1.0, **ok),
             "precondition:beta-positive")
    _refuses(lambda: population_dynamics(("regular", 2), ("pm1", 1), 1.0, pop=5, sweeps=2,
                                         seed=0), "precondition:population-size")
    _refuses(lambda: population_dynamics(("regular", 2), ("pm1", 1), 1.0, pop=300, sweeps=2,
                                         seed=0, runs=1), "precondition:runs")
    _refuses(lambda: population_dynamics(("regular", 2), ("pm1", 1), 1.0, pop=300, sweeps=3,
                                         seed=0), "precondition:sweeps")
    _refuses(lambda: population_dynamics(("regular", "k"), ("pm1", 1), 1.0, **ok),
             "unsupported:degree-law")


def test_sparse_leaves_mpmath_precision_unchanged():
    before = mp.mp.dps          # the Gaussian (mp.quad) path is exercised just above
    viana_bray_tc(("poisson", 2), ("pm1", 1))
    rs_zero_temperature_pm1(3)
    assert mp.mp.dps == before


# ---- slow tier -----------------------------------------------------------------------
@slow
def test_pd_reproduces_mezard_parisi_rs_k5():
    """Mezard-Parisi 2001 Eq. (87), RS, J = +-1, k = 5, T = 0.8: q, U, F (statistical)."""
    t0 = time.time()
    ref = _ref()
    r = population_dynamics(("regular", 5), ("pm1", 1.0), 1.25, pop=10000, sweeps=40,
                            burn_in=150, runs=8, seed=11)
    assert r.phase == "spin-glass", r.note
    for obs, key in (("q", "q_rs_regular_pm1_k5_T0p8"), ("e", "e_rs_regular_pm1_k5_T0p8"),
                     ("f", "f_rs_regular_pm1_k5_T0p8")):
        ok, info = _band(r.value[obs], r.stderr[obs], float(ref[key].value),
                         float(ref[key].tol), 8)
        assert ok, (obs, info)
    assert time.time() - t0 < 120


@slow
def test_pd_bisected_transition_matches_closed_form():
    bc = float(_ref()["beta_c_regular_pm1_k2"].value)
    got = S.pd_transition_beta(("regular", 2), ("pm1", 1.0), lo=0.5, hi=1.5, pop=20000,
                               sweeps=10, seed=9, iters=14)
    assert abs(got - bc) < 0.01 * bc, (got, bc)
    gb = float(viana_bray_beta_c(("poisson", 4), ("gauss", 1.0)))
    got = S.pd_transition_beta(("poisson", 4), ("gauss", 1.0), lo=0.2, hi=1.5, pop=20000,
                               sweeps=10, seed=9, iters=14)
    assert abs(got - gb) < 0.015 * gb, (got, gb)


@slow
def test_pd_zero_temperature_large_population():
    sol = rs_zero_temperature_pm1(2)
    r = population_dynamics(("regular", 2), ("pm1", 1.0), mp.inf, pop=20000, sweeps=30,
                            burn_in=100, runs=8, seed=6)
    ok, info = _band(r.value["e"], r.stderr["e"], float(sol["e"]), 0.0, 8)
    assert ok and r.stderr["e"] < 2e-3, info
    # the banked p0 = 1/3, through PD: q = P(H != 0) = 1 - P(H = 0) computed from p0
    p0 = mp.mpf(_ref()["p0_rs_T0_regular_pm1_k2"].value)
    q_from_p0 = float(1 - (p0**3 + 3 * p0 * ((1 - p0) / 2)**2 * 2))  # H = sum of 3 u's
    assert abs(q_from_p0 - 20 / 27) < 1e-12
    ok, info = _band(r.value["q"], r.stderr["q"], q_from_p0, 0.0, 8)
    assert ok and r.stderr["q"] < 5e-4 and r.phase == "spin-glass", info


@slow
def test_bp_on_sampled_graph_low_temperature_with_damping_is_reported():
    """3-regular, N = 1000, beta = 1.5 > beta_c: BP either converges (to a nonzero
    fixed point) or says it did not; `converged` must match the measured residual."""
    r = bp_fixed_point(("regular", 2), 1.5, ("pm1", 1.0), seed=2, n=1000, damping=0.5,
                       max_iter=2000)
    assert r.converged == (r.residual < 1e-12), r
    r = bp_fixed_point(("poisson", 3), 0.5, ("gauss", 1.0), seed=2, n=2000)
    bc = float(viana_bray_beta_c(("poisson", 3), ("gauss", 1.0)))
    assert 0.5 < bc and r.converged and np.max(np.abs(r.messages)) < 1e-10
