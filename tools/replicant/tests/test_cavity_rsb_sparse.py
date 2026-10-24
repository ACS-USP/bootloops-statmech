"""Sparse 1RSB cavity: population of populations (T > 0) and integer-field surveys (T = 0).

Conventions (cavity/rsb_sparse.py): E = -sum J s s, Parisi parameter m in [0, 1], states
reweighted by z^m = exp(-beta m DeltaF); at T = 0, y = beta m fixed (Mezard-Parisi 2003's
mu), reweighting exp(-y DeltaE).  Reductions to RS: m = 1 at T > 0 (barycentres obey BP),
y = 0 at T = 0 (no reweighting).  All population outputs are statistical.
"""
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.cavity import rsb_sparse as R
from replicant.cavity import sparse as S
from replicant.cavity.rsb_sparse import (factorized_1rsb_t0, pop_of_pops_1rsb,
                                         rsb_ground_state_energy, rsb_sparse_energy)
from replicant.cavity.sparse import population_dynamics, rs_zero_temperature_pm1
from replicant.core import Refusal
from replicant.families.common import CAVITY_CAVEAT
from replicant.refdata import check, load
from replicant.testing import slow

PM = ("pm1", 1.0)
K2 = ("regular", 2)


def _ref():
    return load("viana_bray")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        assert r.detail, "refusal must say why"
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _band(got, err, want, err2, runs, conf=0.99):
    """|got - want| <= t_{runs-1} hypot(err, err2), two-sided Student-t at `conf`."""
    t = S.t_quantile(1 - (1 - conf) / 2, runs - 1)
    return abs(got - want) <= t * np.hypot(err, err2), (got, err, want, err2, t)


_TINY = dict(pop=100, inner=12, sweeps=8, burn_in=16, runs=4)
_MINI = dict(pop=60, inner=8, sweeps=4, burn_in=4, runs=2)


# ---- finite temperature: population of populations ---------------------------------
def test_result_is_labelled_statistical():
    r = pop_of_pops_1rsb(K2, PM, 1.5, 0.5, seed=1, **_MINI)
    assert r.label == "statistical" and r.caveat == R.RSB_CAVEAT != CAVITY_CAVEAT
    assert "single 1RSB" in r.caveat and "statistical" in r.caveat and "not proven" in r.caveat
    assert set(r.value) == set(r.stderr) == {"f", "e", "q0", "q1"}
    assert all(np.isfinite(r.stderr[k]) and r.stderr[k] >= 0 for k in r.stderr)
    assert [e["step"] for e in r.trace.to_json()["entries"]] == list(R.STEP_NAMES)
    t0 = rsb_sparse_energy(K2, PM, 0.7, pop=60, sweeps=4, burn_in=4, runs=2, seed=1)
    assert t0.label == "statistical" and t0.caveat == R.RSB_CAVEAT
    assert set(t0.value) == {"phi", "phi_var", "e", "sigma", "p0", "asym"}


def test_reproducible_with_seed():
    a = pop_of_pops_1rsb(K2, PM, 1.5, 0.5, seed=3, **_MINI)
    b = pop_of_pops_1rsb(K2, PM, 1.5, 0.5, seed=3, **_MINI)
    c = pop_of_pops_1rsb(K2, PM, 1.5, 0.5, seed=4, **_MINI)
    assert a.value == b.value and a.stderr == b.stderr and a.value != c.value
    kw = dict(pop=60, sweeps=4, burn_in=4, runs=2)
    x = rsb_sparse_energy(K2, PM, 0.7, seed=3, **kw)
    y = rsb_sparse_energy(K2, PM, 0.7, seed=3, **kw)
    assert x.value == y.value and x.value != rsb_sparse_energy(K2, PM, 0.7, seed=4, **kw).value


def test_m_equal_one_reduces_to_rs():
    """m = 1 (Parisi convention, reweighting z^m): the reweighted barycentres
    E_w tanh(beta h) obey the RS (BP) recursion exactly, so f(m=1) = f_RS and
    e(m=1) = e_RS (Mezard-Montanari 2009 ch. 19).  Compared with Task 12's RS
    population dynamics (independent code path), named seeds 21 / 22, 99% Student-t
    bands on the combined stderr.  Started from a genuinely spread inner population
    (spread 1.0), so this is not the trivial delta-inner solution."""
    beta = 1.5
    one = pop_of_pops_1rsb(K2, PM, beta, 1.0, pop=150, inner=16, sweeps=8, burn_in=16,
                           runs=4, seed=21, init_spread=1.0)
    rs = population_dynamics(K2, PM, beta, pop=1000, sweeps=10, burn_in=30, runs=4, seed=22)
    assert one.value["q1"] > one.value["q0"] + 0.02, one.value     # inner spread survives
    for obs in ("e", "f"):
        ok, info = _band(one.value[obs], one.stderr[obs], rs.value[obs], rs.stderr[obs], 4)
        assert ok, (obs, info)
    # and q0 (overlap of barycentres) equals the RS q = E tanh^2(beta H)
    ok, info = _band(one.value["q0"], one.stderr["q0"], rs.value["q"], rs.stderr["q"], 4)
    assert ok, ("q0", info)
    # PLANTED ERROR at T > 0: wrong sign z^-m in the inner resampling; the barycentres no
    # longer follow BP and q0 misses RS q by ~0.06, failing the same band (default tier;
    # the dropped-reweighting control, a smaller effect, is the slow m = 1 test)
    bad = pop_of_pops_1rsb(K2, PM, beta, 1.0, pop=100, inner=12, sweeps=6, burn_in=16,
                           runs=4, seed=21, init_spread=1.0, _reweight_sign=-1)
    ok, info = _band(bad.value["q0"], bad.stderr["q0"], rs.value["q"], rs.stderr["q"], 4)
    assert not ok, ("planted q0", info)


def test_m_equal_one_poisson_gaussian_runs_and_matches_rs():
    """The other graph / coupling laws: Poisson(3) degrees (padded parents), Gaussian J,
    m = 1 vs Task 12 RS PD (seeds 23 / 24), 99% Student-t band on e and q0."""
    law, jl, beta = ("poisson", 3), ("gauss", 1.0), 1.2
    one = pop_of_pops_1rsb(law, jl, beta, 1.0, pop=60, inner=8, sweeps=4, burn_in=16,
                           runs=3, seed=23, init_spread=1.0)
    rs = population_dynamics(law, jl, beta, pop=500, sweeps=6, burn_in=20, runs=4, seed=24)
    for obs, ro in (("e", "e"), ("q0", "q")):
        ok, info = _band(one.value[obs], one.stderr[obs], rs.value[ro], rs.stderr[ro], 3)
        assert ok, (obs, info)


def test_finite_t_1rsb_differs_from_rs_below_m_one():
    """k = 5, T = 0.8, m = 0.24 (Mezard-Parisi 2001 Sec. VII setting and their x*): states
    are distinct, q1 > q0 well beyond noise.  (At k = 2, beta = 1.5 small inner populations
    collapse towards q1 = q0 within ~60 sweeps; not used here.)"""
    r = pop_of_pops_1rsb(("regular", 5), PM, 1.25, 0.24, pop=60, inner=16, sweeps=6,
                         burn_in=24, runs=3, seed=5, init_spread=1.0)
    gap = r.value["q1"] - r.value["q0"]
    assert gap > 0.1 and gap > 3 * (r.stderr["q1"] + r.stderr["q0"]), (r.value, r.stderr)
    # 3 runs: the drift label is weak at this size (seed 6 at burn_in 40 says
    # "undetermined"); the stationary label at real sizes is the slow k = 5 test
    assert r.phase in ("1rsb", "undetermined"), r


def test_delta_inner_populations_are_the_rs_solution_at_any_m():
    """init_spread = 0: every inner population is a delta and stays one (couplings are
    common to all states of an edge), so q1 = q0 = q_RS: the trivial 1RSB solution."""
    r = pop_of_pops_1rsb(K2, PM, 1.5, 0.4, seed=6, init_spread=0.0, **_MINI)
    assert abs(r.value["q1"] - r.value["q0"]) < 1e-12 and r.phase == "rs", r


# ---- zero temperature: surveys over integer cavity fields -----------------------------
def test_y_zero_reduces_to_rs_zero_temperature():
    """y = beta m = 0: no reweighting, Phi(0) = e = the exact RS -23/18 (Task 12 bank)."""
    e_rs = float(mp.mpf(_ref()["e0_rs_regular_pm1_k2"].value))
    r = rsb_sparse_energy(K2, PM, 0.0, pop=300, sweeps=10, burn_in=20, runs=4, seed=7)
    for obs in ("phi", "e"):
        ok, info = _band(r.value[obs], r.stderr[obs], e_rs, 0.0, 4)
        assert ok, (obs, info)
    assert r.value["sigma"] == 0.0


def _p0_closed_form(mu):
    """Mezard-Parisi 2003 Eqs. (51)-(52), k = 2 factorized solution (the paper's algebra,
    written here only as a test oracle; the code under test iterates the recursion)."""
    e = mp.e**mu
    p0 = (24 + 3 * e**2 - (8 + e) * mp.sqrt(8 + e**2)) / (4 * (e - 1)**2)
    return p0


def test_factorized_solution_matches_banked_value_and_closed_form():
    ref = _ref()
    sol = factorized_1rsb_t0(2)
    assert check(ref["e0_1rsb_factorized_regular_pm1_k2"], mp.nstr(sol["e0"], 12),
                 label="factorized e0").ok, sol
    assert check(ref["mu_star_1rsb_factorized_regular_pm1_k2"], mp.nstr(sol["y_star"], 12),
                 label="factorized mu*").ok, sol
    with mp.workdps(20):
        for mu in (0.2, 0.4174, 0.9):
            got = factorized_1rsb_t0(2, y=mu)["p0"]
            assert abs(got - _p0_closed_form(mp.mpf(mu))) < 1e-10, (mu, got)
    assert abs(sol["sigma"]) < 1e-8          # Sigma = y (e - Phi) = 0 at the maximum
    assert sol["caveat"] == R.FACTORIZED_CAVEAT and "NOT" in sol["caveat"]


def test_symmetric_start_stays_factorized_random_start_does_not():
    """Population of surveys: a symmetric start stays exactly factorized (asym = 0) and
    reproduces the deterministic factorized phi(y); a random start breaks the factorization
    (Mezard-Parisi 2003 Fig. 5).  That the broken solution has the higher phi is slow-tier."""
    y = 0.7
    fac = factorized_1rsb_t0(2, y=y)
    sym = rsb_sparse_energy(K2, PM, y, pop=60, sweeps=4, burn_in=4, runs=2, seed=8,
                            init="symmetric")
    assert sym.value["asym"] < 1e-12 and sym.phase == "factorized", sym
    for key in ("phi", "phi_var"):
        assert abs(sym.value[key] - float(fac["phi"])) < 1e-9, (key, sym.value, fac)
    assert abs(sym.value["e"] - float(fac["e"])) < 1e-9, (sym.value, fac)
    full = rsb_sparse_energy(K2, PM, y, pop=100, sweeps=8, burn_in=30, runs=3, seed=8)
    # 30 sweeps is not fully relaxed (the drift diagnostic may say "undetermined"; the
    # stationary label is checked in the slow tier), but factorization is clearly broken
    assert full.value["asym"] > 0.2 and full.phase in ("1rsb", "undetermined"), full


def test_planted_reweighting_errors_are_caught():
    """PLANTED ERRORS (_reweight_sign): 0 drops the reweighting exp(-y DeltaE) from the
    recursion, -1 uses exp(+y DeltaE).  The two free-energy forms (variational Eq. (31)
    and the fixed-point Eq. (50) form) then disagree by ~0.09 and the run is refused; the
    correct recursion at the same (small, not fully relaxed) settings is accepted."""
    kw = dict(pop=100, sweeps=6, burn_in=24, runs=3, seed=9, require_consistent=True)
    for sign in (0, -1):
        _refuses(lambda: rsb_sparse_energy(K2, PM, 0.7, _reweight_sign=sign, **kw),
                 "no-solution:estimator-mismatch")
    ok = rsb_sparse_energy(K2, PM, 0.7, **kw)
    assert abs(ok.value["phi"] - ok.value["phi_var"]) < 0.03, ok.value


def test_ground_state_route_labels_and_traces():
    """rsb_ground_state_energy: a symmetric start stays factorized, so its phase is
    "factorized" (not "1rsb") and its e0 is the exact factorized maximum of the parabola;
    the typed-step trace is attached like the other routes."""
    r = rsb_ground_state_energy(K2, PM, ys=(0.2, 0.4, 0.6), pop=50, sweeps=4, burn_in=4,
                                runs=2, seed=0, init="symmetric")
    assert r.phase == "factorized" and r.label == "statistical", (r.phase, r.note)
    assert abs(r.value["e0"] - float(factorized_1rsb_t0(2)["e0"])) < 1e-4, r.value
    assert [e["step"] for e in r.trace.to_json()["entries"]] == list(R.STEP_NAMES)


def test_drifting_nested_population_is_undetermined_or_refused():
    """T = 0: a near-factorized start (symmetric surveys + 1e-6 asymmetry) relaxes away from
    the unstable factorized solution, the asymmetry growing ~10% per sweep; with a short
    burn-in the drift diagnostic (dynamics only) must flag it.  T > 0: 0.95 beta_c (k = 2)
    from a biased start, the barycentres still decaying to h = 0 (as Task 12)."""
    kw = dict(pop=100, sweeps=8, burn_in=4, runs=4, seed=10, init="near-factorized")
    r = rsb_sparse_energy(K2, PM, 0.7, **kw)
    assert r.phase == "undetermined" and "not equilibrated" in r.note, (r.phase, r.note)
    _refuses(lambda: rsb_sparse_energy(K2, PM, 0.7, require_equilibrated=True, **kw),
             "no-convergence:equilibration")
    bc = float(mp.mpf(_ref()["beta_c_regular_pm1_k2"].value))
    f = pop_of_pops_1rsb(K2, PM, 0.95 * bc, 0.5, pop=60, inner=8, sweeps=8, burn_in=16,
                         runs=4, seed=10, init_field=1.0, init_spread=0.5)
    assert f.phase == "undetermined" and "not equilibrated" in f.note, (f.phase, f.note)


def test_refusals():
    ok = dict(pop=100, inner=12, sweeps=8, seed=0)
    _refuses(lambda: pop_of_pops_1rsb(("star", 2), PM, 1.0, 0.5, **ok), "unsupported:degree-law")
    _refuses(lambda: pop_of_pops_1rsb(K2, ("cauchy", 1), 1.0, 0.5, **ok), "unsupported:J-law")
    _refuses(lambda: pop_of_pops_1rsb(K2, PM, mp.inf, 0.5, **ok), "precondition:beta-finite")
    _refuses(lambda: pop_of_pops_1rsb(K2, PM, -1.0, 0.5, **ok), "precondition:beta-positive")
    for m in (0.0, 1.5, -0.1, float("nan")):
        _refuses(lambda: pop_of_pops_1rsb(K2, PM, 1.0, m, **ok), "precondition:parisi-m")
    _refuses(lambda: pop_of_pops_1rsb(K2, PM, 1.0, 0.5, pop=100, inner=1, sweeps=8, seed=0),
             "precondition:inner-size")
    _refuses(lambda: pop_of_pops_1rsb(K2, PM, 1.0, 0.5, pop=10, inner=12, sweeps=8, seed=0),
             "precondition:population-size")
    t0 = dict(pop=200, sweeps=8, seed=0)
    _refuses(lambda: rsb_sparse_energy(("poisson", 3), PM, 0.5, **t0), "unsupported:degree-law")
    _refuses(lambda: rsb_sparse_energy(K2, ("gauss", 1.0), 0.5, **t0), "unsupported:J-law")
    _refuses(lambda: rsb_sparse_energy(K2, PM, -0.5, **t0), "precondition:y")
    _refuses(lambda: rsb_sparse_energy(K2, PM, 0.5, init="bogus", **t0), "precondition:init")
    _refuses(lambda: factorized_1rsb_t0(1), "precondition:k")
    _refuses(lambda: rsb_ground_state_energy(K2, PM, ys=(0.5,), **t0), "precondition:y-grid")


def test_rsb_leaves_mpmath_precision_unchanged():
    before = mp.mp.dps
    factorized_1rsb_t0(2)
    rsb_sparse_energy(K2, PM, 0.3, pop=200, sweeps=4, burn_in=4, runs=2, seed=0)
    assert mp.mp.dps == before


# ---- slow tier -----------------------------------------------------------------------
@slow
def test_energy_matches_banked_reference():
    """Full (non-factorized) 1RSB ground-state energy, k = 2, J = +-1: max over y of phi(y),
    one least-squares parabola through the run-averaged phi on a y grid, jackknife stderr
    over 8 runs, 99% Student-t band against the banked Mezard-Parisi 2003 value (a
    reproduction target, not an input).  The banked tol enters the band as a standard
    error, so that window (~8e-4) is wider than the gap to the factorized maximum
    (6.1e-4): the full-vs-factorized discrimination is the separate e0 - fac > t * stderr
    assertion.  It must also be clearly above RS (-23/18), and the planted
    no-reweighting dynamics must not pass."""
    t0 = time.time()
    ref = _ref()["e0_1rsb_regular_pm1_k2"]
    want, tol = float(ref.value), float(ref.tol)
    kw = dict(ys=(0.3, 0.5, 0.7, 0.9, 1.1), pop=2000, sweeps=300, burn_in=150, runs=8,
              seed=12)
    r = rsb_ground_state_energy(K2, PM, **kw)
    ok, info = _band(r.value["e0"], r.stderr["e0"], want, tol, 8)
    assert ok and r.stderr["e0"] < 4e-4 and r.phase == "1rsb", (info, r.note)
    assert 0.4 < r.value["y_star"] < 1.0, r.value
    e_rs = float(mp.mpf(_ref()["e0_rs_regular_pm1_k2"].value))
    assert r.value["e0"] - e_rs > 10 * r.stderr["e0"], (r.value, e_rs)
    # and above the factorized 1RSB maximum (-1.27231): measured 5.5 stderr at this build
    fac = float(factorized_1rsb_t0(2)["e0"])
    assert r.value["e0"] - fac > S.t_quantile(0.995, 7) * r.stderr["e0"], (r.value, r.stderr, fac)
    try:
        bad = rsb_ground_state_energy(K2, PM, _reweight_sign=0, **{**kw, "sweeps": 50,
                                                                   "burn_in": 50})
        ok_bad, info_bad = _band(bad.value["e0"], bad.stderr["e0"], want, tol, 8)
        assert not ok_bad, info_bad
    except Refusal as e:
        assert e.reason == "no-solution:no-maximum", e
    assert time.time() - t0 < 200


@slow
def test_full_1rsb_lies_above_factorized_at_fixed_y():
    """y = 0.7: random-start (non-factorized) phi exceeds the exact factorized phi(0.7) by
    many stderr (MP03 Fig. 4: full curve above the dotted one), and the two free-energy
    forms agree once relaxed (burn-in 100)."""
    fac = float(factorized_1rsb_t0(2, y=0.7)["phi"])
    full = rsb_sparse_energy(K2, PM, 0.7, pop=2000, sweeps=200, burn_in=100, runs=4, seed=15,
                             require_equilibrated=True, require_consistent=True)
    assert full.value["phi"] - fac > 8 * full.stderr["phi"], (full.value, full.stderr, fac)
    ok, info = _band(full.value["phi"], full.stderr["phi"], full.value["phi_var"],
                     full.stderr["phi_var"], 4)
    assert ok, info


@slow
def test_finite_t_m_one_reduces_to_rs_larger_population():
    """k = 2, beta = 1.5, m = 1, pop 1000, inner 160, 8 runs, vs Task 12 RS PD at pop 40000
    (8 runs; its stderr is ~2e-4, so the reference is not the noise floor).  Every
    comparison is a 99% Student-t band on the COMBINED stderr (1RSB runs and RS runs).
    Checked: the estimators f, e, q0 and the exact m = 1 barycentre functional f_bary,
    e_bary (Bethe f, e of the reweighted barycentres; isolates the identity from
    finite-inner estimator bias).  Measured at this build (offsets from RS):
    inner 160: e +0.0013(21), e_bary +0.0023(15), f -0.0013(22), q0 -0.0003(10).
    OPEN ITEM (not hidden): e_bary sits above RS by about +0.003 in every configuration
    measured -- inner 20: +0.0024(13); inner 40: +0.0029(20) (seed 31), +0.0071(8)
    (seed 13); inner 80: +0.0030(20); inner 160: +0.0023(15) -- each 1.4-1.8 sigma except
    seed 13 / inner 40 (9 sigma); RS PD at pop 1000 shows no such shift.  A candidate
    cause is barycentre sampling noise (q1 - q0) / inner, which stays ~1e-3 over this range
    because the inner spread grows with inner; not established.  The planted control
    (reweighting dropped) is off by ~0.075 and must fail the same band."""
    rs = population_dynamics(K2, PM, 1.5, pop=40000, sweeps=20, burn_in=50, runs=8, seed=17)
    kw = dict(pop=1000, inner=160, sweeps=20, burn_in=40, runs=8, seed=13, init_spread=1.0)
    one = pop_of_pops_1rsb(K2, PM, 1.5, 1.0, **kw)
    assert one.value["q1"] - one.value["q0"] > 0.05, one.value
    for obs, ro in (("e", "e"), ("f", "f"), ("q0", "q"), ("e_bary", "e"), ("f_bary", "f")):
        ok, info = _band(one.value[obs], one.stderr[obs], rs.value[ro], rs.stderr[ro], 8)
        assert ok, (obs, info)
    bad = pop_of_pops_1rsb(K2, PM, 1.5, 1.0, _reweight_sign=0, **{**kw, "inner": 40})
    ok, info = _band(bad.value["e"], bad.stderr["e"], rs.value["e"], rs.stderr["e"], 8)
    assert not ok, info


@slow
def test_finite_t_1rsb_reproduces_mezard_parisi_k5():
    """k = 5, T = 0.8, m = x* = 0.24 (the paper's maximiser, used as an input): U, F, q1
    vs Mezard-Parisi 2001 Eq. (88), 99% Student-t bands (8 runs) on hypot(stderr, paper
    error).  Finite inner = 100 (no extrapolation); measured at this build: e -1.794(3),
    f -1.865(3), q1 0.772 (4 runs), inner 200: e -1.797, f -1.861, q1 0.775."""
    ref = _ref()
    r = pop_of_pops_1rsb(("regular", 5), PM, 1.25, 0.24, pop=1000, inner=100, sweeps=20,
                         burn_in=60, runs=8, seed=16, init_spread=1.0)
    assert r.phase == "1rsb", r.note
    for obs, key in (("e", "e_1rsb_regular_pm1_k5_T0p8"), ("f", "f_1rsb_regular_pm1_k5_T0p8"),
                     ("q1", "q1_1rsb_regular_pm1_k5_T0p8")):
        ok, info = _band(r.value[obs], r.stderr[obs], float(ref[key].value),
                         float(ref[key].tol), 8)
        assert ok, (obs, info)
    # and clearly not RS: the RS e at the same point is -1.8160 (MP 2001 Eq. (87))
    rs = float(ref["e_rs_regular_pm1_k5_T0p8"].value)
    assert r.value["e"] - rs > 5 * r.stderr["e"], (r.value, rs)


@slow
def test_drift_diagnostic_over_seeds():
    """Near-factorized start, short burn-in: never a confident '1rsb' over 12 seeds; long
    burn-in from the random start: '1rsb' (not 'undetermined') in most seeds."""
    for seed in range(12):
        r = rsb_sparse_energy(K2, PM, 0.7, pop=500, sweeps=8, burn_in=4, runs=4, seed=seed,
                              init="near-factorized")
        assert r.phase == "undetermined", (seed, r.phase, r.note)
    good = [rsb_sparse_energy(K2, PM, 0.7, pop=1000, sweeps=20, burn_in=100, runs=8,
                              seed=s).phase for s in range(6)]
    assert good.count("1rsb") >= 5, good
