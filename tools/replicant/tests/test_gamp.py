"""amp/gamp.py, amp/gamp_se.py: GAMP and its state evolution for the teacher-student
perceptron (E3b), an independent finite-N route.

Conventions (amp/gamp.py): teacher w* with i.i.d. components from the prior (gaussian
N(0, 1) or ising +-1, rho = E w*^2 = 1), patterns Phi with i.i.d. N(0, 1) entries,
M = alpha N labels y = sign(Phi w* / sqrt N).  Bayes-optimal (matched prior and channel,
the Nishimori line).  q = (1/N) x_hat . w*; eps_Bayes = arccos(sqrt q)/pi (plug-in =
Bayes-optimal predictor for the sign channel), eps_Gibbs = arccos(q)/pi.
families/teacher_student.py appears HERE only, as the comparison target (spherical
teacher; the Gaussian prior gives the same overlap equation).  Banked thresholds
(reference/gamp.json) are targets, never inputs.  Finite-N checks are statistical: 99%
Student-t bands over the named seeds 0-23 (df = 23).
"""
import inspect
import math
import os
import re
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import amp as A
from replicant.amp import gamp as G
from replicant.amp import gamp_se as GS
from replicant.core import Refusal
from replicant.refdata import check, load
from replicant.testing import slow

ALPHAS = ("1", "2")          # default tier; the slow test adds 0.5 and 5
SEEDS = range(24)
N = 800


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        assert r.detail, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _bayes_gaps(audited=True, alphas=ALPHAS):
    from replicant.families.teacher_student import teacher_student_error
    out = {}
    for a in alphas:
        se = A.perceptron_state_evolution(a, "gaussian")
        assert se.converged == audited, (a, se.note)
        out[a] = (abs(se.eps_bayes - float(teacher_student_error(a, "bayes"))),
                  abs(se.eps_gibbs - float(teacher_student_error(a, "gibbs"))))
    return out


# ---- state evolution (default tier) ---------------------------------------------------
def test_gamp_state_evolution_matches_teacher_student_bayes_error():
    """Gaussian prior: the SE fixed point reached from the GAMP start q = 0 gives the
    Bayes generalization error of families.teacher_student (comparison target only) to
    1e-6 at alpha = 1, 2 (the slow test adds 0.5, 5, 10).  The agreement checks an
    identity: with the Gaussian prior the SE equation is Gyorgyi's spherical Gibbs
    equation (r = q/(1 - q)), so it tests the implementation, not the physics.  The
    Gibbs error arccos(q)/pi (Nishimori: a posterior sample has overlap q with the
    teacher) likewise.  Positive control: a planted factor 1.05
    in the channel hook `_rhat` (restored in finally, identity assert) is detected by
    the comparison, and by the SE's own xi-form audit (an independent quadrature of the
    unplanted channel), which then reports converged False.
    Ising prior: replicant has no RS formula for the binary teacher-student overlap
    (families/teacher_student.py is spherical), so it is checked here only for the
    small-alpha limit both priors share (unit variance: q = r - r^2 + O(r^3),
    r = 2 alpha/pi + O(alpha^2), so the two SE agree to O(alpha^3)); its independent
    checks are the banked thresholds (test_ising_prior_thresholds_match_bank) and the
    finite-N GAMP runs (slow)."""
    before = mp.mp.dps
    gaps = _bayes_gaps()
    for a, (gb, gg) in gaps.items():
        assert gb < 1e-6 and gg < 1e-6, (a, gb, gg)
    assert mp.mp.dps == before
    orig = GS._rhat
    GS._rhat = lambda q, alpha: 1.05 * orig(q, alpha)
    try:
        planted = _bayes_gaps(audited=False)
    finally:
        GS._rhat = orig
    assert GS._rhat is orig
    assert all(gb > 1e-4 for gb, _ in planted.values()), planted
    for a in ("0.01", "0.03"):
        qg = A.perceptron_state_evolution(a, "gaussian").overlap
        qi = A.perceptron_state_evolution(a, "ising").overlap
        assert abs(qg - qi) < 10 * float(a)**3, (a, qg, qi)
        assert abs(qg - 2 * float(a) / math.pi) < 2 * float(a)**2, (a, qg)


def test_channel_and_prior_maps_are_derivatives_of_the_potential():
    """BKMMZ eq. (10): r = 2 alpha Psi_out'(q), q = 2 psi_P0'(r).  The SE maps (Gauss-
    Hermite in the t variable) against central differences of the potentials Psi_out
    and psi_P0 computed independently here by adaptive mpmath quadrature."""
    with mp.workdps(16):
        Phi = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
        D = lambda f: mp.quad(lambda z: f(z) * mp.exp(-z * z / 2),
                              [-mp.inf, -3, 0, 3, mp.inf]) / mp.sqrt(2 * mp.pi)

        def Psi(q):
            s = mp.sqrt(q / (1 - q))
            return 2 * D(lambda v: Phi(s * v) * mp.log(Phi(s * v)))

        def psi_ising(r):
            return D(lambda z: mp.log(mp.cosh(r + mp.sqrt(r) * z))) - r / 2
        psi_gauss = lambda r: r / 2 - mp.log(1 + r) / 2
        d = mp.mpf("1e-6")
        for q in ("0.2", "0.9"):
            q = mp.mpf(q)
            want = 2 * (Psi(q + d) - Psi(q - d)) / (2 * d)
            assert abs(GS._rhat(float(q), 1.0) - want) < 1e-8, (q, want)
        # ising at large r only (where Gauss-Hermite had failed: the trapezoid rule's case)
        for prior, psi, rs in (("ising", psi_ising, ("7",)), ("gaussian", psi_gauss, ("0.3", "7"))):
            for r in map(mp.mpf, rs):
                want = 2 * (psi(r + d) - psi(r - d)) / (2 * d)
                assert abs(GS._prior_map(prior, float(r)) - want) < 1e-8, (prior, r)


def test_ising_prior_thresholds_match_bank():
    """Ising prior, sign channel: the information-theoretic threshold (the free entropy
    f_RS of the GAMP fixed point equals that of perfect recovery, -ln 2) and the GAMP
    (spinodal) threshold, computed from the SE maps and f_RS, against the banked values
    (targets).  SE from q = 0 stays below 0.95 at alpha = 1.45 (hard phase) and reaches
    q = 1 at 1.55; in (alpha_IT, alpha_AMP) GAMP is NOT Bayes-optimal (q* = 1).  Gaussian
    prior: one fixed point, no transition."""
    bank = load("gamp")
    th = A.ising_thresholds()
    for key, got in (("ising_alpha_it", th.alpha_it), ("ising_alpha_amp", th.alpha_amp)):
        c = check(bank[key], mp.nstr(mp.mpf(got), 12), label=key)
        assert c.ok, c
    # regression lock, not a reference: this code's own values (alpha_IT also reproduced by
    # an independent dps-25 mpmath computation in the build, 1.24468399)
    assert 1.2446 < th.alpha_it < 1.2448 and 1.4925 < th.alpha_amp < 1.4928, th
    low, high = A.perceptron_state_evolution("1.45", "ising"), \
        A.perceptron_state_evolution("1.55", "ising")
    assert low.converged and low.overlap < 0.95 and not low.gamp_is_optimal, low.note
    assert low.optimal_overlap == 1.0 and len(low.fixed_points) >= 3, low.fixed_points
    assert high.overlap == 1.0 and high.eps_bayes == 0.0 and high.gamp_is_optimal
    below = A.perceptron_state_evolution("1.2", "ising")
    assert below.gamp_is_optimal and below.optimal_overlap == below.overlap < 1
    g = A.perceptron_state_evolution("1.45", "gaussian")
    assert len(g.fixed_points) == 1 and g.gamp_is_optimal
    assert abs(g.mse - (1 - g.overlap)) < 1e-15


def test_planted_errors_move_the_thresholds():
    """Positive controls for the threshold check, one per threshold: a factor 1.05 planted
    in the channel hook `_rhat` moves alpha_AMP out of its banked window (alpha_IT barely
    moves: f_RS is stationary at the fixed point, so an error in the fixed-point map
    enters it at second order); a factor 1.02 planted in the potential hook `_Psi_out`
    moves alpha_IT out.  Hooks restored in finally (identity asserts)."""
    bank = load("gamp")
    ok = lambda key, x: check(bank[key], mp.nstr(mp.mpf(x), 12), label=key).ok
    for name, factor, key in (("_rhat", 1.05, "ising_alpha_amp"),
                              ("_Psi_out", 1.02, "ising_alpha_it")):
        orig = getattr(GS, name)
        setattr(GS, name, lambda *args, f=factor, o=orig: f * o(*args))
        try:
            th = A.ising_thresholds()
        finally:
            setattr(GS, name, orig)
        assert getattr(GS, name) is orig
        got = th.alpha_amp if key == "ising_alpha_amp" else th.alpha_it
        assert not ok(key, got), (name, key, got)


def test_gamp_refusals_say_why():
    _refuses(lambda: A.perceptron_state_evolution("1", "binary"), "precondition:known-prior")
    _refuses(lambda: A.perceptron_state_evolution("1", "gaussian", teacher=False),
             "not-implemented:mismatched-teacher-student")
    _refuses(lambda: A.perceptron_state_evolution("-1"), "precondition:nonnegative-alpha")
    _refuses(lambda: A.perceptron_state_evolution("x"), "precondition:numeric-alpha")
    X, y, w = A.perceptron_instance(20, 1.5, "ising", 0)
    assert X.shape == (30, 20) and set(np.unique(y)) <= {-1.0, 1.0} and set(np.unique(w)) \
        == {-1.0, 1.0}
    _refuses(lambda: A.gamp_perceptron(X, y[:-1], "ising", iters=3, seed=0),
             "precondition:label-count")
    _refuses(lambda: A.gamp_perceptron(X, 2 * y, "ising", iters=3, seed=0),
             "precondition:sign-labels")
    _refuses(lambda: A.gamp_perceptron(X[0], y, "ising", iters=3, seed=0),
             "precondition:pattern-matrix")
    _refuses(lambda: A.gamp_perceptron(X, y, "laplace", iters=3, seed=0),
             "precondition:known-prior")
    _refuses(lambda: A.gamp_perceptron(X, y, "ising", iters=0, seed=0),
             "precondition:positive-iters")
    _refuses(lambda: A.gamp_perceptron(X, y, "ising", iters=3, seed=0, teacher=w[:-1]),
             "precondition:teacher-length")
    _refuses(lambda: A.perceptron_instance(20, 0, "ising", 0), "precondition:positive-alpha")
    r = A.gamp_perceptron(X, y, "ising", iters=5, seed=0, teacher=w)
    assert r.label == "statistical" and "Nishimori" in r.note and len(r.overlap_trace) == 6
    assert r.overlap_trace[0] == 0.0


def test_gamp_modules_do_not_import_replica_code():
    for mod in (G, GS):
        src = inspect.getsource(mod)
        for line in src.splitlines():
            if re.match(r"\s*(from|import)\s", line):
                assert re.match(r"\s*(import (math|mpmath as mp|numpy as np)|from (dataclasses|"
                                r"scipy\.(special|integrate)|\.\.core|\._stats|\.gamp_se) import)",
                                line), (mod.__name__, line)
        for bad in ("families", "derive", "solve_derived", "teacher_student_error",
                    "gauss_expect", "..cavity", "..solve"):
            assert bad not in src, (mod.__name__, bad)


# ---- slow tier ------------------------------------------------------------------------
@slow
def test_gamp_state_evolution_matches_teacher_student_at_more_loads():
    """The default-tier comparison at alpha = 0.5, 5, 10 (Bayes and Gibbs, 1e-6)."""
    for a, (gb, gg) in _bayes_gaps(alphas=("0.5", "5", "10")).items():
        assert gb < 1e-6 and gg < 1e-6, (a, gb, gg)



def _ensemble(alpha, prior, T):
    return A.gamp_ensemble(N, alpha, prior, seeds=SEEDS, iters=T)


@slow
def test_gamp_finite_N_matches_state_evolution():
    """N = 800, alpha in {1, 2}, named seeds 0-23, t = 30: the GAMP overlap q_t against the
    SE trajectory at the same t from the same start (q = 0), in a 99% Student-t band
    (df = 23): gaussian prior at alpha = 1, 2 and ising prior at alpha = 1 (below
    alpha_IT).  The plug-in generalization error of the GAMP estimate against
    arccos(sqrt q_t)/pi likewise.  Ising at alpha = 2 (> alpha_AMP): SE reaches q = 1,
    but at N = 800 not every instance does (14 of 24 measured); the others freeze a few
    wrong spins (e.g. seed 2: 3 of 800, unchanged for any variance floor 1e-6..1e-12); the
    band (0.9928 +- 0.0074) only just reaches q = 1: observed, not explained.  The deficit 1 - q
    shrinks from N = 800 to 1600 to 3200 (asserted); the perfect fraction does not.
    Band-excludes control: the Onsager coefficient of the omega update scaled by the
    private hook gamp._onsager_scale (restored in finally, identity assert): 0.8, 0.9,
    1.1 and 1.2 all fall outside every band, and each planted band excludes the SE q_t
    (measured: the closest, 0.9 at gaussian alpha = 2, by 0.0035); smaller errors are
    not claimed."""
    T = 30
    bands = {}
    for a, prior in (("1", "gaussian"), ("2", "gaussian"), ("1", "ising")):
        ens = _ensemble(a, prior, T)
        se = A.perceptron_state_evolution(a, prior, iters=T)
        b = ens.band
        print(f"GAMP {prior} alpha={a} t={T}: q = {b.mean:.4f} +- {b.half_width:.4f} "
              f"(stderr {b.stderr:.4f}); SE {se.trajectory[-1]:.4f}; eps {ens.eps_band.mean:.4f}"
              f" +- {ens.eps_band.half_width:.4f}")
        assert abs(b.mean - se.trajectory[-1]) <= b.half_width, (a, prior, b, se)
        eps_t = math.acos(math.sqrt(se.trajectory[-1])) / math.pi
        assert abs(ens.eps_band.mean - eps_t) <= ens.eps_band.half_width, (a, prior)
        bands[(a, prior)] = b
    hard = _ensemble("2", "ising", T)
    assert A.perceptron_state_evolution("2", "ising").overlap == 1.0
    perfect = sum(q > 1 - 1e-7 for q in hard.values)
    print(f"GAMP ising alpha=2: {perfect}/{len(hard.values)} reach q = 1; band "
          f"{hard.band.mean:.4f} +- {hard.band.half_width:.4f}; min {min(hard.values):.4f}")
    # the band (0.9928 +- 0.0074 measured) just reaches 1, so it neither confirms nor
    # excludes the SE value; what is asserted is that not every instance gets there
    assert perfect < len(SEEDS) and hard.band.mean < 1, (perfect, hard.band)
    # N-scaling (observed, not explained): the mean deficit 1 - q shrinks with N, the
    # fraction of perfect instances does not (measured 0.0072 / 0.0039 / 0.0025 and
    # 14 / 13 / 17 of 24 at N = 800 / 1600 / 3200)
    deficits = [1 - A.gamp_ensemble(n, 2, "ising", seeds=SEEDS, iters=T).band.mean
                for n in (800, 1600, 3200)]
    print(f"ising alpha=2 deficit 1 - q at N = 800/1600/3200: {deficits}")
    assert deficits[0] > deficits[1] > deficits[2] > 0, deficits
    orig = G._onsager_scale
    try:
        for scale in (0.8, 1.2, 0.9, 1.1):
            G._onsager_scale = scale
            for key, b in bands.items():
                pb = _ensemble(*key, T).band
                q = pb.mean
                out = abs(q - b.mean) > b.half_width
                se_t = A.perceptron_state_evolution(key[0], key[1], iters=T).trajectory[-1]
                assert abs(q - se_t) > pb.half_width, (scale, key, pb, se_t)
                print(f"Onsager x{scale} {key}: q = {q:.4f} vs band {b.mean:.4f} +- "
                      f"{b.half_width:.4f}: {'outside' if out else 'inside'}")
                assert out, (scale, key, q, b)
    finally:
        G._onsager_scale = orig
    assert G._onsager_scale is orig
