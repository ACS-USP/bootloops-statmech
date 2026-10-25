"""E5b: random K-SAT (csp/ksat.py: instances, exhaustive counts, BP; csp/sp.py: survey
propagation on instances and the SP population dynamics).

Conventions: N boolean variables, M = round(alpha N) clauses, each the OR of K literals on
K distinct uniform variables with independent uniform signs.  Annealed (first-moment) bound
(1/N) ln E Z = ln 2 + (M/N) ln(1 - 2^-K) (Ding-Sly-Sun, Eq. (2.3)).  SP (Braunstein-Mezard-
Zecchina 2005, Eqs. (26)-(27); Mertens-Mezard-Zecchina 2006, Eqs. (1)-(2)) and its complexity
(BMZ Eqs. (39)-(41); MMZ Eqs. (3)-(5), (18)).

Bank: reference/ksat.json.  The banked thresholds are targets: the SP-population crossing
is located by regression on the measured Sigma(alpha) and compared in a Student-t band.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.cavity._stats import t_quantile
from replicant.compose import compose
from replicant.core import Refusal
from replicant.csp import ksat as S
from replicant.csp import sp as SP
from replicant.refdata import load
from replicant.testing import slow

BANK = load("ksat")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _tree_formula(K, M, seed):
    """A K-SAT formula whose factor graph is a tree: every new clause shares exactly one
    variable with the clauses before it (the rest are new)."""
    rng = np.random.default_rng(seed)
    var, n = [list(range(K))], K
    for _ in range(M - 1):
        old = int(rng.integers(n))
        var.append([old] + list(range(n, n + K - 1)))
        n += K - 1
    var = np.array(var, dtype=np.int64)
    for row in var:
        rng.shuffle(row)
    sign = np.where(rng.random(var.shape) < 0.5, -1, 1).astype(np.int8)
    return S.KSatFormula(K, n, M / n, seed, var, sign)


# ---- ensemble, exhaustive counts, annealed bound ------------------------------------------
def test_random_ksat_ensemble():
    f = S.random_ksat(3, 50, "4.2", seed=7)
    assert f.var.shape == (210, 3) and f.sign.shape == (210, 3)
    assert all(len(set(r)) == 3 for r in f.var.tolist())
    assert set(np.unique(f.sign)) == {-1, 1}
    g = S.random_ksat(3, 50, "4.2", seed=7)
    assert np.array_equal(f.var, g.var) and np.array_equal(f.sign, g.sign)
    assert not np.array_equal(f.var, S.random_ksat(3, 50, "4.2", seed=8).var)
    big = S.random_ksat(3, 20000, "4.0", seed=0)
    assert abs(np.mean(big.sign == 1) - 0.5) < 0.01


def test_exhaustive_count_matches_backtracking():
    """Two counters that share nothing: the vectorised enumeration of all 2^N assignments and
    a recursive backtracking count (pure Python); both outcomes SAT/UNSAT exercised."""
    seen = set()
    for seed in range(12):
        f = S.random_ksat(3, 10, "4.5", seed)
        z, ones = S.ksat_exact_marginals(f)
        assert z == S.ksat_count_reference(f), seed
        assert all(0 <= o <= z for o in ones)
        seen.add(z > 0)
    assert seen == {True, False}
    f = S.KSatFormula(2, 3, 1.0, 0, np.array([[0, 1], [1, 2], [0, 2]]), np.ones((3, 2), np.int8))
    assert S.ksat_exact_count(f) == 4      # x0|x1, x1|x2, x0|x2: at least two of three true


def test_annealed_bound_closed_form():
    with mp.workdps(30):
        b = S.ksat_annealed_bound(3, "4.2")
        assert abs(b - (mp.log(2) + mp.mpf("4.2") * mp.log(mp.mpf(7) / 8))) < mp.mpf("1e-28")
        a1 = S.ksat_first_moment_threshold(3)
        assert abs(S.ksat_annealed_bound(3, a1)) < mp.mpf("1e-25")
        assert abs(a1 - mp.log(2) / mp.log(mp.mpf(8) / 7)) < mp.mpf("1e-25")
    assert 5.19 < float(a1) < 5.20


def _counts(K, N, alpha, seeds):
    return np.array([S.ksat_exact_count(S.random_ksat(K, N, alpha, s)) for s in seeds], float)


def _jensen_and_first_moment(K, N, alpha, seeds):
    """(i) mean ln Z <= ln mean Z on the sample: this is AM-GM, true for ANY positive data (a
    tautology, kept because the brief asks for it; every sampled instance satisfiable, so ln Z
    is finite); the real content is (ii): E Z = 2^N (1 - 2^-K)^M exactly for this ensemble:
    the sample mean of Z / E Z contains 1 in its 99% Student-t band, and the positive control
    (1 - 2^-(K-1))^M (a wrong clause-violation probability) does not; (iii) mean ln Z / N is
    not above the annealed bound beyond its band."""
    z = _counts(K, N, alpha, seeds)
    M = round(float(alpha) * N)
    assert z.min() > 0, alpha
    lnz = np.log(z)
    assert lnz.mean() <= math.log(z.mean())
    t = t_quantile(0.995, len(z) - 1)
    for law, should_hold in (((1 - 2.0**-K)**M, True), ((1 - 2.0**-(K - 1))**M, False)):
        r = z / (2.0**N * law)
        band = t * r.std(ddof=1) / math.sqrt(len(r))
        assert (abs(r.mean() - 1) < band) == should_hold, (alpha, r.mean(), band, should_hold)
    se = lnz.std(ddof=1) / math.sqrt(len(lnz)) / N
    assert lnz.mean() / N - t * se < float(S.ksat_annealed_bound(K, M / N)), alpha


def test_small_N_exact_counts_below_annealed_bound():
    """Exhaustive counts on named seeds (independent of BP/SP), N = 12, seeds 0-39, alpha = 2
    and 2.5 (every sampled instance satisfiable: measured).  At this N the quenched-annealed
    gap is not resolved (alpha = 2: mean ln Z / N = 0.4177 +- 0.0042 against 0.4261): only
    the AM-GM inequality on the sample (a tautology) and the exact E Z with its control are
    asserted."""
    for alpha in ("2.0", "2.5"):
        _jensen_and_first_moment(3, 12, alpha, range(40))


@slow
def test_small_N_exact_counts_below_annealed_bound_N18():
    """N = 18 (2^18 assignments), 30 named seeds at alpha = 3 (all satisfiable: measured)."""
    _jensen_and_first_moment(3, 18, "3.0", range(30))


# ---- support of the surveys (MMZ Eq. (39)) -----------------------------------------------
def test_sp_support_threshold_matches_bank_and_xorsat():
    """alpha_t(3) against the bank (MZ 2002: 1.64; MMZ: 1.6; a target).  Cross-check of the
    numerics through separate code: with q = u^(1/(K-1)) the zero-survey equation IS the
    K-XORSAT leaf-removal recursion at density alpha / 2 (the same recursion, not an
    independent derivation), so alpha_t(K) = 2 alpha_d^XOR(K) as computed by csp/xorsat.py.
    The independent evidence is the population's measured zero-survey fraction (next test).
    Positive control: the SP onset (3.927) is not within tolerance."""
    from replicant.csp import xorsat as X
    from replicant.refdata import check
    a3 = S.ksat_sp_support_threshold(3)
    assert check(BANK["alpha_t_k3"], a3, label="alpha_t_k3").ok
    assert not check(BANK["alpha_t_k3"], BANK["alpha_sp_onset_k3"].value, label="ctl").ok
    with mp.workdps(30):
        for K in (3, 4, 5):
            assert abs(S.ksat_sp_support_threshold(K) - 2 * X.xorsat_bp_threshold(K)) < mp.mpf("1e-20")
    t, tau = S.ksat_sp_support(3, "1.6")
    assert t == 1 and tau == 1                                    # below alpha_t
    t, _ = S.ksat_sp_support(3, "1.7")
    assert t < 1


def test_population_zero_fraction_matches_support_equation():
    """The SP population's measured fraction of exactly-zero surveys at alpha = 4.2 against
    t(alpha) of Eq. (39) (0.0037570), in a 99% Student-t band over runs; positive control:
    t at twice the clause density (a factor-2 error in the Poisson mean) is excluded."""
    r = SP.ksat_population_dynamics_sp(3, "4.2", pop=5000, sweeps=6, seed=3, burn_in=6, runs=4)
    t, _ = S.ksat_sp_support(3, "4.2")
    band = t_quantile(0.995, 3) * r.stderr["zero_survey_fraction"]
    assert abs(r.value["zero_survey_fraction"] - float(t)) < band, (r.value, r.stderr, t)
    wrong, _ = S.ksat_sp_support(3, "8.4")
    assert abs(r.value["zero_survey_fraction"] - float(wrong)) > band


# ---- belief propagation ------------------------------------------------------------------
def test_bp_is_exact_on_tree_formulas():
    """BMZ Theorem 2: on a tree factor graph BP converges and its marginals (Eq. (13)) and
    entropy (Eq. (14)) are exact; checked against exhaustive enumeration (independent)."""
    for K, M, seed in ((3, 6, 0), (3, 7, 1), (4, 4, 2), (2, 9, 3)):
        f = _tree_formula(K, M, seed)
        z, ones = S.ksat_exact_marginals(f)
        bp = S.ksat_bp(f, iters=200, damping=0.0, seed=seed)
        assert bp.converged, (K, M, seed, bp.residual)
        assert abs(bp.entropy * f.N - math.log(z)) < 1e-9, (bp.entropy * f.N, math.log(z))
        assert np.max(np.abs(bp.marginals - np.asarray(ones) / z)) < 1e-9


def test_bp_on_loopy_formula_is_not_exact_and_reports_iterations():
    f = S.random_ksat(3, 14, "2.0", seed=3)
    z = S.ksat_exact_count(f)
    bp = S.ksat_bp(f, iters=500, damping=0.0, seed=0)
    assert bp.converged and 0 < bp.iterations < 500
    assert abs(bp.entropy * f.N - math.log(z)) < 2.0       # close, but not exact on loops
    nc = S.ksat_bp(f, iters=2, damping=0.0, seed=0)        # honest flag: 2 iterations are not enough
    assert not nc.converged and nc.iterations == 2 and nc.residual > 1e-8


@slow
def test_bp_converges_below_alpha_d_and_not_above():
    """Finite N, named seeds (see BP_WINDOW for the measured window).  The test BRACKETS
    alpha_d (KMRSZ 2007: 3.86 for K = 3, banked as a target only), it does not measure it: at
    3.5 every seed of 0-19 converges, at 4.0 none does (Clopper-Pearson upper edge < 0.5); at
    3.8 / 3.9 8 / 2 of 20 converge at N = 2000 (and 2/20 near 3.86, reviewer's measurement)."""
    w = S.BP_WINDOW
    lo = S.ksat_bp_converged_fraction(3, w["N"], w["below"], w["seeds"], iters=w["iters"],
                                      damping=w["damping"])
    hi = S.ksat_bp_converged_fraction(3, w["N"], w["above"], w["seeds"], iters=w["iters"],
                                      damping=w["damping"])
    assert float(w["below"]) < float(BANK["alpha_d_k3"].value) < float(w["above"])
    assert lo.n_conv == lo.n, lo
    assert hi.hi < 0.5, hi


# ---- survey propagation: the update rule and its hook --------------------------------------
def test_sp_rule_matches_warning_enumeration():
    """Independent route for the SP update: enumerate every configuration of incoming
    warnings (each clause b warns with probability eta_b, independently), discard the
    contradictory ones, and take P(at least one impeding, no supporting).  This is the
    definition of MMZ Sec. 2.1; the hook implements the closed products of Eq. (2)."""
    assert SP.audit_sp_rule() < 1e-12


def test_planted_error_in_sp_update_is_caught():
    """Planted errors in the SP_UPDATES hook (a 1.003 power on the factor, a 1.1 weight on
    Pi^0) are caught by the warning-enumeration audit (deviations 1.1e-3 and 0.024), and the
    1.1 weight changes the converged SP fixed point of an instance at alpha = 4.2 (instance SP,
    N = 1000: trivial, Sigma = 0, against 0.0113); the registry is restored with an identity
    assert."""
    orig = SP.SP_UPDATES["bmz"]
    try:
        SP.SP_UPDATES["bmz"] = SP.planted_power_rule(1.003)
        assert SP.audit_sp_rule() > 1e-4
        SP.SP_UPDATES["bmz"] = lambda ps, pu: ps * (1 - pu) / (ps + pu - 0.9 * ps * pu)
        assert SP.audit_sp_rule() > 1e-3
        bad = SP.ksat_sp(S.random_ksat(3, 1000, "4.2", seed=0), iters=300, seed=0)
    finally:
        SP.SP_UPDATES["bmz"] = orig
    assert SP.SP_UPDATES["bmz"] is orig
    assert SP.audit_sp_rule() < 1e-12
    good = SP.ksat_sp(S.random_ksat(3, 1000, "4.2", seed=0), iters=300, seed=0)
    assert good.converged and not good.trivial
    assert bad.converged     # same instance and seed: a converged, different fixed point
    assert abs(bad.complexity - good.complexity) > 1e-3, (bad.complexity, good.complexity)


def test_sp_on_instance_trivial_below_onset():
    """BMZ Sec. IV: for alpha below the SP onset (~3.9) SP converges to eta = 0 (complexity
    exactly 0); at alpha = 4.2 to a non-trivial fixed point with positive complexity.  N = 1000
    (measured, seeds 0-3, 300 synchronous sweeps): at 3.5 all four trivial; at 4.2 seeds 0, 2, 3
    converge (Sigma 0.0113, 0.0077, 0.0045) and seed 1 does not (BMZ: at N ~ 1000 "one often
    finds some instances in which SP does not converge"); the flag is checked on seed 1."""
    lo = SP.ksat_sp(S.random_ksat(3, 1000, "3.5", seed=1), iters=300, seed=1)
    assert lo.converged and lo.trivial and abs(lo.complexity) < 1e-12
    nc = SP.ksat_sp(S.random_ksat(3, 1000, "4.2", seed=1), iters=300, seed=1)
    assert not nc.converged and nc.iterations == 300 and "not converged" in nc.note
    hi = SP.ksat_sp(S.random_ksat(3, 1000, "4.2", seed=0), iters=300, seed=0)
    assert hi.converged and not hi.trivial and hi.complexity > 0


def test_population_dynamics_trivial_below_onset_nontrivial_above():
    lo = SP.ksat_population_dynamics_sp(3, "3.6", pop=2000, sweeps=10, seed=0, burn_in=40, runs=2)
    assert lo.phase == "trivial" and abs(lo.value["complexity"]) < 1e-12
    hi = SP.ksat_population_dynamics_sp(3, "4.2", pop=2000, sweeps=10, seed=0, burn_in=40, runs=2)
    assert hi.phase in ("nontrivial", "undetermined") and hi.value["mean_survey"] > 0.1
    assert hi.label == "statistical" and hi.stderr["complexity"] > 0


@slow
def test_complexity_estimators_agree():
    """MMZ Eq. (18) (Sigma_0 + alpha [K I_(K-1) - (K-1) I_K]) and its rewriting through the
    fixed-point identity (17) (E ln(pi + pi' - pi pi') - alpha (K-1) E ln(1 - prod rho)) agree
    on an equilibrated population (they differ only by that identity)."""
    r = SP.ksat_population_dynamics_sp(3, "4.2", pop=20000, sweeps=40, seed=5, burn_in=100, runs=4)
    d = r.value["complexity"] - r.value["complexity_mmz18"]
    se = math.hypot(r.stderr["complexity"], r.stderr["complexity_mmz18"])
    assert abs(d) < t_quantile(0.995, 3) * se, (d, se)


@slow
def test_instance_complexity_matches_population():
    """BMZ Eqs. (39)-(41) on sampled instances (no fixed-point identity, no sampling of the
    degree law) against the population estimate at alpha = 4.2, in a 99% Student-t band with no
    extra allowance.  Measured before the asserts (fix round 1): instances N = 20000, seeds 0-7,
    all converged, 0.005996 +- 0.000232; population pop 50000, 60 sweeps, 8 runs, seed 1,
    0.005983 +- 0.000460; band 0.0018, i.e. 0.30 Sigma (it tells Sigma from 0 and excludes
    |Delta Sigma| >~ 0.002).  Positive control: a planted 1.005 power on the SP factor in the
    population only (hook-level, restored) gives 0.0139 +- 0.0004 (4 runs), outside its band.
    About 60 s (measured 58 s)."""
    runs = [SP.ksat_sp(S.random_ksat(3, 20000, "4.2", seed=s), iters=500, seed=s)
            for s in range(8)]
    assert all(r.converged and not r.trivial for r in runs), [r.note for r in runs]
    vals = [r.complexity for r in runs]
    m, se = np.mean(vals), np.std(vals, ddof=1) / math.sqrt(len(vals))
    pd = SP.ksat_population_dynamics_sp(3, "4.2", pop=50000, sweeps=60, seed=1, burn_in=100,
                                        runs=8)
    assert pd.phase == "nontrivial", pd.note
    band = t_quantile(0.995, 7) * math.hypot(se, pd.stderr["complexity"])
    assert abs(m - pd.value["complexity"]) < band, (m, se, pd.value, pd.stderr, band)
    assert band < 0.5 * m, (band, m)                  # the band resolves Sigma from 0
    orig = SP.SP_UPDATES["bmz"]
    try:
        SP.SP_UPDATES["bmz"] = SP.planted_power_rule(1.005)
        bad = SP.ksat_population_dynamics_sp(3, "4.2", pop=50000, sweeps=60, seed=2,
                                             burn_in=100, runs=4)
    finally:
        SP.SP_UPDATES["bmz"] = orig
    assert SP.SP_UPDATES["bmz"] is orig
    ctl = t_quantile(0.995, 3) * math.hypot(se, bad.stderr["complexity"])
    assert abs(m - bad.value["complexity"]) > ctl, (m, bad.value, bad.stderr, ctl)


@slow
def test_sp_complexity_vanishes_at_banked_threshold():
    """3-SAT: the root of the SP-population complexity Sigma(alpha), located by linear
    regression per run over the CROSSING grid, contains the banked alpha_s (MMZ 2006, Eq. (43):
    4.26675, a TARGET) in the 99% Student-t band over runs.  Positive control: a planted
    1.005 power on the SP factor (hook-level) moves the crossing out of both bands."""
    c = SP.CROSSING
    r = SP.ksat_sp_crossing(3, c["alphas"], pop=c["pop"], sweeps=c["sweeps"], seed=c["seed"],
                            burn_in=c["burn_in"], runs=c["runs"])
    want = float(BANK["alpha_s_k3"].value)
    assert abs(r.alpha_s - want) < r.halfwidth99, r
    assert r.halfwidth99 < c["max_halfwidth"], r
    orig = SP.SP_UPDATES["bmz"]
    try:
        SP.SP_UPDATES["bmz"] = SP.planted_power_rule(1.005)
        bad = SP.ksat_sp_crossing(3, c["alphas"], pop=c["pop"], sweeps=c["sweeps"],
                                  seed=c["seed"], burn_in=c["burn_in"], runs=c["runs"])
    finally:
        SP.SP_UPDATES["bmz"] = orig
    assert SP.SP_UPDATES["bmz"] is orig
    assert abs(bad.alpha_s - want) > r.halfwidth99 + bad.halfwidth99, (bad, r)


# ---- refusals ----------------------------------------------------------------------------
def test_refusals():
    for K in (1, 2.5, True, "3"):
        _refuses(lambda: S.random_ksat(K, 20, "2.0", 0), "precondition:integer-k-at-least-2")
    for a in (0, "-1", "abc"):
        _refuses(lambda: S.random_ksat(3, 20, a, 0), "precondition:positive-alpha")
        _refuses(lambda: S.ksat_annealed_bound(3, a), "precondition:positive-alpha")
    _refuses(lambda: S.ksat_annealed_bound(1, "2"), "precondition:integer-k-at-least-2")
    _refuses(lambda: S.random_ksat(3, 2, "2.0", 0), "precondition:enough-variables")
    for bad in (-1, "x", 1.5, True):
        _refuses(lambda: S.random_ksat(3, 20, "2.0", bad), "precondition:named-seeds")
    _refuses(lambda: S.ksat_bp_converged_fraction(3, 50, "2.0", "x"), "precondition:named-seeds")
    _refuses(lambda: S.ksat_bp_converged_fraction(3, 50, "2.0", []), "precondition:named-seeds")
    _refuses(lambda: S.ksat_exact_count(S.random_ksat(3, 30, "2.0", 0)),
             "precondition:exhaustive-size")
    f = S.random_ksat(3, 30, "2.0", 0)
    _refuses(lambda: S.ksat_bp(f, iters=10, damping=1.0, seed=0), "precondition:damping")
    _refuses(lambda: S.ksat_bp(f, iters=0, damping=0.0, seed=0), "precondition:iterations")
    _refuses(lambda: SP.ksat_sp(f, iters=10, seed=0, rule="nope"), "unsupported:sp-rule")
    _refuses(lambda: SP.ksat_population_dynamics_sp(1, "4.2", pop=1000, sweeps=10, seed=0),
             "precondition:integer-k-at-least-2")
    _refuses(lambda: SP.ksat_population_dynamics_sp(3, 0, pop=1000, sweeps=10, seed=0),
             "precondition:positive-alpha")
    _refuses(lambda: SP.ksat_population_dynamics_sp(3, "4.2", pop=10, sweeps=10, seed=0),
             "precondition:population-size")
    _refuses(lambda: SP.ksat_population_dynamics_sp(3, "4.2", pop=1000, sweeps=10, seed=-3),
             "precondition:named-seeds")
    _refuses(lambda: SP.ksat_population_dynamics_sp(3, "4.2", pop=1000, sweeps=10, seed=0,
                                                    runs=1), "precondition:runs")
    _refuses(lambda: SP.ksat_sp_crossing(3, ("4.2", "4.3"), pop=1000, sweeps=10, seed=0),
             "precondition:crossing-grid")


# ---- catalog entry -----------------------------------------------------------------------
def test_catalog_entry_ksat():
    e = catalog()["ksat"]
    assert e.routes == {"cavity"} and e.default_route == "cavity"
    res = compose(e.spec, "cavity").run()
    assert res.finality["alpha_1"].state == "exact-by-literature"
    assert res.finality["alpha_t"].state == "not-computed"
    assert all(c.ok for c in res.reference_checks())
    assert "Ding" in res.finality["alpha_1"].cite
    pt = compose(e.spec, "cavity").run(alpha="3.0", N=12, seeds=(0, 1, 2, 3))
    v = pt.values
    assert v["sat_fraction"] == 1 and v["ln_mean_z"] >= v["entropy_exact_mean"]   # AM-GM
    assert pt.finality["entropy_exact_mean"].state == "not-computed"
    assert "entropy_exact_mean" in pt.statistical and not pt.final
    r = _refuses(lambda: compose(e.spec, "cavity").run(alfa="3"), "precondition:known-parameter")
    assert "alpha" in r.detail
    _refuses(lambda: compose(e.spec, "cavity").run(N=12), "precondition:state-point")
    _refuses(lambda: compose(e.spec, "cavity").run(alpha="3", seeds=(0,)),
             "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "replica"), "missing-step:route:replica")
    _refuses(lambda: compose(e.spec, "cavity").run(alpha="3.0", N=12, seeds="x"),
             "precondition:named-seeds")
    assert e.derivation("cavity")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "GUIDE.md")) as f:
        guide = f.read()
    assert "| `ksat`" in guide and "precondition:integer-k-at-least-2" in guide


def test_crossing_flag_outside_true_false_is_refused():
    """Fix round 1 (Important 1): a `crossing` value other than true/false is refused by name,
    never reported as "crossing not requested"."""
    import io
    from replicant.cli import main
    e = catalog()["ksat"]
    for bad in ("yes", "on", 2, "TRUE", 1):
        r = _refuses(lambda: compose(e.spec, "cavity").run(crossing=bad),
                     "precondition:known-parameter")
        assert "true" in r.detail and "false" in r.detail, r.detail
    for val in ("yes", "on", "2"):
        err = io.StringIO()
        code = main(["run", "ksat", "--route", "cavity", "--param", f"crossing={val}"],
                    io.StringIO(), err)
        assert code == 2 and "REFUSED: precondition:known-parameter" in err.getvalue(), err.getvalue()
    out = compose(e.spec, "cavity").run(crossing=False)      # false: the step is skipped
    assert "alpha_s_sp" not in out.values


def test_alpha_s_bank_tolerance_is_from_agreeing_estimates():
    """Fix round 1 (Important 2): the alpha_s_k3 tolerance comes from the estimates that agree
    (MMZ 4.26675 +- 0.00015, MZ / BMZ 4.267); Lundow-Markstrom's conditional upper bound is in
    the cite, not in sources_disagree, so the catalog band t hypot(stderr, tol) is the one the
    GUIDE states."""
    ref = BANK["alpha_s_k3"]
    assert float(ref.tol) <= 5e-4
    assert all(abs(float(d["value"]) - float(ref.value)) <= float(ref.tol)
               for d in ref.sources_disagree)
    assert not any("Lundow" in d["cite"] for d in ref.sources_disagree)
    assert "conditional upper bound" in ref.cite
    band = t_quantile(0.995, 5) * math.hypot(0.00267, float(ref.tol))   # measured stderr
    assert band < 0.012, band


def test_undetermined_population_finality_says_so():
    """Fix round 1 (Minor 9): an `undetermined` SP population carries that in its finality
    note; the population seed and runs are stated."""
    e = catalog()["ksat"]
    for key in ("complexity", "complexity_mmz18", "mean_survey", "zero_survey_fraction"):
        und = e.finality[key]({"sp_phase": "undetermined"})
        assert und.state == "not-computed"
        assert "undetermined" in und.note and "not equilibrated" in und.note, und.note
        assert "undetermined" not in e.finality[key]({"sp_phase": "nontrivial"}).note
    pt = compose(e.spec, "cavity").run(alpha="4.2", pop=2000, sweeps=10)
    assert pt.values["sp_phase"] == "nontrivial", pt.values
    assert any("seed 0" in n and "4 runs" in n for n in pt.notes), pt.notes


@slow
def test_catalog_ksat_sp_point_and_crossing_finality():
    e = catalog()["ksat"]
    pt = compose(e.spec, "cavity").run(alpha="4.2", pop=5000, sweeps=10)
    assert pt.finality["complexity"].state == "not-computed"
    assert "complexity" in pt.statistical and pt.errors["complexity"] > 0
