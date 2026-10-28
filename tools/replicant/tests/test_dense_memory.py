"""Dense associative memories with P = e^{alpha N} patterns: the entries `dense_memory` (Lucibello & Mezard,
Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4) and `dense_memory_ising` (Demircigil et al., J. Stat. Phys.
168, 288 (2017), arXiv:1702.01929v2; the binary REM noise is derived here).

Default tier: the four routes to phi agree, the printed numbers are reproduced, the limits, the reduction to the REM
entry (and its exact finite-N quadrature), the independent finite-N checks (exact-law sampler, the paper's gradient
descent, the single-flip dynamics, Gray-code enumeration), planted controls that must fail, named refusals, the
composer, finality labels and the InfoNCE dictionary.  Slow tier: larger N.
"""
import contextlib
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive_dense_memory as DD
from replicant.catalog import catalog, compare
from replicant.compose import compose
from replicant.core import Refusal, State, run_pipeline
from replicant.dense_memory import ensembles as EN
from replicant.dense_memory import infonce as IN
from replicant.dense_memory import ising as IS
from replicant.dense_memory import rem_noise as RN
from replicant.dense_memory import sim as SIM
from replicant.dense_memory import thresholds as TH
from replicant.families.rem import rem_free_energy, rem_quenched_finite_n
from replicant.refdata import load
from replicant.spec import ModelSpec
from replicant.testing import slow

mp.mp.dps = 30
ENS = ("gaussian", "spherical", "binary")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


@contextlib.contextmanager
def planted(module, **kw):
    saved = dict(module._PLANT)
    module._PLANT.update(kw)
    try:
        yield
    finally:
        module._PLANT.clear()
        module._PLANT.update(saved)


def _fresh_check(entry, key):
    """True when the entry's banked key is NOT reproduced by a fresh (uncached) run: a mismatch or a refusal."""
    e = catalog()[entry]
    ref = next(r for r in e.refs if r.key == key)
    try:
        state, _ = e.execute("replica", **ref.params)
    except Refusal:
        return True
    return not compare(e, ref, state["values"], state["errors"]).ok


# ---- the building blocks ---------------------------------------------------------------------------------
def test_zeta_is_the_legendre_transform_of_the_rate_function_and_its_derivative():
    for ens in ENS:
        for lam in ("0.3", "1", "2.5"):
            eps = EN.dzeta(ens, lam)                                  # the dominating level
            assert abs(EN.zeta(ens, lam) - (mp.mpf(lam) * eps - EN.rate(ens, eps))) < mp.mpf("1e-25"), (ens, lam)
            assert abs(EN.drate(ens, eps) - mp.mpf(lam)) < mp.mpf("1e-22"), (ens, lam)
            h = mp.mpf("1e-10")
            fd = (EN.zeta(ens, mp.mpf(lam) + h) - EN.zeta(ens, mp.mpf(lam) - h)) / (2 * h)
            assert abs(fd - eps) < mp.mpf("1e-18"), (ens, lam)


def test_rotation_invariant_normalisation_a0_b1():
    """(B6)-(B7): zeta(lam) = lam^2/2 + O(lam^3) when E|xi|^2 = N and |x|^2 = N: a = 0, b = 1, s(eps) ~ eps^2/2."""
    for ens in ENS:
        l = mp.mpf("1e-6")
        assert abs(EN.zeta(ens, l) / l**2 - mp.mpf(1) / 2) < mp.mpf("1e-5"), ens
        e = mp.mpf("1e-4")
        assert abs(EN.rate(ens, e) / (e**2 / 2) - 1) < mp.mpf("1e-6"), ens


def test_four_routes_to_phi_agree_in_both_phases():
    worst = 0
    for ens in ENS:
        for a in ("0.05", "0.3", "0.6"):
            for l in ("0.2", "1", "2.5", "9"):
                vals = [RN.phi(ens, a, l, method=m) for m in ("closed", "legendre", "1rsb", "laplace")]
                worst = max(worst, max(vals) - min(vals))
    assert worst < mp.mpf("1e-25"), worst


def test_phases_parisi_parameter_and_continuity_at_the_freezing_line():
    for ens in ("gaussian", "spherical", "binary"):
        a = mp.mpf("0.3")
        ls = RN.condensation_lambda(ens, a)
        assert abs(ls - RN.condensation_lambda(ens, a, method="legendre")) < mp.mpf("1e-22")
        lo, hi = RN.noise(ens, a, ls * mp.mpf("0.999")), RN.noise(ens, a, ls * mp.mpf("1.001"))
        assert lo.phase == "uncondensed" and lo.m == 1 and hi.phase == "condensed" and 0 < hi.m < 1
        assert abs(hi.m - ls / (ls * mp.mpf("1.001"))) < mp.mpf("1e-20")           # (A26)-(A27)
        assert abs(lo.phi - hi.phi) < mp.mpf("2e-3")                                # continuous
        assert RN.noise(ens, a, ls).phase == "condensed" and abs(RN.noise(ens, a, ls).m - 1) < mp.mpf("1e-20")
        assert hi.phi == RN.eps_star(ens, a)
        # phi increases with alpha and decreases with lam
        assert RN.phi(ens, "0.2", "1") < RN.phi(ens, "0.4", "1")
        assert RN.phi(ens, "0.3", "0.5") > RN.phi(ens, "0.3", "2") >= RN.phi(ens, "0.3", "20")


def test_alpha1_by_root_and_by_legendre_duality_agree_and_lambda1_inverts_it():
    for ens in ENS:
        for l in ("0.4", "1", "3", "12"):
            a, b = TH.alpha_1(ens, l), TH.alpha_1_closed(ens, l)
            assert abs(a - b) < mp.mpf("1e-24"), (ens, l)
    for ens in ("gaussian", "spherical"):
        for a in ("0.05", "0.2", "0.45"):
            assert abs(TH.alpha_1(ens, TH.lambda_1(ens, a)) - mp.mpf(a)) < mp.mpf("1e-22"), (ens, a)


# ---- the paper's printed numbers ---------------------------------------------------------------------------
def test_printed_numbers_of_the_gaussian_phase_diagram():
    assert abs(TH.alpha_1("gaussian", 3) - mp.mpf(1) / 2) < mp.mpf("1e-25")                    # Sec. V, (D7)
    assert abs(TH.alpha_1("gaussian", "0.5") - mp.mpf(3) / 8) < mp.mpf("1e-25")                # (D7) lam(1 - lam/2)
    assert abs(TH.alpha_c_lb_gaussian(2) - mp.log(2) / 4) < mp.mpf("1e-22")                    # App. D 2
    assert abs(TH.lambda_saturation() - mp.mpf("0.70091")) < mp.mpf("1e-5")                    # printed to 5 decimals
    # alpha_c^lb = log(2)/4 exactly from the saturation point on, strictly below before it
    ls = TH.lambda_saturation()
    assert abs(TH.alpha_c_lb_gaussian(ls * mp.mpf("1.001")) - mp.log(2) / 4) < mp.mpf("1e-20")
    assert TH.alpha_c_lb_gaussian(ls * mp.mpf("0.99")) < mp.log(2) / 4 - mp.mpf("1e-5")
    # alpha_c^lb ~ lam at small lam ("it matches alpha_1"): the ratio is 1 - 2 sqrt(lam) + ..., tending to 1 slowly
    r = [TH.alpha_c_lb_gaussian(l) / mp.mpf(l) for l in ("1e-4", "1e-6", "1e-8")]
    assert r[0] < r[1] < r[2] < 1 and abs(r[2] - 1) < mp.mpf("3e-4")
    assert all(abs(x - (1 - 2 * mp.sqrt(mp.mpf(l)))) < 5 * mp.mpf(l) for x, l in zip(r, ("1e-4", "1e-6", "1e-8")))
    assert TH.alpha_c_lb_gaussian("0.5") < TH.alpha_1("gaussian", "0.5")                      # a gap at any lam


def test_gaussian_lower_bound_by_direct_minimisation_over_the_pattern_norm():
    """alpha_c^lb from a brute-force grid minimisation of A(alpha, lam), not through the stationary point r^2 = 1/2."""
    rr = np.linspace(0.02, 6, 600001)
    It = (rr**2 - 1) / 2 - np.log(rr)
    for lam in (0.3, 0.9, 2.0):
        def A(a):
            r0 = math.sqrt(a / (lam * (1 - lam / 2))) if lam < 1 else math.sqrt(2 * a)
            first = It[rr <= r0].min() if r0 < 1 else 0.0
            second = (It + rr**2 / 2 - a)[rr >= r0].min()
            return min(first, second)
        lo, hi = 1e-6, 0.49
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if A(mid) - mid > 0 else (lo, mid)
        assert abs((lo + hi) / 2 - float(TH.alpha_c_lb_gaussian(lam))) < 2e-6, lam


def test_spherical_capacity_is_alpha1_and_grows_like_half_log_lambda():
    for l in ("1", "5", "50"):
        v, kind = TH.capacity("spherical", l)
        assert kind == "equals-alpha_1" and v == TH.alpha_1("spherical", l)
        a = TH.alpha_1("spherical", l)
        assert mp.mpf(l) < RN.condensation_lambda("spherical", a)       # the boundary is in the uncondensed phase
    # alpha_1(lam) = lam - zeta(lam) = (1/2) log lam + 1/2 + o(1): the paper's "~ 0.5 log lambda" is the leading term
    for l in ("1e6", "1e10"):
        assert abs(TH.alpha_1_closed("spherical", l) - mp.log(mp.mpf(l)) / 2 - mp.mpf(1) / 2) < mp.mpf("1e-5")
    # Gaussian saturates, spherical does not, binary tends to log 2 (the hard-max limit)
    assert TH.alpha_1("gaussian", 100) == mp.mpf(1) / 2
    assert abs(TH.alpha_1("binary", 30) - mp.log(2)) < mp.mpf("1e-12")


def test_ramsauer_bound_asymptotics_and_gap():
    for l in ("1e-6", "1e-5", "1e-4"):
        assert abs(TH.ramsauer_lower_bound(l) / mp.mpf(l) * 10 - 1) < mp.mpf(l) * 4        # ~ lam/10
    assert TH.ramsauer_lower_bound("1000") < TH.alpha_1("spherical", "1000") - 1            # "a large gap"
    r1, r2 = TH.ramsauer_lower_bound("1e6"), TH.ramsauer_lower_bound("1e12")
    assert 0.5 < (r2 - r1) / (mp.log(mp.mpf("1e12")) / 4 - mp.log(mp.mpf("1e6")) / 4) < 1.5  # (1/4) log lam growth


def test_basins_nearest_pattern_few_patterns_and_scaled_regime():
    for ens in ("gaussian", "spherical"):
        a = mp.mpf("0.1")
        assert abs(TH.cos_theta_c(ens, a, 50) - TH.cos_theta_nearest(ens, a)) < mp.mpf("1e-25")   # large lam: c_max
        # cos(theta_c) falls as lam grows (basin widens) until it freezes
        assert TH.cos_theta_c(ens, a, "0.2") > TH.cos_theta_c(ens, a, "0.3") > TH.cos_theta_c(ens, a, 50)
        # small alpha: eps*/sqrt(2 alpha) -> 1, so cos(theta_c) -> sqrt(2 log P/N)  (B8)
        x = mp.mpf("1e-8")
        assert abs(TH.cos_theta_nearest(ens, x) / mp.sqrt(2 * x) - 1) < mp.mpf("1e-7"), ens
    assert abs(TH.cos_theta_few_patterns(1000, 100) - mp.sqrt(2 * mp.log(1000) / 100)) < mp.mpf("1e-25")
    assert TH.scaled_threshold("0.7") == mp.mpf("0.7")
    _refuses(lambda: TH.theta_c(TH.cos_theta_c("gaussian", "0.6", 50)), "no-solution:no-basin")   # alpha > 1/2


def test_limits_hard_max_small_lambda_and_small_alpha():
    for ens in ENS:
        a = mp.mpf("0.2")
        # lam -> inf: the largest level wins, phi -> eps*(alpha)
        assert abs(RN.phi(ens, a, 1e6) - RN.eps_star(ens, a)) < mp.mpf("1e-12")
        # lam -> 0 at fixed alpha: the uncondensed (annealed) paramagnet, lam phi - alpha = zeta(lam) ~ lam^2/2
        l = mp.mpf("1e-4")
        assert RN.noise(ens, a, l).phase == "uncondensed"
        assert abs((l * RN.phi(ens, a, l) - a) / l**2 - mp.mpf(1) / 2) < mp.mpf("1e-3"), ens
    for ens in ("gaussian", "spherical"):                    # alpha -> 0: any finite lam is condensed, phi -> 0
        assert RN.noise(ens, "1e-6", 1).phase == "condensed" and RN.phi(ens, "1e-6", 1) < mp.mpf("2e-3")
    # no retrieval at any lam above the hard-max cap (Gaussian alpha > 1/2)
    _refuses(lambda: TH.lambda_1("gaussian", "0.55"), "no-solution:no-retrieval")


# ---- reduction to the REM entry ------------------------------------------------------------------------------
def test_gaussian_noise_at_alpha_log2_is_the_rem_free_energy_both_phases():
    for lv, s in (("0.9", "1"), ("1.3", "0.5"), ("4", "2")):
        rho = mp.sqrt(mp.mpf(s))
        assert abs(RN.phi("gaussian", mp.log(2), lv, rho) + rem_free_energy(mp.mpf(lv), N_scale=mp.mpf(s))) < mp.mpf("1e-25")


def test_sampler_reproduces_the_exact_finite_N_rem_quadrature_of_the_rem_family():
    """At alpha = log 2 and Gaussian levels the e^{alpha N} = 2^N overlaps ARE the REM: the sampler's
    (1/N) E log Z = lam E phi_hat must equal families/rem.py rem_quenched_finite_n (an exact quadrature, N = 30)."""
    N, s = 30, 1.0
    for lam, tol_sigma in ((0.8, 6), (2.5, 6)):          # uncondensed and frozen (lam_* = sqrt(2 log 2) = 1.18)
        v = np.array(SIM.rem_phi_mc("gaussian", math.log(2), lam, N, range(400), rho=math.sqrt(s)))
        exact = rem_quenched_finite_n(lam, N, s)
        got, se = lam * v.mean(), lam * v.std(ddof=1) / math.sqrt(len(v))
        assert abs(got - exact) < max(tol_sigma * se, 2e-4), (lam, got, exact, se)


# ---- independent finite-N checks -------------------------------------------------------------------------------
def test_exact_law_sampler_converges_to_the_noise_in_both_phases_and_all_ensembles():
    cases = (("gaussian", 0.3, 0.5), ("spherical", 0.3, 1.0), ("binary", 0.3, 0.7),            # uncondensed
             ("gaussian", 0.3, 3.0), ("spherical", 0.3, 8.0), ("binary", 0.3, 4.0))            # condensed
    for ens, a, l in cases:
        theory = float(RN.phi(ens, a, l))
        err = [abs(np.mean(SIM.rem_phi_mc(ens, a, l, N, range(8))) - theory) for N in (100, 400, 1600)]
        if RN.noise(ens, a, l).phase == "uncondensed":
            assert err[2] < 2e-3, (ens, err)
        else:                                                   # Derrida's log N / N correction: shrinks with N
            assert err[2] < err[1] < err[0] and err[2] < 0.012, (ens, err)


def test_the_papers_gradient_descent_retrieves_above_lambda1_and_collapses_below_it():
    """Eq. (2) run directly from x = xi^1 (N = 200, P = 22026): the crossover sits at lambda_1(alpha) (the paper's
    Figs. 6-7, whose finite-N transition is smooth for Gaussian patterns)."""
    for ens in ("spherical", "gaussian"):
        l1 = float(TH.lambda_1(ens, "0.05"))
        above = [SIM.gd_retrieval(ens, 0.05, 2 * l1, 200, s) for s in range(3)]
        below = [SIM.gd_retrieval(ens, 0.05, 0.5 * l1, 200, s) for s in range(3)]
        assert max(above) < 0.05 and min(below) > 0.7, (ens, above, below)


def test_binary_decision_sum_has_the_noise_of_2_lambda_not_lambda():
    """The single-flip decision of Demircigil et al. sums terms with fair random signs: its size is the REM noise
    at 2 lam.  The sampler converges to phi(alpha, 2 lam) and stays clear of phi(alpha, lam)."""
    for a, l in (("0.3", 0.5), ("0.5", 1.0)):
        got = float(np.median(SIM.sign_noise_mc(float(a), l, 1600, range(20))))
        two, one = float(RN.phi("binary", a, 2 * l)), float(RN.phi("binary", a, l))
        assert abs(got - two) < 0.006 and abs(got - one) > 0.025, (a, l, got, two, one)


def test_no_pair_within_hamming_distance_1_implies_all_patterns_are_fixed_points():
    """The mechanism of Demircigil et al. Remark 1.4.  One direction holds in every sampled instance: no close pair =>
    every pattern is a fixed point.  With a close pair the shared coordinate is an exact tie decided by the remaining
    noise, so the instance usually (not always) has an unstable pattern."""
    free, paired = [], []
    for N, P in ((16, 32), (16, 64), (16, 128)):
        for seed in range(25):
            t = SIM.ising_flip_trial(N, P, 1.0, seed)
            if t.n_close_pairs == 0:
                assert t.stable.all(), (N, P, seed)
                free.append(1)
            else:
                paired.append(bool(t.stable.all()))
    assert len(free) > 5 and len(paired) > 20
    assert np.mean(paired) < 0.5, np.mean(paired)


def test_expected_close_pairs_is_exact_and_the_exponent_is_2alpha_minus_log2():
    N, P = 12, 40
    counts = []
    for seed in range(300):
        t = SIM.ising_flip_trial(N, P, 1.0, seed)
        counts.append(t.n_close_pairs)
    exact = IS.expected_close_pairs(N, P, 1)
    assert abs(np.mean(counts) - exact) < 5 * np.std(counts) / math.sqrt(len(counts)), (np.mean(counts), exact)
    # the exponent: (1/N) log E[pairs] -> 2 alpha - log 2, positive exactly above alpha = log(2)/2
    for a, sign in ((0.30, -1), (0.40, 1), (0.60, 1)):
        N = 200000                                         # corrections are log(N + 1)/N ~ 6e-5
        e = IS.log_expected_close_pairs(N, a) / N
        assert abs(e - (2 * a - math.log(2))) < 2e-4 and np.sign(e) == sign, (a, e)


def test_gray_code_enumeration_gives_the_exact_overlap_law_and_zeta_of_the_binary_ensemble():
    for N in (10, 12):
        hist, Z = SIM.enumerate_overlap_law(N)
        assert hist == {N - 2 * k: math.comb(N, k) for k in range(N + 1)}
        for lam in (0.3, 1.0, 2.0):
            assert abs(Z(lam) / (2 * math.cosh(lam)) ** N - 1) < 1e-12                  # sum_x e^{lam x.xi} = (2 cosh)^N
            assert abs(math.log(Z(lam) / 2**N) / N - float(EN.zeta("binary", lam))) < 1e-12
    _refuses(lambda: SIM.enumerate_overlap_law(30), "precondition:size")


def test_dhluv_exponent_from_the_exact_binomial_tail():
    """I(1 - 2 rho) = log 2 - H(rho) is the exponent of the Hamming-ball volume: the banked formula against
    log-gamma sums at N = 10^6 (independent of the formula)."""
    from scipy.special import gammaln
    N = 10**6
    for rho in (0.05, 0.1, 0.25):
        k = np.arange(0, int(rho * N) + 1)
        lg = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) - N * math.log(2)
        tail = float(np.logaddexp.reduce(lg)) / N
        assert abs(-tail - float(IS.fixed_point_exponent(rho) * 2)) < 3e-5, rho
    assert abs(IS.fixed_point_exponent(0) - mp.log(2) / 2) < mp.mpf("1e-25")
    _refuses(lambda: IS.fixed_point_exponent("0.5"), "precondition:known-parameter")


def test_binary_thresholds_order_and_limits():
    for l in ("0.5", "1", "3"):
        assert IS.alpha_1_energy(l) < IS.alpha_1_flip(l) < mp.log(2)
    assert IS.fixed_point_exponent(0) < IS.alpha_1_flip(1)                    # log(2)/2 against the typical-pattern value
    assert abs(IS.alpha_1_flip(40) - mp.log(2)) < mp.mpf("1e-20")
    # alpha >= log 2: eps* is the edge, lam* infinite, never condensed, no retrieval at any lam
    assert RN.eps_star("binary", "0.8") == 1 and RN.condensation_lambda("binary", "0.8") == mp.inf
    assert RN.noise("binary", "0.8", 50).phase == "uncondensed" and RN.phi("binary", "0.8", 50) > 1


# ---- planted controls: every one must fail ---------------------------------------------------------------------
def test_planted_wrong_zeta_fails_the_bank_and_the_derivation():
    with planted(EN, zeta=2):
        assert _fresh_check("dense_memory", "spherical_phi_uncondensed")
        assert _fresh_check("dense_memory", "gaussian_phi_uncondensed")


def test_planted_wrong_condensation_threshold_fails():
    with planted(RN, lam_star=1.5):
        assert _fresh_check("dense_memory", "gaussian_lam_star_a0p1")
        assert _fresh_check("dense_memory", "spherical_lam_star_a0p3")
        assert abs(RN.phi("gaussian", "0.1", "0.6") - RN.phi("gaussian", "0.1", "0.6", method="legendre")) > 0.01


def test_planted_wrong_closed_threshold_disagrees_with_the_root():
    with planted(TH, alpha_1=2):
        assert abs(TH.alpha_1_closed("spherical", 1) - TH.alpha_1("spherical", 1)) > mp.mpf("0.1")


def test_gaussian_zeta_is_not_the_spherical_one():
    assert abs(RN.phi("gaussian", "0.3", 1) - mp.mpf(load("dense_memory")["spherical_phi_uncondensed"].value)) > 0.01


def test_planted_wrong_block_count_is_refused_by_the_derivation():
    with planted(DD, blocks=1):
        _refuses(lambda: run_pipeline(DD.label_replica_steps(), State(params={"ensemble": "spherical"})),
                 "check:block-count")


def test_planted_signal_noise_exponent_is_caught_by_the_sampler():
    """Claiming the decision sum has the noise at lam (not 2 lam) fails the sampled exponent."""
    got = float(np.median(SIM.sign_noise_mc(0.3, 0.5, 1600, range(20))))
    assert abs(got - float(RN.phi("binary", "0.3", "0.5"))) > 0.025


# ---- refusals, composer, finality ----------------------------------------------------------------------------------
def test_named_refusals():
    e, ei = catalog()["dense_memory"], catalog()["dense_memory_ising"]
    _refuses(lambda: e.run(alpha="0.2", lam="1", interaction="polynomial", p=3), "unsupported:interaction")
    _refuses(lambda: ei.run(alpha="0.2", lam="1", interaction="polynomial"), "unsupported:interaction")
    _refuses(lambda: e.run(ensemble="binary", alpha="0.2", lam="1"), "unsupported:ensemble")
    _refuses(lambda: e.run(ensemble="hidden-manifold", alpha="0.2", lam="1"), "precondition:known-parameter")
    _refuses(lambda: e.run(), "precondition:state-point")
    _refuses(lambda: e.run(alpha="-0.1", lam="1"), "precondition:known-parameter")
    _refuses(lambda: e.run(alpha="0.1", lam="0"), "precondition:known-parameter")
    _refuses(lambda: e.run(alpha="0.1", lam="1", rho="x"), "precondition:known-parameter")
    _refuses(lambda: EN.zeta("binary", 1, rho=2), "unsupported:ensemble-rho")
    _refuses(lambda: TH.alpha_1("spherical", 1, level=1.2), "no-solution:level-beyond-support")
    _refuses(lambda: IS.expected_close_pairs(0, 5), "precondition:size")
    v = e.run(ensemble="gaussian", alpha="0.6", lam="2")                  # a result, not a refusal: no retrieval phase
    assert v["lambda_1_status"] == "no-solution:no-retrieval" and "lambda_1" not in v


def test_composer_catalog_run_unknown_parameters_and_the_original_refusal():
    e = catalog()["dense_memory"]
    res = compose(e.spec, "replica").run(ensemble="gaussian", alpha="0.2", lam="1.5")
    assert res.report.kind == "catalog" and "dm-extremize" in res.report.reused
    assert res.values["phase"] == "condensed" and res.values["m_star"] < 1
    _refuses(lambda: compose(e.spec, "replica").run(alpha="0.2", lam="1", beta="2"), "precondition:known-parameter")
    # a dense_memory spec that is not a catalog spec is still refused by name, and names the nearest entry
    r = _refuses(lambda: compose(ModelSpec("dense_memory", "real", "gaussian"), "replica"),
                 "missing-step:family:dense_memory")
    assert "dense_memory" in r.detail
    assert compose(catalog()["dense_memory_ising"].spec, "replica").run(alpha="0.2", lam="1").values["phase"]


def test_derivation_steps_run_for_all_ensembles_and_reuse_the_rem_family():
    names = None
    for ens in ENS:
        st, tr = run_pipeline(DD.label_replica_steps(), State(params={"ensemble": ens, "alpha": "0.3", "lam": "1.5"}))
        names = [e.step for e in tr.entries]
        assert st["dm_ens"] == ens and st["dm_ext"]["num"]["m"] < 1
        assert (st["dm_reused"] is not None) == (ens == "gaussian")                   # the REM functional is reused
        assert (st["dm_reduction"] is not None) == (ens == "gaussian")
    assert names == ["dm-noise-split", "dm-replicate", "dm-average-patterns", "dm-1rsb-labels",
                     "dm-replica-limit", "dm-extremize", "dm-rem-reduction"]


def test_finality_labels_published_derived_and_extension():
    e = catalog()["dense_memory"]
    r = compose(e.spec, "replica").run(ensemble="spherical", alpha="0.2", lam="1.5", lam_tilde="0.7")
    assert r.final and all(st.state == "exact-by-literature" for st in r.finality.values()), r.finality
    r = compose(e.spec, "replica").run(ensemble="spherical", alpha="0.2", lam="1.5", N=60, seeds=(0, 1, 2))
    assert r.finality["phi_mc"].state == "not-computed" and "phi_mc" in r.statistical and not r.final
    r = compose(e.spec, "replica").run(ensemble="spherical", alpha="0.2", lam="1.5", level="0.7071067811865476")
    assert r.finality["alpha_1_level"].state == "not-computed" and not r.final
    ei = compose(catalog()["dense_memory_ising"].spec, "replica").run(alpha="0.2", lam="1", radius="0")
    assert ei.finality["alpha_fixed_points"].state == "exact-by-literature"
    assert ei.finality["phi"].state == "not-computed" and ei.finality["alpha_1_flip"].state == "not-computed"
    assert not ei.final
    # Thm 1.3 is proved for lam = 1 only: the fixed-point step is skipped at any other lam
    assert "alpha_fixed_points" not in catalog()["dense_memory_ising"].run(alpha="0.2", lam="2")


def test_bank_files_cite_their_sources():
    for entry in ("dense_memory", "dense_memory_ising"):
        for k, v in load(entry).items():
            assert ("PDF read" in v.cite) and "arXiv" in v.cite and v.independent_route, (entry, k)


# ---- the InfoNCE dictionary -----------------------------------------------------------------------------------------
def test_infonce_dictionary_and_caps():
    assert abs(IN.lam_from_tau("0.1", 20) - mp.mpf("0.5")) < mp.mpf("1e-25")
    assert abs(IN.tau_from_lam(IN.lam_from_tau("0.07", 33), 33) - mp.mpf("0.07")) < mp.mpf("1e-25")
    assert abs(IN.mixture_level(0) - 1 / mp.sqrt(2)) < mp.mpf("1e-25") and IN.mixture_level("0.5") > IN.mixture_level(0)
    lvl = IN.mixture_level(0)
    # tau -> 0: the cap (1 - level^2)^(-d/2) = 2^(d/2) at level 1/sqrt(2)
    assert abs(IN.n_cap(lvl, 10) - 2**5) < mp.mpf("1e-18")
    assert abs(IN.n_threshold(10, "1e-6", lvl) / IN.n_cap(lvl, 10) - 1) < mp.mpf("1e-12")
    # single positive, large d at fixed tau: alpha_1 -> lam, so n_threshold -> e^{1/tau} (the annealed comparison)
    n = IN.n_threshold(10**5, "0.5", 1)
    assert abs(mp.log(n) - 2) < 1e-3
    # tau_threshold inverts n_threshold (n below the cap)
    d, tau = 20, mp.mpf("0.1")
    assert abs(IN.tau_threshold(IN.n_threshold(d, tau, lvl), d, lvl) - tau) < mp.mpf("1e-18")
    _refuses(lambda: IN.tau_threshold(2**11, 20, lvl), "no-solution:no-retrieval")      # above the cap 2^10


def test_level_extension_closed_forms_gaussian():
    """The extension at level l (not in the paper): Gaussian alpha = lam l - lam^2/2 below lam = l, l^2/2 above."""
    l = mp.mpf(1) / mp.sqrt(2)
    for lam in ("0.3", "0.7"):
        assert abs(TH.alpha_1("gaussian", lam, l) - (mp.mpf(lam) * l - mp.mpf(lam) ** 2 / 2)) < mp.mpf("1e-24")
    assert abs(TH.alpha_1("gaussian", 5, l) - mp.mpf(1) / 4) < mp.mpf("1e-24")


# ---- slow tier -----------------------------------------------------------------------------------------------------
@slow
def test_slow_sampler_at_large_N_and_gradient_descent_sharpens():
    for ens, a, l in (("gaussian", 0.3, 0.5), ("spherical", 0.3, 8.0), ("binary", 0.3, 4.0)):
        theory = float(RN.phi(ens, a, l))
        assert abs(np.mean(SIM.rem_phi_mc(ens, a, l, 6400, range(8))) - theory) < 0.004, ens
    l1 = float(TH.lambda_1("spherical", "0.035"))                  # N = 300: P = e^{10.5} = 36315 patterns
    above = [SIM.gd_retrieval("spherical", 0.035, 1.4 * l1, 300, s) for s in range(3)]
    below = [SIM.gd_retrieval("spherical", 0.035, 0.7 * l1, 300, s) for s in range(3)]
    assert max(above) < 0.05 and min(below) > 0.8, (above, below)


@slow
def test_slow_birthday_equivalence_at_n20_and_the_landscape_has_no_spurious_minima_at_n14():
    for P in (200, 800, 1600):
        for seed in range(20):
            t = SIM.ising_flip_trial(20, P, 1.0, seed)
            assert bool(t.stable.all()) == (t.n_close_pairs == 0), (P, seed)
    # Gray-code enumeration of the whole 2^14 landscape: every single-flip local minimum of E_lam is a pattern
    # (an observation at this size, recorded; neither paper claims it)
    from replicant.numerics.enumeration import gray_states
    N, rng = 14, np.random.default_rng(3)
    S = gray_states(N)
    code = ((S > 0).astype(np.int64) * (1 << np.arange(N))).sum(axis=1)
    for lam, P in ((1.0, 100), (3.0, 400)):
        xi = rng.choice([-1.0, 1.0], size=(P, N))
        ov = S @ xi.T
        mx = ov.max(axis=1)
        E = np.empty(2**N)
        E[code] = -(mx + np.log(np.exp(lam * (ov - mx[:, None])).sum(axis=1)) / lam)
        codes = np.arange(2**N)
        is_min = np.ones(2**N, bool)
        for k in range(N):
            is_min &= E <= E[codes ^ (1 << k)] + 1e-12
        pset = set(((xi > 0).astype(np.int64) * (1 << np.arange(N))).sum(axis=1).tolist())
        assert set(np.nonzero(is_min)[0].tolist()) <= pset, (lam, P)
