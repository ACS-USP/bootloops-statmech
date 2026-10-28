"""M3, default tier: population annealing (montecarlo/population.py) against exact ln Z, with planted positive controls
restored by identity asserts.  Measured powers (named seed sets other than the ones used here) are stated in each
docstring and in the task report; larger systems and populations: tests/test_mc_pa_slow.py.
  - the SK Metropolis sweep (montecarlo/sk_metropolis.py): stationarity chi^2 against exact Boltzmann draws, planted
    +-5% acceptance; its exact ln Z = E11's integer spectrum on a +-J matrix; named-seed blocks independent
  - the family statistics against WMK's stated exponential-family values (rho_t = 2 mu, rho_s ~ 1.53 mu)
  - ln Z and u at the final beta on the 4 x 4 Ising torus (Kaufman) and on two Gaussian and two +-J SK matrices
    (N = 10, enumeration): one Bonferroni family over the ten comparisons; planted: a 3% wrong reweighting factor,
    the resampling step missing
  - the finite-R bias of one run (WMK Eq. (16)): resolved at R = 5, removed by s^2/2 and by Machta's weighted
    average; the labels (too few families, var(beta F~) > 1/2) and a planted broken family bookkeeping
  - the catalog entry: labels, finality, refusals; the displayed reweighting block can fail"""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.catalog import catalog
from replicant.compose import compose
from replicant.core import Refusal
from replicant.montecarlo import population as PA
from replicant.montecarlo import sk_metropolis as SKM
from replicant.montecarlo import stats as ST

ALPHA = 0.01
SEEDS = tuple(range(8))
MAIN = {"R": 600, "temperatures": 10, "sweeps": 1}       # every run of the main check: equilibrated (asserted)


def _refused(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"not refused: expected {reason}")


def _planted(hooks, key, value, fn):
    saved = hooks[key]
    hooks[key] = value
    try:
        return fn()
    finally:
        hooks[key] = saved
        assert hooks[key] is saved


@lru_cache(maxsize=None)
def _systems():
    """(name, system, beta, exact ln Z, exact u): Kaufman for the torus, E11's integer spectrum for +-J, the 2^N
    enumeration of sk_metropolis (equal to E11's on +-J: test_sk_metropolis...) for the Gaussian matrices."""
    from replicant.lattice.kaufman import kaufman_free_energy
    from replicant.lattice.torus_thermo import torus_thermo_kaufman
    from replicant.numerics.enumeration import enumerate_pm_one_sk
    from replicant.numerics.finite_size import gaussian_sk_couplings, pm_one_sk_couplings
    with mp.workdps(20):
        out = [("ising 4 x 4", PA.Ising2D(4), 0.4, float(kaufman_free_energy(4, 4, "2/5").mid()) * 16,
                float(torus_thermo_kaufman(4, 4, "2/5")[0].mid()))]
    for d, (G, P) in enumerate(zip(gaussian_sk_couplings(10, 2, 7), pm_one_sk_couplings(10, 2, 7))):
        out.append((f"gaussian SK N = 10 #{d}", PA.SK(G), 1.5, SKM.exact_log_z(G, 1.5), SKM.exact_energy(G, 1.5)))
        sp = enumerate_pm_one_sk(P)
        out.append((f"+-J SK N = 10 #{d}", PA.SK(P), 1.5, sp.log_Z(1.5), sp.mean_energy_per_spin(1.5)))
    return tuple(out)


def _comparisons(systems, seeds=SEEDS):
    """Per system: (t of ln Z, t of u, status), t = (estimate - exact) / stderr over seeds.  ln Z: the simple mean
    of the runs' ln Z~ plus the leading bias s^2/2 (WMK Eq. (16)); at these R, s^2 ~ 1e-3 and the correction and its
    own error (s^2 / sqrt(2 (M - 1))) are below 5% of the stderr."""
    rows = []
    for name, sysm, beta, lnz, u in systems:
        run = PA.anneal(sysm, beta, MAIN["R"], MAIN["temperatures"], MAIN["sweeps"], seeds)
        s, x = PA.summary(run), run.ln_z[:, -1]
        b = ST.band(x)
        rows.append((name, (b.mean + s.var_beta_f / 2 - lnz) / b.stderr, (s.energy.mean - u) / s.energy.stderr,
                     s.status))
    return rows


def _joint_ok(rows):
    """One claim (the free energy and energy of every system): Bonferroni over the 2 x len(rows) comparisons,
    family-wise alpha = 0.01, Student-t with df = seeds - 1."""
    t = ST.t_quantile(1 - ALPHA / (2 * 2 * len(rows)), len(SEEDS) - 1)
    return all(abs(a) <= t and abs(b) <= t for _, a, b, _ in rows)


# ---- the SK Metropolis sweep -----------------------------------------------------------------------------------
def test_sk_metropolis_is_stationary_its_exact_ln_z_is_e11s_and_its_blocks_are_independent():
    """chi^2 at alpha = 0.01, N = 5 Gaussian SK, beta = 1, 20000 exact draws, one sweep.  Measured on seeds 100-139:
    unplanted 0/40 rejected; acceptance x 0.95 rejected 40/40, x 1.05 39/40 (power >= 0.97 against 5%)."""
    from replicant.numerics.enumeration import enumerate_pm_one_sk, random_pm_one_J
    from replicant.numerics.finite_size import gaussian_sk_couplings
    J = gaussian_sk_couplings(5, 1, 11)[0]
    assert SKM.stationarity_chi2(J, 1.0, 20000, 7, ALPHA).ok
    for a in (0.95, 1.05):
        assert not _planted(SKM._HOOKS, "acceptance_scale", a, lambda: SKM.stationarity_chi2(J, 1.0, 20000, 7, ALPHA)).ok
    P = random_pm_one_J(10, 3)
    sp = enumerate_pm_one_sk(P)
    for b in (0.5, 2.0):
        assert abs(SKM.exact_log_z(P, b) - sp.log_Z(b)) < 1e-9 and abs(SKM.exact_energy(P, b)
                                                                        - sp.mean_energy_per_spin(b)) < 1e-12
    _refused(lambda: SKM.exact_log_z(random_pm_one_J(21, 0), 1.0), "precondition:enumeration-size")
    s0 = np.random.default_rng(1).choice((-1.0, 1.0), size=(30, 10))
    both = SKM.sk_sweeps(P, s0.copy(), 1.0, 3, [np.random.default_rng(5), np.random.default_rng(6)], [12, 18])
    alone = SKM.sk_sweeps(P, s0[12:].copy(), 1.0, 3, [np.random.default_rng(6)], [18])
    assert np.array_equal(both[12:], alone)


# ---- families ---------------------------------------------------------------------------------------------------
def test_family_statistics_reproduce_wmks_exponential_values():
    """WMK Sec. IV B: exponential family sizes of mean mu give rho_t = 2 mu and rho_s ~ 1.53 mu (exactly
    mu e^{1 - gamma} = 1.5262 mu in the continuum, derived in the report).  10^5 families: the relative standard
    error of rho_t is about 0.7%, the tolerance 3%.  Equal sizes give rho_t = rho_s = mu exactly."""
    mu = 50.0
    x = np.random.default_rng(3).exponential(mu, 100_000)
    rho_t, n_eff, rho_s, eS = PA.family_stats(x)
    assert abs(rho_t / (2 * mu) - 1) < 0.03 and abs(rho_s / (mu * np.exp(1 - np.euler_gamma)) - 1) < 0.03
    assert abs(n_eff * rho_t - x.sum()) < 1e-6 * x.sum() and rho_s < rho_t
    assert np.allclose(PA.family_stats(np.full(40, 50))[:3], (50, 40, 50)) and np.isclose(
        PA.family_stats(np.full(40, 50))[3], 40)


# ---- the free energy against exact ln Z -------------------------------------------------------------------------
def test_free_energy_and_energy_match_the_exact_targets_and_the_plants_fail():
    """R = 600, 10 temperatures, 1 sweep, seeds 0-7; all five runs equilibrated (asserted, so the band is fed by
    determined runs only; the margin is thin on the torus: the smallest R~ / rho_t over 4000 seeds is 104.9 against
    the threshold 100, so a change of schedule or sweeps may fail the label assert before the band).  Calibration
    (the reviewer's 500 disjoint 8-seed sets, seeds 100-4099, which contain my first 200): unplanted 6/500 rejected,
    1.2% (binomial 95% interval about 0.4-2.6%), consistent with the nominal family-wise 1%.  ln Z~ is Gaussian
    over seeds (|skew| <= 0.07) and the within-set variance follows chi^2_7 / 7 (KS p = 0.40), so the set 1404-1411
    (t = -15, its eight values within 0.005) is a chance small-s tail of Student-t on 8 seeds (about 4e-6 per set),
    not a defect.  The plants are run on the Gaussian matrix #0 alone (a family of its two comparisons); measured on
    the sets 100-419: reweighting factor x 0.97 rejected 40/40 (x 0.99: 11/40), resampling missing 40/40; the same
    on the +-J matrix #0.  Not resolved here: a normalisation error below about 0.05 in total ln Z (e.g. Q divided by
    R~ + 1, ln Z~ low by about 0.017, about 1 stderr); that is caught exactly by
    test_the_normalisation_is_the_mean_at_the_run_s_own_population_size."""
    rows = _comparisons(_systems())
    assert all(st == "equilibrated" for *_, st in rows), rows
    assert _joint_ok(rows), rows
    one = _systems()[1:2]
    for key, value in (("reweight_scale", 0.97), ("resample", False)):
        bad = _planted(PA._HOOKS, key, value, lambda: _comparisons(one))
        assert not _joint_ok(bad), (key, bad)


def test_estimator_property_finite_population_bias_and_its_removal():
    """ESTIMATOR-PROPERTY TEST (controller ruling, M3 fix round 1): it uses runs deliberately labelled undetermined
    by the run path (asserted below: too few families and var(beta F~) > 1/2), and it ASSERTS A VALUE BAND for the
    corrected estimators against the exact ln Z on those runs.  It tests the estimators' finite-R bias and its
    removal; no run-path output takes a value from such runs (the entry labels them undetermined).
    WMK Eqs. (15), (16): one run's ln Z~ is biased low by var(beta F~)/2 + higher cumulants.  At R = 5 (Gaussian SK
    N = 16, beta = 2, 10 temperatures, 1 sweep, 256 seeds 1000-1255): the simple mean's 99% band lies below the
    exact ln Z (the bias is resolved); mean + s^2/2 (jackknifed over seeds) and Machta's weighted average (Eq. (7),
    nominal-R normalisation, jackknifed) each lie within their 99% band of it.  Measured on 20 disjoint 256-seed
    sets (1000-6119): simple band below 20/20; s^2/2-corrected covers 19/20; weighted covers 20/20."""
    from replicant.numerics.finite_size import gaussian_sk_couplings
    J = gaussian_sk_couplings(16, 1, 3)[0]
    ex = SKM.exact_log_z(J, 2.0)
    run = PA.anneal(PA.SK(J), 2.0, 5, 10, 1, range(1000, 1256))
    x = run.ln_z[:, -1]
    M = len(x)
    t = ST.t_quantile(1 - ALPHA / 2, M - 1)
    b = ST.band(x)
    assert b.mean + b.half_width < ex
    corr = ST.jackknife(lambda a: a[0].mean() + a[0].var(ddof=1) / 2, [x], n_blocks=M)
    wtd = PA._jackknife_weighted(run.ln_z_nominal)
    assert abs(corr.estimate - ex) <= t * corr.stderr and abs(wtd.estimate - ex) <= t * wtd.stderr
    s = PA.summary(run)
    assert s.status == "undetermined" and any("families" in r for r in s.reasons) and any(
        "var(beta F~)" in r for r in s.reasons)
    assert abs(s.bias_estimate * 16 + x.var(ddof=1) / 2) < 1e-12 and abs(s.ln_z_weighted * 16 - wtd.estimate) < 1e-9


def test_the_normalisation_is_the_mean_at_the_run_s_own_population_size():
    """Q must be the mean of the weights over the population actually present (WMK Eq. (2)).  Checked exactly, at
    the run's own size: the first annealing step of every seed against ln mean exp(-beta_1 E) of that seed's
    regenerated uniform start (R = 600), and _resample on a population of fluctuated size R~ = 613 != R.  The planted
    Q divided by R~ + 1 (ln Z~ low by about temperatures / R, below the main check's resolution) fails both."""
    from replicant.numerics.finite_size import gaussian_sk_couplings
    J = gaussian_sk_couplings(10, 1, 7)[0]
    seeds, R, beta = (0, 1, 2), 600, 1.5

    def first_steps():
        run = PA.anneal(PA.SK(J), beta, R, 10, 1, seeds)
        out = []
        for i, sd in enumerate(seeds):
            E = SKM.sk_energy(J, PA.SK(J).init(np.random.default_rng([sd, PA.STREAM]), R))
            want = np.log(np.mean(np.exp(-run.betas[1] * E)))
            out.append(abs(run.ln_z[i, 1] - run.ln_z[i, 0] - want))
        x = -0.15 * SKM.sk_energy(J, np.random.default_rng(4).choice((-1.0, 1.0), size=(613, 10)))
        out.append(abs(PA._resample(x, R, np.random.default_rng(5))[0] - np.log(np.mean(np.exp(x)))))
        return max(out)
    assert first_steps() < 1e-12
    assert _planted(PA._HOOKS, "q_offset", 1, first_steps) > 1e-3


def test_labels_follow_the_diagnostics_and_a_broken_family_bookkeeping_is_caught():
    """A population collapsed by two large steps without sweeps (4 x 4 at beta_c, R = 1000) has few families and
    must be labelled undetermined; the planted bookkeeping that refounds every family at each step hides the
    collapse (the label flips), so the family check depends on the tracking.  A synthetic run isolates each
    reason: var(beta F~) > 1/2 alone, an extinct population alone, fewer than two surviving seeds (labelled with NaN
    values, never refused).  Machta's weighted average keeps an extinct run as a zero weight (Eq. (7) over all M)."""
    bc = 0.4406867935097715
    fn = lambda: PA.summary(PA.anneal(PA.Ising2D(4), bc, 1000, 2, 0, (0, 1)))
    s = fn()
    assert s.status == "undetermined" and len(s.reasons) == 1 and "families" in s.reasons[0], s.reasons
    assert _planted(PA._HOOKS, "families", "reset", fn).status == "equilibrated"
    fam = np.tile([1.0, 500.0, 1.0, 500.0], (4, 1))
    base = dict(N=16, R=500, betas=np.array([0.0, 1.0]), seeds=(0, 1, 2, 3), energy=np.full((4, 2), -1.0),
                sizes=np.full((4, 2), 500), families=fam, extinct=np.zeros(4, bool))
    calm = PA.PARun(ln_z=np.array([[0, 10.0], [0, 10.1], [0, 9.9], [0, 10.0]]), **base)
    assert PA.summary(calm).status == "equilibrated"
    wide = PA.PARun(ln_z=np.array([[0, 10.0], [0, 11.5], [0, 9.0], [0, 10.0]]), **base)
    assert PA.summary(wide).reasons == (f"var(beta F~) over seeds {np.var([10, 11.5, 9, 10], ddof=1):.3g} > 0.5",)
    dead = PA.PARun(ln_z=calm.ln_z.copy(), **{**base, "extinct": np.array([False, False, False, True])})
    assert PA.summary(dead).status == "undetermined" and "died out" in PA.summary(dead).reasons[0]
    lone = PA.PARun(ln_z=np.array([[0, 10.0], [0, -np.inf], [0, -np.inf], [0, -np.inf]]),
                    **{**base, "extinct": np.array([False, True, True, True]),
                       "energy": np.array([[-1.0, -1.0]] + [[-1.0, np.nan]] * 3)})
    ls = PA.summary(lone)
    assert ls.status == "undetermined" and "fewer than two surviving seeds (1 of 4)" in ls.reasons[-1]
    assert np.isnan(ls.ln_z_weighted) and np.isnan(ls.ln_z_mean.mean)
    assert np.isclose(PA.weighted_ln_z([10.0, -np.inf]), 10.0 - np.log(2)) and PA.weighted_ln_z([-np.inf]) == -np.inf
    gone = PA.PARun(ln_z=np.array([[0, 5, 10.0], [0, 5, 10.1], [0, 5, 9.9], [0, -np.inf, -np.inf]]),
                    **{**base, "betas": np.array([0.0, 0.5, 1.0]), "extinct": np.array([False, False, False, True]),
                       "energy": np.array([[-1.0, -1, -1]] * 3 + [[-1.0, np.nan, np.nan]]),
                       "sizes": np.array([[500, 500, 500]] * 3 + [[500, 0, 0]])})
    gs = PA.summary(gone)                # three survivors: the weighted average keeps the extinct run as a zero
    assert gs.status == "undetermined" and np.isfinite(gs.ln_z_weighted) and gs.ln_z_mean.df == 2
    plain = PA.weighted_ln_z([10.0, 10.1, 9.9, -np.inf])          # ln(3/4) below the three survivors' average
    assert abs(plain - PA.weighted_ln_z([10.0, 10.1, 9.9]) - np.log(0.75)) < 1e-12
    assert abs(gs.ln_z_weighted - plain / 16) <= gs.ln_z_weighted_stderr   # the jackknife's bias step


def test_named_seeds_are_independent():
    from replicant.numerics.finite_size import pm_one_sk_couplings
    for sysm, beta in ((PA.Ising2D(4), 0.4), (PA.SK(pm_one_sk_couplings(8, 1, 0)[0]), 1.0)):
        a = PA.anneal(sysm, beta, 50, 3, 1, (0, 1))
        b = PA.anneal(sysm, beta, 50, 3, 1, (7, 1))
        assert np.array_equal(a.ln_z[1], b.ln_z[1]) and not np.array_equal(a.ln_z[0], b.ln_z[0])


# ---- the catalog entry -------------------------------------------------------------------------------------------
def _run(**p):
    return compose(catalog()["population_annealing_mc"].spec, "sample").run(**p)


def test_entry_labels_finality_and_refusals():
    from replicant.catalog_mc_pa import CHEAP
    before = mp.mp.dps
    r = _run(**CHEAP)
    assert mp.mp.dps == before
    assert r.rigor == "statistical" and not r.final and r.finality["beta_c"].final
    assert {"ln_z_per_site_pa", "ln_z_per_site_weighted", "energy_per_site_pa"} <= set(r.errors)
    assert r.values["pa_status"] == "undetermined" and "UNDETERMINED" in r.finality["ln_z_per_site_pa"].note
    assert "ln_z_per_site_exact" in r.values["exact_targets"]
    k = _run(**{**CHEAP, "model": "sk_pm_one", "N": 6})
    ex = k.values["exact_targets"]
    assert abs(ex["ln_z_per_site_exact"] - ex["ln_z_per_site_e11"]) < 1e-12 and "beta_c" not in k.values
    for bad, reason in (({"model": "x"}, "precondition:model"), ({"L": 3}, "precondition:checkerboard"),
                        ({"L": 2}, "precondition:checkerboard"), ({"model": "sk_gaussian", "N": 1}, "precondition:size"),
                        ({"model": "sk_gaussian", "beta": "beta_c"}, "precondition:beta"), ({"beta": "0"},
                        "precondition:beta"), ({"population": 1}, "precondition:population"),
                        ({"temperatures": 0}, "precondition:temperatures"), ({"sweeps": -1}, "precondition:sweeps"),
                        ({"seeds": (0,)}, "precondition:named-seeds"), ({"q": 3}, "precondition:known-parameter")):
        _refused(lambda: _run(**{**CHEAP, **bad}), reason)


def test_displayed_reweighting_block_can_fail():
    from replicant.latex_mc_pa import population_annealing_mc
    from replicant.tex import verify_block
    blocks = [b for b in population_annealing_mc("sample") if b.check is not None]
    assert blocks and all(not verify_block(b) for b in blocks)
    assert _planted(PA._HOOKS, "reweight_scale", 0.97, lambda: verify_block(blocks[0]))
