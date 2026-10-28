"""M3, slow tier: population annealing on larger systems and populations, on named seeds; each test < 2 min.
  - the 8 x 8 and 16 x 16 Ising tori at beta_c and 1/2: ln Z / N (simple mean + s^2/2) and u against Kaufman, on
    equilibrated runs only; planted at 8 x 8: a 1% wrong reweighting factor, the resampling step missing
  - SK at N = 18, four Gaussian and four +-J matrices at beta = 5/2 (below T_c = 1): ln Z and u against enumeration
    (+-J: E11's integer spectrum), one Bonferroni family
  - estimator-property tests (controller ruling, M3 fix round 1: they assert value bands for the corrected
    estimators on runs the tool labels undetermined, asserted so, and no run-path output takes a value from them):
    the finite-R bias at R = 10, 20, 40 (4 x 4 at beta = 1/2, 1024 seeds each), the simple mean resolved below
    Kaufman, mean + s^2/2 and Machta's weighted average covering it, R var(beta F~) (WMK's rho_f) and rho_t, rho_s
    printed; E[Z~ / Z] at R = 5 over 200000 runs: the nominal-R form unbiased, WMK's actual-R~ form resolved high"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.lattice.kaufman import kaufman_free_energy
from replicant.lattice.torus_thermo import torus_thermo_kaufman
from replicant.montecarlo import population as PA
from replicant.montecarlo import sk_metropolis as SKM
from replicant.montecarlo import stats as ST
from replicant.testing import slow

ALPHA = 0.01
SEEDS = tuple(range(8))
BETA_C = 0.4406867935097715


def _planted(key, value, fn):
    saved = PA._HOOKS[key]
    PA._HOOKS[key] = value
    try:
        return fn()
    finally:
        PA._HOOKS[key] = saved
        assert PA._HOOKS[key] is saved


def _kaufman(L, beta):
    with mp.workdps(20):
        return float(kaufman_free_energy(L, L, beta).mid()) * L * L, float(torus_thermo_kaufman(L, L, beta)[0].mid())


def _ts(system, beta, lnz, u, R, K, sweeps, seeds=SEEDS):
    run = PA.anneal(system, beta, R, K, sweeps, seeds)
    s, x = PA.summary(run), run.ln_z[:, -1]
    b = ST.band(x)
    return (b.mean + s.var_beta_f / 2 - lnz) / b.stderr, (s.energy.mean - u) / s.energy.stderr, s


def _family_ok(ts):
    t = ST.t_quantile(1 - ALPHA / (2 * 2 * len(ts)), len(SEEDS) - 1)
    return all(abs(a) <= t and abs(b) <= t for a, b, _ in ts)


@slow
def test_ising_tori_against_kaufman_and_plants():
    cases = [(8, "beta_c", BETA_C, 2000, 20, 2), (8, "1/2", 0.5, 2000, 20, 2), (16, "beta_c", BETA_C, 5000, 40, 2),
             (16, "1/2", 0.5, 5000, 40, 2)]
    ts = []
    for L, name, b, R, K, sw in cases:
        lnz, u = _kaufman(L, name)
        t = _ts(PA.Ising2D(L), b, lnz, u, R, K, sw)
        print(f"L = {L}, beta = {name}: t(ln Z) = {t[0]:+.2f}, t(u) = {t[1]:+.2f}; {t[2].text}")
        assert t[2].status == "equilibrated", t[2].reasons
        ts.append(t)
    assert _family_ok(ts)
    lnz, u = _kaufman(8, "beta_c")
    for key, value in (("reweight_scale", 0.99), ("resample", False)):
        bad = _planted(key, value, lambda: _ts(PA.Ising2D(8), BETA_C, lnz, u, 2000, 20, 2))
        print(f"planted {key} = {value}: t(ln Z) = {bad[0]:+.2f}, t(u) = {bad[1]:+.2f}")
        assert not _family_ok([bad])


@slow
def test_sk_and_pm_one_matrices_at_low_temperature():
    from replicant.numerics.enumeration import enumerate_pm_one_sk
    from replicant.numerics.finite_size import gaussian_sk_couplings, pm_one_sk_couplings
    beta, ts = 2.5, []
    for d, (G, P) in enumerate(zip(gaussian_sk_couplings(18, 4, 21), pm_one_sk_couplings(18, 4, 21))):
        sp = enumerate_pm_one_sk(P)
        for name, J, lnz, u in (("gaussian", G, SKM.exact_log_z(G, beta), SKM.exact_energy(G, beta)),
                                ("+-J", P, sp.log_Z(beta), sp.mean_energy_per_spin(beta))):
            t = _ts(PA.SK(J), beta, lnz, u, 2000, 50, 4)
            print(f"{name} #{d}: t(ln Z) = {t[0]:+.2f}, t(u) = {t[1]:+.2f}; {t[2].text}")
            assert t[2].status == "equilibrated", (name, d, t[2].reasons)
            ts.append(t)
    assert _family_ok(ts)


@slow
def test_estimator_property_finite_population_bias_scales_and_is_removed():
    """Estimator-property test on runs labelled undetermined (asserted): value bands for the corrected estimators."""
    lnz, _ = _kaufman(4, "1/2")
    for R in (10, 20, 40):
        run = PA.anneal(PA.Ising2D(4), 0.5, R, 10, 1, range(10_000, 11_024))
        x = run.ln_z[:, -1]
        b = ST.band(x)
        corr = ST.jackknife(lambda a: a[0].mean() + a[0].var(ddof=1) / 2, [x], n_blocks=64)
        wtd = PA._jackknife_weighted(run.ln_z_nominal)
        rho_t, rho_s = np.nanmean(run.families[:, 0]), np.nanmean(run.families[:, 2])
        print(f"R = {R}: mean - exact = {b.mean - lnz:+.4f} +- {b.half_width:.4f} (99%), -s^2/2 = {-x.var(ddof=1) / 2:+.4f};"
              f" corrected {corr.estimate - lnz:+.4f} +- {corr.stderr:.4f}; weighted {wtd.estimate - lnz:+.4f} +- "
              f"{wtd.stderr:.4f}; R var(beta F~) = {R * x.var(ddof=1):.3g}; mean rho_t {rho_t:.3g}, rho_s {rho_s:.3g}")
        assert PA.summary(run).status == "undetermined"
        assert b.mean + b.half_width < lnz
        assert abs(corr.estimate - lnz) <= ST.t_quantile(1 - ALPHA / 2, 63) * corr.stderr
        assert abs(wtd.estimate - lnz) <= ST.t_quantile(1 - ALPHA / 2, 63) * wtd.stderr


@slow
def test_estimator_property_the_nominal_normalisation_is_unbiased_for_z():
    """E[Z~ / Z] over 200000 independent runs at R = 5 (Gaussian SK N = 16, beta = 2, 10 temperatures, 1 sweep;
    seeds 100000-299999; standard errors from 100 chunks of 2000 runs; extinct runs count as 0).  Measured:
    nominal-R form 0.9995 +- 0.0027, WMK's actual-R~ form 1.0153 +- 0.0020.  Asserted: the nominal form within its
    99% band of 1, the WMK form above 1 beyond its 99% band.  These runs are undetermined (R = 5): an
    estimator-property test asserting value bands on them."""
    from replicant.numerics.finite_size import gaussian_sk_couplings
    J = gaussian_sk_couplings(16, 1, 3)[0]
    ex = SKM.exact_log_z(J, 2.0)
    run = PA.anneal(PA.SK(J), 2.0, 5, 10, 1, range(100_000, 300_000))
    t = ST.t_quantile(1 - ALPHA / 2, 99)
    out = {}
    for name, v in (("nominal", run.ln_z_nominal), ("wmk", run.ln_z[:, -1])):
        b = ST.band(np.exp(v - ex).reshape(100, -1).mean(axis=1))
        out[name] = b
        print(f"{name}: E[Z~ / Z] = {b.mean:.4f} +- {b.stderr:.4f}")
    assert PA.summary(run).status == "undetermined"
    assert abs(out["nominal"].mean - 1) <= t * out["nominal"].stderr
    assert out["wmk"].mean - 1 > t * out["wmk"].stderr
