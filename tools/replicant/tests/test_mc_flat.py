"""M2, default tier: Wang-Landau (1/t and the original schedule), multicanonical sampling and Ferrenberg-Swendsen
reweighting (montecarlo/), every check against an exact target with a planted positive control restored by an
identity assert.  Measured powers (named seed sets outside the ones used here) are stated in each docstring and in
the task report; larger tori and real canonical chains: tests/test_mc_flat_slow.py.
  - exact g(E) (Kaufman's integer polynomial) = the enumerated histogram; its canonical u, C = torus_thermo's
  - WL on the 3 x 3 torus against the exact ln g: the per-level bias band and the error-against-time slope;
    planted: a wrong acceptance rule, a refinement faster than 1/t, a wrong flatness criterion
  - the fixed-weight walk: stationarity and ergodicity chi^2 against the exact multicanonical law, planted +-5%
    acceptance and a non-ergodic proposal; multicanonical reweighting against exact u, C, <|m|>, a planted wrong
    reweighting factor, the window refusal
  - FS single and multiple histogram on exact iid canonical draws (L = 4), planted wrong factor and unsolved free
    energies, the overlap-window refusals
  - the two catalog entries: labels, finality, refusals; named-seed independence of the walkers"""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.catalog import catalog
from replicant.compose import compose
from replicant.core import Refusal
from replicant.montecarlo import dos_exact as DX
from replicant.montecarlo import flat_walk as FW
from replicant.montecarlo import multicanonical as MU
from replicant.montecarlo import reweight as RW
from replicant.montecarlo import stats as ST
from replicant.montecarlo import wang_landau as WL

ALPHA = 0.01
SEEDS = tuple(range(8))
BETA_C = 0.4406867935097715
WL_T, WL_T1 = 500, 125                 # the 3 x 3 WL runs: F_final = 1/WL_T (reached at WL_T), the first snapshot


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


# ---- the exact density of states ---------------------------------------------------------------------------
def test_exact_dos_is_the_enumerated_histogram_and_gives_torus_thermo():
    from replicant.lattice.torus_thermo import torus_thermo_polynomial
    from replicant.montecarlo.exact_small import enumerate_torus
    before = mp.mp.dps
    for L in (2, 3):
        d = DX.exact_dos(L)
        N = L * L
        en = enumerate_torus(L, 2)
        hist = np.bincount((2 * N - en.n_sat) // 2, minlength=N + 1)
        assert tuple(int(x) for x in hist[d.k]) == d.g and hist.sum() == sum(d.g) == 2 ** N, L
        assert (hist[np.setdiff1d(np.arange(N + 1), d.k)] == 0).all()
    for beta in ("2/5", "beta_c"):
        from replicant.montecarlo.fk import beta_value
        ex = DX.exact_canonical(4, beta_value(2, beta))
        u, c = torus_thermo_polynomial(4, 4, beta)
        assert abs(ex.u - float(u.mid())) < 1e-12 and abs(ex.C - float(c.mid())) < 1e-12, beta
        fl = DX.canonical(DX.exact_dos(4).E, DX.exact_dos(4).ln_g, float(beta_value(2, beta)), 16)
        assert abs(fl.u - ex.u) < 1e-12 and abs(fl.C - ex.C) < 1e-11 and abs(fl.ln_z - ex.ln_z) < 1e-12
    assert abs(DX.exact_canonical(4, mp.mpf("0.5")).u - ex.u) > 1e-2          # control: another beta
    from replicant.lattice.kaufman import kaufman_polynomial
    for L in (3, 4):                     # the L >= 13 route (kaufman._z interpolated here) = kaufman_polynomial
        assert DX._interpolated(L) == kaufman_polynomial(L, L).coefficients, L
    assert mp.mp.dps == before
    _refused(lambda: DX.exact_dos(17), "precondition:size")


# ---- Wang-Landau against the exact ln g ----------------------------------------------------------------------
def _wl(schedule="one_over_t", walkers=8):
    """A fresh run (planted runs must never be served from a cache)."""
    return WL.wang_landau(3, SEEDS, walkers, schedule, WL_T + WL_T // 8, f"1/{WL_T}", snapshots=(WL_T1, WL_T))


@lru_cache(maxsize=1)
def _wl_unplanted():
    return _wl()


def _checks(run):
    ex = DX.exact_dos(3)
    return WL.error_profile(run, ex, first_walker=True), WL.error_slope(run, ex, WL_T1, WL_T)


def test_one_over_t_converges_to_the_exact_dos():
    """3 x 3 torus, 8 seeds x 8 walkers, 1/t to MC time 500 (F = 1/500).  Bias: each seed's first walker against the
    exact ln g, a Bonferroni 99% band per level (resolution: the printed half-widths, ~0.5); the 1/t estimate's own
    finite-time bias at the edge levels (~0.05 at t = 1000, measured on 640 walkers, slow tier) is below it.  Slope:
    <eps> from t = 125 to 500 over all walkers, band upper edge < -1/4 (wang_landau.error_slope).  Measured on seed
    sets 100-419 (40 sets of 8): 1/40 false alarms for the bias band, 0/40 for the slope rule."""
    run = _wl_unplanted()
    prof, slope = _checks(run)
    assert run.status == "converged" and np.array_equal(run.k, DX.exact_dos(3).k), run.text
    assert prof.within, prof.text
    assert slope.decreasing, slope.text
    assert np.allclose(run.S[:, 0], np.log(2))                     # the ground-state normalisation


def test_a_wrong_acceptance_rule_fails_both_checks():
    """Planted: accept iff ln u < 0.8 (S1 - S2) (a 20% error in the exponent of WL Eq. (1)).  Measured on seed sets
    100-419 (40 sets of 8): rejected by the bias band 40/40 and by the slope rule 38/40 (at a = 0.9: 28/40, 9/40;
    the chi^2 of the fixed-weight walk below rejects a 5% error with power >= 0.92)."""
    prof, slope = _planted(FW._HOOKS, "acceptance_power", 0.8, lambda: _checks(_wl()))
    assert not prof.within and not slope.decreasing, (prof.text, slope.text)


def test_a_refinement_faster_than_one_over_t_saturates():
    """Planted F = t^-2 (BP: decreasing faster than 1/t saturates the error): the slope rule fails, 40/40 on seed
    sets 100-419."""
    slope = _planted(WL._HOOKS, "time_exponent", 2.0, lambda: _checks(_wl())[1])
    assert not slope.decreasing, slope.text


def test_a_wrong_flatness_criterion_fails():
    """Original schedule (80% flatness, f -> sqrt f), 8 seeds x 4 walkers: its error falls (slope band below -1/4; 0/40
    false alarms on seed sets 100-419).  Planted: 'flat' when max H >= 0.8 <H> (true at every check), so f is reduced
    every check and the error freezes: the slope rule fails 40/40."""
    assert _checks(_wl("wang_landau", 4))[1].decreasing
    slope = _planted(WL._HOOKS, "flatness", "max", lambda: _checks(_wl("wang_landau", 4))[1])
    assert not slope.decreasing, slope.text


def test_wl_labels_and_refusals():
    short = WL.wang_landau(3, (0, 1), 2, "one_over_t", 10, "1/1000")
    assert short.status == "undetermined" and "UNDETERMINED" in short.text      # t_max before F_final: labelled
    for args, reason in (((3, (0, 1), 2, "heatbath", 100, "1/100"), "precondition:schedule"),
                         ((3, (0, 1), 2, "one_over_t", 100, "2"), "precondition:f-final"),
                         ((3, (0, 1), 0, "one_over_t", 100, "1/100"), "precondition:walkers"),
                         ((3, (0,), 2, "one_over_t", 100, "1/100"), "precondition:named-seeds"),
                         ((40, (0, 1), 2, "one_over_t", 100, "1/100"), "precondition:size")):
        _refused(lambda: WL.wang_landau(*args), reason)
    wrong = WL.wang_landau(2, (0, 1), 2, "one_over_t", 20, "1/10")
    _refused(lambda: WL.error_profile(wrong, DX.exact_dos(3)), "no-solution:level-set")


def test_walkers_of_a_seed_do_not_depend_on_the_other_seeds():
    both = FW.Walkers(3, (5, 9), 4)
    solo = FW.Walkers(3, (9,), 4)
    S = np.zeros(both.R * both.K)
    both.run(200, S, np.full(both.R, 0.1), np.zeros(both.R * both.K))
    S1 = np.zeros(solo.R * solo.K)
    solo.run(200, S1, np.full(solo.R, 0.1), np.zeros(solo.R * solo.K))
    assert np.array_equal(both.spins()[4:], solo.spins()) and np.array_equal(S[4 * both.K:], S1)
    both.check()


# ---- the fixed-weight walk and the multicanonical run --------------------------------------------------------
def test_fixed_weight_walk_balance_and_ergodicity_chi2():
    """One sweep from 20000 exact draws of the multicanonical law pi ~ 1/g(E) (3 x 3, seed 7, alpha = 0.01).
    Measured on seeds 100-139: unplanted 0/40 rejected; acceptance exponent x 0.95 rejected 39/40, x 1.05 37/40
    (power >= 0.92 against a 5% error; x 0.9, x 1.1: 40/40).  Ergodicity: 1000 walkers from all +1 after 12
    sweeps; planted proposal of the first N/2 sites only (the rest never flip) must fail."""
    ex = DX.exact_dos(3)
    ok = MU.stationarity_chi2(3, ex.ln_g, ex.k, 20000, 7, ALPHA)
    assert ok.ok, ok.text
    for a in (0.95, 1.05):
        r = _planted(FW._HOOKS, "acceptance_power", a, lambda: MU.stationarity_chi2(3, ex.ln_g, ex.k, 20000, 7, ALPHA))
        assert not r.ok, (a, r.text)
    erg = MU.ergodicity_chi2(3, ex.ln_g, ex.k, 1000, 12, 7, ALPHA)
    assert erg.ok, erg.text
    bad = _planted(FW._HOOKS, "sites", "half", lambda: MU.ergodicity_chi2(3, ex.ln_g, ex.k, 1000, 12, 7, ALPHA))
    assert not bad.ok, bad.text


MUCA_SEEDS = tuple(range(16))


@lru_cache(maxsize=1)
def _muca():
    """16 seeds x 8 walkers: WL (1/t) to t = 300 on 3 x 3, then 100 multicanonical sweeps on each seed's weights."""
    run = WL.wang_landau(3, MUCA_SEEDS, 8, "one_over_t", 300 + 300 // 8, "1/300")
    return run, MU.muca_run(3, run.per_seed, run.k, run.seeds, 8, 100, 10)


def _agrees(rw, beta, m):
    from replicant.montecarlo.exact_small import exact_distribution
    ex = DX.exact_canonical(3, mp.mpf(beta))
    am = exact_distribution(3, 2, beta).abs_m
    t = ST.t_quantile(1 - (1 - 0.99) / (2 * m), len(rw.per_seed["u"]) - 1)
    return all(abs(rw.bands[k].mean - want) <= t * rw.bands[k].stderr for k, want in
               (("u", ex.u), ("C", ex.C), ("abs_m", am)))


def _wl_canonical_biased(run, betas, m):
    """The Wang-Landau canonical estimate (u, C from each seed's ln g) outside some Bonferroni band."""
    t = ST.t_quantile(1 - (1 - 0.99) / (2 * m), len(run.seeds) - 1)
    for b in betas:
        ex = DX.exact_canonical(3, mp.mpf(b))
        cs = [DX.canonical(run.E, row, b, 9) for row in run.per_seed]
        for k, v in (("u", [c.u for c in cs]), ("C", [c.C for c in cs])):
            bd = ST.band(v)
            if abs(bd.mean - getattr(ex, k)) > t * bd.stderr:
                return True
    return False


def test_multicanonical_reweighting_matches_the_exact_torus_and_removes_the_wl_bias():
    """u, C, <|m|> at four beta against the exact g(E) and the enumeration in Bonferroni 99% bands over 16 seeds (12
    comparisons).  The same bands resolve the WL canonical estimate (from ln g alone) as biased, so dropping the
    multicanonical histogram from the reweighting (planted: the estimate becomes e^{S_WL - beta E}, the WL one) must
    fail.  Measured on 40 sets of 16 seeds (100-739): the multicanonical estimate passes 40/40, the WL canonical one
    (= the dropped-histogram plant) is rejected 38/40.  A planted factor w^{-0.9} must also fail."""
    run, m = _muca()
    assert run.status == "converged" and m.status == "equilibrated", (run.text, m.text)
    betas = (0.2, 0.4, 0.6, 1.0)
    assert all(_agrees(MU.reweight(m, b), b, 12) for b in betas), [MU.reweight(m, b).text for b in betas]
    assert _wl_canonical_biased(run, betas, 12)
    for key, value in (("histogram", "drop"), ("weight_power", 0.9)):
        bad = _planted(MU._HOOKS, key, value, lambda: [MU.reweight(m, b) for b in betas])
        assert not all(_agrees(r, b, 12) for r, b in zip(bad, betas)), key
    short = MU.muca_run(3, m.ln_w, m.k, m.seeds, 1, 8, 0)
    _refused(lambda: MU.reweight(short, 3.0), "precondition:reweight-window")
    soft = MU.reweight(short, 3.0, enforce=False)
    assert not soft.in_window and "OUTSIDE" in soft.text


# ---- Ferrenberg-Swendsen on exact canonical draws --------------------------------------------------------------
def _draws(beta, n, seed, L=4):
    d = DX.exact_dos(L)
    a = d.ln_g - beta * d.E
    p = np.exp(a - a.max())
    return np.random.default_rng(seed).choice(d.E, size=n, p=p / p.sum()).astype(float)


def _fs_ok(rw, beta, m):
    ex = DX.exact_canonical(4, mp.mpf(beta))
    t = ST.t_quantile(1 - (1 - 0.99) / (2 * m), len(rw.per_seed["u"]) - 1)
    return all(abs(rw.bands[k].mean - getattr(ex, k)) <= t * rw.bands[k].stderr for k in ("u", "C"))


def test_single_histogram_reweighting_and_its_window():
    """8 seeds x 2000 iid exact draws at beta_c (4 x 4), reweighted to 0.38 and 0.5: u, C in Bonferroni bands (4
    comparisons).  Measured on seed sets 100-419 (40 sets): 40/40 pass; a factor e^{-1.1 (beta - beta0) E} (planted
    10% error) fails 38/40, e^{-1.25 ...} 40/40.  Outside the overlap window: refused."""
    ser = [_draws(BETA_C, 2000, s) for s in SEEDS]
    assert all(_fs_ok(RW.single_histogram(ser, BETA_C, b, 16), b, 4) for b in (0.38, 0.5))
    assert not _planted(RW._HOOKS, "factor_scale", 1.1, lambda: all(
        _fs_ok(RW.single_histogram(ser, BETA_C, b, 16), b, 4) for b in (0.38, 0.5)))
    lo, hi = RW.window(ser, BETA_C)
    assert lo < 0.38 and 0.5 < hi and abs(RW.overlap(ser, BETA_C, hi) - RW.O_MIN) < 1e-6
    grid = np.linspace(lo, hi, 25)
    O = [RW.overlap(ser, BETA_C, b) for b in grid]
    k = int(np.argmax(O))
    assert all(np.diff(O[:k + 1]) >= -1e-12) and all(np.diff(O[k:]) <= 1e-12)     # monotone away from beta0
    for b in (lo - 0.02, hi + 0.02):
        _refused(lambda: RW.single_histogram(ser, BETA_C, b, 16), "precondition:reweight-window")
    _refused(lambda: RW.window([np.full(20, -32.0)] * 2, BETA_C), "precondition:histogram")


def test_multiple_histogram_reweighting_and_overlap():
    """Runs at beta = 0.3, 0.4, 0.5, 0.6 (8 seeds x 2000 iid exact draws each): u, C at 0.32, 0.45, 0.55, 0.62 in
    Bonferroni bands (8 comparisons); 40/40 seed sets pass, and with the free energies left at 0 (planted) 0/40.
    Runs whose windows do not chain: precondition:histogram-overlap."""
    runs = [(b, [_draws(b, 2000, (s, i)) for s in SEEDS]) for i, b in enumerate((0.3, 0.4, 0.5, 0.6))]
    targets = (0.32, 0.45, 0.55, 0.62)
    assert all(_fs_ok(r, b, 8) for r, b in zip(RW.multiple_histogram(runs, targets, 16), targets))
    assert not _planted(RW._HOOKS, "free_energies", "zero", lambda: all(
        _fs_ok(r, b, 8) for r, b in zip(RW.multiple_histogram(runs, targets, 16), targets)))
    _refused(lambda: RW.multiple_histogram(runs, 0.05, 16), "precondition:reweight-window")
    from fractions import Fraction
    two = [(b, ser[:2]) for b, ser in runs[1:3]]                # a small combination: only the beta types matter
    want = RW.multiple_histogram(two, 0.45, 16).bands["u"].mean
    for b in (Fraction(9, 20), mp.mpf("0.45")):                 # the package's rationals and mpf are numbers
        assert abs(RW.multiple_histogram(two, b, 16).bands["u"].mean - want) < 1e-12
    _refused(lambda: RW.multiple_histogram(runs, "x", 16), "precondition:beta")
    far = [(0.1, [_draws(0.1, 2000, (s, 7)) for s in SEEDS]), runs[-1]]
    _refused(lambda: RW.multiple_histogram(far, 0.3, 16), "precondition:histogram-overlap")


# ---- the catalog entries -----------------------------------------------------------------------------------------
def _run(name, **p):
    return compose(catalog()[name].spec, "sample").run(**p)


def test_entries_labels_finality_and_refusals():
    from replicant.catalog_mc_fs import CHEAP as FS
    from replicant.catalog_mc_wl import CHEAP as W
    r = _run("wang_landau_mc", **W)
    assert r.rigor == "statistical" and not r.final and r.finality["beta_c"].final
    assert {"energy_per_site_muca", "energy_per_site_wl"} <= set(r.errors) and r.values["ln_g_profile"]
    # CHEAP stops at t_max = 200 before F_final = 1/1000: labelled (WL and the multicanonical run on its weights)
    assert r.values["wl_status"] == "undetermined" and r.values["muca_status"] == "undetermined"
    assert "UNDETERMINED" in r.finality["energy_per_site_muca"].note
    # a default beta outside the multicanonical window is labelled, an explicit one refused (ruling M5)
    short = {**W, "muca_sweeps": 8, "muca_burn_in": 0}
    o = _run("wang_landau_mc", **short)
    assert o.values["muca_status"] == "undetermined" and "OUTSIDE" in o.values["reweight_summary"]
    _refused(lambda: _run("wang_landau_mc", **{**short, "beta": "beta_c"}), "precondition:reweight-window")
    z = _run("wang_landau_mc", **{**W, "t_max": 10})                 # no ground level yet: labelled, not refused
    assert z.values["muca_status"] == "undetermined" and "energy_per_site_muca" not in z.values
    h = _run("histogram_reweighting_mc", **FS)
    assert h.rigor == "statistical" and not h.final and "energy_per_site_multihistogram" in h.errors
    assert h.values["single_window"][0] <= 0.44 <= h.values["single_window"][1]
    for name, bad, reason in (("wang_landau_mc", {"schedule": "x"}, "precondition:schedule"),
                              ("wang_landau_mc", {"muca_sweeps": 2}, "precondition:muca-sweeps"),
                              ("wang_landau_mc", {"beta": "0"}, "precondition:beta"),
                              ("histogram_reweighting_mc", {"beta": "3"}, "precondition:reweight-window"),
                              ("histogram_reweighting_mc", {"betas": ("2/5", "2/5")}, "precondition:betas"),
                              ("histogram_reweighting_mc", {"algorithm": "metropolis", "L": 3},
                               "precondition:algorithm"),
                              ("histogram_reweighting_mc", {"q": 3}, "precondition:known-parameter")):
        cheap = W if name == "wang_landau_mc" else FS
        _refused(lambda: _run(name, **{**cheap, **bad}), reason)


def test_displayed_acceptance_block_can_fail():
    from replicant.latex_mc_flat import wang_landau_mc
    from replicant.tex import verify_block
    blocks = [b for b in wang_landau_mc("sample") if b.check is not None]
    assert blocks and all(not verify_block(b) for b in blocks)
    acc = blocks[0]
    assert _planted(FW._HOOKS, "acceptance_power", 0.9, lambda: verify_block(acc))
