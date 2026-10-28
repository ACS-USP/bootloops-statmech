"""M2, slow tier: Wang-Landau, multicanonical and Ferrenberg-Swendsen on larger tori and real canonical chains, on
named seeds; each test < 2 min.
  - WL (1/t) on the 4 x 4, 8 x 8, 12 x 12 tori against the exact g(E): converged, the exact level set, the
    first-walker bias band, the error still falling; the walker-mean profile printed (its edge bias)
  - the 1/t estimate's systematic edge bias resolved on 640 walkers at t = 1000 and its decay by t = 4000
  - the 16 x 16 torus: the exact g(E) by the interpolation of dos_exact.py (35 s), its checks (sum 2^256, the
    even-L symmetry, u and C against torus_thermo_kaufman), and WL against it
  - the error-against-time slope of the 1/t schedule (printed against Belardinelli & Pereyra's -0.488(8), L = 8)
    and of the original schedule
  - the multicanonical run on the 8 x 8 torus: u, C at five beta against the exact g(E); <|m|> printed; a planted
    reweighting factor; the stationarity chi^2 at L = 4
  - FS with Metropolis chains (L = 8, 16) and Swendsen-Wang chains (L = 8): single and multiple histogram against
    Kaufman's torus u, C in the window (equilibrated runs only, the C chain-variance bias 2 tau / n stated as a
    systematic, as M1); beyond the window the reweighted u, its overlap and Kish effective sample size printed"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.lattice.torus_thermo import torus_thermo_kaufman
from replicant.montecarlo import dos_exact as DX
from replicant.montecarlo import flat_walk as FW
from replicant.montecarlo import multicanonical as MU
from replicant.montecarlo import reweight as RW
from replicant.montecarlo import stats as ST
from replicant.montecarlo import wang_landau as WL
from replicant.testing import slow

BANK = json.loads((Path(__file__).resolve().parent.parent / "reference" / "wang_landau_mc.json").read_text())
SEEDS = tuple(range(8))
BETA_C = 0.4406867935097715


@lru_cache(maxsize=None)
def _wl(L, T, walkers=8, schedule="one_over_t", F_final=None, snaps=(), t_max=None):
    """F_final = 1/T, reached at MC time T once in the 1/t phase; t_max (default T + T/8) leaves room for the last
    block, and at L = 16 for walkers still halving F (every halving needs a visit to all 255 levels)."""
    return WL.wang_landau(L, SEEDS, walkers, schedule, t_max or T + T // 8, F_final or f"1/{T}", snapshots=snaps)


def _kaufman(L, beta):
    u, c = torus_thermo_kaufman(L, L, beta)
    with mp.workdps(20):
        return float(u.mid()), float(c.mid())


@slow
def test_wang_landau_against_the_exact_dos_up_to_l12():
    for L, T in ((4, 4000), (8, 16000), (12, 16000)):
        r = _wl(L, T, snaps=(T // 4, T))
        ex = DX.exact_dos(L)
        first = WL.error_profile(r, ex, first_walker=True)
        allw = WL.error_profile(r, ex)
        slope = WL.error_slope(r, ex, T // 4, T)
        print(f"L = {L}: {r.text}; restarts per walker {int(r.restarts.min())} .. {int(r.restarts.max())}\n  "
              f"{first.text}"
              f"\n  {allw.text}\n  {slope.text}")
        assert r.status == "converged", r.text
        assert first.within and slope.decreasing, (L, first.text, slope.text)


@slow
def test_the_one_over_t_edge_bias_is_resolved_and_decays():
    """80 seeds x 8 walkers on the 4 x 4 torus: the walker-mean deviation at the level with the largest |deviation| /
    band at t = 1000 lies outside its 99% band (a systematic, not noise) and is smaller at t = 4000.  With the
    ground-state normalisation the rare edge levels' deficit appears as an excess in the middle of the spectrum."""
    seeds = tuple(range(1000, 1080))
    r = WL.wang_landau(4, seeds, 8, "one_over_t", 4500, "1/4000", snapshots=(1000.0, 4000.0))
    ex = DX.exact_dos(4)
    out = {}
    for t in (1000.0, 4000.0):
        d = r.snapshots[t].reshape(len(seeds), 8, -1).mean(axis=1) - ex.ln_g
        out[t] = (d.mean(axis=0), ST.t_quantile(0.995, len(seeds) - 1) * d.std(axis=0, ddof=1) / np.sqrt(len(seeds)))
    m, hw = out[1000.0]
    j = int(np.argmax(np.abs(m) / np.maximum(hw, 1e-300)))
    m4, hw4 = out[4000.0]
    print(f"largest bias (ln g(ground) = ln 2 fixed, so an edge-level deficit shows as a middle excess) at E = "
          f"{int(ex.E[j])}: t = 1000 {m[j]:+.4f} +- {hw[j]:.4f}; t = 4000 {m4[j]:+.4f} +- "
          f"{hw4[j]:.4f}; "
          f"all levels at t = 1000: {np.round(m, 3).tolist()}")
    assert abs(m[j]) > hw[j] and abs(m4[j]) < abs(m[j]), (m[j], hw[j], m4[j])


@slow
def test_the_16x16_exact_dos_and_wang_landau_on_it():
    d = DX.exact_dos(16)
    assert sum(d.g) == 2 ** 256 and d.g == d.g[::-1] and d.k.size == 255
    ex = DX.exact_canonical(16, mp.mpf(2) / 5)
    u, c = _kaufman(16, "2/5")
    assert abs(ex.u - u) < 1e-12 and abs(ex.C - c) < 1e-10, (ex, u, c)
    r = _wl(16, 12000, t_max=40000)
    first = WL.error_profile(r, d, first_walker=True)
    print(f"L = 16: {r.text}; restarts per walker {int(r.restarts.min())} .. {int(r.restarts.max())}\n  {first.text}"
          f"\n  {WL.error_profile(r, d).text}")
    assert r.status == "converged" and first.within, (r.text, first.text)


@slow
def test_error_saturation_and_its_slope():
    """<eps> against t on the 8 x 8 torus, both schedules run to t = 32000 (F_final 1e-9, never reached).  The 1/t
    slope over t = 8000 .. 32000 is printed with Belardinelli & Pereyra's -0.488 +- 0.008 (their Fig. 3, L = 8) and
    asserted below the saturation rule -1/4; the original schedule's slopes are printed."""
    ts = (500, 1000, 2000, 4000, 8000, 16000, 32000)
    ex = DX.exact_dos(8)
    bank = BANK["bp_one_over_t_eps_slope_l8"]
    for schedule in ("one_over_t", "wang_landau"):
        r = WL.wang_landau(8, SEEDS, 8, schedule, 32000, "1e-9", snapshots=ts)
        lines = [WL.error_slope(r, ex, a, b).text for a, b in zip(ts, ts[1:])]
        late = WL.error_slope(r, ex, 8000, 32000)
        print(f"{schedule}: {r.text}\n  " + "\n  ".join(lines) + f"\n  over 8000 .. 32000: {late.text}"
              + (f" (Belardinelli & Pereyra: {bank['value']} +- {bank['tol']})" if schedule == "one_over_t" else ""))
        if schedule == "one_over_t":
            assert late.decreasing, late.text


@slow
def test_multicanonical_on_the_8x8_torus():
    r = _wl(8, 16000, snaps=(4000, 16000))
    m = MU.muca_run(8, r.per_seed, r.k, r.seeds, 8, 2000, 200)
    assert m.status == "equilibrated", m.text
    betas = (0.2, 0.3, BETA_C, 0.5, 0.7)
    t = ST.t_quantile(1 - 0.01 / (2 * 2 * len(betas)), len(SEEDS) - 1)
    for b in betas:
        rw = MU.reweight(m, b)
        ex = DX.exact_canonical(8, mp.mpf(b))
        print(f"{rw.text}; exact u = {ex.u:.6f}, C = {ex.C:.5f}; <|m|> {rw.bands['abs_m'].mean:.4f} +- "
              f"{rw.bands['abs_m'].half_width:.4f} (no exact target at L = 8)")
        for k, want in (("u", ex.u), ("C", ex.C)):
            assert abs(rw.bands[k].mean - want) <= t * rw.bands[k].stderr, (b, k, rw.text, want)
    saved = MU._HOOKS["weight_power"]
    MU._HOOKS["weight_power"] = 0.9
    try:
        bad = [MU.reweight(m, b) for b in betas]
    finally:
        MU._HOOKS["weight_power"] = saved
        assert MU._HOOKS["weight_power"] is saved
    assert any(abs(rw.bands["u"].mean - DX.exact_canonical(8, mp.mpf(b)).u) > t * rw.bands["u"].stderr
               for rw, b in zip(bad, betas))
    ex4 = DX.exact_dos(4)
    assert MU.stationarity_chi2(4, ex4.ln_g, ex4.k, 20000, 11, 0.01).ok
    saved = FW._HOOKS["acceptance_power"]
    FW._HOOKS["acceptance_power"] = 0.9
    try:
        assert not MU.stationarity_chi2(4, ex4.ln_g, ex4.k, 20000, 11, 0.01).ok
    finally:
        FW._HOOKS["acceptance_power"] = saved
        assert FW._HOOKS["acceptance_power"] is saved


@lru_cache(maxsize=None)
def _metropolis(L, beta, sweeps, burn_in, tag):
    from replicant.montecarlo.metropolis import metropolis_run
    seeds = tuple(s + 7919 * tag for s in SEEDS)
    r = metropolis_run(L, beta, burn_in + sweeps, seeds)
    es = [x.astype(float) for x in r["E"][:, burn_in:]]
    taus = [ST.tau_int(x) for x in es]
    eq = ST.equilibration(np.array(es) / (L * L))
    ok = eq.equilibrated and all(ti.reliable for ti in taus)
    return es, ok, float(np.mean([ti.tau for ti in taus])), eq.text


@lru_cache(maxsize=None)
def _sw(L, beta, sweeps, burn_in, tag):
    from replicant.montecarlo.run import chain_series
    es = [chain_series("swendsen_wang", 2, L, beta, sweeps, burn_in, s + 7919 * tag)["E"].astype(float) for s in SEEDS]
    taus = [ST.tau_int(x) for x in es]
    eq = ST.equilibration(np.array(es) / (L * L))
    return es, eq.equilibrated and all(ti.reliable for ti in taus), float(np.mean([ti.tau for ti in taus])), eq.text


def _inside(rw, beta, L, n, tau, m):
    u, C = _kaufman(L, beta)
    t = ST.t_quantile(1 - 0.01 / (2 * m), len(SEEDS) - 1)
    sysC = 2 * tau / n * C                        # the chain-variance bias of C (stated systematic, as M1)
    return (abs(rw.bands["u"].mean - u) <= t * rw.bands["u"].stderr
            and abs(rw.bands["C"].mean - C) <= t * rw.bands["C"].stderr + sysC), (u, C)


@slow
def test_histogram_reweighting_with_canonical_chains():
    compared = 0
    for name, chain, L, n in (("metropolis", _metropolis, 8, 20000), ("metropolis", _metropolis, 16, 20000),
                              ("swendsen_wang", _sw, 8, 2000)):
        es, ok, tau, eqt = chain(L, BETA_C, n, n // 10, 0)
        if not ok:
            print(f"{name} L = {L} at beta_c: undetermined, not compared ({eqt})")
            continue
        compared += 1
        lo, hi = RW.window(es, BETA_C)
        targets = (BETA_C + 0.6 * (lo - BETA_C), BETA_C + 0.6 * (hi - BETA_C))
        for b in targets:
            rw = RW.single_histogram(es, BETA_C, b, L * L)
            good, want = _inside(rw, b, L, n, tau, 2 * len(targets))
            print(f"{name} L = {L}: {rw.text}; Kaufman u = {want[0]:.6f}, C = {want[1]:.5f}")
            assert good, (name, L, b, rw.text, want)
        # beyond the window (printed, not asserted): the seed band stays honest here, the window rule is conservative;
        # the rule's positive control is the default-tier refusal
        for k in (3, 4):
            for far in (BETA_C + k * (lo - BETA_C), BETA_C + k * (hi - BETA_C)):
                per = [RW._moments(x, RW.log_factor(BETA_C, far, x), far, L * L)[0] for x in es]
                w = [np.exp(RW.log_factor(BETA_C, far, x) - RW.log_factor(BETA_C, far, x).max()) for x in es]
                kish = np.mean([v.sum() ** 2 / (v * v).sum() / v.size for v in w])
                bd, u_far = ST.band(per), _kaufman(L, far)[0]
                print(f"  beyond the window: beta = {far:.5f}, overlap {RW.overlap(es, BETA_C, far):.3f}, Kish "
                      f"n_eff / n {kish:.3g}, u = {bd.mean:.5f} +- {bd.half_width:.5f} vs Kaufman {u_far:.5f}")
        # multiple histogram across beta_c
        runs, oks = [], []
        for tag, b in enumerate((BETA_C - 2 * (BETA_C - lo), BETA_C, BETA_C + 2 * (hi - BETA_C)), start=1):
            es_b, ok_b, tau_b, _ = chain(L, b, n, n // 10, tag)
            runs.append((b, es_b))
            oks.append(ok_b)
        if not all(oks):
            print(f"{name} L = {L}: a multiple-histogram run is undetermined, not compared")
            continue
        grid = np.linspace(runs[0][0], runs[-1][0], 5)
        for rw, b in zip(RW.multiple_histogram(runs, tuple(grid), L * L), grid):
            good, want = _inside(rw, b, L, n, tau, 2 * grid.size)
            print(f"{name} L = {L}: {rw.text}; Kaufman u = {want[0]:.6f}, C = {want[1]:.5f}")
            assert good, (name, L, b, rw.text, want)
    assert compared >= 2, compared
