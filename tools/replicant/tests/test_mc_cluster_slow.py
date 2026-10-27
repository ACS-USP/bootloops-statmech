"""M1, slow tier: cluster algorithms on larger tori, on named seeds.
  - exact small tori (L = 3, 4; Potts q = 3 at L = 3): <e>, <|m|>, <m^2>, C of SW / Wolff / Metropolis against the
    enumeration in 99% Student-t bands over 16 seeds, SW's FK estimator of <m^2>; ergodicity and stationarity chi^2
    at L = 4 (alpha = 0.01); Kaufman's integer polynomial = the 2^16-state enumeration
  - L = 8, 16, 32 at beta_c: <e> and C against Kaufman's torus values (torus_thermo_kaufman) in the band plus the
    stated variance bias 2 tau / n of C, for the runs labelled equilibrated (an undetermined size is printed and
    skipped; at least two sizes per algorithm must be compared); <|m|>, <m^2> printed with bands (no exact target: ruling A-R6); a planted
    wrong bond probability is rejected
  - critical slowing down: tau_int at beta_c against L for Metropolis (L = 16, 32, 64), SW (8 .. 64), Wolff
    (8 .. 32, five sizes) and SW at q = 3 (8 .. 32); z per seed in a Student-t band, printed next to the fetched literature
    values (reference/cluster_mc.json) with the estimator each measures; only runs labelled equilibrated enter a
    fit (an undetermined size is dropped and printed); asserted only where the band decides: every Metropolis
    size is equilibrated, its z band lies above 1.5 and contains a literature value (assuming Metropolis and
    heat-bath share z); the SW and Wolff bands lie below 1.5; their literature comparison is printed (not
    separable from a logarithm).  Wolff sizes 8, 12, 16, 24, 32
Every run is memoised per process (one simulation serves several tests)."""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.lattice.torus_thermo import torus_thermo_kaufman
from replicant.montecarlo import balance as BL
from replicant.montecarlo import exact_small as XS
from replicant.montecarlo import fk as FK
from replicant.montecarlo import stats as ST
from replicant.montecarlo.run import simulate
from replicant.testing import slow

BANK = json.loads((Path(__file__).resolve().parent.parent / "reference" / "cluster_mc.json").read_text())
SEEDS8, SEEDS16 = tuple(range(8)), tuple(range(100, 116))


@lru_cache(maxsize=None)
def _sim(alg, q, L, sweeps, seeds=SEEDS8, beta="beta_c", burn_in=None):
    return simulate(alg, q, L, beta, sweeps, sweeps // 10 if burn_in is None else burn_in, seeds)


def _inside(r, key, want, systematic=0.0):
    b = r.bands[key]
    return abs(b.mean - want) <= b.half_width + systematic


def _bank(key):
    rec = BANK[key]
    return [float(rec["value"])] + [float(s["value"]) for s in rec.get("sources_disagree", [])], float(rec["tol"])


@slow
def test_small_tori_against_the_enumeration():
    cases = [("swendsen_wang", 2, 3), ("wolff", 2, 3), ("swendsen_wang", 2, 4), ("wolff", 2, 4),
             ("metropolis", 2, 4), ("swendsen_wang", 3, 3), ("wolff", 3, 3)]
    for alg, q, L in cases:
        r = _sim(alg, q, L, 3000, SEEDS16)
        ex = XS.exact_distribution(L, q, "beta_c")
        sysC = 2 * r.tau["e"] / r.sweeps * ex.C
        for key, want, s in (("e", ex.e, 0.0), ("abs_m", ex.abs_m, 0.0), ("m2", ex.m2, 0.0), ("C", ex.C, sysC)):
            assert _inside(r, key, want, s), (alg, q, L, key, r.bands[key], want)
        if alg == "swendsen_wang":
            assert _inside(r, "m2_improved", ex.m2), (q, L, r.bands["m2_improved"], ex.m2)
        assert r.status == "equilibrated", (alg, q, L, r.equilibration.text)
    holds, detail = XS.kaufman_histogram_identity(4)
    assert holds, detail
    for alg in ("swendsen_wang", "wolff", "metropolis"):
        assert BL.stationarity_chi2(alg, 2, 4, "beta_c", 3000, 11, 0.01).ok, alg
        assert BL.ergodicity_chi2(alg, 2, 4, "beta_c", 2000, 30, 11, 0.01).ok, alg
    for alg in ("swendsen_wang", "wolff"):
        assert BL.stationarity_chi2(alg, 3, 3, "beta_c", 3000, 11, 0.01).ok, alg
        assert BL.ergodicity_chi2(alg, 3, 3, "beta_c", 2000, 20, 11, 0.01).ok, alg


@slow
def test_energy_and_specific_heat_against_kaufman_and_a_planted_bond_probability():
    for alg, Ls in (("swendsen_wang", (8, 16, 32)), ("wolff", (8, 16, 32))):
        compared = 0
        for L in Ls:
            r = _sim(alg, 2, L, 4000)
            if r.status != "equilibrated":          # an undetermined run never feeds an asserted band
                print(f"{alg} L = {L}: undetermined, not compared ({r.equilibration.text})")
                continue
            compared += 1
            u, c = torus_thermo_kaufman(L, L, "beta_c")
            e, C = float(u.mid()), float(c.mid())
            sysC = 2 * r.tau["e"] / r.sweeps * C          # the chain-variance bias of C (stated systematic)
            assert _inside(r, "e", e) and _inside(r, "C", C, sysC), (alg, L, r.bands["e"], e, r.bands["C"], C)
            print(f"{alg} L = {L}: <e> {r.mean['e']:.5f} +- {r.bands['e'].half_width:.5f} (Kaufman {e:.5f}); C "
                  f"{r.mean['C']:.4f} +- {r.bands['C'].half_width:.4f} (Kaufman {C:.4f}); <|m|> {r.mean['abs_m']:.4f} "
                  f"+- {r.bands['abs_m'].half_width:.4f}, <m^2> {r.mean['m2']:.4f} +- {r.bands['m2'].half_width:.4f} "
                  "(no exact target)")
        assert compared >= 2, (alg, compared)
    saved = FK._HOOKS["coupling_factor"]
    FK._HOOKS["coupling_factor"] = 0.5
    try:
        bad = simulate("swendsen_wang", 2, 16, "beta_c", 1000, 100, SEEDS8)
    finally:
        FK._HOOKS["coupling_factor"] = saved
        assert FK._HOOKS["coupling_factor"] is saved
    u, _ = torus_thermo_kaufman(16, 16, "beta_c")
    assert not _inside(bad, "e", float(u.mid())), bad.bands["e"]


# what each banked z measures, printed next to the comparison (the estimators differ from this package's tau_int)
ESTIMATOR = {"metropolis_z_2d_ising": "Nightingale & Blote: tau_exp of heat-bath (random-site) dynamics from the "
                                      "subdominant eigenvalue; LPS: Kibble-Zurek quench scaling of Metropolis; the "
                                      "identification with Metropolis tau_int(m^2) is ASSUMED (universality of local "
                                      "single-spin-flip dynamics), not proved",
             "sw_z_2d_ising": "LPS 0.297: Kibble-Zurek quench scaling of <m^2>, a different estimator; 0.25 and 0.35: "
                              "z_int,E as quoted by Salas & Sokal (secondary)",
             "wolff_z_2d_ising": "LPS: quench scaling, single-cluster time shifted to sweeps by their Eq. (27)",
             "potts3_sw_z_int_e": "Salas & Sokal Eq. (5.12): z_int,E, fit L >= 128 (this run: L <= 32)"}


def _z(alg, q, Ls, sweeps, obs, burn_in=None):
    """The fit over the runs labelled equilibrated only (an undetermined run never feeds a band); the dropped sizes
    and their diagnostics are returned and printed.  None when fewer than 3 sizes remain."""
    runs = {L: _sim(alg, q, L, sweeps, burn_in=(burn_in or {}).get(L)) for L in Ls}
    kept = [L for L in Ls if runs[L].status == "equilibrated"]
    dropped = {L: runs[L].equilibration.text for L in Ls if L not in kept}
    taus = np.array([runs[L].per_seed[f"tau_{obs}"] for L in kept]).T
    return (ST.power_fit(kept, taus) if len(kept) >= 3 else None), kept, dropped


ZSEP = 1.5      # the separation asserted: Metropolis's z band lies above it, every cluster algorithm's below it


def _report(name, fit, kept, dropped, key):
    vals, _ = _bank(key)
    if dropped:
        print(f"{name}: dropped as undetermined (not fitted): {dropped}")
    if fit is None:
        print(f"{name}: fewer than 3 equilibrated sizes ({kept}): no fit")
        return
    hit = [v for v in vals if abs(v - fit.z.mean) <= fit.z.half_width]       # the band alone, no tolerance
    print(f"{name}: {fit.text}; literature {vals} inside the 99% band: {hit or 'none'} ({ESTIMATOR[key]})")


@slow
def test_metropolis_z_fit():
    # burn-in 10^4 sweeps at L = 64 (60 tau_int(m^2)); 4000 left the energy drifting (fix round 1)
    metro, mk, md = _z("metropolis", 2, (16, 32, 64), 40000, "m2", burn_in={64: 10000})
    _report("Metropolis tau_int(m^2)", metro, mk, md, "metropolis_z_2d_ising")
    # asserted over equilibrated runs only, and every Metropolis size must be equilibrated
    assert metro is not None and not md, md
    assert metro.separable_from_zero and metro.z.mean - metro.z.half_width > ZSEP, metro.text
    vals, _ = _bank("metropolis_z_2d_ising")
    assert any(abs(v - metro.z.mean) <= metro.z.half_width for v in vals), metro.text   # assumes universality


@slow
def test_cluster_z_fits():
    """SW and Wolff: fitted over the equilibrated sizes (>= 3 required), asserted only below ZSEP (so below the
    Metropolis band of test_metropolis_z_fit); the literature comparison is printed.  Wolff has five sizes so that
    one undetermined size (on these seeds L = 8 drifts at the 1% level) still leaves a fit."""
    sw = _z("swendsen_wang", 2, (8, 16, 32, 64), 4000, "e")
    wolff = _z("wolff", 2, (8, 12, 16, 24, 32), 4000, "e")
    potts = _z("swendsen_wang", 3, (8, 16, 32), 4000, "e")
    _report("SW tau_int(e)", *sw, "sw_z_2d_ising")
    _report("Wolff tau_int(e), sweep units", *wolff, "wolff_z_2d_ising")
    _report("SW q = 3 tau_int(e)", *potts, "potts3_sw_z_int_e")
    for fit, kept, dropped in (sw, wolff):
        assert fit is not None, dropped
        assert fit.z.mean + fit.z.half_width < ZSEP, fit.text
