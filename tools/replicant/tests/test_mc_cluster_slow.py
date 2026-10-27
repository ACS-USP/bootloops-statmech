"""M1, slow tier: cluster algorithms on larger tori, on named seeds.
  - exact small tori (L = 3, 4; Potts q = 3 at L = 3): <e>, <|m|>, <m^2>, C of SW / Wolff / Metropolis against the
    enumeration in 99% Student-t bands over 16 seeds, SW's FK estimator of <m^2>; ergodicity and stationarity chi^2
    at L = 4 (alpha = 0.01); Kaufman's integer polynomial = the 2^16-state enumeration
  - L = 8, 16, 32 at beta_c: <e> and C against Kaufman's torus values (torus_thermo_kaufman) in the band plus the
    stated variance bias 2 tau / n of C; <|m|>, <m^2> printed with bands (no exact target: ruling A-R6); a planted
    wrong bond probability is rejected
  - critical slowing down: tau_int at beta_c against L for Metropolis (L = 16, 32, 64), SW (8 .. 64), Wolff
    (8 .. 32) and SW at q = 3 (8 .. 32); z per seed in a Student-t band, printed next to the fetched literature
    values (reference/cluster_mc.json); asserted only where the band decides: Metropolis's z excludes 0 and the
    cluster algorithms' bands, and contains a literature value; SW / Wolff are printed (their bands cannot
    separate z from a logarithm at these sizes)
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
def _sim(alg, q, L, sweeps, seeds=SEEDS8, beta="beta_c"):
    return simulate(alg, q, L, beta, sweeps, sweeps // 10, seeds)


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
        for L in Ls:
            r = _sim(alg, 2, L, 4000)
            u, c = torus_thermo_kaufman(L, L, "beta_c")
            e, C = float(u.mid()), float(c.mid())
            sysC = 2 * r.tau["e"] / r.sweeps * C          # the chain-variance bias of C (stated systematic)
            assert _inside(r, "e", e) and _inside(r, "C", C, sysC), (alg, L, r.bands["e"], e, r.bands["C"], C)
            print(f"{alg} L = {L}: <e> {r.mean['e']:.5f} +- {r.bands['e'].half_width:.5f} (Kaufman {e:.5f}); C "
                  f"{r.mean['C']:.4f} +- {r.bands['C'].half_width:.4f} (Kaufman {C:.4f}); <|m|> {r.mean['abs_m']:.4f} "
                  f"+- {r.bands['abs_m'].half_width:.4f}, <m^2> {r.mean['m2']:.4f} +- {r.bands['m2'].half_width:.4f} "
                  "(no exact target)")
    saved = FK._HOOKS["coupling_factor"]
    FK._HOOKS["coupling_factor"] = 0.5
    try:
        bad = simulate("swendsen_wang", 2, 16, "beta_c", 1000, 100, SEEDS8)
    finally:
        FK._HOOKS["coupling_factor"] = saved
        assert FK._HOOKS["coupling_factor"] is saved
    u, _ = torus_thermo_kaufman(16, 16, "beta_c")
    assert not _inside(bad, "e", float(u.mid())), bad.bands["e"]


def _z(alg, q, Ls, sweeps, obs):
    runs = [_sim(alg, q, L, sweeps) for L in Ls]
    return ST.power_fit(Ls, np.array([r.per_seed[f"tau_{obs}"] for r in runs]).T), runs


@slow
def test_critical_slowing_down_z_fits():
    metro, mruns = _z("metropolis", 2, (16, 32, 64), 40000, "m2")
    sw, _ = _z("swendsen_wang", 2, (8, 16, 32, 64), 4000, "e")
    wolff, _ = _z("wolff", 2, (8, 16, 32), 4000, "e")
    potts, _ = _z("swendsen_wang", 3, (8, 16, 32), 4000, "e")
    for name, fit, key in (("Metropolis tau_int(m^2)", metro, "metropolis_z_2d_ising"),
                           ("SW tau_int(e)", sw, "sw_z_2d_ising"), ("Wolff tau_int(e), sweep units", wolff,
                                                                     "wolff_z_2d_ising"),
                           ("SW q = 3 tau_int(e)", potts, "potts3_sw_z_int_e")):
        vals, tol = _bank(key)
        hit = [v for v in vals if abs(v - fit.z.mean) <= fit.z.half_width + tol]
        print(f"{name}: {fit.text}; literature {vals} (tol {tol}): inside the band: {hit or 'none'}")
    # asserted where the band decides: the local dynamics is critically slow, far beyond the cluster algorithms
    assert metro.separable_from_zero and metro.z.mean - metro.z.half_width > 1.5
    for fit in (sw, wolff):
        assert fit.z.mean + fit.z.half_width < metro.z.mean - metro.z.half_width
    vals, tol = _bank("metropolis_z_2d_ising")
    assert any(abs(v - metro.z.mean) <= metro.z.half_width + tol for v in vals), metro.text
    assert all(r.status == "equilibrated" for r in mruns[:2]), [r.equilibration.text for r in mruns]
